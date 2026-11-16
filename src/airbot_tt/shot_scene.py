"""Driven AIRBOT and low-spin ball sharing one PhysX scene."""
import json
import numpy as np
from pxr import UsdShade, UsdPhysics
import isaaclab.sim as sim_utils
from .ball import load_ball_config
from .ball_scene import BallTestbed
from .config import ROOT
from .runtime import AirbotSimulation, ROBOT_PRIM
from .tool import racket_model
from .single_shot import ShotEvents
from isaaclab.utils.math import quat_rotate_inverse


class SingleShot:
    def __init__(self, device='cpu'):
        self.config = json.loads((ROOT/'configs/single_shot.json').read_text())
        self.ball_config = c = load_ball_config()
        self.events = ShotEvents(c, self.config['timeout_s'])
        self.running = False
        self.time = 0.
        self.state = None
        model = racket_model(ROOT/'configs/racket.json')
        self.body_names = list(model.links)
        self.filters = ['/World/Table', '/World/Net'] + [f'{ROBOT_PRIM}/{n}' for n in self.body_names]
        self.control = AirbotSimulation(device=device, model=model, base_position=self.config['base_position_m'],
                                       initial=self.config['initial_q_rad'], task_scene=self, dt=1/self.config['physics_hz'])
        # The shared table/net geometry is also used by offline path preflight.
        from .collision import CollisionChecker
        obstacles = [dict(name='table', center_m=[0,0,c['table_top_m']-c['table_size_m'][2]/2], size_m=c['table_size_m']),
                     dict(name='net', center_m=[0,0,c['table_top_m']+c['net_height_m']/2],
                          size_m=[c['net_thickness_m'],c['table_size_m'][1],c['net_height_m']])]
        self.control.checker = CollisionChecker(model, obstacles, base_position=self.config['base_position_m'],
                                                base_mount_obstacle='table')
        self.control.checker.require_clear(self.control.initial)

    def build(self, control):
        self.world = BallTestbed(control.dt, str(control.sim.device), sim=control.sim, static_paddle=False,
                                contact_filters=self.filters)
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=0, dynamic_friction=0,
                restitution=self.ball_config['paddle_restitution'], restitution_combine_mode='min', friction_combine_mode='min')
        mat.func('/World/ShotPaddleMaterial',mat)
        stage=control.sim.stage
        # Fixed mounting interface only; table contacts with moving links remain active.
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(ROBOT_PRIM+'/base_link')).CreateFilteredPairsRel().AddTarget('/World/Table')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(ROBOT_PRIM+'/paddle/collisions'):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/ShotPaddleMaterial'),materialPurpose='physics')

    def prepare_step(self):
        self.world.prepare_step()

    def after_step(self, control, forces):
        self.state, _ = self.world.read_step()
        matrix = self.world.sensor.data.force_matrix_w[0,0].cpu().numpy()
        names=['table','net']+self.body_names
        touching={name for name,force in zip(names,matrix) if np.linalg.norm(force)>1e-5}
        if touching.intersection(set(self.body_names)-{'paddle','fixer'}):
            touching.add('arm')
        # Action/reaction: subtract only the measured ball-paddle contribution.
        # Other contacts on that paddle and all other bodies remain guarded.
        adjusted=forces.copy()
        paddle_force=matrix[names.index('paddle')]
        sensor_index=control.sensor.body_names.index('paddle')
        adjusted[sensor_index] += paddle_force
        if 'paddle' in touching:
            # Contact with the handle is not a valid striking-face event.
            index=control.robot.body_names.index('paddle')
            local=quat_rotate_inverse(control.robot.data.body_quat_w[:,index],
                self.world.ball.data.root_pos_w-control.robot.data.body_pos_w[:,index])[0].cpu().numpy()
            split=control.model.tool_geometry['components']['paddle']['collision_split_link6_z_m']
            if local[2] < split:
                touching.add('handle')
        self.last_contacts=touching
        if self.running:
            self.time += control.dt
            self.events.update(self.time,self.state,touching,control.fault)
        return adjusted

    def reset(self, serve):
        self.running=False
        self.control.reset()
        self.world.reset([3,0,2],[0,0,0])
        for _ in range(round(self.config['settle_s']/self.control.dt)):
            self.control.step()
        if self.control.fault:
            raise RuntimeError('Reset settling failed: '+self.control.fault)
        self.events.reset()
        self.time=0.
        self.world.reset(serve['position_m'],serve['velocity_m_s'])
        self.state=np.r_[serve['position_m'],serve['velocity_m_s']]
        self.running=True

    def forecast(self, horizon_s=2.0, *, first_table_event='bounce'):
        """Forecast from current measurement; future robot contacts are not modeled."""
        from .prediction import predict_ball
        if self.state is None or not self.running:
            raise RuntimeError('Reset a serve before forecasting.')
        return predict_ball(self.state, horizon_s, config=self.ball_config,
                            first_table_event=first_table_event)

    def step(self):
        if not self.running:
            raise RuntimeError('Reset a serve before stepping.')
        if self.events.outcome is not None:
            raise RuntimeError('Episode terminated; reset before stepping.')
        self.control.step()
        if self.control.fault and self.events.outcome is None:
            self.events.update(self.time,self.state,arm_fault=self.control.fault)
        return dict(time_s=self.time,ball_state=self.state.copy(),phase=self.events.phase,
                    ball_omega_rad_s=self.world.ball.data.root_ang_vel_w[0].cpu().numpy().copy(),
                    q_rad=self.control.q(),qd_rad_s=self.control.qd(),
                    outcome=self.events.outcome,contacts=sorted(self.last_contacts),
                    terminated=self.events.outcome is not None and self.events.outcome!='timeout',
                    truncated=self.events.outcome=='timeout', reward=float(self.events.outcome=='legal_return'))
