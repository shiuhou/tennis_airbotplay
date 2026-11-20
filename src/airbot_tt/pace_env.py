"""M7 fixed-arm vector scene using Isaac Lab InteractiveScene.

Explicit terminal observations precede partial resets. No normal-step state writes.
The initial A/B integration uses CPU teacher only at serve reset; no online MLP.
"""
import json
import math
import numpy as np
import torch
from pxr import UsdPhysics, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import combine_frame_transforms, quat_rotate, quat_rotate_inverse
from .config import ROOT, CONTACT_TRIP_N
from .runtime import _articulation_config, configure_racket_colliders
from .tool import racket_model
from .ball_scene import BallTestbed
from .pace_control import PositionCommand, TensorShotEvents
from .pace_guidance import apex_teacher


class PaceEnv:
    def __init__(self,num_envs=1,device='cuda:0',seed=7101,group='A',decimation=None,serve_y_half_range_m=None,serve_z_m=None):
        if group not in ('A','B'): raise ValueError('Only initial A/B integration is implemented')
        self.cfg=json.loads((ROOT/'configs/pace_rl.json').read_text())
        self.strike=json.loads((ROOT/'configs/strike.json').read_text())
        if serve_y_half_range_m is not None:
            if not np.isfinite(serve_y_half_range_m) or serve_y_half_range_m<0:
                raise ValueError('Serve Y half range must be finite and nonnegative')
            self.strike['position_half_range_m'][1]=float(serve_y_half_range_m)
        if serve_z_m is not None:
            if not np.isfinite(serve_z_m) or serve_z_m<=0:
                raise ValueError('Serve Z must be finite and positive')
            self.strike['serve_position_m'][2]=float(serve_z_m)
        self.shot=json.loads((ROOT/'configs/single_shot.json').read_text())
        self.guidance=json.loads((ROOT/'configs/pace.json').read_text())['reward']
        self.n=num_envs;self.device=device;self.group=group
        self.decimation=self.cfg['decimation'] if decimation is None else decimation
        if self.decimation<=0 or self.cfg['physics_hz']%self.decimation: raise ValueError('Integer fixed policy rate required')
        self.dt=1/self.cfg['physics_hz'];self.policy_dt=self.dt*self.decimation
        self.model=racket_model(ROOT/'configs/racket.json')
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt,device=device,
            physx=sim_utils.PhysxCfg(enable_ccd=True,bounce_threshold_velocity=.01)))
        self.scene=InteractiveScene(InteractiveSceneCfg(num_envs=num_envs,env_spacing=self.cfg['env_spacing_m'],replicate_physics=True))
        root='/World/envs/env_.*';source='/World/envs/env_0'
        art_cfg=_articulation_config(self.model,self.shot['base_position_m'],self.strike['ready_q_rad'],self.strike['simulation_speed_limit_rad_s'])
        art_cfg.prim_path=root+'/AIRBOT'
        self.robot=Articulation(art_cfg)
        configure_racket_colliders(self.model,source+'/AIRBOT')
        filters=[root+'/Table',root+'/Net']+[root+'/AIRBOT/'+n for n in self.model.links]
        self.world=BallTestbed(self.dt,device,sim=self.sim,static_paddle=False,contact_filters=filters,root=root)
        self.ball=self.world.ball;self.ball_sensor=self.world.sensor;self.bc=self.world.config
        mat=sim_utils.RigidBodyMaterialCfg(static_friction=0,dynamic_friction=0,restitution=self.bc['paddle_restitution'],
                                           restitution_combine_mode='min',friction_combine_mode='min')
        mat.func('/World/ShotPaddleMaterial',mat)
        stage=self.sim.stage
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(source+'/AIRBOT/base_link')).CreateFilteredPairsRel().AddTarget(source+'/Table')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(source+'/AIRBOT/paddle/collisions'):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/ShotPaddleMaterial'),materialPurpose='physics')
        self.arm_sensor=ContactSensor(ContactSensorCfg(prim_path=root+'/AIRBOT/(base_link|link[1-6]|fixer|paddle)',update_period=0))
        self.scene.articulations['robot']=self.robot
        self.scene.rigid_objects['ball']=self.ball
        self.scene.sensors['ball']=self.ball_sensor;self.scene.sensors['arm']=self.arm_sensor
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions([])
        light=sim_utils.DomeLightCfg(intensity=1500);light.func('/World/Light',light)
        self.sim.reset()
        self.origins=self.scene.env_origins
        self.ids=torch.arange(num_envs,device=device)
        self.joints,names=self.robot.find_joints(list(self.model.joint_names),preserve_order=True)
        if tuple(names)!=tuple(self.model.joint_names) or self.robot.num_joints!=6: raise RuntimeError('Wrong joint schema')
        self.paddle=self.robot.body_names.index('paddle')
        self.contact_paddle=self.arm_sensor.body_names.index('paddle')
        self.ball_paddle=2+list(self.model.links).index('paddle')
        self.ball_fixer=2+list(self.model.links).index('fixer')
        self.ball_arm=[2+i for i,n in enumerate(self.model.links) if n not in ('paddle','fixer')]
        self.nominal=self.tensor(self.strike['ready_q_rad'])
        self.command=PositionCommand(self.nominal,self.tensor(self.cfg['action_scale_rad']),self.tensor(self.model.lower),
            self.tensor(self.model.upper),num_envs,self.strike['velocity_limit_rad_s'],self.strike['acceleration_limit_rad_s2'],self.dt,self.policy_dt)
        self.effort=self.tensor(self.model.effort)
        self.tcp_offset=self.tensor(self.model.link6_tcp[:3,3]).repeat(num_envs,1)
        from .kinematics import pose_dict
        self.tcp_rotation=self.tensor(pose_dict(self.model.link6_tcp)['quaternion_wxyz']).repeat(num_envs,1)
        self.events=TensorShotEvents(num_envs,device,self.bc,self.shot['timeout_s'])
        self.elapsed=torch.zeros(num_envs,device=device)
        self.previous_action=torch.zeros(num_envs,6,device=device)
        self.actor_history=torch.zeros(num_envs,5,23,device=device)
        self.critic_history=torch.zeros(num_envs,5,40,device=device)
        self.teacher_point=torch.zeros(num_envs,3,device=device)
        self.teacher_time=torch.zeros(num_envs,device=device)
        self.teacher_valid=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.serve_id=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(num_envs)]
        self.state_writes=0
        self.distance_min=torch.full((num_envs,),float('inf'),device=device)
        self.valid_seconds=torch.zeros(num_envs,device=device);self.plateau_seconds=self.valid_seconds.clone()
        self.episode_reward=self.valid_seconds.clone();self.episode_guidance=self.valid_seconds.clone()
        self.reset(self.ids)

    def tensor(self,x):
        if isinstance(x,torch.Tensor):return x.to(device=self.device,dtype=torch.float32)
        return torch.tensor(np.array(x,copy=True),dtype=torch.float32,device=self.device)

    def state(self):
        return torch.cat((self.ball.data.root_pos_w-self.origins,self.ball.data.root_lin_vel_w),dim=1)

    def tcp(self):
        p,q=combine_frame_transforms(self.robot.data.body_link_pos_w[:,self.paddle],self.robot.data.body_link_quat_w[:,self.paddle],self.tcp_offset,self.tcp_rotation)
        # Body linear velocity is measured at COM; convert it to TCP.
        r=p-self.robot.data.body_com_pos_w[:,self.paddle]
        v=self.robot.data.body_lin_vel_w[:,self.paddle]+torch.cross(self.robot.data.body_ang_vel_w[:,self.paddle],r,dim=-1)
        return p-self.origins,q,v

    def reset(self,ids,serve=None):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        if ids.numel()==0:return
        values=[]
        for i in ids.cpu().tolist():
            c=self.strike
            values.append(np.r_[np.array(c['serve_position_m'])+self.rng[i].uniform(-1,1,3)*c['position_half_range_m'],
                                np.array(c['serve_velocity_m_s'])+self.rng[i].uniform(-1,1,3)*c['velocity_half_range_m_s']])
        states=self.tensor(values) if serve is None else self.tensor(serve).reshape(-1,6).expand(len(ids),-1)
        # Reset-only CPU reference: no per-policy-step SciPy loop, no hidden online predictor.
        teachers=[apex_teacher(row) for row in states.cpu().numpy()]
        self.teacher_point[ids]=self.tensor([t['point'] for t in teachers])
        self.teacher_time[ids]=self.tensor([t['time_s'] for t in teachers])
        self.teacher_valid[ids]=torch.tensor([t['valid'] for t in teachers],device=self.device)
        q=self.nominal.repeat(len(ids),1)
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=self.joints,env_ids=ids)
        ball=self.ball.data.default_root_state[ids].clone()
        ball[:,:3]=states[:,:3]+self.origins[ids];ball[:,7:10]=states[:,3:];ball[:,10:]=0
        self.ball.write_root_state_to_sim(ball,env_ids=ids)
        self.scene.reset(ids);self.state_writes+=1
        self.sim.forward()  # Lab reset contract: update link kinematics without advancing physics.
        self.command.reset(ids);self.elapsed[ids]=0;
        self.previous_action[ids]=0;self.events.reset(ids);self.serve_id[ids]+=1
        self.distance_min[ids]=float('inf');self.valid_seconds[ids]=0;self.plateau_seconds[ids]=0;self.episode_reward[ids]=0;self.episode_guidance[ids]=0
        actor,critic=self.frames()
        self.actor_history[ids]=actor[ids,None,:].expand(-1,5,-1)
        self.critic_history[ids]=critic[ids,None,:].expand(-1,5,-1)
        return self.observations()

    def frames(self):
        ball=self.state();p,q,v=self.tcp()
        valid=self.teacher_valid & (self.elapsed<self.teacher_time) & (self.events.phase<2) & (self.events.outcome==0)
        actor=torch.cat((self.robot.data.joint_pos[:,self.joints]-self.nominal,self.robot.data.joint_vel[:,self.joints],
                         self.previous_action,ball[:,:3],torch.ones(self.n,1,device=self.device),torch.zeros(self.n,1,device=self.device)),1)
        critic=torch.cat((actor[:,:18],ball,self.teacher_point*valid[:,None],valid[:,None],
                          torch.clamp(self.teacher_time-self.elapsed,min=0)[:,None]*valid[:,None],p,v,
                          torch.nn.functional.one_hot(torch.where(self.events.outcome>0,3,self.events.phase.clamp_max(2)),4), (self.elapsed/self.shot['timeout_s'])[:,None]),1)
        return actor.clamp(-self.cfg['observation_clip'],self.cfg['observation_clip']),critic.clamp(-self.cfg['observation_clip'],self.cfg['observation_clip'])

    def observations(self):return self.actor_history.flatten(1).clone(),self.critic_history.flatten(1).clone()

    def physics_step(self):
        q,v=self.command.step()
        delayed=self.elapsed<self.strike['reaction_delay_s']
        q=torch.where(delayed[:,None],self.nominal,q);v=torch.where(delayed[:,None],torch.zeros_like(v),v)
        self.command.q=q;self.command.v=v
        gravity=self.robot.root_physx_view.get_gravity_compensation_forces()[:,self.joints].clamp(-self.effort,self.effort)
        self.robot.set_joint_effort_target(gravity,joint_ids=self.joints)
        self.robot.set_joint_position_target(q,joint_ids=self.joints);self.robot.set_joint_velocity_target(v,joint_ids=self.joints)
        velocity=self.ball.data.root_lin_vel_w
        c=self.bc;coefficient=.5*c['air_density_kg_m3']*c['drag_coefficient']*math.pi*c['radius_m']**2
        drag=-coefficient*torch.linalg.vector_norm(velocity,dim=1,keepdim=True)*velocity
        force=quat_rotate_inverse(self.ball.data.root_quat_w,drag)
        self.ball.set_external_force_and_torque(force[:,None,:],torch.zeros_like(force[:,None,:]))
        self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.dt)
        state=self.state();matrix=self.ball_sensor.data.force_matrix_w[:,0]
        touching=torch.linalg.vector_norm(matrix,dim=-1)>1e-5
        local=quat_rotate_inverse(self.robot.data.body_quat_w[:,self.paddle],self.ball.data.root_pos_w-self.robot.data.body_pos_w[:,self.paddle])
        handle=touching[:,self.ball_paddle] & (local[:,2]<self.model.tool_geometry['components']['paddle']['collision_split_link6_z_m'])
        contacts=torch.stack((touching[:,0],touching[:,1],touching[:,self.ball_paddle],touching[:,self.ball_fixer]|touching[:,self.ball_arm].any(1)|handle),1)
        forces=self.arm_sensor.data.net_forces_w.clone();forces[:,self.contact_paddle]+=matrix[:,self.ball_paddle]
        positions=self.robot.data.joint_pos[:,self.joints]
        fault=(torch.linalg.vector_norm(forces,dim=-1).amax(1)>CONTACT_TRIP_N)|((positions<self.command.lower-.004)|(positions>self.command.upper+.004)).any(1)
        if not torch.isfinite(state).all() or not torch.isfinite(positions).all():raise RuntimeError('Non-finite physical state')
        self.elapsed+=self.dt
        first_hit=self.events.update(self.elapsed,state,contacts,fault)
        p,_,_=self.tcp()
        distance=torch.linalg.vector_norm(p-self.teacher_point,dim=1)
        self.distance_min=torch.minimum(self.distance_min,torch.linalg.vector_norm(p-state[:,:3],dim=1))
        valid=self.teacher_valid & (self.elapsed<self.teacher_time) & (self.events.phase<2) & (self.events.outcome==0)
        self.valid_seconds+=valid*self.dt;self.plateau_seconds+=(valid & (distance<=.15))*self.dt
        g=self.guidance
        self.episode_guidance+=torch.exp(-distance.clamp_min(g['hit_floor_m'])/g['hit_std_m']**2)*valid*g['hit_weight']*self.dt
        return first_hit

    def step(self,action,*,auto_reset=True):
        if action.shape!=(self.n,6) or not torch.isfinite(action).all():raise ValueError('Expected finite Nx6 action')
        if (self.events.outcome>0).any():raise RuntimeError('Reset finished episodes before stepping')
        old=self.previous_action.clone();self.command.set_action(action)
        previous_guidance=self.episode_guidance.clone()
        hit=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        for _ in range(self.decimation): hit|=self.physics_step()
        self.previous_action=action.clone()
        p,_,_=self.tcp();state=self.state()
        distance=torch.linalg.vector_norm(p-self.teacher_point,dim=1)
        r=self.cfg['reward'];dense=self.episode_guidance-previous_guidance
        reward=hit*r['contact']+(self.events.outcome==6)*r['legal_return']+r['action_rate']*((action-old)**2).sum(1)*self.policy_dt
        reward+=(self.events.outcome==2)*r['fault']
        if self.group=='B':reward+=dense
        self.episode_reward+=reward
        a,c=self.frames();self.actor_history=torch.cat((self.actor_history[:,1:],a[:,None]),1);self.critic_history=torch.cat((self.critic_history[:,1:],c[:,None]),1)
        terminal=self.observations()
        terminated=(self.events.outcome>0)&(self.events.outcome!=9);truncated=self.events.outcome==9
        done=terminated|truncated
        info=dict(terminal_actor=terminal[0],terminal_critic=terminal[1],terminated=terminated.clone(),truncated=truncated.clone(),
                  q_request=self.command.request.clone(),q_applied=self.command.q.clone(),q_measured=self.robot.data.joint_pos[:,self.joints].clone(),
                  guidance=dense.clone(),apex_distance=distance.clone(),episodes=[])
        ids=done.nonzero().flatten()
        for i in ids.cpu().tolist():
            row=dict(env=i,serve=int(self.serve_id[i]),outcome=self.events.names[int(self.events.outcome[i])],hit=bool(self.events.hit[i]),
                     net_touched=bool(self.events.net_touched[i]),duration_s=float(self.elapsed[i]),
                     landing_xy_m=self.events.landing[i,:2].cpu().tolist() if self.events.outcome[i]==6 else None,
                     min_ball_distance_m=float(self.distance_min[i]),guidance_valid_s=float(self.valid_seconds[i]),
                     plateau_s=float(self.plateau_seconds[i]),guidance_sum=float(self.episode_guidance[i]),reward=float(self.episode_reward[i]),
                     target_success=bool(self.events.outcome[i]==6 and torch.all(abs(self.events.landing[i,:2]-self.tensor([.75,0]))<=self.strike['goal_half_width_m'])),
                     landing_error_m=float(torch.linalg.vector_norm(self.events.landing[i,:2]-self.tensor([.75,0]))) if self.events.outcome[i]==6 else None)
            info['episodes'].append(row)
        if auto_reset and ids.numel():self.reset(ids)
        return (*self.observations(),reward,done,info)
