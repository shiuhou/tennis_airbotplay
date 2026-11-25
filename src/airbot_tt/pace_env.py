"""M7 fixed-arm vector scene using Isaac Lab InteractiveScene.

Explicit terminal observations precede partial resets. No normal-step state writes.
Explicit E-online / E-frozen / E-mean add observable-history hints; critic and D rewards are unchanged.
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
from .pace_control import PositionCommand, TensorShotEvents, TensorReleaseLatch
from .pace_guidance import apex_teacher, return_teacher, guidance_rewards


class PaceEnv:
    def __init__(self,num_envs=1,device='cuda:0',seed=7101,group='A',decimation=None,serve_y_half_range_m=None,serve_z_m=None,serve_bank=None,frozen_predictor=None,mean_predictor=None):
        if group not in ('A','B','C','D','E','E-online','E-frozen','E-mean'):
            raise ValueError('Implemented groups: A, B, C, D, E (online), E-online, E-frozen, E-mean')
        if (group=='E-frozen') != (frozen_predictor is not None):
            raise ValueError('E-frozen requires an explicit frozen predictor artifact; other groups reject it')
        if (group=='E-mean') != (mean_predictor is not None):
            raise ValueError('E-mean requires an explicit mean artifact; other groups reject it')
        self.predictor_mode='mean' if group=='E-mean' else ('frozen' if group=='E-frozen' else ('online' if group in ('E','E-online') else 'none'))
        self.online=None;self.frozen=None
        self.serve_bank=None
        if serve_bank is not None:
            if group!='A' or serve_y_half_range_m is not None or serve_z_m is not None:
                raise ValueError('Fixed serve-bank diagnostic requires A without synthetic serve overrides')
            from .pace_serves import FixedServeBank
            self.serve_bank=FixedServeBank(serve_bank)
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
        if self.predictor_mode in ('frozen','mean'):
            from .pace_predictor import FrozenPredictor,FixedMeanPredictor
            self.frozen=(FixedMeanPredictor(num_envs,artifact=mean_predictor) if self.predictor_mode=='mean'
                         else FrozenPredictor(num_envs,artifact=frozen_predictor))
            self.frozen.validate_environment(self.strike,self.cfg['physics_hz'],self.decimation)
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
        from .pace_predictor import OnlinePredictor
        self.online=OnlinePredictor(num_envs) if self.predictor_mode=='online' else None
        self.prediction_collect=True
        self.actor_width=27 if self.predictor_module is not None else 23
        self.prediction=torch.zeros(num_envs,4,device=device)
        self.policy_index=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.first_prediction=torch.full((num_envs,),-1.,device=device)
        self.first_hint_used=self.first_prediction.clone()
        self.hint_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.prehit_steps=self.hint_steps.clone();self.hit_reward_steps=self.hint_steps.clone()
        self.hint_missing=torch.zeros(num_envs,3,dtype=torch.long,device=device)
        self.hint_reference_steps=self.hint_steps.clone()
        self.hint_error_sq_sum=torch.zeros(num_envs,device=device)
        self.actor_history=torch.zeros(num_envs,5,self.actor_width,device=device)
        self.critic_history=torch.zeros(num_envs,5,40,device=device)
        self.teacher_point=torch.zeros(num_envs,3,device=device)
        self.teacher_time=torch.zeros(num_envs,device=device)
        self.teacher_valid=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.serve_id=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(num_envs)]
        self.bank_index=np.full(num_envs,-1,dtype=np.int64)
        self.bank_initial_states=np.zeros((num_envs,6),dtype=float)
        self.bank_draws=[]
        self.bank_cycle=None
        if self.serve_bank is not None:self.seed_bank(seed)
        self.state_writes=0
        self.distance_min=torch.full((num_envs,),float('inf'),device=device)
        self.valid_seconds=torch.zeros(num_envs,device=device);self.plateau_seconds=self.valid_seconds.clone()
        self.episode_reward=self.valid_seconds.clone();self.episode_guidance=self.valid_seconds.clone()
        self.release=TensorReleaseLatch(num_envs,device)
        self.episode_return=self.valid_seconds.clone()
        self.return_prediction_valid=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.return_records=[None for _ in range(num_envs)]
        self.reset(self.ids)

    @property
    def predictor_module(self):
        return self.frozen if self.predictor_mode in ('frozen','mean') else self.online

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

    def seed_bank(self,seed):
        """Restart bank schedules/logs; caller must next reset all environments."""
        if self.serve_bank is None:raise ValueError('No fixed serve bank is configured')
        from .pace_serves import ShuffledServeCycles
        self.bank_cycle=ShuffledServeCycles(len(self.serve_bank.rows),self.n,seed)
        self.bank_index.fill(-1)
        self.bank_initial_states.fill(0)
        self.bank_draws=[]

    def reset(self,ids,serve=None,bank_indices=None):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        if ids.numel()==0:return
        selected=ids.cpu().tolist()
        if self.serve_bank is None:
            if bank_indices is not None:raise ValueError('Bank indices require a fixed serve bank')
            # Preserve the original synthetic RNG consumption, including explicit-state resets.
            values=[]
            for i in selected:
                c=self.strike
                values.append(np.r_[np.array(c['serve_position_m'])+self.rng[i].uniform(-1,1,3)*c['position_half_range_m'],
                                    np.array(c['serve_velocity_m_s'])+self.rng[i].uniform(-1,1,3)*c['velocity_half_range_m_s']])
            states=self.tensor(values) if serve is None else self.tensor(serve).reshape(-1,6).expand(len(ids),-1)
        else:
            if serve is not None:raise ValueError('Use bank_indices for explicit fixed-bank evaluation')
            if bank_indices is None:
                indices=np.array([self.bank_cycle.next(i) for i in selected],dtype=np.int64)
            else:
                if isinstance(bank_indices,torch.Tensor):bank_indices=bank_indices.detach().cpu().numpy()
                indices=np.asarray(bank_indices)
                if indices.shape!=(len(selected),) or not np.issubdtype(indices.dtype,np.integer):
                    raise ValueError('Provide one integer bank index per selected environment')
                if np.any(indices<0) or np.any(indices>=len(self.serve_bank.rows)):
                    raise ValueError('Bank index out of range')
            values=self.serve_bank.states[indices]
            self.bank_index[selected]=indices
            self.bank_initial_states[selected]=values
            for i,index in zip(selected,indices):
                self.bank_draws.append(dict(env=i,raw_id=self.serve_bank.rows[int(index)]['id'],
                    serve=int(self.serve_id[i])+1,explicit=bank_indices is not None))
            states=self.tensor(values)
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
        self.release.reset(ids);self.episode_return[ids]=0;self.return_prediction_valid[ids]=False
        for i in ids.cpu().tolist():self.return_records[i]=None
        self.policy_index[ids]=0;self.prediction[ids]=0
        self.first_prediction[ids]=-1;self.first_hint_used[ids]=-1
        self.hint_steps[ids]=0;self.prehit_steps[ids]=0;self.hit_reward_steps[ids]=0;self.hint_missing[ids]=0
        self.hint_reference_steps[ids]=0;self.hint_error_sq_sum[ids]=0
        if self.predictor_module is not None:
            self.predictor_module.hook.reset(ids.cpu().tolist());self.refresh_prediction(ids)
        actor,critic=self.frames()
        self.actor_history[ids]=actor[ids,None,:].expand(-1,5,-1)
        self.critic_history[ids]=critic[ids,None,:].expand(-1,5,-1)
        return self.observations()

    def refresh_prediction(self,ids=None):
        selected=self.ids if ids is None else ids
        positions=self.state()[:,:3].detach().cpu().numpy()
        # Integer policy indices avoid floating-point drift from 960 Hz elapsed sums.
        times=self.policy_index.cpu().numpy()*self.policy_dt
        output=self.predictor_module.observe(positions,times,self.serve_id.cpu().numpy(),
                                   np.isfinite(positions).all(1),selected.cpu().tolist())
        self.prediction[selected]=output.to(self.device)[selected]
        first=(self.first_prediction[selected]<0)&(self.prediction[selected,3]>0)
        chosen=selected[first];self.first_prediction[chosen]=self.policy_index[chosen]*self.policy_dt
        if ids is None and self.prediction_collect and self.online is not None:
            label_valid=self.teacher_valid & (self.elapsed<self.teacher_time) & (self.events.phase<2) & (self.events.outcome==0)
            self.online.collect(self.teacher_point.cpu().numpy(),label_valid.cpu().numpy())

    def refresh_after_predictor_update(self):
        if self.online is None:
            raise RuntimeError('Only E-online may refresh after a predictor update')
        # Recompute only the latest frame; do not append the same sample twice.
        self.prediction=self.online.hook.current().to(self.device)
        first=(self.first_prediction<0)&(self.prediction[:,3]>0)
        self.first_prediction[first]=self.policy_index[first]*self.policy_dt
        self.actor_history[:,-1,23:]=self.prediction.clamp(-self.cfg['observation_clip'],self.cfg['observation_clip'])
        return self.observations()

    def frames(self):
        ball=self.state();p,q,v=self.tcp()
        valid=self.teacher_valid & (self.elapsed<self.teacher_time) & (self.events.phase<2) & (self.events.outcome==0)
        actor=torch.cat((self.robot.data.joint_pos[:,self.joints]-self.nominal,self.robot.data.joint_vel[:,self.joints],
                         self.previous_action,ball[:,:3],torch.ones(self.n,1,device=self.device),torch.zeros(self.n,1,device=self.device)),1)
        critic=torch.cat((actor[:,:18],ball,self.teacher_point*valid[:,None],valid[:,None],
                          torch.clamp(self.teacher_time-self.elapsed,min=0)[:,None]*valid[:,None],p,v,
                          torch.nn.functional.one_hot(torch.where(self.events.outcome>0,3,self.events.phase.clamp_max(2)),4), (self.elapsed/self.shot['timeout_s'])[:,None]),1)
        if self.predictor_module is not None:actor=torch.cat((actor,self.prediction),1)
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
        if self.group in ('C','D','E','E-online','E-frozen','E-mean'):
            tcp,quat,tcp_velocity=self.tcp()
            axis=torch.zeros_like(tcp);axis[:,2]=1
            normal=quat_rotate(quat,axis)
            legal=self.events.hit & (self.events.outcome==0) & ~contacts[:,3] & ~contacts[:,0] & ~contacts[:,1]
            released=self.release.update(self.elapsed,state,contacts[:,2],legal,tcp,normal,tcp_velocity)
            for i in released.nonzero().flatten().cpu().tolist():
                measured=state[i].cpu().numpy();prediction=return_teacher(measured,config=self.bc)
                self.return_prediction_valid[i]=prediction['valid']
                parts=guidance_rewards(tcp[i].cpu().numpy(),{'valid':False},prediction,cfg=self.guidance)
                value=parts['weighted']['landing']+parts['weighted']['net']
                self.episode_return[i]+=value
                self.return_records[i]=dict(time_s=float(self.elapsed[i]),state=measured.tolist(),
                    prediction=prediction,reward=value,components=parts['weighted'])
        return first_hit

    def step(self,action,*,auto_reset=True):
        if action.shape!=(self.n,6) or not torch.isfinite(action).all():raise ValueError('Expected finite Nx6 action')
        if (self.events.outcome>0).any():raise RuntimeError('Reset finished episodes before stepping')
        used=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        prehit=~self.events.hit & (self.events.outcome==0)
        self.prehit_steps+=prehit.long()
        if self.predictor_module is not None:
            used=prehit & (self.actor_history[:,-1,26]>0)
            first=used & (self.first_hint_used<0)
            self.first_hint_used[first]=self.policy_index[first]*self.policy_dt
            self.hint_steps+=used.long()
            lengths=torch.tensor([len(h) for h in self.predictor_module.hook.history],device=self.device)
            self.hint_missing[:,0]+=(prehit & ~used & (lengths<5)).long()
            self.hint_missing[:,1]+=(prehit & ~used & (lengths==5) & (not self.predictor_module.predictor.trained)).long()
            self.hint_missing[:,2]+=(prehit & ~used & (lengths==5) & self.predictor_module.predictor.trained).long()
        hint_reference=used & self.teacher_valid & (self.elapsed<self.teacher_time)
        hint_error=((self.prediction[:,:3]-self.teacher_point)**2).sum(1)*hint_reference
        self.hint_reference_steps+=hint_reference.long();self.hint_error_sq_sum+=hint_error
        old=self.previous_action.clone();self.command.set_action(action)
        previous_release=self.release.reason.clone()
        previous_return=self.episode_return.clone()
        previous_guidance=self.episode_guidance.clone()
        hit=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        for _ in range(self.decimation): hit|=self.physics_step()
        self.previous_action=action.clone()
        p,_,_=self.tcp();state=self.state()
        distance=torch.linalg.vector_norm(p-self.teacher_point,dim=1)
        r=self.cfg['reward'];dense=self.episode_guidance-previous_guidance
        reward=hit*r['contact']+(self.events.outcome==6)*r['legal_return']+r['action_rate']*((action-old)**2).sum(1)*self.policy_dt
        reward+=(self.events.outcome==2)*r['fault']
        if self.group in ('B','D','E','E-online','E-frozen','E-mean'):reward+=dense
        if self.group in ('C','D','E','E-online','E-frozen','E-mean'):reward+=self.episode_return-previous_return
        self.episode_reward+=reward
        self.hit_reward_steps+=(dense>0).long()
        self.policy_index+=1
        if self.predictor_module is not None:self.refresh_prediction()
        a,c=self.frames();self.actor_history=torch.cat((self.actor_history[:,1:],a[:,None]),1);self.critic_history=torch.cat((self.critic_history[:,1:],c[:,None]),1)
        terminal=self.observations()
        terminated=(self.events.outcome>0)&(self.events.outcome!=9);truncated=self.events.outcome==9
        done=terminated|truncated
        info=dict(terminal_actor=terminal[0],terminal_critic=terminal[1],terminated=terminated.clone(),truncated=truncated.clone(),
                  q_request=self.command.request.clone(),q_applied=self.command.q.clone(),q_measured=self.robot.data.joint_pos[:,self.joints].clone(),
                  guidance=dense.clone(),apex_distance=distance.clone(),episodes=[],
                  prehit=prehit.clone(),prediction_used=used.clone(),hint_reference=hint_reference.clone(),hint_error_sq=hint_error.clone(),
                  return_events=((self.release.reason==2)&(previous_release!=2)).clone(),
                  return_valid_events=((self.release.reason==2)&(previous_release!=2)&self.return_prediction_valid).clone(),
                  return_increment=(self.episode_return-previous_return).clone())
        ids=done.nonzero().flatten()
        for i in ids.cpu().tolist():
            row=dict(env=i,serve=int(self.serve_id[i]),outcome=self.events.names[int(self.events.outcome[i])],hit=bool(self.events.hit[i]),
                     net_touched=bool(self.events.net_touched[i]),duration_s=float(self.elapsed[i]),
                     landing_xy_m=self.events.landing[i,:2].cpu().tolist() if self.events.outcome[i]==6 else None,
                     min_ball_distance_m=float(self.distance_min[i]),guidance_valid_s=float(self.valid_seconds[i]),
                     plateau_s=float(self.plateau_seconds[i]),guidance_sum=float(self.episode_guidance[i]),reward=float(self.episode_reward[i]),
                     target_success=bool(self.events.outcome[i]==6 and torch.all(abs(self.events.landing[i,:2]-self.tensor([.75,0]))<=self.strike['goal_half_width_m'])),
                     landing_error_m=float(torch.linalg.vector_norm(self.events.landing[i,:2]-self.tensor([.75,0]))) if self.events.outcome[i]==6 else None)
            if self.serve_bank is not None:
                row.update(raw_id=self.serve_bank.rows[int(self.bank_index[i])]['id'],
                           initial_state=self.bank_initial_states[i].tolist())
            row.update(prehit_policy_steps=int(self.prehit_steps[i]),hit_guidance_policy_steps=int(self.hit_reward_steps[i]))
            if self.predictor_module is not None:
                row.update(first_prediction_s=float(self.first_prediction[i]) if self.first_prediction[i]>=0 else None,
                           first_hint_used_s=float(self.first_hint_used[i]) if self.first_hint_used[i]>=0 else None,
                           prehit_hint_steps=int(self.hint_steps[i]),hint_missing=dict(zip(('history_short','untrained','invalid'),self.hint_missing[i].cpu().tolist())),
                           predictor_updates=self.predictor_module.predictor.updates,
                           predictor_mode=self.predictor_mode,hint_reference_steps=int(self.hint_reference_steps[i]),
                           hint_error_sq_sum_m2=float(self.hint_error_sq_sum[i]),
                           hint_teacher_rmse_m=float(torch.sqrt(self.hint_error_sq_sum[i]/self.hint_reference_steps[i])) if self.hint_reference_steps[i]>0 else None)
            if self.group in ('C','D','E','E-online','E-frozen','E-mean'):
                row.update(return_guidance=float(self.episode_return[i]),release_reason=self.release.reasons[int(self.release.reason[i])],return_record=self.return_records[i])
            info['episodes'].append(row)
        if auto_reset and ids.numel():self.reset(ids)
        return (*self.observations(),reward,done,info)
