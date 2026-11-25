"""Same-serve measured PhysX motion examples; separate M5/PPO processes, no training."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from airbot_tt.config import ROOT

DEST=ROOT/'.agents/runs/foundation/m7/motion_comparison'
SERVE=np.array([.5,-.0036606420645922664,2.2062970962371953,-3.301360144032454,0.,-1.40866137948719])
POLICIES={name:ROOT/f'.agents/runs/foundation/m7/m74/ab/{name}/policy.pt' for name in ('A_7101','B_7101')}


def snapshot(robot,paddle,tcp,state,t,origin,contacts,index=0):
    p,quat=tcp
    from scipy.spatial.transform import Rotation
    normal=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()[:,2]
    omega=robot.data.body_ang_vel_w[index,paddle].cpu().numpy()
    com=robot.data.body_com_pos_w[index,paddle].cpu().numpy()-origin
    velocity=robot.data.body_lin_vel_w[index,paddle].cpu().numpy()+np.cross(omega,np.asarray(p)-com)
    ids=[robot.body_names.index(n) for n in ['base_link','link1','link2','link3','link4','link5','link6']]
    joints=robot.data.body_link_pos_w[index,ids].cpu().numpy()-origin
    return dict(t=float(t),ball=np.asarray(state).tolist(),tcp=np.asarray(p).tolist(),quaternion_wxyz=np.asarray(quat).tolist(),
          normal=normal.tolist(),tcp_velocity=velocity.tolist(),omega=omega.tolist(),arm=joints.tolist(),contacts=list(map(bool,contacts)))


def impact(before,after,radius):
    normal=np.array(before['normal']);delta=np.array(before['ball'][:3])-before['tcp']
    if normal@delta<0:normal=-normal
    # A geometric proxy, explicitly not a PhysX solver contact point.
    point=np.array(before['ball'][:3])-normal*(normal@delta)
    surface=np.array(before['tcp_velocity'])+np.cross(before['omega'],point-before['tcp'])
    relative=np.array(before['ball'][3:])-surface
    return dict(time_s=after['t'],pre_sample=before,post_impulse_sample=after,
                normal_facing_ball=normal.tolist(),tcp_speed_m_s=float(np.linalg.norm(before['tcp_velocity'])),
                point_estimate_m=point.tolist(),surface_velocity_estimate_m_s=surface.tolist(),
                surface_speed_estimate_m_s=float(np.linalg.norm(surface)),relative_normal_closing_speed_estimate_m_s=float(-relative@normal),
                contact_point_source='Ball-centre projection onto measured TCP face plane; geometric estimate, not solver contact data')


def observe(run,before,after,radius,tick,terminal):
    old,new=before['contacts'],after['contacts']
    if new[2] and not old[2] and run.get('impact') is None:run['impact']=impact(before,after,radius)
    if run.get('impact') is not None:
        if old[2] and not new[2] and run.get('release') is None:run['release']=after
        if new[0] and not old[0] and run.get('post_hit_table_contact') is None:run['post_hit_table_contact']=after
    if new[1] and not old[1]:run['net_contacts'].append(after)
    if tick%8==0 or terminal:run['trace'].append(after)


def setup(kind):
    from pace_m7 import source_hashes
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/f'{kind}.json').exists():raise FileExistsError('Preserve existing motion capture')
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICIES.values()}
    protocol=dict(scope='Descriptive same-serve examples, not independent success-rate ranking',serve=SERVE.tolist(),
         serve_source='Existing M5 validation_0',m5='Original nominal precomputed centre-return script; no per-serve replanning',
         policy_selection='Existing A7101 (previous legal example) and B7101 (previous failure example), selected before replay',
         policy_sha256=inputs,source_sha256=source_hashes(),capture_hz=120,physics_hz=960,
         contact_speed='Last pre-contact sample at 960 Hz; TCP exact rigid-body velocity, contact-point proxy labelled',
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=DEST/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        assert {k:v for k,v in existing.items() if k!='code_sha256'}=={k:v for k,v in protocol.items() if k!='code_sha256'}
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def capture(device='cuda:0',m5=False):
    from pace_m7 import source_hashes
    started=time.monotonic();torch.set_num_threads(1);kind='m5' if m5 else 'policies';protocol=setup(kind);runs=[]
    if m5:
        from airbot_tt.shot_scene import SingleShot
        from airbot_tt.planning import load_strike_config,plan_return
        cfg=load_strike_config();task=SingleShot(device,speed_limit=cfg['simulation_speed_limit_rad_s'],physics_hz=960,initial=cfg['ready_q_rad'])
        c=task.control
        plan=plan_return(c.model,c.checker,cfg['ready_q_rad'],np.r_[cfg['serve_position_m'],cfg['serve_velocity_m_s']],
                         [.75,0],np.array(task.config['base_position_m']),task.ball_config,cfg)
        assert plan['accepted'],plan
        task.reset(dict(position_m=SERVE[:3].tolist(),velocity_m_s=SERVE[3:].tolist()))
        def snap():
            tcp=c.state()['tcp_world'];contacts=getattr(task,'last_contacts',set()) if task.time>0 else set()
            contact=[bool(('table','net','paddle','arm')[j] in contacts) for j in range(4)]
            return snapshot(c.robot,c.racket,(tcp['position_m'],tcp['quaternion_wxyz']),task.state,task.time,np.zeros(3),contact)
        run=dict(name='M5',trace=[snap()],impact=None,release=None,post_hit_table_contact=None,net_contacts=[])
        writes=c.state_writes;begun=False;tick=0
        while task.events.outcome is None:
            if not begun and task.time+1e-10>=cfg['reaction_delay_s']:
                c.start_prepared_strike(plan['path']);begun=True
            before=snap();task.step();after=snap();tick+=1
            observe(run,before,after,task.ball_config['radius_m'],tick,task.events.outcome is not None)
        assert c.state_writes==writes
        run.update(outcome=task.events.outcome,actual_hit=any(e['kind']=='paddle_contact' for e in task.events.events),
             events=task.events.events,controller='M5 original prepared StrikeTrajectory, position + velocity targets')
        runs.append(run)
    else:
        from airbot_tt.pace_env import PaceEnv
        from train_pace import original_ppo
        env=PaceEnv(1,device);ActorCritic,_=original_ppo();cfg=env.cfg['ppo']
        original=env.physics_step;active=None;tick=0
        def snap():
            p,q,_=env.tcp()
            return snapshot(env.robot,env.paddle,(p[0].cpu().numpy(),q[0].cpu().numpy()),env.state()[0].cpu().numpy(),
                 float(env.elapsed[0]),env.origins[0].cpu().numpy(),env.events.contacts[0].cpu().tolist())
        def step():
            nonlocal tick
            before=snap();result=original();after=snap();tick+=1
            observe(active,before,after,env.bc['radius_m'],tick,bool(env.events.outcome[0]))
            return result
        env.physics_step=step
        for name in ('zero','A_7101','B_7101'):
            policy=None;weights=None
            if name!='zero':
                ck=torch.load(POLICIES[name],map_location=device,weights_only=False)
                policy=ActorCritic(115,200,6,actor_hidden_dims=cfg['hidden'],critic_hidden_dims=cfg['hidden'],activation='elu',init_noise_std=cfg['init_std']).to(device)
                policy.load_state_dict(ck['policy']);weights={k:v.clone() for k,v in policy.state_dict().items()}
            env.group='A' if name=='zero' else name[0];env.reset(env.ids,SERVE)
            active=dict(name=name,trace=[snap()],impact=None,release=None,post_hit_table_contact=None,net_contacts=[])
            writes=env.state_writes;tick=0
            for _ in range(120):
                with torch.no_grad():
                    obs,_=env.observations();action=torch.zeros((1,6),device=device) if policy is None else policy.act_inference(obs)
                    env.step(action,auto_reset=False)
                if env.events.outcome[0]:break
            assert env.events.outcome[0] and env.state_writes==writes
            if policy is not None:assert all(torch.equal(weights[k],v) for k,v in policy.state_dict().items())
            active.update(outcome=env.events.names[int(env.events.outcome[0])],actual_hit=bool(env.events.hit[0]),
                 controller='48 Hz bounded joint-position residual; no learning',policy_unchanged=True)
            runs.append(active)
    for run in runs:
        np.testing.assert_allclose(run['trace'][0]['ball'],SERVE,atol=2e-6,rtol=0)
        landing=run['post_hit_table_contact']
        run['landing_xy_m']=None if landing is None else landing['ball'][:2]
        run['duration_s']=run['trace'][-1]['t']
        run['tcp_displacement_max_m']=float(max(np.linalg.norm(np.array(s['tcp'])-run['trace'][0]['tcp']) for s in run['trace']))
        run['tcp_path_length_m']=float(np.linalg.norm(np.diff([s['tcp'] for s in run['trace']],axis=0),axis=1).sum())
        if run['name']=='M5':run['actual_hit']=run['impact'] is not None
        print('MOTION',run['name'],run['outcome'],'landing',run['landing_xy_m'],'speed',None if run['impact'] is None else run['impact']['tcp_speed_m_s'],flush=True)
    assert source_hashes()==protocol['source_sha256']
    for p,h in protocol['policy_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    result=dict(status='PASS',protocol=protocol,runs=runs,seconds=time.monotonic()-started,ppo_updates=0,predictor_updates=0,direct_state_writes_during_motion=0)
    (DEST/f'{kind}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
