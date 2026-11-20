"""Actual PhysX M7 lifecycle/control gates; invoked after AppLauncher."""
import time
import torch
from airbot_tt.pace_env import PaceEnv


def _verify(device='cuda:0',num_envs=2,check='lifecycle'):
    started=time.monotonic()
    if check=='reference':return reference(device)
    env=PaceEnv(num_envs,device)
    if check=='control':return control_gate(env)
    if check=='parity':return parity(env)
    report=dict(status='PASS',checks={},num_envs=num_envs,device=device)
    obs,critic=env.observations()
    assert obs.shape==(num_envs,115) and critic.shape==(num_envs,200)
    report['checks']['schema']=True
    nominal=torch.tensor([.5,0,2.2,-3.3,0,-1.4],device=device)
    env.reset(env.ids,nominal)
    obs,critic=env.observations()
    assert torch.max(abs(obs-obs[0]))<2e-5
    assert torch.max(abs(critic-critic[0]))<2e-5
    report['checks']['translation_invariant_observations']=True
    writes=env.state_writes
    for _ in range(5):
        obs,critic,r,d,info=env.step(torch.zeros(num_envs,6,device=device),auto_reset=False)
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
        assert not d.any(),(env.events.outcome,info['episodes'])
    assert env.state_writes==writes
    assert float(env.robot.data.joint_vel[:,env.joints].abs().max())<.01
    report['checks']['reset_pose_stable_under_drive']=True
    report['checks']['normal_step_no_state_writes']=True
    for _ in range(3):env.step(torch.full((num_envs,6),.05,device=device),auto_reset=False)
    if num_envs>1:
        before=(env.robot.data.joint_pos[1:].clone(),env.state()[1:].clone(),env.actor_history[1:].clone(),env.serve_id[1:].clone())
        env.reset([0],nominal)
        after=(env.robot.data.joint_pos[1:],env.state()[1:],env.actor_history[1:],env.serve_id[1:])
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        report['checks']['partial_reset_isolation']=True
        import numpy as np
        expected=env.tensor(env.model.fk(env.strike['ready_q_rad'])[:3,3]+np.array(env.shot['base_position_m']))
        assert torch.linalg.vector_norm(env.tcp()[0][0]-expected)<2e-5
        report['checks']['reset_tcp_kinematics']=True
    env.reset(env.ids,nominal)
    steps=0;events=[];t=time.monotonic()
    for _ in range(100):
        obs,critic,r,d,info=env.step(torch.zeros(num_envs,6,device=device))
        steps+=1;events+=info['episodes']
        if len(events)>=num_envs:break
    assert len(events)>=num_envs,'Episodes did not finish'
    assert all(torch.isfinite(x).all() for x in (obs,critic,r))
    report['checks']['episode_and_reset']=True
    throughput=steps*num_envs/(time.monotonic()-t)
    env.reset(env.ids,nominal);env.events.timeout=env.policy_dt/2
    obs,critic,r,d,info=env.step(torch.zeros(num_envs,6,device=device))
    assert info['truncated'].all() and not info['terminated'].any()
    assert not torch.equal(info['terminal_actor'],obs)
    env.events.timeout=env.shot['timeout_s']
    obs,critic,r,d,info=env.step(torch.zeros(num_envs,6,device=device))
    assert not info['truncated'].any() and not d.any()
    report['checks']['terminal_before_reset_and_fresh_timeout']=True
    report.update(episodes=events,policy_samples_s=throughput,seconds=time.monotonic()-started,
                  body_names=env.robot.body_names,ball_filter_shape=list(env.ball_sensor.data.force_matrix_w.shape))
    return report


def control_gate(env):
    import numpy as np
    from airbot_tt.planning import plan_return
    from airbot_tt.collision import CollisionChecker
    from airbot_tt.kinematics import pose_matrix, pose_error
    c=env.bc
    obstacles=[dict(name='table',center_m=[0,0,c['table_top_m']-c['table_size_m'][2]/2],size_m=c['table_size_m']),
               dict(name='net',center_m=[0,0,c['table_top_m']+c['net_height_m']/2],size_m=[c['net_thickness_m'],c['table_size_m'][1],c['net_height_m']])]
    checker=CollisionChecker(env.model,obstacles,base_position=env.shot['base_position_m'],base_mount_obstacle='table')
    plan=plan_return(env.model,checker,env.strike['ready_q_rad'],np.r_[env.strike['serve_position_m'],env.strike['serve_velocity_m_s']],
                     [0.75,0],np.array(env.shot['base_position_m']),env.bc,env.strike)
    assert plan['accepted'],plan
    path=plan['path'];delay=env.strike['reaction_delay_s']
    report=dict(status='PASS',checks={},trials=[],scope='M5 centre endpoint position requests through bounded cubic action interpolation',path_hit_time_s=path.hit_time)
    samples=np.array([path.sample(t)[0] for t in np.linspace(0,path.duration,1001)])
    residual=(samples-np.array(env.strike['ready_q_rad']))/np.array(env.cfg['action_scale_rad'])
    report['reference_action_abs_max']=np.max(abs(residual),axis=0).tolist()
    report['checks']['action_envelope']=bool(np.max(abs(residual))<=1)
    for decimation in (env.cfg['decimation'],):
        env.decimation=decimation;env.policy_dt=decimation*env.dt
        env.reset(env.ids,[3,0,2,0,0,0])
        # Parked ball is outside task bounds; control diagnostics ignore task outcomes.
        rows=[];peak_contact=0.;peak_tracking=0.;peak_request_gap=0.
        count=round((path.duration+delay+.2)/env.dt)
        for k in range(count):
            t=k*env.dt
            if k%decimation==0:
                # Desired position at the end of this policy interval, no velocity action.
                target=path.sample(max(0,t-delay+env.policy_dt))[0]
                a=env.tensor((target-np.array(env.strike['ready_q_rad']))/np.array(env.cfg['action_scale_rad'])).repeat(env.n,1)
                env.command.set_action(a)
            env.physics_step()
            force=float(torch.linalg.vector_norm(env.arm_sensor.data.net_forces_w,dim=-1).max())
            peak_contact=max(peak_contact,force)
            assert force<5.,f'Unexpected robot contact: {force}'
            peak_tracking=max(peak_tracking,float((env.robot.data.joint_pos[:,env.joints]-env.command.q).abs().max()))
            peak_request_gap=max(peak_request_gap,float((env.command.request-env.command.q).abs().max()))
            pos,quat,velocity=env.tcp()
            rows.append((pos[0].cpu().numpy(),quat[0].cpu().numpy(),velocity[0].cpu().numpy(),env.robot.data.joint_pos[0,env.joints].cpu().numpy()))
        j=round((path.hit_time+delay)/env.dt)-1
        pose=pose_matrix(np.r_[rows[j][0]-np.array(env.shot['base_position_m']),rows[j][1]])
        q,qd,_=path.sample(path.hit_time)
        error=pose_error(pose,env.model.fk(q));desired=env.model.jacobian(q)@qd
        linear=(rows[j+1][0]-rows[j-1][0])/(2*env.dt)
        from scipy.spatial.transform import Rotation
        angular=(Rotation.from_quat(np.roll(rows[j+1][1],-1))*Rotation.from_quat(np.roll(rows[j-1][1],-1)).inv()).as_rotvec()/(2*env.dt)
        progress=np.array([np.dot(row[3]-q,qd) for row in rows])
        crossings=[k for k in range(max(0,j-100),min(len(rows)-1,j+100)) if progress[k]<=0<progress[k+1]]
        timing=None
        if crossings:
            k=min(crossings,key=lambda k:abs(k-j));cross=(k+1-progress[k]/(progress[k+1]-progress[k]))*env.dt
            timing=float(abs(cross-(path.hit_time+delay)))
        trial=dict(policy_hz=960/decimation,peak_contact_N=peak_contact,peak_tracking_rad=peak_tracking,peak_request_applied_gap_rad=peak_request_gap,position_error_m=float(np.linalg.norm(error[:3])),orientation_error_rad=float(np.linalg.norm(error[3:])),
                   linear_velocity_error_m_s=float(np.linalg.norm(linear-desired[:3])),angular_velocity_error_rad_s=float(np.linalg.norm(angular-desired[3:])),timing_error_s=timing)
        trial['passed']=bool(trial['position_error_m']<.002 and trial['orientation_error_rad']<.02 and trial['linear_velocity_error_m_s']<.03 and trial['angular_velocity_error_rad_s']<.08 and timing is not None and timing<2*env.dt)
        report['trials'].append(trial)
        if trial['passed']:break
    report['checks']['tracking']=any(t['passed'] for t in report['trials'])
    report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
    return report


def reference(device):
    from airbot_tt.shot_scene import SingleShot
    from airbot_tt.planning import load_strike_config
    cfg=load_strike_config()
    task=SingleShot(device,speed_limit=cfg['simulation_speed_limit_rad_s'],physics_hz=960,initial=cfg['ready_q_rad'])
    task.reset(dict(position_m=cfg['serve_position_m'],velocity_m_s=cfg['serve_velocity_m_s']))
    trace=[]
    for i in range(3000):
        row=task.step()
        if (i+1)%20==0 or row['outcome'] is not None:trace.append(dict(time_s=task.time,state=task.state.tolist(),phase=task.events.phase))
        if row['outcome'] is not None:break
    return dict(status='PASS',event_rules=task.events.rules,outcome=task.events.outcome,events=task.events.events,trace=trace,scope='Old SingleShot zero-action centre-ready reference')


def parity(env):
    import json
    import numpy as np
    from airbot_tt.config import ROOT
    reference=json.loads((ROOT/'.agents/runs/foundation/m7/reference.json').read_text())
    env.reset(env.ids,[.5,0,2.2,-3.3,0,-1.4])
    rows=[]
    for k in range(3000):
        env.physics_step()
        if (k+1)%20==0 or env.events.outcome[0]>0:
            rows.append(dict(time_s=float(env.elapsed[0]),state=env.state()[0].cpu().tolist(),phase=int(env.events.phase[0])))
        if env.events.outcome[0]>0:break
    # Compare on common pre-contact timestamps; impacts also require same event sequence/outcome.
    a={round(r['time_s'],5):r for r in reference['trace']};diff=[]
    for row in rows:
        key=round(row['time_s'],5)
        if key in a:diff.append(np.linalg.norm(np.array(row['state'][:3])-a[key]['state'][:3]))
    outcome=env.events.names[int(env.events.outcome[0])]
    checks=dict(rule_version=reference.get('event_rules')==env.events.rules,outcome=outcome==reference['outcome'],position=bool(diff and max(diff)<.01),
                end_time=abs(rows[-1]['time_s']-reference['trace'][-1]['time_s'])<.01)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,outcome=outcome,reference_outcome=reference['outcome'],
                max_position_difference_m=float(max(diff)) if diff else None,end_time_s=rows[-1]['time_s'],trace=rows)


def source_hashes():
    import hashlib
    from airbot_tt.config import ROOT
    names=['src/airbot_tt/'+n+'.py' for n in ['pace_control','pace_env','runtime','ball_scene','single_shot','pace_guidance','planning','trajectory','collision']]
    names+=['configs/'+n+'.json' for n in ['pace_rl','pace','strike','single_shot','ball','racket']]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def verify(device='cuda:0',num_envs=2,check='lifecycle'):
    before=source_hashes()
    report=_verify(device,num_envs,check)
    assert before==source_hashes(),'Source changed during verification'
    report['source_sha256']=before
    return report
