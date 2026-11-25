"""Pinned initial-state distribution and bounded AIRBOT reachability audit (no PPO)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from airbot_tt.config import ROOT

BASE=ROOT/'.agents/runs/foundation/deepmind_audit'
CONFIG=ROOT/'configs/deepmind_audit.json'
FIELDS=['pos_x','pos_y','pos_z','vel_x','vel_y','vel_z','w_vel_x','w_vel_y','w_vel_z']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    modules=['ball','ball_scene','collision','kinematics','pace_control','pace_env','planning','prediction','tool','trajectory']
    paths=['src/airbot_tt/'+n+'.py' for n in modules]+['configs/'+n+'.json' for n in ['ball','pace_rl','racket','single_shot','strike']]
    return {name:sha(ROOT/name) for name in paths}


def quantile(x):
    a=np.asarray(x)
    return dict(n=len(a),percentiles={str(p):v for p,v in zip([0,1,5,25,50,75,95,99,100],np.percentile(a,[0,1,5,25,50,75,95,99,100],axis=0).tolist())}) if len(a) else dict(n=0)


def load():
    c=json.loads(CONFIG.read_text());manifest=json.loads((BASE/'source/manifest.json').read_text())
    assert manifest['commit']==c['commit']
    for item in manifest['files']:assert sha(BASE/'source'/item['path'])==item['sha256']
    rows=[]
    for name in ('serves','rallies'):
        for row in json.loads((BASE/f'source/{name}.json').read_text()):
            source=np.array([row[k] for k in FIELDS],dtype=float)
            assert np.isfinite(source).all() and source[1]*source[4]<0
            sign=1 if source[1]>0 else -1
            rotation=np.array([[0,sign,0],[-sign,0,0],[0,0,1]],dtype=float)
            assert np.linalg.det(rotation)==1 and np.array_equal(rotation.T@rotation,np.eye(3))
            position=rotation@source[:3]+[0,0,.76];velocity=rotation@source[3:6];omega=rotation@source[6:]
            np.testing.assert_allclose(rotation.T@(position-[0,0,.76]),source[:3],atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(omega),np.linalg.norm(source[6:]),atol=1e-12)
            bucket=int.from_bytes(hashlib.sha256(f"{c['split_seed']}:{row['id']}".encode()).digest()[:8],'big')/2**64
            split='train' if bucket<c['split_ratio'][0] else ('validation' if bucket<sum(c['split_ratio'][:2]) else 'test')
            rows.append(dict(id=row['id'],kind=name,source_state=source.tolist(),state=np.r_[position,velocity].tolist(),
                 omega=omega.tolist(),opposite_direction=sign<0,split=split,low_spin=bool(np.linalg.norm(omega)<=c['low_spin_norm_max_rad_s'])))
    assert len(rows)==15792 and len(set(r['id'] for r in rows))==len(rows)
    return c,rows


def forecast_row(row):
    from airbot_tt.prediction import predict_ball
    from airbot_tt.ball import load_ball_config
    c=load_ball_config();state=np.array(row['state']);surface=c['table_top_m']+c['radius_m']
    result=dict(id=row['id'],kind=row['kind'],split=row['split'],low_spin=row['low_spin'])
    if state[2]<surface:return dict(result,termination='initial_below_contact_plane',own_bounce=False)
    forecast=predict_ball(state,horizon_s=2.,config=c)
    first_table=next((e for e in forecast['events'] if e['kind'] in ('bounce','out_of_table','unsupported_table_edge')),None)
    bounce=first_table if first_table and first_table['kind']=='bounce' else None
    net=next((e for e in forecast['events'] if e['kind']=='net_crossing'),None)
    result.update(termination=forecast['termination'],first_table_event=first_table,net=net,
        own_bounce=bool(bounce and bounce['position_m'][0]<0),
        current_task_bounds=bool(abs(state[0])<=2 and abs(state[1])<=1.2))
    if bounce:
        result['bounce']=bounce
        result['post_bounce_end']=next((e for e in forecast['events'] if e['time_s']>bounce['time_s']+1e-8 and e['kind'] in ('landing','out_of_table','unsupported_table_edge')),None)
    return result


def distribution(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    started=time.monotonic();c,rows=load();BASE.mkdir(parents=True,exist_ok=True)
    destination=BASE/'distribution.json'
    if destination.exists():raise FileExistsError('Preserve completed distribution audit')
    protocol=dict(config=c,config_sha256=sha(CONFIG),source_manifest_sha256=sha(BASE/'source/manifest.json'),
           transform='Proper rotation: X=s*source_Y, Y=-s*source_X, Z=source_Z+0.76; s=sign(source_Y); same R for v and omega',
           spin='Raw omega retained; forecasts/replays explicitly zero-spin approximations, no Magnus/spin transfer',
           source_sha256={f:sha(ROOT/f) for f in ['src/airbot_tt/prediction.py','src/airbot_tt/ball.py','configs/ball.json','configs/strike.json','configs/pace_rl.json']})
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (BASE/'transformed.json').write_text(json.dumps(rows)+'\n')
    summaries={}
    for name,subset in [('all',rows)]+[(k,[r for r in rows if r['kind']==k]) for k in ('serves','rallies')]:
        x=np.array([r['state'] for r in subset]);w=np.array([r['omega'] for r in subset])
        summaries[name]=dict(count=len(subset),position_m=quantile(x[:,:3]),velocity_m_s=quantile(x[:,3:]),speed_m_s=quantile(np.linalg.norm(x[:,3:],axis=1)),
                heading_deg=quantile(np.degrees(np.arctan2(x[:,4],-x[:,3]))),spin_rad_s=quantile(np.linalg.norm(w,axis=1)),
                low_spin=sum(r['low_spin'] for r in subset),exact_zero_spin=int(sum(np.linalg.norm(w,axis=1)==0)),
                opposite_direction=sum(r['opposite_direction'] for r in subset),splits=dict(Counter(r['split'] for r in subset)))
    forecasts=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(forecast_row,rows,chunksize=32),1):
            forecasts.append(result)
            if i%2000==0:print('DATASET_FORECAST',i,len(rows),flush=True)
    (BASE/'forecasts.json').write_text(json.dumps(forecasts)+'\n')
    by_kind={}
    for kind in ('serves','rallies'):
        subset=[f for f in forecasts if f['kind']==kind];own=[f for f in subset if f['own_bounce']]
        by_kind[kind]=dict(termination=dict(Counter(f['termination'] for f in subset)),own_first_bounce=len(own),
              first_own_bounce_time_s=quantile([f['bounce']['time_s'] for f in own]),
              first_own_bounce_position_m=quantile([f['bounce']['position_m'] for f in own]),
              net_crossing_time_s=quantile([f['net']['time_s'] for f in subset if f.get('net')]),
              compatible_low_spin_ids=[f['id'] for f in own if f['low_spin'] and f.get('current_task_bounds')])
    report=dict(status='PASS',protocol=protocol,raw=summaries,zero_spin_forecasts=by_kind,
                seconds=time.monotonic()-started,code_sha256=sha(__file__))
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print('DATASET_DISTRIBUTION', {k:(v['count'],v['low_spin']) for k,v in summaries.items()},flush=True)
    print('COMPATIBLE_LOW_SPIN',{k:len(v['compatible_low_spin_ids']) for k,v in by_kind.items()},flush=True)
    return report


_SEARCH=None


def search_context(model=None):
    from airbot_tt.tool import racket_model
    from airbot_tt.ball import load_ball_config
    from airbot_tt.collision import CollisionChecker
    model=racket_model(ROOT/'configs/racket.json') if model is None else model;bc=load_ball_config()
    strike=json.loads((ROOT/'configs/strike.json').read_text());rl=json.loads((ROOT/'configs/pace_rl.json').read_text())
    shot=json.loads((ROOT/'configs/single_shot.json').read_text());base=np.array(shot['base_position_m'])
    obstacles=[dict(name='table',center_m=[0,0,bc['table_top_m']-bc['table_size_m'][2]/2],size_m=bc['table_size_m']),
       dict(name='net',center_m=[0,0,bc['table_top_m']+bc['net_height_m']/2],size_m=[bc['net_thickness_m'],bc['table_size_m'][1],bc['net_height_m']])]
    checker=CollisionChecker(model,obstacles,base_position=base,base_mount_obstacle='table')
    return model,checker,bc,strike,rl,shot,base


def action_preflight(model,checker,path,arrival,pose,strike,rl,shot):
    import torch
    from airbot_tt.pace_control import PositionCommand
    from airbot_tt.kinematics import pose_error
    torch.set_num_threads(1);nom=np.array(strike['ready_q_rad']);scale=np.array(rl['action_scale_rad'])
    def tensor(x):return torch.tensor(x,dtype=torch.float32)
    command=PositionCommand(tensor(nom),tensor(scale),tensor(model.lower),tensor(model.upper),1,
              tensor(strike['velocity_limit_rad_s']),tensor(strike['acceleration_limit_rad_s2']),1/960,period=1/48)
    actions=[];max_action=0.;samples=[];previous=nom.copy();last_checked=nom.copy();peak_v=0.;peak_a=0.;old_v=np.zeros(6)
    hit_tick=round(arrival*960);q_at_hit=None
    for policy_step in range(round(shot['timeout_s']*48)):
        t=policy_step/48;requested=path.sample(max(0,t-strike['reaction_delay_s']+1/48))[0]
        action=(requested-nom)/scale;max_action=max(max_action,float(max(abs(action))))
        if max_action>1+1e-7:return dict(accepted=False,reason='action_envelope',action_abs_max=max_action)
        actions.append(action.tolist());command.set_action(tensor(action[None]))
        for sub in range(20):
            tick=policy_step*20+sub;t0=tick/960
            q,v=command.step()
            if t0<strike['reaction_delay_s']:
                command.q=tensor(nom[None]);command.v=torch.zeros_like(command.v);q,v=command.q,command.v
            qq=q[0].numpy().copy();vv=v[0].numpy().copy()
            peak_v=max(peak_v,float(max(abs(vv))));peak_a=max(peak_a,float(max(abs(vv-old_v)))*960);old_v=vv
            if max(abs(qq-last_checked))>=.005 or tick+1==hit_tick or tick+1==round(shot['timeout_s']*960):
                checker.require_clear(qq);last_checked=qq.copy();samples.append(qq.tolist())
            if tick+1==hit_tick:q_at_hit=qq.copy()
    assert q_at_hit is not None
    error=pose_error(model.fk(q_at_hit),pose)
    assert peak_v<=strike['velocity_limit_rad_s']+1e-5 and peak_a<=strike['acceleration_limit_rad_s2']+.002
    if np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:])>.02:
        return dict(accepted=False,reason='command_tracking_at_intercept',position_error_m=float(np.linalg.norm(error[:3])),orientation_error_rad=float(np.linalg.norm(error[3:])))
    return dict(accepted=True,actions=actions,action_abs_max=max_action,peak_command_velocity_rad_s=peak_v,
         peak_command_acceleration_rad_s2=peak_a,command_position_error_m=float(np.linalg.norm(error[:3])),
         command_orientation_error_rad=float(np.linalg.norm(error[3:])),command_collision_samples=len(samples))


def candidate_search(item):
    from scipy.optimize import brentq
    from airbot_tt.planning import flight_state
    from airbot_tt.trajectory import Trajectory
    row,forecast,c=item;model,checker,bc,strike,rl,shot,base=_SEARCH
    started=time.monotonic();nom=np.array(strike['ready_q_rad']);ready=model.fk(nom);normal=ready[:3,2]
    bounce=forecast['bounce'];post=np.r_[bounce['position_m'],bounce['velocity_m_s']];post[5]*=-bc['table_restitution']
    ending=forecast.get('post_bounce_end');duration=(ending['time_s']-bounce['time_s']) if ending else 1.
    reasons=Counter();attempts=[]
    for plane in c['candidate_planes_x_m']:
        f=lambda t:flight_state(post,t,bc)[0]-plane
        if f(0)*f(duration)>0:reasons['no_plane_crossing_before_next_surface']+=1;continue
        tau=brentq(f,0,duration);state=flight_state(post,tau,bc);arrival=bounce['time_s']+tau
        pose=ready.copy();pose[:3,3]=state[:3]-bc['radius_m']*normal-base
        if c.get('position_only_diagnostic'):
            from scipy.optimize import least_squares
            lower=np.maximum(model.lower,nom-np.array(rl['action_scale_rad']))
            upper=np.minimum(model.upper,nom+np.array(rl['action_scale_rad']))
            def residual(q):
                fk=model.fk(q)
                return np.r_[fk[:3,3]+bc['radius_m']*fk[:3,2]+base-state[:3],1e-7*(q-nom)]
            solution=least_squares(residual,nom,bounds=(lower,upper),max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            ik=dict(success=bool(np.linalg.norm(residual(solution.x)[:3])<=2e-4),q_rad=solution.x.tolist())
            pose=model.fk(solution.x)
        else:ik=model.ik(pose,nom)
        attempt=dict(plane_x_m=plane,arrival_s=arrival,ball_state_at_intercept=state.tolist(),ik_success=ik['success'])
        attempts.append(attempt)
        if not ik['success']:reasons['candidate_ik_not_solved']+=1;continue
        q=np.array(ik['q_rad']);envelope=float(max(abs((q-nom)/np.array(rl['action_scale_rad']))))
        attempt['action_abs_max']=envelope
        if envelope>1:reasons['action_envelope']+=1;continue
        # A rigorous per-joint lower bound for rest-to-rest displacement under v/a.
        distance=abs(q-nom);v=strike['velocity_limit_rad_s'];a=strike['acceleration_limit_rad_s2']
        lower=np.where(distance<=v*v/a,2*np.sqrt(distance/a),distance/v+v/a)
        attempt['joint_displacement_time_lower_bound_s']=float(max(lower))
        if max(lower)+strike['reaction_delay_s']>arrival:reasons['candidate_joint_time_bound']+=1;continue
        try:
            path=Trajectory(model,[nom,q],vmax=v,amax=a,require_clear=checker.require_clear)
            attempt['path_duration_s']=path.duration
            if path.duration+strike['reaction_delay_s']>arrival:reasons['trajectory_deadline']+=1;continue
            checked=action_preflight(model,checker,path,arrival,pose,strike,rl,shot)
        except ValueError as error:
            message=str(error)
            if 'solver' in message.lower() or 'linear program' in message.lower():raise
            reasons['collision_or_path_limit']+=1;attempt['rejection_detail']=message;continue
        if not checked['accepted']:reasons[checked['reason']]+=1;attempt.update(checked);continue
        return dict(id=row['id'],kind=row['kind'],split=row['split'],state=row['state'],omega_raw_transformed=row['omega'],
              accepted=True,spin_approximation='forced_zero_from_low_spin_raw',plane_x_m=plane,arrival_s=arrival,
              ball_state_at_intercept=state.tolist(),target_q=q.tolist(),target_tcp_pose=pose.tolist(),
              path_duration_s=path.duration,path_collision_samples=path.checked_samples,
              attempts=attempts,preflight=checked,seconds=time.monotonic()-started)
    return dict(id=row['id'],kind=row['kind'],split=row['split'],state=row['state'],omega_raw_transformed=row['omega'],
         accepted=False,reason='no_witness_in_finite_search',rejections=dict(reasons),attempts=attempts,seconds=time.monotonic()-started)


def search(workers=4,position_only=False,remaining=False):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    global _SEARCH
    started=time.monotonic();c,rows=load();c['position_only_diagnostic']=position_only;forecasts=json.loads((BASE/'forecasts.json').read_text());lookup={r['id']:r for r in rows}
    distribution=json.loads((BASE/'distribution.json').read_text())
    assert distribution['protocol']['config_sha256']==sha(CONFIG)
    compatible=[f for f in forecasts if f['own_bounce'] and f['low_spin'] and f.get('current_task_bounds')]
    # Uniform deterministic hash ordering, fixed before any search outcomes.
    compatible.sort(key=lambda f:hashlib.sha256(f"{c['selection_seed']}:{f['id']}".encode()).digest())
    chosen=compatible[c['search_max_rows']:] if remaining else compatible[:c['search_max_rows']]
    prefix='position_remaining' if remaining else ('position_search' if position_only else 'search')
    before=source_hashes();protocol=dict(eligible=len(compatible),selected_ids=[f['id'] for f in chosen],
        selection='Remaining eligible rows in same predeclared SHA256 order, after 96-row diagnostic' if remaining else 'First 96 by predeclared SHA256(seed:raw_id); no outcome-based selection',
        source_sha256=before,code_sha256=sha(__file__),config_sha256=sha(CONFIG),
        distribution_sha256=sha(BASE/'distribution.json'),forecasts_sha256=sha(BASE/'forecasts.json'),
        search_type='Finite position-only bounded IK then rest-to-rest interception' if position_only else 'Finite fixed-face rest-to-rest interception',
        refinement='Position-only refinement declared after fixed-face search found 0/96; same rows/planes/physical constraints, no training')
    if (BASE/f'{prefix}_protocol.json').exists():raise FileExistsError('Preserve existing search')
    (BASE/f'{prefix}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    _SEARCH=search_context();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for index,result in enumerate(pool.map(candidate_search,[(lookup[f['id']],f,c) for f in chosen]),1):
            results.append(result)
            if index%8==0:print('REACHABILITY_SEARCH',index,len(chosen),'accepted',sum(r['accepted'] for r in results),flush=True)
    assert before==source_hashes()
    report=dict(status='PASS',protocol=protocol,rows=results,accepted=sum(r['accepted'] for r in results),
       unresolved=len(compatible)-len(chosen),rejection_counts=dict(sum((Counter(r.get('rejections',{})) for r in results),Counter())),
       seconds=time.monotonic()-started,classification='Accepted means command/dynamic preflight only; physical returns require independent replay')
    (BASE/f'{prefix}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SEARCH_COMPLETE',report['accepted'],len(results),report['rejection_counts'],flush=True)
    return report


def candidate_striker(item):
    from airbot_tt.planning import plan_return
    row,_forecast,c=item;model,checker,bc,strike,rl,shot,base=_SEARCH
    started=time.monotonic()
    plan=plan_return(model,checker,strike['ready_q_rad'],row['state'],[.75,0],base,bc,strike)
    result=dict(id=row['id'],kind=row['kind'],split=row['split'],state=row['state'],omega_raw_transformed=row['omega'],
          method='Original M5 finite return planner per raw row, offline oracle; baseline M5 script unchanged',
          planner_accepted=plan['accepted'],rejections=plan.get('rejections',{}))
    if not plan['accepted']:
        result.update(accepted=False,reason='no_witness_in_original_m5_search',seconds=time.monotonic()-started);return result
    try:checked=action_preflight(model,checker,plan['path'],plan['hit_time_s'],plan['desired_pose'],strike,rl,shot)
    except ValueError as error:checked=dict(accepted=False,reason='full_path_collision_or_limit',detail=str(error))
    result.update(accepted=checked['accepted'],preflight=checked,hit_time_s=plan['hit_time_s'],
         ball_state_at_intercept=plan['incoming'].tolist(),target_q=plan['strike_q'].tolist(),
         target_tcp_pose=plan['desired_pose'].tolist(),seconds=time.monotonic()-started)
    return result


def striker_search(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    global _SEARCH
    started=time.monotonic();c,rows=load();lookup={r['id']:r for r in rows};fs=json.loads((BASE/'forecasts.json').read_text())
    fs=[f for f in fs if f['own_bounce'] and f['low_spin'] and f.get('current_task_bounds')]
    fs.sort(key=lambda f:hashlib.sha256(f"{c['selection_seed']}:{f['id']}".encode()).digest())
    protocol=dict(ids=[f['id'] for f in fs],method='Existing M5 nonzero-speed return planner, original candidate times/goals/physics/limits; followed by unchanged M7 action preflight',
         source_sha256=source_hashes(),config_sha256=sha(CONFIG),code_sha256=sha(__file__),
         reason='Complement rest-to-rest intercept audit with existing active-strike family; no new algorithm/parameter sweep or PPO')
    if (BASE/'striker_protocol.json').exists():raise FileExistsError('Preserve original striker audit')
    (BASE/'striker_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    _SEARCH=search_context();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for index,result in enumerate(pool.map(candidate_striker,[(lookup[f['id']],f,c) for f in fs]),1):
            results.append(result)
            if index%16==0:print('M5_AUDIT_SEARCH',index,len(fs),'accepted',sum(r['accepted'] for r in results),flush=True)
    report=dict(status='PASS',protocol=protocol,rows=results,accepted=sum(r['accepted'] for r in results),seconds=time.monotonic()-started)
    assert source_hashes()==protocol['source_sha256']
    (BASE/'striker_search.json').write_text(json.dumps(report,indent=2)+'\n')
    print('M5_AUDIT_COMPLETE',report['accepted'],len(fs),flush=True)
    return report


def replay(device='cuda:0',num_envs=16):
    """Independent zero-spin PhysX returns and frozen action cross-matrix."""
    import torch
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.trajectory import Trajectory
    from airbot_tt.planning import plan_return
    if num_envs!=16:raise ValueError('Audit replay uses 16 environments')
    if (BASE/'physics.json').exists():raise FileExistsError('Preserve existing physical audit')
    started=time.monotonic();torch.set_num_threads(1);c,raw=load();raw_by_id={r['id']:r for r in raw}
    first=json.loads((BASE/'position_search.json').read_text());rest=json.loads((BASE/'position_remaining.json').read_text())
    combined=first['rows']+rest['rows'];ids=[r['id'] for r in combined]
    assert len(ids)==len(set(ids))==343
    plans=[r for r in combined if r['accepted']]
    before=source_hashes();protocol=dict(eligible_ids=ids,preflight_candidate_ids=[r['id'] for r in plans],
        zero_baseline_ids=ids[:c['search_max_rows']],matrix_serve_ids=[r['id'] for r in plans[:c['cross_replay_max_rows']]],
        scope='Zero-spin approximation, offline precomputed actions; no PPO or sensor/controller changes',
        selection='Matrix chosen from command candidates before any PhysX return outcomes; no diagonal cherry-picking',
        source_sha256=before,code_sha256=sha(__file__),config_sha256=sha(CONFIG),
        search_sha256={n:sha(BASE/f'{n}.json') for n in ['position_search','position_remaining']})
    (BASE/'physics_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=PaceEnv(num_envs,device);model,checker,bc,strike,rl,shot,base=search_context(env.model)
    full_preflight=[];active_plans=[]
    for plan in plans:
        path=Trajectory(model,[strike['ready_q_rad'],plan['target_q']],vmax=strike['velocity_limit_rad_s'],
            amax=strike['acceleration_limit_rad_s2'],require_clear=checker.require_clear)
        try:checked=action_preflight(model,checker,path,plan['arrival_s'],np.array(plan['target_tcp_pose']),strike,rl,shot)
        except ValueError as error:checked=dict(accepted=False,reason='full_path_collision',detail=str(error))
        full_preflight.append(dict(id=plan['id'],result=checked))
        if checked['accepted']:
            np.testing.assert_allclose(checked['actions'],plan['preflight']['actions'],atol=1e-12,rtol=0)
            active_plans.append(plan)
    # Normal-step writes are rejected; explicit resets retain their existing contract.
    resetting=False;original_reset=env.reset
    def reset(*args,**kwargs):
        nonlocal resetting
        resetting=True
        try:return original_reset(*args,**kwargs)
        finally:resetting=False
    env.reset=reset
    for obj,name in [(env.robot,'write_joint_state_to_sim'),(env.ball,'write_root_state_to_sim')]:
        fn=getattr(obj,name)
        def guarded(*args,_fn=fn,**kwargs):
            if not resetting:raise AssertionError('Normal-motion direct state write')
            return _fn(*args,**kwargs)
        setattr(obj,name,guarded)
    original_physics=env.physics_step;captures=[]
    def step():
        old_phase=env.events.phase.clone();old_contacts=env.events.contacts.clone()
        result=original_physics();state=env.state()
        for i,row in enumerate(captures):
            if row['finished']:continue
            if int(old_phase[i])==0 and int(env.events.phase[i])==1:
                row['bounce']=dict(time_s=float(env.elapsed[i]),state=state[i].cpu().tolist())
            if env.events.hit[i] and 'hit' not in row:row['hit']=dict(time_s=float(env.elapsed[i]),state=state[i].cpu().tolist())
            if 'hit' in row and bool(env.events.contacts[i,0]) and not bool(old_contacts[i,0]) and 'landing' not in row:
                row['landing']=dict(time_s=float(env.elapsed[i]),state=state[i].cpu().tolist())
            if env.events.outcome[i]:row['finished']=True
        return result
    env.physics_step=step
    def evaluate(cases,label):
        nonlocal captures
        results=[]
        for start in range(0,len(cases),num_envs):
            batch=cases[start:start+num_envs];padded=batch+[batch[0]]*(num_envs-len(batch))
            env.reset(env.ids,np.array([raw_by_id[z['id']]['state'] for z in padded]))
            assert not env.ball.data.root_ang_vel_w.any()
            captures=[dict(finished=False) for _ in range(num_envs)];episodes={}
            for k in range(round(shot['timeout_s']*48)+1):
                actions=np.array([z['actions'][min(k,len(z['actions'])-1)] for z in padded])
                assert np.max(abs(actions))<=1+1e-7
                with torch.no_grad():_,_,_,_,info=env.step(env.tensor(actions))
                for episode in info['episodes']:episodes.setdefault(episode['env'],episode)
                if len(episodes)==num_envs:break
            assert len(episodes)==num_envs
            for i,case in enumerate(batch):
                results.append(dict(id=case['id'],controller=case['controller'],split=raw_by_id[case['id']]['split'],
                      episode=episodes[i],measured=captures[i]))
            print('DATASET_PHYSICS',label,len(results),len(cases),'legal',sum(z['episode']['outcome']=='legal_return' for z in results),flush=True)
        return results
    zero_actions=np.zeros((round(shot['timeout_s']*48),6)).tolist()
    zeros=evaluate([dict(id=i,controller='zero',actions=zero_actions) for i in protocol['zero_baseline_ids']],'zero')
    nominal=plan_return(model,checker,strike['ready_q_rad'],np.r_[strike['serve_position_m'],strike['serve_velocity_m_s']],
              [.75,0],base,bc,strike)
    assert nominal['accepted']
    nom=np.array(strike['ready_q_rad']);scale=np.array(rl['action_scale_rad'])
    m5_actions=np.array([(nominal['path'].sample(max(0,k/48-strike['reaction_delay_s']+1/48))[0]-nom)/scale for k in range(len(zero_actions))])
    assert abs(m5_actions).max()<=1+1e-7
    matrix_plans=active_plans[:c['cross_replay_max_rows']]
    controllers=[('zero',zero_actions),('M5_reference_via_M7',m5_actions.tolist())]+[(f"script_{r['id']}",r['preflight']['actions']) for r in matrix_plans]
    cases=[dict(id=serve['id'],controller=name,actions=actions) for name,actions in controllers for serve in matrix_plans]
    matrix=evaluate(cases,'cross') if cases else []
    diagonal=[z for z in matrix if z['controller']==f"script_{z['id']}"]
    forecasts={f['id']:f for f in json.loads((BASE/'forecasts.json').read_text())}
    ball_errors=[]
    for z in zeros:
        if 'bounce' in z['measured']:
            actual=z['measured']['bounce'];reference=forecasts[z['id']]['bounce']
            ball_errors.append(dict(id=z['id'],position_error_m=float(np.linalg.norm(np.array(actual['state'][:3])-reference['position_m'])),
                 time_error_s=abs(actual['time_s']-reference['time_s'])))
    legal_ids=[z['id'] for z in diagonal if z['episode']['outcome']=='legal_return']
    outcome_counts=dict(Counter(z['episode']['outcome'] for z in diagonal))
    assert before==source_hashes()
    summary=dict(status='PASS',protocol=protocol,full_preflight=full_preflight,zero_baseline=zeros,matrix=matrix,
         diagonal=diagonal,certified_return_ids=legal_ids,diagonal_outcomes=outcome_counts,
         eligible=len(combined),candidate_count=len(plans),full_preflight_count=len(active_plans),matrix_serve_count=len(matrix_plans),
         finite_search_no_witness=sum(not r['accepted'] for r in combined),unsearched=0,
         zero_baseline_outcomes=dict(Counter(z['episode']['outcome'] for z in zeros)),bounce_errors=ball_errors,
         seconds=time.monotonic()-started,normal_motion_state_writes=0,ppo_updates=0,
         conclusion='Certified returns are finite-search lower bounds under zero-spin approximation; other cases are not proven unreachable')
    (BASE/'physics.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Export an audit-only manifest. No environment/trainer consumes it.
    export=dict(schema='deepmind-airbot-audited-candidates-v1',source_commit=c['commit'],source_license='CC-BY-4.0',
          attribution='Google LLC, Competitive Robot Table Tennis: Initial Ball States (2025)',
          changes='Proper rotation, +0.76m Z; original omega retained, simulated omega forced zero; finite controller filtering',
          usage='Audit candidates only; not promoted into M7 training; test IDs must not be used for training',
          split_limit='Raw-row split; source session grouping unavailable; feasibility inspected, not untouched policy heldout',
          rows=[dict(raw_by_id[i],physics_legal_return=True) for i in legal_ids])
    (BASE/'audited_candidates.json').write_text(json.dumps(export,indent=2)+'\n')
    print('DATASET_AUDIT_COMPLETE',len(legal_ids),len(diagonal),outcome_counts,flush=True)
    return summary


def summarize():
    from scipy.optimize import brentq
    from airbot_tt.planning import flight_state
    from airbot_tt.ball import load_ball_config
    c,raw=load();by_id={r['id']:r for r in raw};bc=load_ball_config()
    distribution=json.loads((BASE/'distribution.json').read_text());physics=json.loads((BASE/'physics_gate.json').read_text())
    assert physics['status']=='PASS' and physics['process_exit_code']==0
    first=json.loads((BASE/'position_search.json').read_text());rest=json.loads((BASE/'position_remaining.json').read_text())
    striker=json.loads((BASE/'striker_search.json').read_text());all_rows=first['rows']+rest['rows']
    assert {r['id'] for r in all_rows}=={r['id'] for r in striker['rows']}
    forecasts=json.loads((BASE/'forecasts.json').read_text());eligible=[f for f in forecasts if f['low_spin'] and f['own_bounce'] and f.get('current_task_bounds')]
    arrivals=[]
    for f in eligible:
        bounce=f['bounce'];post=np.r_[bounce['position_m'],bounce['velocity_m_s']];post[5]*=-bc['table_restitution']
        ending=f.get('post_bounce_end');duration=ending['time_s']-bounce['time_s'] if ending else 1.
        fun=lambda t:flight_state(post,t,bc)[0]+1.1
        if fun(0)*fun(duration)>0:continue
        tau=brentq(fun,0,duration);state=flight_state(post,tau,bc)
        arrivals.append(dict(id=f['id'],time_s=bounce['time_s']+tau,state=state.tolist()))
    rows=physics['zero_baseline']+physics['matrix'];legal_ids=sorted({r['id'] for r in rows if r['episode']['outcome']=='legal_return'})
    labels=list(dict.fromkeys(r['controller'] for r in physics['matrix']));serve_ids=physics['protocol']['matrix_serve_ids']
    matrix=[[[int(z['episode']['hit']),int(z['episode']['outcome']=='legal_return')]
             for i in serve_ids for z in physics['matrix'] if z['id']==i and z['controller']==label] for label in labels]
    errors=physics['bounce_errors']
    fingerprints={}
    for r in raw:
        key=tuple(r['state']+r['omega']);fingerprints.setdefault(key,[]).append(r['id'])
    duplicates=[ids for ids in fingerprints.values() if len(ids)>1]
    summary=dict(status='PASS',scope='Dataset feasibility audit, not PPO evaluation or real-spin validation',
      raw_count=len(raw),low_spin_count=sum(r['low_spin'] for r in raw),eligible_count=len(eligible),
      search_count=len(all_rows),position_command_candidates=sum(r['accepted'] for r in all_rows),
      m5_planner_candidates=striker['accepted'],unsearched_eligible=0,
      physical_distinct_raw_rows=len({r['id'] for r in rows}),physical_cases=len(rows),
      certified_return_ids=legal_ids,certified_return_count=len(legal_ids),
      direct_scripts_hit_count=sum(r['episode']['hit'] for r in physics['diagonal']),
      direct_scripts_legal_count=sum(r['episode']['outcome']=='legal_return' for r in physics['diagonal']),
      diagonal_outcomes=physics['diagonal_outcomes'],zero_outcomes=physics['zero_baseline_outcomes'],
      eligible_splits=dict(Counter(by_id[f['id']]['split'] for f in eligible)),
      contact_case_splits=dict(Counter(r['split'] for r in physics['diagonal'] if r['episode']['hit'])),
      matrix_labels=labels,matrix_serve_ids=serve_ids,matrix_hit_legal=matrix,
      max_one_tested_script_hit_count=max((sum(z[0] for z in row) for row in matrix),default=0),
      arrival_reference_x_m=-1.1,arrivals=arrivals,arrival_time_s=quantile([r['time_s'] for r in arrivals]),
      intercept_yz_m=quantile(np.array([r['state'][1:3] for r in arrivals])),
      bounce_position_error_m=quantile([e['position_error_m'] for e in errors]),
      bounce_time_error_s=quantile([e['time_error_s'] for e in errors]),
      exact_duplicate_groups=duplicates,duplicate_cross_split_groups=[g for g in duplicates if len({by_id[i]['split'] for i in g})>1],
      promotion_status='NOT_READY_NO_LEGAL_RETURNS' if not legal_ids else 'CANDIDATE_ONLY_REQUIRES_DISTRIBUTION_REVIEW',
      conclusions=['Current finite candidate families found ball-contact witnesses but no legal return witnesses',
        'No single tested frozen script contacts all matrix balls; this does not prove all possible scripts require conditioning',
        'Source has no session grouping; raw-ID splits are not proven independent sessions; inspected test states are not untouched policy tests'],
      input_sha256={name:sha(BASE/name) for name in ['distribution.json','search.json','position_search.json','position_remaining.json','striker_search.json','physics_gate.json','forecasts.json']},
      config_sha256=sha(CONFIG),code_sha256=sha(__file__))
    assert max(e['position_error_m'] for e in errors)<.02 and max(e['time_error_s'] for e in errors)<.01
    # Separate task-ready returns from contact-only diagnostic cases.
    exported=json.loads((BASE/'audited_candidates.json').read_text())
    exported['rows']=[dict(by_id[i],physics_legal_return=True) for i in legal_ids]
    exported['promotion_status']=summary['promotion_status']
    (BASE/'audited_candidates.json').write_text(json.dumps(exported,indent=2)+'\n')
    contacts=dict(schema='deepmind-airbot-contact-diagnostics-v1',source_commit=c['commit'],source_license='CC-BY-4.0',
        attribution='Google LLC, Competitive Robot Table Tennis: Initial Ball States (2025)',
        scope='Contact diagnostics only; no successful-return training dataset; raw omega retained, simulated omega zero',
        rows=[dict(by_id[r['id']],outcome=r['episode']['outcome'],controller=r['controller']) for r in physics['diagonal']])
    (BASE/'contact_cases.json').write_text(json.dumps(contacts,indent=2)+'\n')
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('AUDIT_SUMMARY',summary['position_command_candidates'],summary['certified_return_count'],summary['physical_cases'],summary['promotion_status'],flush=True)
    return summary


def return_witness_search(radius=None):
    """Bounded, roll-free impact search on three previously inspected train rows.

    Diagnostic only: preserves production M5 planner and all M7 controls.
    A failed search is not an infeasibility certificate.
    """
    from scipy.optimize import least_squares
    from airbot_tt.planning import flight_state, return_velocity
    from airbot_tt.trajectory import StrikeTrajectory
    started=time.monotonic()
    model,checker,bc,strike,rl,shot,base=search_context()
    if radius is not None:
        rl=dict(rl,action_scale_rad=[radius]*6)
    previous=[r for name in ('position_search','position_remaining')
              for r in json.loads((BASE/f'{name}.json').read_text())['rows']]
    chosen=[r for r in previous if r['accepted'] and r['split']=='train'][:3]
    forecasts={r['id']:r for r in json.loads((BASE/'forecasts.json').read_text())}
    protocol=dict(ids=[r['id'] for r in chosen],selection='First three train contact candidates in existing fixed hash order',
        offsets_s=[-.04,0,.04],goals_xy=[[.3,0],[.75,0],[1.1,0]],flight_s=[.35,.5,.65],
        action_scale_rad=rl['action_scale_rad'],velocity_limit_rad_s=strike['velocity_limit_rad_s'],
        acceleration_limit_rad_s2=strike['acceleration_limit_rad_s2'],
        ik_seeds=['existing_contact_q','ready'],ik_max_nfev=160,
        method='Ball-center and face-normal constrained IK; in-plane roll free; M5 normal-speed equation and StrikeTrajectory',
        source_sha256=source_hashes(),code_sha256=sha(__file__),
        input_sha256={n:sha(BASE/f'{n}.json') for n in ('position_search','position_remaining','forecasts')},
        scope='Offline zero-spin feasibility; only diagnostic action radius varies; no production/PPO/ready/physics/v/a changes')
    prefix='return_witness' if radius is None else f'return_radius_{radius:.1f}'
    destination=BASE/f'{prefix}_search.json'
    if destination.exists():raise FileExistsError('Preserve completed return search')
    (BASE/f'{prefix}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    nominal=np.array(strike['ready_q_rad']);scale=np.array(rl['action_scale_rad'])
    lower=np.maximum(model.lower,nominal-scale);upper=np.minimum(model.upper,nominal+scale)
    rows=[]
    for contact in chosen:
        bounce=forecasts[contact['id']]['bounce'];post=np.r_[bounce['position_m'],bounce['velocity_m_s']]
        post[5]*=-bc['table_restitution'];attempts=[];candidates=[]
        for offset in protocol['offsets_s']:
            arrival=contact['arrival_s']+offset
            incoming=flight_state(post,arrival-bounce['time_s'],bc)
            for goal in protocol['goals_xy']:
                for duration in protocol['flight_s']:
                    a=dict(arrival_s=arrival,goal_xy=goal,flight_s=duration);attempts.append(a)
                    try:outgoing=return_velocity(incoming[:3],goal,duration,bc)
                    except ValueError as error:a.update(reason=str(error));continue
                    normal=outgoing-incoming[3:];normal/=np.linalg.norm(normal)
                    a.update(incoming=incoming.tolist(),outgoing=outgoing.tolist(),normal=normal.tolist())
                    def residual(q):
                        fk=model.fk(q)
                        return np.r_[fk[:3,3]+base+bc['radius_m']*fk[:3,2]-incoming[:3],
                                     .2*(fk[:3,2]-normal)]
                    solutions=[least_squares(residual,np.clip(seed,lower+1e-9,upper-1e-9),bounds=(lower,upper),
                        max_nfev=protocol['ik_max_nfev'],ftol=1e-10,xtol=1e-10,gtol=1e-10)
                        for seed in (contact['target_q'],nominal)]
                    sol=min(solutions,key=lambda z:np.linalg.norm(z.fun));q=sol.x;pose=model.fk(q)
                    pe=float(np.linalg.norm(sol.fun[:3]));ne=float(np.arccos(np.clip(pose[:3,2]@normal,-1,1)))
                    a.update(position_error_m=pe,normal_error_rad=ne,ik_q_rad=q.tolist(),
                        active_bound_joints_1based=(np.flatnonzero(np.minimum(q-lower,upper-q)<1e-5)+1).tolist())
                    if pe>.0002 or ne>.002:a['reason']='impact_ik_not_solved';continue
                    # At a face-center sphere contact r=radius*n: n dot (omega cross r)=0.
                    # Thus the original M5 TCP normal-velocity equation is exact for this ideal point.
                    jacrow=normal@model.jacobian(q)[:3];e=bc['paddle_restitution']
                    speed=float((outgoing+e*incoming[3:])@normal/(1+e))
                    if jacrow@jacrow<1e-10:
                        a['reason']='normal_velocity_singular';continue
                    qd=jacrow*speed/(jacrow@jacrow);end=q+.5*strike['recovery_s']*qd
                    a.update(required_normal_speed_m_s=speed,strike_q=q.tolist(),strike_qd=qd.tolist())
                    try:
                        path=StrikeTrajectory(model,nominal,q,qd,end,arrival-strike['reaction_delay_s'],strike['recovery_s'],
                            vmax=strike['velocity_limit_rad_s'],amax=strike['acceleration_limit_rad_s2'],require_clear=checker.require_clear)
                        checked=action_preflight(model,checker,path,arrival,pose,strike,rl,shot)
                    except ValueError as error:
                        if 'solver' in str(error).lower() or 'linear program' in str(error).lower():raise
                        a.update(reason='trajectory_rejected',detail=str(error));continue
                    a['reason']='accepted' if checked['accepted'] else checked['reason']
                    if checked['accepted']:candidates.append(dict(id=contact['id'],attempt=len(attempts)-1,preflight=checked))
        rows.append(dict(id=contact['id'],split=contact['split'],attempts=attempts,candidates=candidates,
                         reasons=dict(Counter(a['reason'] for a in attempts))))
        print('RETURN_WITNESS_SEARCH',contact['id'],rows[-1]['reasons'],flush=True)
    assert source_hashes()==protocol['source_sha256']
    result=dict(status='PASS',protocol=protocol,rows=rows,candidate_count=sum(len(r['candidates']) for r in rows),
        seconds=time.monotonic()-started,ppo_updates=0,
        physics_status='NOT_RUN',conclusion='Command candidates require independent PhysX; zero candidates do not prove infeasibility')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_return_radius():
    """Paired finite search; original constraints except the action radius."""
    destination=BASE/'return_radius_comparison.json'
    if destination.exists():raise FileExistsError('Preserve completed radius comparison')
    reports=[return_witness_search(radius) for radius in (.6,1.)]
    original=json.loads((BASE/'return_witness_search.json').read_text())
    for old,new in zip(original['rows'],reports[0]['rows'],strict=True):
        assert old['id']==new['id'] and old['reasons']==new['reasons']
        for a,b in zip(old['attempts'],new['attempts'],strict=True):
            assert a['reason']==b['reason']
            np.testing.assert_allclose([a['position_error_m'],a['normal_error_rad']],
                [b['position_error_m'],b['normal_error_rad']],atol=1e-10,rtol=0)
    pairs=[]
    for left,right in zip(reports[0]['rows'],reports[1]['rows'],strict=True):
        assert left['id']==right['id']
        for a,b in zip(left['attempts'],right['attempts'],strict=True):
            for key in ('arrival_s','goal_xy','flight_s','incoming','outgoing','normal'):
                assert a[key]==b[key]
        pairs.append(dict(id=left['id'],r06=left['reasons'],r10=right['reasons']))
    result=dict(status='PASS',baseline_reproduced=True,paired_inputs_equal=True,rows=pairs,
        source_sha256=source_hashes(),input_sha256={f:sha(BASE/f) for f in
            ['return_radius_0.6_search.json','return_radius_1.0_search.json','return_witness_search.json']},
        candidate_counts=[r['candidate_count'] for r in reports],ppo_updates=0,physics_status='NOT_RUN')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print('RETURN_RADIUS_COMPARISON',result['rows'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--stage',choices=['distribution','search','position-search','position-remaining','striker-search','return-witness','return-radius','summarize'],default='distribution')
    args=parser.parse_args()
    if args.stage=='return-radius':compare_return_radius()
    elif args.stage=='return-witness':return_witness_search()
    elif args.stage=='distribution':distribution(args.workers)
    elif args.stage=='striker-search':striker_search(args.workers)
    elif args.stage=='summarize':summarize()
    else:search(args.workers,position_only=args.stage in ('position-search','position-remaining'),remaining=args.stage=='position-remaining')
