"""C/D release/reward and free-flight prediction against frozen-policy PhysX replays."""
import hashlib
import json
import time
import numpy as np
import torch
from airbot_tt.config import ROOT


def verify(device='cuda:0',num_envs=16,group='C'):
    if group not in ('C','D'):raise ValueError('Expected C or D')
    reward_group=group
    if num_envs!=16:raise ValueError('Frozen reference has 16 environments')
    from train_pace import original_ppo,evaluate
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    before=source_hashes();started=time.monotonic()
    env=PaceEnv(16,device,group=reward_group,serve_z_m=2.3,serve_y_half_range_m=.015);torch.set_num_threads(1)
    ActorCritic,_=original_ppo();p=env.cfg['ppo'];base=ROOT/'.agents/runs/foundation/m7/m74'
    reference=json.loads((base/'final_summary.json').read_text());runs=[];original=env.physics_step;records={};finished=set()
    c_reference=None
    if reward_group=='D':
        c_path=ROOT/'.agents/runs/foundation/m7/m74c/return_gate.json'
        protocol=json.loads((c_path.parent/'protocol.json').read_text())
        assert hashlib.sha256(c_path.read_bytes()).hexdigest()==protocol['return_gate_sha256']
        c_reference=json.loads(c_path.read_text());assert c_reference['status']=='PASS'
    reset_original=env.reset;reset_counts={'all':0,'partial':0}
    def checked_reset(ids):
        keep=torch.ones(env.n,dtype=torch.bool,device=device);keep[ids]=False
        fields=[env.episode_return,env.episode_guidance,env.release.reason,env.release.started]
        saved=[x.clone() for x in fields];records_before=list(env.return_records)
        result=reset_original(ids)
        for current,previous in zip(fields,saved):assert torch.equal(current[keep],previous[keep])
        assert not env.episode_return[ids].any() and not env.episode_guidance[ids].any()
        assert not env.release.reason[ids].any() and (env.release.started[ids]==-1).all()
        for i in range(env.n):
            if bool(keep[i]):assert env.return_records[i] is records_before[i]
            else:assert env.return_records[i] is None
        reset_counts['all' if len(ids)==env.n else 'partial']+=1
        return result
    env.reset=checked_reset
    def step():
        previous=env.state().clone();times=env.elapsed.clone();reward_before=env.episode_return.clone()
        result=original();state=env.state()
        for i in range(16):
            if i in finished:continue
            record=env.return_records[i]
            if record is not None:
                row=records.setdefault(i,dict(release=record,reward_updates=0,intervening_contact=False))
                if abs(float(env.episode_return[i]-reward_before[i]))>1e-8:row['reward_updates']+=1
                assert row['reward_updates']<=1
                assert record['time_s']>float(env.release.started[i])
                if float(env.elapsed[i])>record['time_s']+1e-6:
                    row['intervening_contact'] |= bool(env.events.contacts[i,1:].any()) or int(env.events.outcome[i])==2
                if float(previous[i,0])<0<=float(state[i,0]) and float(state[i,3])>0 and 'net' not in row:
                    fraction=float(-previous[i,0]/(state[i,0]-previous[i,0]));point=previous[i,:3]+fraction*(state[i,:3]-previous[i,:3])
                    row['net']=dict(point=point.cpu().tolist(),time_s=float(times[i])+fraction*env.dt)
            if int(env.events.outcome[i]):
                row=records.setdefault(i,dict(release=None,reward_updates=0,intervening_contact=False))
                row.update(outcome=env.events.names[int(env.events.outcome[i])],time_s=float(env.elapsed[i]),landing=env.events.landing[i].cpu().tolist(),return_reward=float(env.episode_return[i]),release_reason=env.release.reasons[int(env.release.reason[i])])
                finished.add(i)
        return result
    env.physics_step=step
    errors=[];unexplained=[];valid_count=0;interrupted=0
    for group in ['A','B']:
        for seed in [7101,7102,7103]:
            records={};finished=set();path=base/'ab'/f'{group}_{seed}'/'policy.pt'
            sha=hashlib.sha256(path.read_bytes()).hexdigest();saved=json.loads(path.with_name('result.json').read_text());assert sha==saved['checkpoint_sha256']
            ck=torch.load(path,map_location=device,weights_only=False)
            policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device);policy.load_state_dict(ck['policy'])
            evaluation=evaluate(env,policy,7511,16)
            expected=next(r for r in reference['runs'] if r['group']==group and r['seed']==seed)['evaluations']['active']['episodes'][:16]
            matches=all((a['outcome'],a['hit'])==(b['outcome'],b['hit']) for a,b in zip(evaluation['episodes'],expected))
            composition_error=0.
            if c_reference is not None:
                c_run=next(r for r in c_reference['runs'] if r['group']==group and r['seed']==seed)
                assert c_run['checkpoint_sha256']==sha
                for actual,c in zip(evaluation['episodes'],c_run['evaluation']['episodes']):
                    assert (actual['outcome'],actual['hit'])==(c['outcome'],c['hit'])
                    assert abs(actual['return_guidance']-c['return_guidance'])<1e-5
                    assert abs(actual['guidance_sum']-c['guidance_sum'])<1e-5
                    composition_error=max(composition_error,abs(actual['reward']-c['reward']-c['guidance_sum']))
                assert composition_error<1e-4
            for i,row in records.items():
                release=row['release']
                if release is None:continue
                assert abs(row['return_reward']-release['reward'])<1e-5
                pred=release['prediction']
                if not pred['valid']:
                    assert row['return_reward']==0
                    continue
                valid_count+=1
                if row['intervening_contact']:
                    interrupted+=1;continue
                if row['outcome']!='legal_return' or 'net' not in row:
                    unexplained.append(dict(group=group,seed=seed,env=i,record=row));continue
                errors.append(dict(group=group,seed=seed,env=i,
                    landing_m=float(np.linalg.norm(np.array(pred['landing'])-row['landing'])),
                    net_height_m=abs(pred['net_z']-row['net']['point'][2]),
                    landing_time_s=abs(release['time_s']+pred['landing_time_s']-row['time_s']),
                    net_time_s=abs(release['time_s']+pred['net_time_s']-row['net']['time_s'])))
            runs.append(dict(group=group,seed=seed,matches_saved=matches,composition_error=composition_error,evaluation=evaluation,records=records,checkpoint_sha256=sha))
            print('RETURN_GATE',group,seed,matches,evaluation['outcomes'],flush=True)
    p95={k:float(np.percentile([r[k] for r in errors],95)) if errors else None for k in ['landing_m','net_height_m','landing_time_s','net_time_s']}
    checks=dict(replay=all(r['matches_saved'] for r in runs),coverage=len(errors)>=8,no_unexplained=not unexplained,
        position=bool(errors and p95['landing_m']<=.02 and p95['net_height_m']<=.02),
        timing=bool(errors and p95['landing_time_s']<=.01 and p95['net_time_s']<=.01))
    checks['reset_isolation']=reset_counts['partial']>0
    if c_reference is not None:checks['combined_reward']=all(r['composition_error']<1e-4 for r in runs)
    assert before==source_hashes()
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,reward_group=reward_group,reset_counts=reset_counts,runs=runs,errors=errors,p95=p95,
        valid_predictions=valid_count,intervening_contacts=interrupted,unexplained=unexplained,source_sha256=before,seconds=time.monotonic()-started,
        scope='All 96 frozen replay serves retained; free-flight errors separately exclude measured intervening contacts')
