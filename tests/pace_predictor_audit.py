"""Cross-evaluate archived predictors and audit live data contracts; never trains."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from airbot_tt.config import ROOT
from airbot_tt.pace_predictor import Predictor, windows

BASE=ROOT/'.agents/runs/foundation/m7/m74e'
DEST=BASE/'data_audit'
SEEDS=(7101,7102,7103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(prediction,target):
    error=np.asarray(prediction,dtype=float)-target
    if not len(error):return dict(n=0)
    bias=error.mean(0);centered=error-bias
    rmse2=np.mean(np.sum(error**2,axis=1));bias2=float(bias@bias)
    centered2=np.mean(np.sum(centered**2,axis=1))
    assert np.isclose(rmse2,bias2+centered2,atol=1e-12)
    return dict(n=len(error),rmse_m=float(np.sqrt(rmse2)),mean_error_xyz_m=bias.tolist(),
                xyz_rmse_m=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                bias_norm_m=float(np.sqrt(bias2)),centered_rmse_m=float(np.sqrt(centered2)),
                p95_m=float(np.percentile(np.linalg.norm(error,axis=1),95)))


def stats(value):
    a=np.asarray(value)
    return dict(mean=a.mean(0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())


def data(name):
    d=dict(np.load(BASE/'data'/f'{name}.npz'))
    x,y,ends=windows(d['state'][:,:3],d['time'],d['serve'],np.ones(len(d['time']),bool),d['target'],d['valid'])
    return d,x,y,ends


def phases(d,ends):
    cross=np.array([np.any((d['state'][e-4:e,5]<0)&(d['state'][e-3:e+1,5]>0)) for e in ends])
    return np.where(cross,'across_bounce',np.where(d['state'][ends,5]>0,'post_bounce','pre_bounce'))


def predictors():
    training=json.loads((BASE/'offline/training.json').read_text())
    assert training['status']=='PASS' and sha(BASE/'offline/predictor.pt')==training['checkpoint_sha256']
    offline=Predictor();offline.load(BASE/'offline/predictor.pt')
    result={'offline':offline};buffers={};inputs={str(BASE/'offline/predictor.pt'):training['checkpoint_sha256']}
    for seed in SEEDS:
        pair=[]
        for batch in (1024,32):
            path=BASE/'batch_comparison'/f'predictor_{"batch32_" if batch==32 else ""}D_{seed}.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['config']['batch_size']==batch and state['attempts']==20
            model=Predictor();model.load_state_dict(state['predictor'])
            result[f'online_{seed}_b{batch}']=model;inputs[str(path)]=sha(path)
            pair.append(state['buffer'])
        assert len(pair[0])==len(pair[1])==128
        assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(*pair))
        x=torch.cat([a for a,b in pair[0]]).numpy();y=torch.cat([b for a,b in pair[0]]).numpy()
        assert np.isfinite(x).all() and np.isfinite(y).all()
        buffers[seed]=(x,y)
    return result,buffers,inputs


def cross_evaluate():
    started=time.monotonic();torch.set_num_threads(1);DEST.mkdir(parents=True,exist_ok=True)
    collection=json.loads((BASE/'data/dataset.json').read_text())
    for name in ('train','validation'):
        assert sha(BASE/'data'/f'{name}.npz')==collection['splits'][name]['sha256']
    train,tx,ty,te=data('train');val,x,y,ends=data('validation');phase=phases(val,ends)
    average=np.mean([ty[train['serve'][te]==sid].mean(0) for sid in np.unique(train['serve'][te])],axis=0)
    models,buffers,inputs=predictors()
    inputs.update({str(BASE/'data'/f'{n}.npz'):sha(BASE/'data'/f'{n}.npz') for n in ('train','validation')})
    inputs[str(BASE/'batch_comparison.json')]=sha(BASE/'batch_comparison.json')
    protocol=dict(scope='Archived inference only; no fitting, heldout access, checkpoint selection or policy evaluation',
                  input_sha256=inputs,code_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                  'src/airbot_tt/pace_predictor.py':sha(ROOT/'src/airbot_tt/pace_predictor.py')},
                  teacher_comparison='Teacher labels on both datasets; PhysX truth only for validation',
                  online_metadata='Checkpoint contains x,y only; env/time/phase unavailable, not inferred as facts')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report=dict(status='PASS',protocol=protocol,train_mean_apex_m=average.tolist(),cross=[],validation_by_phase={})
    report['validation_stats']=dict(last_position_m=stats(x.reshape(-1,5,3)[:,-1]),teacher_m=stats(y),
                                    phase_counts={p:int(sum(phase==p)) for p in np.unique(phase)})
    for name,model in list(models.items())+[('mean',None)]:
        prediction=model.predict(x).numpy() if model else np.tile(average,(len(x),1))
        report['cross'].append(dict(model=name,dataset='validation',teacher=metric(prediction,y),truth=metric(prediction,val['truth'][ends])))
        report['validation_by_phase'][name]={p:dict(teacher=metric(prediction[phase==p],y[phase==p]),
                 truth=metric(prediction[phase==p],val['truth'][ends][phase==p])) for p in np.unique(phase)}
    report['buffer_stats']={};report['archived_samples']={};report['validation_x_coverage']={}
    previous=json.loads((BASE/'batch_comparison.json').read_text())
    for seed,(bx,by) in buffers.items():
        report['buffer_stats'][str(seed)]=dict(windows=len(bx),last_position_m=stats(bx.reshape(-1,5,3)[:,-1]),teacher_m=stats(by))
        min_x,max_x=bx[:,-3].min(),bx[:,-3].max()
        inside=(x[:,-3]>=min_x)&(x[:,-3]<=max_x)
        report['validation_x_coverage'][str(seed)]=dict(
            definition='Last X within archived buffer min/max only; not full multidimensional support',
            range_m=[float(min_x),float(max_x)],within=int(inside.sum()),outside=int((~inside).sum()),
            by_phase={p:dict(within=int(sum(inside&(phase==p))),total=int(sum(phase==p))) for p in np.unique(phase)},
            models={})
        for name in ['offline',f'online_{seed}_b32','mean']:
            prediction=models[name].predict(x).numpy() if name!='mean' else np.tile(average,(len(x),1))
            report['validation_x_coverage'][str(seed)]['models'][name]=dict(
                within=metric(prediction[inside],y[inside]),outside=metric(prediction[~inside],y[~inside]))
        names=['offline',f'online_{seed}_b1024',f'online_{seed}_b32','mean']
        for name in names:
            prediction=models[name].predict(bx).numpy() if name!='mean' else np.tile(average,(len(bx),1))
            report['cross'].append(dict(model=name,dataset=f'buffer_{seed}',teacher=metric(prediction,by)))
        report['archived_samples'][str(seed)]=[dict(index=int(i),positions_m=bx[i].reshape(5,3).tolist(),
            teacher_m=by[i].tolist(),metadata='env/serve/time/phase not saved') for i in sorted({0,len(bx)//2,len(bx)-1})]
        old=next(run for run in previous['runs'] if run['behavior_seed']==seed)['history'][-1]
        for batch in (1024,32):
            row=next(z for z in report['cross'] if z['model']==f'online_{seed}_b{batch}' and z['dataset']=='validation')
            assert np.isclose(row['truth']['rmse_m'],(old if batch==1024 else old['batch32'])['validation_rmse_m'],atol=1e-7)
    report['seconds']=time.monotonic()-started
    for path,h in inputs.items():assert sha(path)==h
    (DEST/'cross.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in report['cross']:
        print(row['dataset'],row['model'],'teacher_rmse_m',row['teacher']['rmse_m'],flush=True)
    return report


def replay(device='cuda:0',num_envs=16):
    """One 72-step frozen-D replay with metadata; zero MSE/PPO updates."""
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import OnlinePredictor
    from airbot_tt.pace_guidance import apex_teacher
    from pace_m7 import source_hashes
    from train_pace import original_ppo
    if num_envs!=16:raise ValueError('Contract replay uses 16 environments')
    if (DEST/'replay_samples.json').exists():raise FileExistsError('Preserve replay evidence')
    started=time.monotonic();torch.set_num_threads(1);before=source_hashes()
    seed=7102;path=ROOT/f'.agents/runs/foundation/m7/m74d/d/D_{seed}/policy.pt'
    policy_sha=sha(path);checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    ActorCritic,_=original_ppo()
    env=PaceEnv(num_envs,device,group='E',serve_z_m=2.3,serve_y_half_range_m=.015)
    cfg=env.cfg['ppo'];policy=ActorCritic(115,200,6,actor_hidden_dims=cfg['hidden'],
           critic_hidden_dims=cfg['hidden'],activation='elu',init_noise_std=cfg['init_std']).to(device)
    policy.load_state_dict(checkpoint['policy'])
    env.online=OnlinePredictor(num_envs);env.prediction_collect=True
    samples=[];history=[[] for _ in range(num_envs)];initial={};stored=[]
    original_observe=env.online.observe;original_collect=env.online.collect
    def observe(position,times,serves,valid,ids=None):
        selected=range(num_envs) if ids is None else ids
        state=env.state().detach().cpu().numpy()
        for i in selected:
            stamp=(int(serves[i]),float(times[i]))
            if history[i] and (history[i][-1]['serve']!=stamp[0] or abs(stamp[1]-history[i][-1]['time_s']-env.policy_dt)>1e-7):history[i]=[]
            row=dict(serve=stamp[0],time_s=stamp[1],state_m_m_s=state[i].tolist(),phase=int(env.events.phase[i]))
            if stamp[1]==0:initial[(i,stamp[0])]=state[i].tolist()
            history[i].append(row);history[i]=history[i][-5:]
        return original_observe(position,times,serves,valid,ids)
    def collect(target,valid):
        original_collect(target,valid)
        bx,by=env.online.buffer[-1];offset=0
        for i,h in enumerate(env.online.hook.history):
            if not (valid[i] and len(h)==5 and np.isfinite(target[i]).all()):continue
            mh=history[i];assert len(mh)==5
            assert len({r['serve'] for r in mh})==1
            assert np.allclose(np.diff([r['time_s'] for r in mh]),env.policy_dt,atol=1e-7,rtol=0)
            positions=np.array([r['state_m_m_s'][:3] for r in mh])
            assert np.array_equal(positions.reshape(15).astype(np.float32),bx[offset].numpy())
            assert np.array_equal(target[i],by[offset].numpy())
            phase='across_bounce' if mh[0]['phase']==0 and mh[-1]['phase']==1 else ('post_bounce' if mh[-1]['phase']==1 else 'pre_bounce')
            current=np.array(mh[-1]['state_m_m_s'])
            reteacher=apex_teacher(current,bounced=mh[-1]['phase']==1)
            origin=env.origins[i].cpu().numpy();world=env.ball.data.root_pos_w[i].cpu().numpy()
            assert np.allclose(world-origin,current[:3],atol=1e-6,rtol=0)
            samples.append(dict(env=i,serve=mh[-1]['serve'],origin_m=origin.tolist(),world_position_m=world.tolist(),
                  time_s=mh[-1]['time_s'],elapsed_s=float(env.elapsed[i]),window=mh.copy(),positions_m=positions.tolist(),
                  phase=phase,event_phase=mh[-1]['phase'],outcome=int(env.events.outcome[i]),hit=bool(env.events.hit[i]),
                  initial_state=initial[(i,mh[-1]['serve'])],teacher_m=target[i].tolist(),recomputed_teacher=reteacher,
                  teacher_absolute_time_s=float(env.teacher_time[i])))
            assert not samples[-1]['hit'] and samples[-1]['outcome']==0 and env.elapsed[i]<env.teacher_time[i]
            offset+=1
        assert offset==len(bx)
        stored.append((bx,by,bx.clone(),by.clone()))
    env.online.observe=observe;env.online.collect=collect
    env.rng=[np.random.default_rng(np.random.SeedSequence([7540,i])) for i in range(num_envs)]
    env.reset(env.ids);episodes=[]
    for _ in range(72):
        with torch.no_grad():
            actor,_=env.observations()
            action=policy.act_inference(actor.reshape(num_envs,5,27)[:,:,:23].reshape(num_envs,115))
            _,_,_,_,info=env.step(action);episodes.extend(info['episodes'])
    assert env.online.attempts==0 and env.online.optimizer_steps==0
    assert all(torch.equal(a,c) and torch.equal(b,d) for a,b,c,d in stored)
    assert all(torch.equal(v.cpu(),checkpoint['policy'][k]) for k,v in policy.state_dict().items())
    assert before==source_hashes() and sha(path)==policy_sha
    models,_,inputs=predictors();train,_,ty,te=data('train')
    mean=np.mean([ty[train['serve'][te]==sid].mean(0) for sid in np.unique(train['serve'][te])],axis=0)
    x=np.array([z['positions_m'] for z in samples],dtype=np.float32).reshape(-1,15)
    y=np.array([z['teacher_m'] for z in samples]);phase=np.array([z['phase'] for z in samples]);env_ids=np.array([z['env'] for z in samples])
    recomputed=np.array([z['recomputed_teacher']['valid'] for z in samples])
    report=dict(status='PASS',scope='Fresh 72-step metadata replay, not reconstruction of archived final buffer; no training',
                behavior_seed=seed,serve_seed=7540,transitions=72*num_envs,predictor_updates=0,ppo_updates=0,
                source_sha256=before,policy_sha256=policy_sha,code_sha256=sha(__file__),windows=len(samples),episodes=len(episodes),
                snapshot_immutability=True,policy_unchanged=True,phase_counts={p:int(sum(phase==p)) for p in np.unique(phase)},
                by_model={},recomputed_teacher_invalid_reasons={})
    for z in samples:
        t=z['recomputed_teacher']
        if not t['valid']:report['recomputed_teacher_invalid_reasons'][t['reason']]=report['recomputed_teacher_invalid_reasons'].get(t['reason'],0)+1
    rt=np.array([z['recomputed_teacher']['point'] for z in samples])
    report['reset_vs_current_teacher']=metric(y[recomputed],rt[recomputed])
    report['teacher_difference_by_phase']={p:metric(y[(phase==p)&recomputed],rt[(phase==p)&recomputed]) for p in np.unique(phase)}
    for name in ['offline',f'online_{seed}_b1024',f'online_{seed}_b32','mean']:
        prediction=models[name].predict(x).numpy() if name!='mean' else np.tile(mean,(len(x),1))
        report['by_model'][name]=dict(all=metric(prediction,y),
              by_phase={p:metric(prediction[phase==p],y[phase==p]) for p in np.unique(phase)},
              by_env={str(i):metric(prediction[env_ids==i],y[env_ids==i]) for i in range(num_envs)})
    selected=[]
    for i in (0,num_envs-1):
        for p in ('pre_bounce','across_bounce','post_bounce'):
            candidates=[z for z in samples if z['env']==i and z['phase']==p]
            if candidates:selected.append(candidates[len(candidates)//2])
    report['representative_samples']=selected
    report['seconds']=time.monotonic()-started
    (DEST/'replay_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    (DEST/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    cross_evaluate()
