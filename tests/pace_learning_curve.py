"""Bounded online predictor diagnosis under frozen D policies; no PPO updates."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from airbot_tt.config import ROOT


def verify(device='cuda:0',num_envs=16,compare_batches=False):
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import OnlinePredictor,windows
    from pace_m7 import source_hashes
    from train_pace import original_ppo
    if num_envs!=16:raise ValueError('Frozen diagnosis uses 16 environments')
    destination=ROOT/'.agents/runs/foundation/m7/m74e'/('batch_comparison' if compare_batches else 'learning_curve')
    destination.mkdir(parents=True,exist_ok=False)
    before=source_hashes();started=time.monotonic();torch.set_num_threads(1)
    data_dir=destination.parent/'data'
    collection=json.loads((data_dir/'dataset.json').read_text())
    for name in ('train','validation'):
        assert hashlib.sha256((data_dir/f'{name}.npz').read_bytes()).hexdigest()==collection['splits'][name]['sha256']
    assert collection['distribution']['serve_position_m'][2]==2.3
    def read(name):
        data=dict(np.load(data_dir/f'{name}.npz'))
        x,y,ends=windows(data['state'][:,:3],data['time'],data['serve'],np.ones(len(data['time']),bool),data['target'],data['valid'])
        return data,x,y,ends
    train,tx,ty,te=read('train');validation,x,y,ends=read('validation')
    truth=validation['truth'][ends]
    average=np.mean([ty[train['serve'][te]==sid].mean(0) for sid in np.unique(train['serve'][te])],axis=0)
    mean_rmse=float(np.sqrt(np.mean(np.sum((truth-average)**2,axis=1))))
    seeds=[7101,7102,7103];paths={seed:ROOT/f'.agents/runs/foundation/m7/m74d/d/D_{seed}/policy.pt' for seed in seeds}
    hashes={str(seed):hashlib.sha256(path.read_bytes()).hexdigest() for seed,path in paths.items()}
    protocol=dict(scope='Frozen D behavior, online MSE only; no PPO, no heldout reopening, no policy efficacy comparison',
        training_seeds=seeds,serve_seed=7540,num_envs=16,rollout_steps=24,update_opportunities=20,
        transitions_per_stream=7680,source_sha256=before,policy_sha256=hashes,
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((data_dir/'dataset.json').read_bytes()).hexdigest(),
        batch_sizes=[1024,32] if compare_batches else [1024],
        pairing='Identical initial weights, buffer tensors and per-epoch shuffle RNG; batch size only; unequal optimizer work',
        checkpoint_selection='last; no candidate tuning',config=json.loads((ROOT/'configs/pace_online.json').read_text()))
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=PaceEnv(num_envs,device,group='E',serve_z_m=2.3,serve_y_half_range_m=.015)
    ActorCritic,_=original_ppo();p=env.cfg['ppo'];runs=[]
    for seed in seeds:
        checkpoint=torch.load(paths[seed],map_location='cpu',weights_only=False)
        saved=json.loads(paths[seed].with_name('result.json').read_text())
        assert saved['checkpoint_sha256']==hashes[str(seed)] and checkpoint['group']=='D'
        assert checkpoint['cfg']['serve_z_m']==2.3 and checkpoint['cfg']['serve_y_half_range_m']==.015
        policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
        policy.load_state_dict(checkpoint['policy'])
        env.online=OnlinePredictor(num_envs);env.prediction_collect=True
        shadow=OnlinePredictor(num_envs) if compare_batches else None
        if shadow is not None:
            shadow.cfg['batch_size']=32
            assert env.online.cfg['batch_size']==1024
            assert all(torch.equal(v,shadow.predictor.model.state_dict()[k]) for k,v in env.online.predictor.model.state_dict().items())
        env.rng=[np.random.default_rng(np.random.SeedSequence([7540,i])) for i in range(num_envs)]
        env.reset(env.ids);history=[];episodes=[]
        for iteration in range(20):
            counts=dict(prehit_steps=0,hint_steps=0,hint_reference_steps=0,hint_error_sq=0.,release_events=0,return_valid_events=0)
            with torch.no_grad():
                for _ in range(24):
                    actor,_=env.observations()
                    # D never receives the new features; its policy remains frozen.
                    action=policy.act_inference(actor.reshape(num_envs,5,27)[:,:,:23].reshape(num_envs,115))
                    _,_,_,_,info=env.step(action)
                    episodes.extend(info['episodes'])
                    for key,field in [('prehit_steps','prehit'),('hint_steps','prediction_used'),('hint_reference_steps','hint_reference'),('release_events','return_events'),('return_valid_events','return_valid_events')]:counts[key]+=int(info[field].sum())
                    counts['hint_error_sq']+=float(info['hint_error_sq'].sum())
            if shadow is not None:
                # One simulation stream supplies both learners; no second collection.
                shadow.buffer.clear();shadow.buffer.extend(env.online.buffer)
                assert torch.equal(env.online.predictor.rng.get_state(),shadow.predictor.rng.get_state())
                digest=hashlib.sha256()
                for a,b in env.online.buffer:
                    digest.update(np.asarray(a.shape,dtype=np.int64).tobytes())
                    digest.update(a.numpy().tobytes());digest.update(b.numpy().tobytes())
            update=env.online.update()
            if shadow is not None:
                small_update=shadow.update()
                assert update['samples']==small_update['samples']
                assert torch.equal(env.online.predictor.rng.get_state(),shadow.predictor.rng.get_state())
            env.refresh_after_predictor_update()
            prediction=env.online.predictor.predict(x).numpy()
            row=dict(iteration=iteration+1,transitions=(iteration+1)*24*num_envs,update=update,coverage=counts,
                validation_rmse_m=float(np.sqrt(np.mean(np.sum((prediction-truth)**2,axis=1)))),
                validation_teacher_rmse_m=float(np.sqrt(np.mean(np.sum((prediction-y)**2,axis=1)))))
            if shadow is not None:
                small_prediction=shadow.predictor.predict(x).numpy()
                row['batch32']=dict(update=small_update,
                    validation_rmse_m=float(np.sqrt(np.mean(np.sum((small_prediction-truth)**2,axis=1)))),
                    validation_teacher_rmse_m=float(np.sqrt(np.mean(np.sum((small_prediction-y)**2,axis=1)))))
                row['paired_buffer_sha256']=digest.hexdigest()
            history.append(row)
            with (destination/f'D_{seed}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        weights={k:v.clone() for k,v in env.online.predictor.model.state_dict().items()}
        optimizer_steps=env.online.optimizer_steps;rng=env.online.predictor.rng.get_state().clone()
        cutoff=env.online.update()
        assert cutoff['status']=='cutoff' and env.online.attempts==20 and env.online.optimizer_steps==optimizer_steps
        assert torch.equal(rng,env.online.predictor.rng.get_state())
        assert all(torch.equal(weights[k],v) for k,v in env.online.predictor.model.state_dict().items())
        assert all(torch.equal(v.cpu(),checkpoint['policy'][k]) for k,v in policy.state_dict().items())
        torch.save(env.online.state_dict(),destination/f'predictor_D_{seed}.pt')
        run=dict(behavior_seed=seed,history=history,episodes=episodes,cutoff=cutoff,policy_unchanged=True)
        if shadow is not None:
            shadow_weights={k:v.clone() for k,v in shadow.predictor.model.state_dict().items()}
            shadow_rng=shadow.predictor.rng.get_state().clone();shadow_steps=shadow.optimizer_steps
            small_cutoff=shadow.update()
            assert small_cutoff['status']=='cutoff' and shadow.attempts==20 and shadow.optimizer_steps==shadow_steps
            assert torch.equal(shadow_rng,shadow.predictor.rng.get_state())
            assert all(torch.equal(shadow_weights[k],v) for k,v in shadow.predictor.model.state_dict().items())
            torch.save(shadow.state_dict(),destination/f'predictor_batch32_D_{seed}.pt')
            run['batch32_cutoff']=small_cutoff
            run['paired_initialization_data_shuffle']=True
            print('PREDICTOR_BATCH32',seed,shadow.optimizer_steps,history[-1]['batch32']['validation_rmse_m'],flush=True)
        runs.append(run)
        (destination/f'D_{seed}.json').write_text(json.dumps(run,indent=2)+'\n')
        print('PREDICTOR_CURVE',seed,optimizer_steps,history[-1]['validation_rmse_m'],flush=True)
    assert before==source_hashes()
    return dict(status='PASS',scope=protocol['scope'],protocol=protocol,runs=runs,validation_windows=len(x),
        train_mean_validation_rmse_m=mean_rmse,source_sha256=before,seconds=time.monotonic()-started,
        scientific_conclusion='Report predictor precision separately from execution PASS; frozen D does not consume hints')
