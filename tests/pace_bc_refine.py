"""Bounded corrective BC candidates and the original paired noise diagnostic.

Use --launch for the pinned host Docker launcher; fit/noise run inside Isaac.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'.agents/runs/foundation/m7/deepmind_five'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def finish_manifest(directory):
    """Only called after the fit subprocess really exited zero."""
    result,protocol=read(directory/'results.json'),read(directory/'protocol.json')
    assert result['status']=='PASS' and result['process_exit_code']==0
    manifest=dict(schema='airbot-contact-refinement-v1',status='PASS',
        contact_gate_pass=result['contact_gate']=='PASS',
        **{k:protocol[k] for k in ('cfg','recipe','source_sha256','bc_source_sha256','study_source_sha256','baseline_directory')},
        serve_bank_sha256=protocol['cfg']['serve_bank_sha256'],demonstration_sha256=digest(directory/'dataset.pt'),
        protocol_sha256=digest(directory/'protocol.json'),evaluation_sha256=digest(directory/'results.json'),
        actors={str(r['seed']):dict(path=f"actor_{r['seed']}.pt",sha256=r['actor_sha256'],contacts=r['contacts'],
                    original_initial_sha256=digest(Path(protocol['baseline_directory'])/f"A_{r['seed']}/initial.pt")) for r in result['runs']})
    save(directory/'initialization.json',manifest)


def launch():
    argv=[a for a in sys.argv[1:] if a!='--launch']
    stage=argv[argv.index('--stage')+1]
    output=Path(argv[argv.index('--output')+1]).resolve()
    report=output/('results.json' if stage=='fit' else 'noise.json')
    if report.exists() or (stage=='fit' and output.exists()):
        raise ValueError('Completed or partial study exists; preserve it and inspect before resuming')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'airbot-isaac'
    for sub in ('home','kit','logs'):(cache/sub).mkdir(parents=True,exist_ok=True)
    logfile=output.parent/f'{output.name}_{stage}.log'
    if logfile.exists():raise ValueError('Existing execution log; do not overwrite')
    image='sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e'
    command=['docker','run','--rm','--gpus','all','--shm-size=2g','--user',f'{os.getuid()}:{os.getgid()}',
        '-v',f'{ROOT}:{ROOT}','-w',str(ROOT),'-e',f'PYTHONPATH={ROOT}/src:{ROOT}/tests:{ROOT}/scripts',
        '-e','PYTHONUNBUFFERED=1','-e',f'TMPDIR={ROOT}/.agents/runs/foundation/tmp',
        '-e','PRIVACY_CONSENT=N','-e','HOME=/tmp/airbot-home','-e','NVIDIA_DRIVER_CAPABILITIES=all',
        '-v',f'{cache}/home:/tmp/airbot-home','-v',f'{cache}/kit:/isaac-sim/kit/cache',
        '-v',f'{cache}/logs:/isaac-sim/kit/logs','--tmpfs',f'/isaac-sim/kit/data:uid={os.getuid()},gid={os.getgid()},mode=0700',
        '--entrypoint','/bin/bash',image,'/workspace/isaaclab/isaaclab.sh','-p',str(Path(__file__).resolve()),*argv]
    print('REFINEMENT_LOG',logfile,flush=True)
    with logfile.open('x') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if completed.returncode:raise SystemExit(completed.returncode)
    if f'REFINEMENT_COMPLETE {stage} PASS' not in logfile.read_text():raise RuntimeError('Missing application completion marker')
    result=read(report);assert result['status']=='PASS'
    result['process_exit_code']=0;save(report,result)
    if stage=='fit':finish_manifest(output)
    logfile.rename(output/f'{stage}.log')
    print('REFINEMENT_EXIT0',report,flush=True)


if __name__=='__main__' and '--launch' in sys.argv:
    launch()
    raise SystemExit(0)

if __name__=='__main__':
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['fit','noise'],required=True)
    parser.add_argument('--candidate',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    app=AppLauncher(args,fast_shutdown=True).app

import numpy as np
import torch
from airbot_tt.pace_bc import bc_source_hashes,recipe
from airbot_tt.pace_serves import FIVE_SERVE_IDS
from contact_recovery_bc import training_data,fit_weighted_actor


def historical_hash(path, expected):
    """Old expected digests stay exact; only named development tools may reconstruct."""
    path=(ROOT/path).resolve()
    current=digest(path)
    if current==expected:return
    patches={ROOT/'src/airbot_tt/pace_bc.py':BASE/'refinement/legacy_pace_bc.patch',
             ROOT/'tests/pace_recovery_probe.py':BASE/'recovery_bc/probe/source_probe.patch',
             ROOT/'scripts/run_recovery_bc.sh':BASE/'recovery_bc/probe/source_launcher.patch'}
    if path not in patches:raise ValueError(f'Historical input changed: {path}')
    with tempfile.TemporaryDirectory() as tmp:
        old=Path(tmp)/'old'
        subprocess.run(['patch','--batch','--reverse','--fuzz=0','--output',str(old),str(path)],
                       input=patches[path].read_bytes(),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if digest(old)!=expected:raise ValueError(f'Historical reconstruction differs: {path}')


def assemble(original, recoveries, additions):
    """Preserve equal ball/source/trajectory mass while adding measured suffixes."""
    obs,actions,weights,meta=training_data(original,recoveries)
    counts={raw_id:sum(r['raw_id']==raw_id for r in additions) for raw_id in FIVE_SERVE_IDS}
    identities=set()
    offset=0
    for row in meta:
        n=row['samples']
        if row['source']=='recovery':
            old_count=sum(r['raw_id']==row['raw_id'] for r in recoveries)
            factor=old_count/(old_count+counts[row['raw_id']])
            weights[offset:offset+n]*=factor
            row['weight']*=factor;row['row_weight']*=factor
        row['identity']=f"original:{row['raw_id']}:{row['switch_tick']}"
        offset+=n
    all_obs,all_actions,all_weights=[obs],[actions],[weights]
    for row in additions:
        raw_id=row['raw_id'];ticks=row['ticks'];n=len(ticks)
        key=f"{row['collector_sha256']}:{raw_id}:{row['switch_tick']}"
        if raw_id not in counts or key in identities or n==0:raise ValueError('Invalid or repeated corrective trajectory')
        identities.add(key)
        if not torch.equal(ticks,torch.arange(row['switch_tick'],row['switch_tick']+n)):raise ValueError('Nonconsecutive corrective suffix')
        if row['observations'].shape!=(n,115) or row['actions'].shape!=(n,6):raise ValueError('Unpaired correction samples')
        if not torch.isfinite(row['observations']).all() or not torch.isfinite(row['actions']).all() or row['actions'].abs().max()>1+1e-7:raise ValueError('Invalid corrective values')
        count=sum(r['raw_id']==raw_id for r in recoveries)+counts[raw_id]
        mass=.1/count
        all_obs.append(row['observations']);all_actions.append(row['actions'])
        all_weights.append(torch.full((n,),mass/n,dtype=obs.dtype))
        meta.append(dict(raw_id=raw_id,source='recovery',switch_tick=row['switch_tick'],first_tick=int(ticks[0]),
                         last_tick=int(ticks[-1]),samples=n,weight=mass,row_weight=mass/n,identity=key))
    return dict(observations=torch.cat(all_obs),actions=torch.cat(all_actions),weights=torch.cat(all_weights),trajectories=meta)


def inputs_and_data():
    """Read physically verified historical data without changing any old receipt."""
    from pace_m7 import source_hashes
    original_bc=BASE/'bc';recovery=BASE/'recovery_bc'
    reports=[recovery/'fit/results.json',recovery/'recovery.json',recovery/'probe/probe.json',recovery/'remaining_probe/probe.json']
    for path in reports:
        report=read(path);assert report['status']=='PASS' and report['process_exit_code']==0
    fit=read(reports[0]);assert [r['contacts'] for r in fit['runs']]==[2,4,4]
    verification=read(recovery/'remaining_probe/verification.json')
    assert verification['status']=='PASS' and verification['process_exit_code']==0 and verification['old_9378']['status']=='PASS'
    assert verification['report_sha256']==digest(reports[-1])
    paths=set(reports+[recovery/'remaining_probe/verification.json'])
    for path in [original_bc/'protocol.json',recovery/'fit/protocol.json',recovery/'probe/protocol.json',recovery/'remaining_probe/protocol.json']:
        p=read(path);paths.add(path)
        assert p['source_sha256']==source_hashes()
        for field in ('input_sha256','bc_source_sha256'):
            for name,expected in p.get(field,{}).items():
                historical_hash(Path(name),expected);paths.add((ROOT/name).resolve())
    for directory in (recovery/'probe',recovery/'remaining_probe'):
        r=read(directory/'probe.json')
        assert r['protocol_sha256']==digest(directory/'protocol.json')
        assert r['traces_sha256']==digest(directory/'traces.npz')
        paths.update([directory/'probe.json',directory/'protocol.json',directory/'traces.npz'])
    assert read(recovery/'remaining_probe/probe.json')['candidates_sha256']==digest(recovery/'remaining_probe/candidates.pt')
    additions=torch.load(recovery/'remaining_probe/candidates.pt',map_location='cpu',weights_only=True)
    cases=read(recovery/'remaining_probe/probe.json')['runs']
    with np.load(recovery/'remaining_probe/traces.npz') as traces:
        for row in additions:
            case=next(c for c in cases if c['seed']==row['seed'] and c['raw_id']==row['raw_id'])
            assert case['episodes']['takeover']['hit'] and len(row['ticks'])==case['candidate_samples']
            prefix=f"s{row['seed']}_{row['raw_id']}_takeover_"
            indices=np.flatnonzero((traces[prefix+'tick']>=12)&traces[prefix+'prehit'])
            np.testing.assert_array_equal(row['ticks'],indices)
            np.testing.assert_array_equal(row['observations'],traces[prefix+'actual_observation'][indices])
            np.testing.assert_array_equal(row['actions'],traces[prefix+'action'][indices])
            row['collector_sha256']=digest(recovery/f"fit/actor_{row['seed']}.pt")
    with np.load(recovery/'probe/traces.npz') as traces:
        indices=np.flatnonzero((traces['takeover_tick']>=12)&traces['takeover_prehit'])
        assert len(indices)==33
        additions.insert(0,dict(seed=7101,raw_id=9378,switch_tick=12,ticks=torch.from_numpy(indices),
            observations=torch.from_numpy(traces['takeover_actual_observation'][indices].copy()),
            actions=torch.from_numpy(traces['takeover_action'][indices].copy()),collector_sha256=digest(recovery/'fit/actor_7101.pt')))
    assert [len(r['ticks']) for r in additions]==[33,21,15,34]
    original=torch.load(original_bc/'demonstrations.pt',map_location='cpu',weights_only=True)
    recoveries=torch.load(recovery/'recoveries.pt',map_location='cpu',weights_only=True)
    dataset=assemble(original,recoveries,additions)
    assert len(dataset['weights'])==792 and len(dataset['trajectories'])==24
    assert abs(float(dataset['weights'].sum())-1)<1e-6
    for raw_id in FIVE_SERVE_IDS:
        assert abs(sum(r['weight'] for r in dataset['trajectories'] if r['raw_id']==raw_id)-.2)<1e-8
    paths.update([original_bc/'demonstrations.pt',recovery/'recoveries.pt',recovery/'remaining_probe/candidates.pt',
                  BASE/'refinement/legacy_pace_bc.patch',ROOT/'scripts/contact_recovery_bc.py'])
    paths.update(recovery/'fit'/f'actor_{seed}.pt' for seed in (7101,7102,7103))
    return dataset,{str(p.resolve()):digest(p) for p in paths},read(original_bc/'protocol.json')


def new_policy(seed,cfg,device,baseline):
    from train_pace import original_ppo
    ActorCritic,_=original_ppo()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    p=cfg['ppo']
    policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
    initial=torch.load(baseline/f'A_{seed}/initial.pt',map_location='cpu',weights_only=False)
    assert initial['cfg']==cfg and initial['seed']==seed
    assert all(torch.equal(v.cpu(),initial['policy'][k]) for k,v in policy.state_dict().items())
    return policy


def fit_scheduled_actor(policy,observations,actions,weights):
    """Same weighted Adam fit, with official PyTorch cosine decay only."""
    cfg=recipe();before={k:v.detach().clone() for k,v in policy.state_dict().items()}
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['updates'],eta_min=1e-5)
    started=time.monotonic();history=[]
    for step in range(1,cfg['updates']+1):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.dot((policy.actor(observations)-actions).square().mean(1),weights)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite scheduled BC loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in policy.actor.parameters()):raise RuntimeError('Nonfinite scheduled BC gradient')
        used_lr=optimizer.param_groups[0]['lr']
        optimizer.step();scheduler.step()
        if step==1 or step%cfg['log_interval']==0:
            history.append(dict(update=step,loss=float(loss.detach()),learning_rate=used_lr))
        if time.monotonic()-started>cfg['maximum_fit_seconds']:raise RuntimeError('Fixed fit wall-time guard exceeded')
    assert scheduler.last_epoch==cfg['updates']
    current=policy.state_dict()
    assert all(torch.isfinite(v).all() for v in current.values())
    assert all(torch.equal(v,before[k]) for k,v in current.items() if not k.startswith('actor.'))
    with torch.no_grad():
        loss=torch.dot((policy.actor(observations)-actions).square().mean(1),weights)
        delta=torch.cat([(v-before[k]).flatten() for k,v in current.items() if k.startswith('actor.')]).norm()
    return dict(final_loss=float(loss),history=history,optimizer_steps=cfg['updates'],actor_update_norm=float(delta),
                critic_unchanged=True,std_unchanged=True,seconds=time.monotonic()-started,
                scheduler='CosineAnnealingLR',scheduler_steps=scheduler.last_epoch,
                final_next_learning_rate=optimizer.param_groups[0]['lr'])


def fit_candidate(output,candidate,device):
    from airbot_tt.pace_env import PaceEnv
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from contact_bc import same_episodes
    if candidate not in (1,2):raise ValueError('Only candidates1/2 have frozen recipes; decide later candidates from evidence first')
    started=time.monotonic();dataset,inputs,original=inputs_and_data()
    if candidate==2:
        previous=BASE/'refinement/candidate_1'
        old_result=read(previous/'results.json')
        assert old_result['status']=='PASS' and old_result['process_exit_code']==0 and old_result['contact_gate']=='FAIL'
        frozen=torch.load(previous/'dataset.pt',map_location='cpu',weights_only=True)
        assert frozen['trajectories']==dataset['trajectories']
        assert all(torch.equal(frozen[k],dataset[k]) for k in ('observations','actions','weights'))
        for name in ('dataset.pt','protocol.json','results.json','study_source.patch'):
            inputs[str((previous/name).resolve())]=digest(previous/name)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(dataset,output/'dataset.pt')
    candidate_recipe=dict(recipe(),candidate=candidate,trajectory_weighting='equal_ball_then_source_then_trajectory',
                          changed_factor='add_103_verified_new_policy_recovery_samples')
    if candidate==2:
        candidate_recipe.update(changed_factor='cosine_learning_rate_decay_only',scheduler='CosineAnnealingLR',
                                scheduler_T_max=2000,scheduler_eta_min=1e-5)
    protocol=dict(schema='airbot-contact-refinement-v1',recipe=candidate_recipe,cfg=original['cfg'],
        source_sha256=source_hashes(),bc_source_sha256=bc_source_hashes(),
        study_source_sha256={'tests/pace_bc_refine.py':digest(__file__)},input_sha256=inputs,
        dataset_sha256=digest(output/'dataset.pt'),baseline_directory=original['baseline_directory'],
        original_samples=237,old_recovery_samples=452,new_recovery_samples=103,
        trajectories=dataset['trajectories'],scope='Five inspected train balls; final checkpoints only; no changed control, observation, physics or success rules.')
    save(output/'protocol.json',protocol)
    torch.set_num_threads(1)
    env=PaceEnv(16,device,group='A',serve_bank=BASE/'manifest.json')
    obs,actions,weights=(dataset[k].to(device) for k in ('observations','actions','weights'))
    runs=[]
    for seed in (7101,7102,7103):
        policy=new_policy(seed,protocol['cfg'],device,Path(protocol['baseline_directory']))
        fit=fit_weighted_actor if candidate==1 else fit_scheduled_actor
        fitting=fit(policy,obs,actions,weights)
        artifact=output/f'actor_{seed}.pt'
        torch.save(dict(seed=seed,actor={k:v.detach().cpu().clone() for k,v in policy.actor.state_dict().items()},
                        protocol_sha256=digest(output/'protocol.json'),fitting=fitting),artifact)
        evaluation=evaluate_bank(env,policy,capture=True)
        restored=new_policy(seed,protocol['cfg'],device,Path(protocol['baseline_directory']))
        restored.actor.load_state_dict(torch.load(artifact,map_location='cpu',weights_only=True)['actor'])
        with torch.no_grad():assert torch.equal(policy.act_inference(obs),restored.act_inference(obs))
        replay=evaluate_bank(env,restored,capture=True);same_episodes(evaluation,replay)
        with torch.no_grad():
            errors=(policy.actor(obs)-actions).square().mean(1);offset=0;fits=[]
            for row in dataset['trajectories']:
                fits.append(dict(identity=row['identity'],raw_id=row['raw_id'],source=row['source'],rmse=float(errors[offset:offset+row['samples']].mean().sqrt())))
                offset+=row['samples']
        contacts=sum(r['hit'] for r in evaluation['episodes'])
        run=dict(seed=seed,actor_sha256=digest(artifact),fitting=fitting,evaluation=evaluation,replay_evaluation=replay,
                 contacts=contacts,contact_gate_pass=contacts>=3,trajectory_fits=fits)
        runs.append(run);save(output/f'bc_{seed}.json',run)
        print('REFINEMENT_CANDIDATE',candidate,seed,contacts,'legal',evaluation['legal_return_rate'],'loss',fitting['final_loss'],flush=True)
    assert all(digest(p)==sha for p,sha in inputs.items())
    assert protocol['source_sha256']==source_hashes() and protocol['bc_source_sha256']==bc_source_hashes()
    assert protocol['study_source_sha256']['tests/pace_bc_refine.py']==digest(__file__)
    result=dict(status='PASS',candidate=candidate,contact_gate='PASS' if all(r['contact_gate_pass'] for r in runs) else 'FAIL',
        runs=runs,scored_episodes=15,reload_episodes=15,bc_updates=sum(r['fitting']['optimizer_steps'] for r in runs),
        ppo_transitions=0,noise_status='NOT_RUN',protocol_sha256=digest(output/'protocol.json'),seconds=time.monotonic()-started)
    save(output/'results.json',result)
    return result


def noise(directory,device):
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_bc import validate_initialization,load_actor_initialization
    from contact_bc import noisy_evaluation
    from pace_m7 import source_hashes
    protocol=read(directory/'protocol.json')
    manifest=validate_initialization(directory,protocol['cfg'],source_hashes())
    env=PaceEnv(16,device,group='A',serve_bank=BASE/'manifest.json')
    torch.set_num_threads(1);started=time.monotonic();runs=[]
    for seed in (7101,7102,7103):
        for kind in ('original','refined_bc'):
            policy=new_policy(seed,protocol['cfg'],device,Path(protocol['baseline_directory']))
            if kind=='refined_bc':load_actor_initialization(policy,directory,seed,protocol['cfg'],source_hashes())
            before={k:v.clone() for k,v in policy.state_dict().items()}
            trials=[noisy_evaluation(env,policy,seed,rep) for rep in range(recipe()['noise_repetitions'])]
            assert all(torch.equal(v,before[k]) for k,v in policy.state_dict().items())
            runs.append(dict(seed=seed,kind=kind,trials=trials))
            print('NOISE',seed,kind,sum(sum(e['hit'] for e in t['episodes']) for t in trials),flush=True)
    total=sum(t['total'] for r in runs for t in r['trials']);assert total==300
    result=dict(status='PASS',total=total,runs=runs,diagnostic_only=True,ppo_veto=False,
                initialization_sha256=digest(directory/'initialization.json'),seconds=time.monotonic()-started)
    save(directory/'noise.json',result)
    return result


if __name__=='__main__':
    code=0
    try:
        result=fit_candidate(args.output,args.candidate,args.device) if args.stage=='fit' else noise(args.output,args.device)
        print('REFINEMENT_COMPLETE',args.stage,result['status'],flush=True)
    except BaseException:
        import traceback
        traceback.print_exc();code=1
        raise
    finally:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance();app.app.post_quit(code);app.close()
