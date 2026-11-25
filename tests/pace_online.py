"""E observable history and independent-MSE contracts in the actual vector scene."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from airbot_tt.config import ROOT


def verify(device='cuda:0',num_envs=2):
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import OnlinePredictor
    from pace_m7 import source_hashes
    if num_envs<2:raise ValueError('At least two environments required for isolation')
    before=source_hashes();started=time.monotonic()
    env=PaceEnv(num_envs,device,group='E',serve_z_m=2.3,serve_y_half_range_m=.015)
    action=torch.zeros(num_envs,6,device=device);checks={}
    actor,critic=env.observations()
    assert actor.shape==(num_envs,135) and critic.shape==(num_envs,200)
    assert not actor.reshape(num_envs,5,27)[:,:,26].any()
    for _ in range(8):env.step(action,auto_reset=False)
    assert not env.prediction[:,3].any() and env.online.predictor.updates==0
    checks['untrained_is_invalid']=True
    status=env.online.update();assert status['status']=='updated'
    old_histories=copy.deepcopy(env.online.hook.history)
    obs,_=env.refresh_after_predictor_update()
    assert all(np.array_equal(a,b) for a,b in zip(old_histories,env.online.hook.history))
    assert obs.reshape(num_envs,5,27)[:,-1,26].all()
    checks['update_refresh_without_duplicate_sample']=True
    # Controlled real observations with the now-trained network, no offline pretrain.
    env.reset(env.ids);positions=[env.state()[:,:3].cpu().numpy().copy()]
    for _ in range(8):
        obs,_,_,_,_=env.step(action,auto_reset=False)
        positions.append(env.state()[:,:3].cpu().numpy().copy())
    frames=obs.reshape(num_envs,5,27)
    for frame,end in enumerate(range(4,9)):
        x=np.stack(positions[end-4:end+1],axis=1).reshape(num_envs,15)
        expected=env.online.predictor.predict(x).to(device).clamp(-10,10)
        assert torch.allclose(frames[:,frame,23:26],expected,atol=1e-6,rtol=0)
        assert frames[:,frame,26].all()
    checks['same_step_and_nested_nine_samples']=True
    assert torch.allclose(env.first_prediction,torch.full_like(env.first_prediction,4/48))
    assert env.hint_steps.min()==4  # actions at t4,t5,t6,t7 consume a valid latest hint
    checks['availability_and_consumption_are_distinct']=True
    original_teacher=env.teacher_point.clone();a,_=env.frames()
    env.teacher_point+=.123;b,_=env.frames();env.teacher_point.copy_(original_teacher)
    assert torch.equal(a,b);checks['teacher_not_in_actor']=True
    saved_history=copy.deepcopy(env.online.hook.history[1:]);saved_actor=env.actor_history[1:].clone()
    env.reset([0]);assert len(env.online.hook.history[0])==1
    assert all(np.array_equal(a,b) for a,b in zip(saved_history,env.online.hook.history[1:]))
    assert torch.equal(saved_actor,env.actor_history[1:]);assert not env.actor_history[0,:,26].any()
    checks['partial_reset_isolation']=True
    # Invalid sensor input clears only that environment and cannot reuse a stale hint.
    h=env.online.hook;prior=copy.deepcopy(h.history[1:]);p=env.state()[:,:3].cpu().numpy();p[0]=np.nan
    out=h.observe(p,[0.]*num_envs,env.serve_id.cpu().numpy(),[False]+[True]*(num_envs-1),ids=[0])
    assert not out[0].any() and h.history[0]==[]
    assert all(np.array_equal(a,b) for a,b in zip(prior,h.history[1:]))
    checks['invalid_observation_clears_history']=True
    # Empty windows cannot cause an optimizer update; exact cutoff persists in a checkpoint.
    empty=OnlinePredictor(num_envs);assert empty.update()['status']=='no_valid_windows'
    restored=OnlinePredictor(num_envs);restored.load_state_dict(copy.deepcopy(env.online.state_dict()))
    expected=env.online.update();actual=restored.update()
    assert expected['mse']==actual['mse']
    assert all(torch.equal(a,b) for a,b in zip(env.online.predictor.model.parameters(),restored.predictor.model.parameters()))
    restored.attempts=20;assert restored.update()['status']=='cutoff'
    checks['checkpoint_optimizer_rng_and_cutoff']=True
    assert before==source_hashes()
    return dict(status='PASS',checks=checks,source_sha256=before,seconds=time.monotonic()-started,
                actor_dim=135,critic_dim=200,first_prediction_s=4/48,first_consumed_action_s=4/48,
                predictor_window_span_s=4/48,nested_actor_history_span_s=8/48,
                scope='Real vector interface with one diagnostic online MSE update; not predictor accuracy or E policy efficacy')


def verify_frozen(device='cuda:0',num_envs=2,artifact=None):
    """Real vector gate, inference only. Must be run after AppLauncher, never mocked."""
    import tempfile
    from pathlib import Path
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import FrozenPredictor
    from pace_m7 import source_hashes
    if num_envs<2 or artifact is None:raise ValueError('Frozen gate requires 2+ envs and explicit artifact')
    before=source_hashes();started=time.monotonic();checks={}
    env=PaceEnv(num_envs,device,group='E-frozen',frozen_predictor=artifact,serve_z_m=2.3,serve_y_half_range_m=.015)
    frozen=env.frozen;weights={k:v.clone() for k,v in frozen.predictor.model.state_dict().items()}
    assert env.online is None and frozen.predictor.optimizer is None
    action=torch.zeros(num_envs,6,device=device);obs,critic=env.observations()
    assert obs.shape==(num_envs,135) and critic.shape==(num_envs,200)
    assert not obs.reshape(num_envs,5,27)[:,:,26].any()
    checks['explicit_frozen_no_optimizer']=True
    positions=[env.state()[:,:3].cpu().numpy().copy()];writes=env.state_writes
    snapshots=[]
    for tick in range(1,9):
        obs,critic,_,done,_=env.step(action,auto_reset=False)
        assert not done.any()
        assert bool(env.prediction[:,3].all())==(tick>=4)
        positions.append(env.state()[:,:3].cpu().numpy().copy());snapshots.append((obs,obs.clone()))
    assert env.state_writes==writes
    frames=obs.reshape(num_envs,5,27)
    for frame,end in enumerate(range(4,9)):
        x=np.stack(positions[end-4:end+1],axis=1).reshape(num_envs,15)
        expected=frozen.predictor.predict(x).to(device).clamp(-env.cfg['observation_clip'],env.cfg['observation_clip'])
        assert torch.equal(frames[:,frame,23:26],expected) and frames[:,frame,26].all()
    assert torch.allclose(env.first_prediction,torch.full_like(env.first_prediction,4/48))
    assert torch.equal(env.hint_steps,torch.full_like(env.hint_steps,4))
    for actual,saved in snapshots:assert torch.equal(actual,saved)
    checks['same_step_nine_instants_snapshot_no_state_write']=True
    teacher=env.teacher_point.clone();a,c=env.frames();env.teacher_point+=.123;b,_=env.frames();env.teacher_point.copy_(teacher)
    assert torch.equal(a,b);checks['critic200_teacher_excluded_from_actor']=True
    kept=copy.deepcopy(frozen.hook.history[1:]);kept_actor=env.actor_history[1:].clone();kept_critic=env.critic_history[1:].clone()
    env.reset([0]);assert len(frozen.hook.history[0])==1
    assert all(np.array_equal(a,b) for a,b in zip(kept,frozen.hook.history[1:]))
    assert torch.equal(kept_actor,env.actor_history[1:]) and torch.equal(kept_critic,env.critic_history[1:])
    assert not env.actor_history[0,:,26].any();checks['partial_reset_isolation']=True
    bad=env.state()[:,:3].cpu().numpy().copy();bad[0]=np.nan
    out=frozen.observe(bad,env.policy_index.cpu().numpy()*env.policy_dt,env.serve_id.cpu().numpy(),[False]+[True]*(num_envs-1),ids=[0])
    assert not out[0].any() and not frozen.hook.history[0]
    assert all(np.array_equal(a,b) for a,b in zip(kept,frozen.hook.history[1:]))
    checks['invalid_clears_selected_history']=True
    probe=np.stack(positions[4:9],axis=1).reshape(num_envs,15)
    with tempfile.TemporaryDirectory(dir=ROOT/'.agents/runs/foundation/tmp') as directory:
        path=Path(directory)/'frozen.pt';torch.save(frozen.state_dict(),path)
        restored=FrozenPredictor(num_envs,artifact)
        restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        assert torch.equal(frozen.predictor.predict(probe),restored.predictor.predict(probe))
        assert all(not h for h in restored.hook.history)
    assert all(torch.equal(v,weights[k]) for k,v in frozen.predictor.model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in frozen.predictor.model.parameters())
    assert not frozen.predictor.model.training;checks['reload_immutable_eval_no_grad']=True
    assert before==source_hashes()
    return dict(status='PASS',checks=checks,source_sha256=before,predictor_provenance=frozen.provenance,
        seconds=time.monotonic()-started,actor_dim=135,critic_dim=200,mode='frozen',
        first_prediction_s=4/48,predictor_window_span_s=4/48,nested_actor_history_span_s=8/48,
        scope='Actual vector inference/reset gate; no PPO or predictor updates, not a learning result')


def evaluate_frozen(device,num_envs,source,artifact):
    """Predeclared Z2.3 paired development evaluation; all320 episodes retained."""
    import hashlib
    from pathlib import Path
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import FrozenPredictor
    from train_pace import original_ppo,evaluate,validate_predictor_checkpoint,set_predictor,frozen_experiment_metadata
    from pace_m7 import source_hashes
    if num_envs!=16 or artifact is None:raise ValueError('Paired evaluation requires16 envs and explicit frozen artifact')
    source=Path(source)
    spec_path=ROOT/'configs/pace_frozen_experiment.json';spec=json.loads(spec_path.read_text())
    protocol=json.loads((source/'protocol.json').read_text())
    cfg=spec['common_cfg'];frozen=FrozenPredictor(num_envs,artifact);before=source_hashes()
    experiment=frozen_experiment_metadata(cfg)
    # All input contracts are checked before starting physical evaluation.
    if (protocol['groups']!=['E-frozen'] or protocol['seeds']!=spec['seeds']
        or protocol['num_envs']!=16 or protocol['transitions_per_trial']!=6144
        or protocol['iterations']!=16 or protocol['rollout_steps']!=24
        or protocol['predictor_mode']!='frozen'
        or protocol['predictor_provenance']!=frozen.provenance
        or protocol['source_sha256']!=before
        or any(protocol.get(k)!=v for k,v in experiment.items())):
        raise ValueError('Paired evaluation requires the complete fixed frozen training protocol/current source')
    if protocol['cfg']!=cfg:raise ValueError('Paired PPO/control/serve configuration mismatch')
    inputs=[]
    for seed in spec['seeds']:
        row=spec['fixed_d'][str(seed)];path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Fixed D checkpoint changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['group']!='D' or ck['seed']!=seed or ck['cfg']!=cfg or ck['iteration']!=16:
            raise ValueError('Fixed D training contract mismatch')
        inputs.append(dict(label='D',seed=seed,path=path,sha256=row['sha256'],checkpoint=ck))
        trial=source/f'E-frozen_{seed}'
        final=torch.load(trial/'policy.pt',map_location='cpu',weights_only=False)
        result=json.loads((trial/'result.json').read_text())
        if (result['status']!='PASS' or result['transitions']!=6144 or final['iteration']!=16
            or result['source_sha256']!=protocol['source_sha256']):
            raise ValueError('E final checkpoint is not the complete fixed16-update trial')
        for label,name,sha in [('E_initial','initial.pt',final['initial_sha256']),('E_final','policy.pt',result['checkpoint_sha256'])]:
            path=trial/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('E checkpoint changed')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if (ck['seed']!=seed or ck['group']!='E-frozen' or ck['cfg']!=protocol['cfg']
                or ck['source_sha256']!=protocol['source_sha256']
                or any(ck.get(k)!=v for k,v in experiment.items())):
                raise ValueError('E checkpoint/protocol mismatch')
            # Helper guards mode, actor schema and exact embedded model provenance.
            validate_predictor_checkpoint(ck,'E-frozen')
            if ck['predictor_provenance']!=frozen.provenance:raise ValueError('E predictor provenance mismatch')
            frozen.load_state_dict(ck['predictor'])
            inputs.append(dict(label=label,seed=seed,path=path,sha256=sha,checkpoint=ck))
    started=time.monotonic();ActorCritic,_=original_ppo();p=cfg['ppo']
    env=PaceEnv(num_envs,device,group='E-frozen',frozen_predictor=artifact,serve_z_m=2.3,serve_y_half_range_m=.015)
    torch.set_num_threads(1);rows=[]
    class Zero:
        def act_inference(self,obs):return torch.zeros(len(obs),6,device=obs.device)
    zero=evaluate(env,Zero(),spec['evaluation']['seed'],32)
    for row in inputs:
        ck=row['checkpoint'];is_d=row['label']=='D'
        set_predictor(env,'D' if is_d else 'E-frozen',None if is_d else artifact,checkpoint=ck)
        env.actor_width=23 if is_d else 27
        env.actor_history=torch.zeros(num_envs,5,env.actor_width,device=device)
        policy=ActorCritic(env.actor_width*5,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
        policy.load_state_dict(ck['policy']);policy.eval()
        evaluation=evaluate(env,policy,spec['evaluation']['seed'],32)
        assert all(torch.equal(v.detach().cpu(),ck['policy'][k]) for k,v in policy.state_dict().items())
        if env.frozen:env.frozen.state_dict()  # Exact frozen-weight audit after physical replay.
        rows.append(dict(label=row['label'],seed=row['seed'],checkpoint_sha256=row['sha256'],evaluation=evaluation))
    differences=[]
    for seed in spec['seeds']:
        by_label={r['label']:r['evaluation'] for r in rows if r['seed']==seed}
        differences.append(dict(seed=seed,final_minus_d={k:by_label['E_final'][k]-by_label['D'][k] for k in ('legal_return_rate','hit_rate','target_success_rate')},
            final_minus_initial={k:by_label['E_final'][k]-by_label['E_initial'][k] for k in ('legal_return_rate','hit_rate','target_success_rate')}))
    assert sum(r['evaluation']['total'] for r in rows)+zero['total']==320
    assert before==source_hashes()
    for row in inputs:assert hashlib.sha256(row['path'].read_bytes()).hexdigest()==row['sha256']
    return dict(status='PASS',scope='Paired development evaluation; engineering completion does not establish efficacy',
        protocol=spec,protocol_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),source_sha256=before,
        predictor_provenance=frozen.provenance,runs=rows,zero=zero,differences=differences,
        total_scored_episodes=320,seconds=time.monotonic()-started,scientific_conclusion='REQUIRES_ALL_SEED_INTERPRETATION')


def _mean_inputs(num_envs,artifact):
    """Validate historical receipts against their pinned sources, never rewrite them."""
    from airbot_tt.pace_predictor import FixedMeanPredictor,FrozenPredictor
    from train_pace import frozen_experiment_metadata,validate_predictor_checkpoint
    if artifact is None:raise ValueError('An explicit mean artifact is required')
    spec_path=ROOT/'configs/pace_mean_experiment.json'
    spec=json.loads(spec_path.read_text());cfg=spec['common_cfg']
    metadata=frozen_experiment_metadata(cfg,'E-mean')
    learned_metadata=frozen_experiment_metadata(cfg)
    mean=FixedMeanPredictor(num_envs,artifact)
    if mean.provenance['artifact_sha256']!=spec['mean_artifact']['sha256']:
        raise ValueError('Mean artifact differs from the fixed experiment')
    frozen_artifact=mean.cfg['files']['frozen_artifact']['path']
    frozen=FrozenPredictor(num_envs,frozen_artifact)
    references=[spec['mean_artifact'],*spec['baseline_reports'].values()]
    references.extend(row[name] for row in spec['fixed_learned'].values() for name in ('initial','final','result'))
    hashes={item['path']:item['sha256'] for item in references}
    hashes[str(spec_path.relative_to(ROOT))]=metadata['experiment_spec_sha256']
    reports={name:json.loads((ROOT/item['path']).read_text()) for name,item in spec['baseline_reports'].items()}
    historical=reports['training_protocol'];old=reports['paired_evaluation']
    prefix=['zero']
    for seed in spec['seeds']:prefix.extend([f'D_{seed}',f'learned_initial_{seed}',f'learned_final_{seed}'])
    tail=[f'mean_{stage}_{seed}' for seed in spec['seeds'] for stage in ('initial','final')]
    evaluation=spec['evaluation'];amendment=spec['sequence_amendment']
    if (evaluation['execution_order']!=prefix+tail or amendment['historical_prefix']!=prefix
        or evaluation['context_policies']!=[f'D_{seed}' for seed in spec['seeds']]
        or evaluation['context_scored_episodes']!=96 or evaluation['actual_scored_episodes']!=512
        or evaluation['compatibility_primary_episodes']!=224 or evaluation['compatibility_actual_episodes']!=320):
        raise ValueError('Mean historical sequence/context budget contract mismatch')
    for item in amendment['evidence'].values():
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Replay-order diagnostic evidence changed')
        hashes[item['path']]=item['sha256']
    if (historical['cfg']!=cfg or historical['groups']!=['E-frozen']
        or historical['seeds']!=spec['seeds'] or historical['num_envs']!=16
        or historical['iterations']!=16 or historical['rollout_steps']!=24
        or historical['transitions_per_trial']!=6144
        or historical['predictor_provenance']!=frozen.provenance
        or any(historical.get(k)!=v for k,v in learned_metadata.items())
        or old['status']!='PASS' or old['source_sha256']!=historical['source_sha256']
        or old['predictor_provenance']!=frozen.provenance
        or old['protocol']!=reports['frozen_experiment']
        or old['protocol_sha256']!=learned_metadata['experiment_spec_sha256']):
        raise ValueError('Historical learned experiment contract mismatch')
    inputs=[]
    for seed in spec['seeds']:
        # D is part of the original simulator execution history. Its episodes
        # remain separately accounted context, never extra primary comparisons.
        item=reports['frozen_experiment']['fixed_d'][str(seed)]
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Historical D context checkpoint changed')
        hashes[item['path']]=item['sha256']
        ck=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
        validate_predictor_checkpoint(ck,'D')
        if (ck['seed']!=seed or ck['group']!='D' or ck['cfg']!=cfg or ck['iteration']!=16
            or ck['policy']['actor.0.weight'].shape[1]!=115
            or ck['policy']['critic.0.weight'].shape[1]!=200):
            raise ValueError('Historical D context policy/config mismatch')
        reference=next(r for r in old['runs'] if r['label']=='D' and r['seed']==seed)
        if reference['checkpoint_sha256']!=item['sha256']:
            raise ValueError('Historical D context evaluation/checkpoint mismatch')
        inputs.append(dict(label='D',seed=seed,group='D',artifact=None,context=True,
                           path=ROOT/item['path'],sha256=item['sha256'],checkpoint=ck,
                           reference=reference['evaluation']))
        row=spec['fixed_learned'][str(seed)]
        result=json.loads((ROOT/row['result']['path']).read_text())
        if (result['status']!='PASS' or result['transitions']!=6144
            or result['checkpoint_sha256']!=row['final']['sha256']
            or result['source_sha256']!=historical['source_sha256']):
            raise ValueError('Historical learned result/checkpoint mismatch')
        for stage in ('initial','final'):
            item=row[stage];ck=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
            validate_predictor_checkpoint(ck,'E-frozen')
            if (ck['seed']!=seed or ck['cfg']!=cfg or ck['critic_dim']!=200
                or ck['source_sha256']!=historical['source_sha256']
                or ck['predictor_provenance']!=frozen.provenance
                or any(ck.get(k)!=v for k,v in learned_metadata.items())):
                raise ValueError('Historical learned policy/source/provenance mismatch')
            if stage=='final' and (ck['iteration']!=16 or ck['initial_sha256']!=row['initial']['sha256']):
                raise ValueError('Historical learned checkpoint is not the fixed last model')
            frozen.load_state_dict(ck['predictor'])
            label='learned_'+stage
            reference=next(r for r in old['runs'] if r['label']=='E_'+stage and r['seed']==seed)
            if reference['checkpoint_sha256']!=item['sha256']:
                raise ValueError('Historical evaluated checkpoint mismatch')
            inputs.append(dict(label=label,seed=seed,group='E-frozen',artifact=frozen_artifact,
                               path=ROOT/item['path'],sha256=item['sha256'],checkpoint=ck,
                               reference=reference['evaluation']))
    return spec,metadata,mean,inputs,old,hashes


def _rng_equal(cpu,cuda):
    return (torch.equal(cpu,torch.get_rng_state())
            and len(cuda)==torch.cuda.device_count()
            and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all(),strict=True)))


def verify_mean(device='cuda:0',num_envs=2,artifact=None):
    """Actual CUDA/PhysX gate; constant hints share the observable history contract."""
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_predictor import FixedMeanPredictor,FrozenPredictor
    from train_pace import original_ppo,pair_mean_initialization
    from pace_m7 import source_hashes
    if num_envs<2 or torch.device(device).type!='cuda' or not torch.cuda.is_available():
        raise ValueError('Mean gate requires 2+ actual CUDA environments')
    started=time.monotonic();before=source_hashes();checks={}
    spec,metadata,mean,_,_,hashes=_mean_inputs(num_envs,artifact)
    ActorCritic,_=original_ppo();p=spec['common_cfg']['ppo'];rng_rows=[]
    saved_cpu=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all()
    try:
        for seed in spec['seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
            learned=FrozenPredictor(num_envs,mean.cfg['files']['frozen_artifact']['path'])
            assert _rng_equal(cpu,cuda), 'Learned constructor changed seeded policy RNG'
            candidate=FixedMeanPredictor(num_envs,artifact)
            assert _rng_equal(cpu,cuda), 'Mean constructor changed seeded policy RNG'
            candidate.load_state_dict(copy.deepcopy(candidate.state_dict()))
            assert _rng_equal(cpu,cuda), 'Mean checkpoint load changed seeded policy RNG'
            policy=ActorCritic(135,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],
                               activation='elu',init_noise_std=p['init_std']).to(device)
            assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all(),strict=True))
            paired=pair_mean_initialization(policy,seed,spec['common_cfg'])
            paired['post_actor_cuda_rng_sha256']=[hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                                 for v in torch.cuda.get_rng_state_all()]
            rng_rows.append(paired)
    finally:
        torch.set_rng_state(saved_cpu);torch.cuda.set_rng_state_all(saved_cuda)
    checks['actual_cpu_cuda_rng_and_all_three_initial_policy_tensors']=True
    env=PaceEnv(num_envs,device,group='E-mean',mean_predictor=artifact,serve_z_m=2.3,serve_y_half_range_m=.015)
    mean=env.predictor_module;model=mean.predictor.model
    fixed={k:v.clone() for k,v in model.state_dict().items()}
    assert env.online is None and mean.predictor.optimizer is None and mean.predictor.rng is None
    assert not list(model.parameters()) and not hasattr(mean,'buffer')
    action=torch.zeros(num_envs,6,device=device);obs,critic=env.observations()
    assert obs.shape==(num_envs,135) and critic.shape==(num_envs,200)
    assert not obs.reshape(num_envs,5,27)[:,:,26].any()
    expected=torch.tensor(mean.cfg['constant_point_m'],dtype=torch.float32,device=device)
    shadow=FrozenPredictor(num_envs,mean.cfg['files']['frozen_artifact']['path'])
    positions=[env.state()[:,:3].cpu().numpy().copy()]
    shadow.observe(positions[0],env.policy_index.cpu().numpy()*env.policy_dt,
                   env.serve_id.cpu().numpy(),np.ones(num_envs,bool))
    writes=env.state_writes;snapshots=[]
    for tick in range(1,9):
        obs,critic,_,done,_=env.step(action,auto_reset=False)
        assert not done.any() and bool(env.prediction[:,3].all())==(tick>=4)
        positions.append(env.state()[:,:3].cpu().numpy().copy())
        other=shadow.observe(positions[-1],env.policy_index.cpu().numpy()*env.policy_dt,
                             env.serve_id.cpu().numpy(),np.ones(num_envs,bool)).to(device)
        assert torch.equal(env.prediction[:,3],other[:,3])
        frames=obs.reshape(num_envs,5,27);valid=frames[:,:,26].bool()
        assert torch.equal(frames[:,:,23:26][valid],expected.expand(int(valid.sum()),3))
        assert not frames[:,:,23:26][~valid].any()
        snapshots.append((obs,obs.clone()))
    assert env.state_writes==writes
    assert torch.allclose(env.first_prediction,torch.full_like(env.first_prediction,4/48))
    assert torch.equal(env.hint_steps,torch.full_like(env.hint_steps,4))
    assert all(torch.equal(actual,saved) for actual,saved in snapshots)
    checks['same_step_constant_learned_validity_snapshots_no_state_writes']=True
    teacher=env.teacher_point.clone();a,_=env.frames();env.teacher_point+=.123;b,_=env.frames();env.teacher_point.copy_(teacher)
    assert torch.equal(a,b);checks['critic200_teacher_excluded_from_actor']=True
    kept=copy.deepcopy(mean.hook.history[1:]);kept_actor=env.actor_history[1:].clone();kept_critic=env.critic_history[1:].clone()
    env.reset([0]);assert len(mean.hook.history[0])==1
    assert all(np.array_equal(a,b) for a,b in zip(kept,mean.hook.history[1:]))
    assert torch.equal(kept_actor,env.actor_history[1:]) and torch.equal(kept_critic,env.critic_history[1:])
    assert not env.actor_history[0,:,26].any();checks['partial_reset_isolation']=True
    position=env.state()[:,:3].cpu().numpy().copy();serve=env.serve_id.cpu().numpy()
    for invalid_kind in ('position','valid','time','gap'):
        mean.hook.reset([0])
        for tick in range(5):
            output=mean.observe(position,np.full(num_envs,tick/48),serve,np.ones(num_envs,bool),ids=[0])
        assert output[0,3]==1
        bad=position.copy();valid=np.ones(num_envs,bool);stamp=np.full(num_envs,5/48)
        if invalid_kind=='position':bad[0]=np.nan
        elif invalid_kind=='valid':valid[0]=False
        elif invalid_kind=='time':stamp[0]=np.nan
        else:stamp[0]=7/48
        output=mean.observe(bad,stamp,serve,valid,ids=[0])
        assert not output[0].any() and len(mean.hook.history[0])==(1 if invalid_kind=='gap' else 0)
        assert all(np.array_equal(a,b) for a,b in zip(kept,mean.hook.history[1:]))
    checks['invalid_nonfinite_and_gaps_clear_selected_history']=True
    probe=np.stack(positions[4:9],axis=1).reshape(num_envs,15)
    with tempfile.TemporaryDirectory(dir=ROOT/'.agents/runs/foundation/tmp') as directory:
        path=Path(directory)/'mean.pt';torch.save(mean.state_dict(),path)
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        restored=FixedMeanPredictor(num_envs,artifact)
        restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        assert _rng_equal(cpu,cuda)
        assert torch.equal(mean.predictor.predict(probe),restored.predictor.predict(probe))
        assert all(not h for h in restored.hook.history)
    assert all(torch.equal(v,fixed[k]) for k,v in model.state_dict().items())
    assert not model.training and mean.predictor.updates==0
    assert not any(v.requires_grad or v.grad is not None for v in model.buffers())
    checks['disk_reload_immutable_constant_no_optimizer_no_grad']=True
    assert before==source_hashes()
    for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    return dict(status='PASS',checks=checks,source_sha256=before,predictor_provenance=mean.provenance,
        **metadata,baseline_sha256=hashes,initialization=rng_rows,seconds=time.monotonic()-started,
        actor_dim=135,critic_dim=200,mode='mean',runtime=dict(torch=torch.__version__,device=device),
        constant_point_m=mean.cfg['constant_point_m'],
        first_prediction_s=4/48,predictor_window_span_s=4/48,actor_history_span_s=4/48,
        scope='Actual CUDA/PhysX constant-hint interface and RNG gate; zero predictor/PPO updates')


def _compare_learned_replay(current,previous):
    """Pair by predeclared batch/env; lifetime serve counters may differ."""
    if current['event_rules']!=previous['event_rules'] or current['total']!=previous['total'] or current['total']!=32:
        raise AssertionError('Learned replay event rules/denominator changed')
    old=previous['episodes'];new=current['episodes'];landing_max=0.;matched=[]
    assert len(old)==len(new)==32
    for index,(a,b) in enumerate(zip(new,old,strict=True)):
        assert a['env']==b['env']==index%16, 'Replay order changed'
        for key in ('outcome','hit','net_touched','target_success','release_reason'):
            assert a[key]==b[key], (index//16,index%16,key,a[key],b[key])
        assert (a['landing_xy_m'] is None)==(b['landing_xy_m'] is None)
        if a['landing_xy_m'] is not None:
            difference=float(np.max(np.abs(np.asarray(a['landing_xy_m'])-b['landing_xy_m'])))
            assert np.isfinite(difference) and difference<=1e-5, ('landing',index,difference)
            landing_max=max(landing_max,difference)
        matched.append(dict(batch=index//16,env=index%16,classifications_equal=True))
    for key in ('hit_rate','legal_return_rate','target_success_rate','outcomes'):assert current[key]==previous[key]
    return dict(status='PASS',episodes=32,pairing='batch/env; lifetime serve ID excluded',
                exact_fields=['outcome','hit','net_touched','target_success','release_reason'],
                landing_tolerance_m=1e-5,max_landing_component_error_m=landing_max,matched=matched)


def _mean_summary(evaluation):
    rows=evaluation['episodes'];n=len(rows)
    assert evaluation['total']==n==32
    summary=dict(total=n,hits=sum(r['hit'] for r in rows),legal=sum(r['outcome']=='legal_return' for r in rows),
                 targets=sum(r['target_success'] for r in rows),outcomes=evaluation['outcomes'])
    values=[r['landing_error_m'] for r in rows if r['landing_error_m'] is not None]
    assert len(values)==summary['legal']
    summary['landing_error']=dict(scope='legal returns only; all failures retained in task denominator',
        n=len(values),mean_m=float(np.mean(values)) if values else None,p95_m=float(np.percentile(values,95)) if values else None)
    summary['floors']=[key for key in ('hits','legal','targets') if summary[key]==0]
    summary['ceilings']=[key for key in ('hits','legal','targets') if summary[key]==n]
    if rows and 'prehit_hint_steps' in rows[0]:
        used=sum(r['prehit_hint_steps'] for r in rows);prehit=sum(r['prehit_policy_steps'] for r in rows)
        count=sum(r['hint_reference_steps'] for r in rows);sq=sum(r['hint_error_sq_sum_m2'] for r in rows)
        summary['hint']=dict(used_steps=used,prehit_steps=prehit,coverage=used/prehit if prehit else None,
            reference_steps=count,teacher_rmse_m=float(np.sqrt(sq/count)) if count else None,
            first_available_s=sorted({r['first_prediction_s'] for r in rows if r['first_prediction_s'] is not None}),
            missing={key:sum(r['hint_missing'][key] for r in rows) for key in ('history_short','untrained','invalid')})
    return summary


def _evaluate_mean_comparison(device,num_envs,artifact,source=None):
    from airbot_tt.pace_env import PaceEnv
    from train_pace import original_ppo,evaluate,set_predictor,validate_predictor_checkpoint
    from pace_m7 import source_hashes
    if num_envs!=16:raise ValueError('Mean paired evaluation requires exactly 16 environments')
    started=time.monotonic();before=source_hashes()
    spec,metadata,mean,inputs,old,hashes=_mean_inputs(num_envs,artifact)
    cfg=spec['common_cfg'];p=cfg['ppo']
    if source is not None:
        source=Path(source);protocol_path=source/'protocol.json';protocol=json.loads(protocol_path.read_text())
        if (protocol['groups']!=['E-mean'] or protocol['seeds']!=spec['seeds'] or protocol['cfg']!=cfg
            or protocol['num_envs']!=16 or protocol['iterations']!=16 or protocol['rollout_steps']!=24
            or protocol['transitions_per_trial']!=6144 or protocol['predictor_mode']!='mean'
            or protocol['predictor_provenance']!=mean.provenance or protocol['source_sha256']!=before
            or protocol['runner_sha256']!=hashlib.sha256((ROOT/'scripts/train_pace.py').read_bytes()).hexdigest()
            or any(protocol.get(k)!=v for k,v in metadata.items())):
            raise ValueError('Mean evaluation requires current complete fixed-budget training')
        hashes[str(protocol_path.resolve())]=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        for seed in spec['seeds']:
            trial=source/f'E-mean_{seed}';result_path=trial/'result.json';result=json.loads(result_path.read_text())
            final=torch.load(trial/'policy.pt',map_location='cpu',weights_only=False)
            if (result['status']!='PASS' or result['transitions']!=6144 or final['iteration']!=16
                or result['source_sha256']!=before or result['group']!='E-mean' or result['training_seed']!=seed
                or any(result.get(k)!=v for k,v in metadata.items())):
                raise ValueError('Incomplete or incompatible mean final trial')
            hashes[str(result_path.resolve())]=hashlib.sha256(result_path.read_bytes()).hexdigest()
            for stage,name,sha in [('initial','initial.pt',final['initial_sha256']),('final','policy.pt',result['checkpoint_sha256'])]:
                path=trial/name
                if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Mean checkpoint changed')
                hashes[str(path.resolve())]=sha
                ck=torch.load(path,map_location='cpu',weights_only=False);validate_predictor_checkpoint(ck,'E-mean')
                if (ck['seed']!=seed or ck['cfg']!=cfg or ck['critic_dim']!=200 or ck['source_sha256']!=before
                    or ck['predictor_provenance']!=mean.provenance or any(ck.get(k)!=v for k,v in metadata.items())):
                    raise ValueError('Mean checkpoint/protocol mismatch')
                mean.load_state_dict(ck['predictor'])
                if stage=='initial':
                    learned=next(r['checkpoint'] for r in inputs if r['seed']==seed and r['label']=='learned_initial')
                    assert ck['policy'].keys()==learned['policy'].keys()
                    assert all(torch.equal(v,learned['policy'][k]) for k,v in ck['policy'].items())
                    rng=ck['pre_rollout_torch_rng'];cuda=ck['pre_rollout_cuda_rng']
                    assert hashlib.sha256(rng.numpy().tobytes()).hexdigest()==spec['fixed_learned'][str(seed)]['post_actor_cpu_rng_sha256']
                    assert cuda and all(v.dtype==torch.uint8 and v.ndim==1 for v in cuda)
                    assert ck['paired_initialization']['all_policy_tensors_equal']
                inputs.append(dict(label='mean_'+stage,seed=seed,group='E-mean',artifact=artifact,
                                   path=path,sha256=sha,checkpoint=ck))
    rows=[];context_rows=[];compatibility=[];zero=None;execution_order=[];current_label=None
    report=dict(scope='Fixed-mean paired development evaluation' if source is not None else 'Current-source learned-baseline compatibility replay',
        protocol=spec,protocol_sha256=metadata['experiment_spec_sha256'],**metadata,source_sha256=before,
        historical_source_sha256=old['source_sha256'],baseline_sha256=hashes,predictor_provenance=mean.provenance,
        runs=rows,context_runs=context_rows,compatibility=compatibility,execution_order=execution_order,
        expected_primary_episodes=416 if source is not None else 224,expected_context_episodes=96,
        expected_actual_episodes=512 if source is not None else 320)
    def counts():
        primary=(zero['total'] if zero is not None else 0)+sum(r['evaluation']['total'] for r in rows)
        context=sum(r['evaluation']['total'] for r in context_rows)
        return dict(total_scored_episodes=primary,context_scored_episodes=context,actual_scored_episodes=primary+context)
    try:
        ActorCritic,_=original_ppo();torch.set_num_threads(1)
        frozen_artifact=mean.cfg['files']['frozen_artifact']['path']
        env=PaceEnv(num_envs,device,group='E-frozen',frozen_predictor=frozen_artifact,serve_z_m=2.3,serve_y_half_range_m=.015)
        class Zero:
            def act_inference(self,obs):return torch.zeros(len(obs),6,device=obs.device)
        current_label='zero'
        zero=evaluate(env,Zero(),spec['evaluation']['seed'],32)
        report['zero']=zero;execution_order.append('zero')
        compatibility.append(dict(label='zero',seed=None,**_compare_learned_replay(zero,old['zero'])))
        report['zero_summary']=_mean_summary(zero)
        for row in inputs:
            current_label=row['label']+'_'+str(row['seed'])
            ck=row['checkpoint'];set_predictor(env,row['group'],row['artifact'],checkpoint=ck)
            env.actor_width=23 if row['group']=='D' else 27
            env.actor_history=torch.zeros(num_envs,5,env.actor_width,device=device)
            policy=ActorCritic(env.actor_width*5,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],
                               activation='elu',init_noise_std=p['init_std']).to(device)
            policy.load_state_dict(ck['policy']);policy.eval()
            evaluation=evaluate(env,policy,spec['evaluation']['seed'],32)
            # Preserve the entire completed policy evaluation before any check
            # can fail. run_airbot serializes exc.report without replacing it.
            result=dict(label=row['label'],seed=row['seed'],checkpoint_sha256=row['sha256'],evaluation=evaluation)
            (context_rows if row.get('context') else rows).append(result)
            execution_order.append(current_label)
            assert all(torch.equal(v.detach().cpu(),ck['policy'][k]) for k,v in policy.state_dict().items())
            if env.predictor_module is not None:env.predictor_module.state_dict()
            if 'reference' in row:
                compatibility.append(dict(label=row['label'],seed=row['seed'],context=bool(row.get('context')),
                                          **_compare_learned_replay(evaluation,row['reference'])))
            result['summary']=_mean_summary(evaluation)
        assert before==source_hashes()
        for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
        total=32+sum(r['evaluation']['total'] for r in rows)
        assert total==(416 if source is not None else 224)
        assert sum(r['evaluation']['total'] for r in context_rows)==96
        expected_order=spec['evaluation']['execution_order'] if source is not None else spec['sequence_amendment']['historical_prefix']
        assert execution_order==expected_order
        differences=[]
        if source is not None:
            assert {'zero'}|{r['label']+'_'+str(r['seed']) for r in rows}==set(spec['evaluation']['policies'])
            for seed in spec['seeds']:
                by={r['label']:r['evaluation'] for r in rows if r['seed']==seed}
                metrics=('legal_return_rate','hit_rate','target_success_rate')
                differences.append(dict(seed=seed,
                    learned_final_minus_mean_final={k:by['learned_final'][k]-by['mean_final'][k] for k in metrics},
                    mean_final_minus_own_initial={k:by['mean_final'][k]-by['mean_initial'][k] for k in metrics},
                    learned_final_minus_own_initial={k:by['learned_final'][k]-by['learned_initial'][k] for k in metrics}))
        report.update(status='PASS',completed=True,differences=differences,seconds=time.monotonic()-started,**counts(),
                      scientific_conclusion='REQUIRES_ALL_SEED_INTERPRETATION' if source is not None else 'COMPATIBILITY_ONLY_NO_TRAINING')
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='FAIL',failed_policy=current_label,completed=False,seconds=time.monotonic()-started,**counts())
        exc.report=report
        raise
    return report


def verify_mean_compat(device='cuda:0',num_envs=16,artifact=None):
    return _evaluate_mean_comparison(device,num_envs,artifact)


def evaluate_mean(device,num_envs,source,artifact):
    return _evaluate_mean_comparison(device,num_envs,artifact,source=source)
