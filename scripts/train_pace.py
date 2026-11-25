"""Thin A/B/C/D/E runner around pinned PACE ActorCritic/PPO; launched by run.sh.

E (legacy alias) and E-online use independent online MSE. E-frozen explicitly
loads a verified offline artifact; E-mean uses its paired train-only constant.
Neither immutable hint updates. A five-serve BC variant loads Actor-only
initialization; PPO itself is unchanged. Terminal-value timeout bootstrap is
supplied once before the original storage call. No hardware operation.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from airbot_tt.config import ROOT
from airbot_tt.pace_control import bootstrap_timeout


def original_ppo():
    source=ROOT/'.agents/runs/foundation/pace/source'
    manifest=json.loads((ROOT/'configs/pace_sources.json').read_text())
    for item in manifest['files']:
        if hashlib.sha256((source/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Official PACE source mismatch: '+item['path'])
    if 'rsl_rl' in sys.modules and not Path(sys.modules['rsl_rl'].__file__).resolve().is_relative_to(source):
        raise RuntimeError('A different RSL-RL is already loaded')
    sys.path.insert(0,str(source/'rsl_rl'))
    return importlib.import_module('rsl_rl.modules').ActorCritic,importlib.import_module('rsl_rl.algorithms').PPO


def predictor_mode(group):
    if group in ('E','E-online'):return 'online'
    if group=='E-frozen':return 'frozen'
    if group=='E-mean':return 'mean'
    if group in ('A','B','C','D'):return 'none'
    raise ValueError('Unknown PACE group: '+str(group))


def online_offline_validation():
    """Audit the old Z2.3 result without loading its model into E-online.

    The edited predictor module may reference its hash-bound historical snapshot.
    This is evidence about the old offline fit, never a current interface PASS.
    All other source entries still have to match the working tree exactly.
    """
    path=ROOT/'.agents/runs/foundation/m7/m74e/offline/validation.json'
    offline=json.loads(path.read_text());report_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if offline['status']!='PASS' or offline['collection'].get('distribution',{}).get('serve_position_m',[0,0,0])[2]!=2.3:
        raise RuntimeError('E-online requires passing Z=2.3 offline validation')
    mismatches=[name for name,sha in offline['source_sha256'].items()
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    evidence=dict(validation_sha256=report_sha,historical_predictor_source=False)
    if not mismatches:return evidence
    predictor_path='src/airbot_tt/pace_predictor.py'
    if mismatches!=[predictor_path]:raise RuntimeError('Stale offline predictor validation: '+', '.join(mismatches))
    artifact_path=ROOT/'configs/pace_frozen.json';artifact=json.loads(artifact_path.read_text())
    report=artifact['files']['validation_report'];snapshot_record=artifact['files']['source_snapshot']
    snapshot_path=ROOT/snapshot_record['path']
    snapshot_sha=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if ((ROOT/report['path']).resolve()!=path.resolve() or report['sha256']!=report_sha
        or snapshot_sha!=snapshot_record['sha256']
        or any(artifact['historical_source_sha256'].get(name)!=sha for name,sha in offline['source_sha256'].items())):
        raise RuntimeError('Offline historical provenance mismatch')
    snapshot=json.loads(snapshot_path.read_text())
    historical={item['path']:item['sha256'] for item in snapshot['files']}
    if snapshot['schema']!='e-frozen-source-snapshot-v1' or historical.get(predictor_path)!=offline['source_sha256'][predictor_path]:
        raise RuntimeError('Offline predictor snapshot mismatch')
    evidence.update(historical_predictor_source=True,source_snapshot_sha256=snapshot_sha,
        historical_predictor_sha256=historical[predictor_path],
        provenance_manifest_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest())
    return evidence


def frozen_experiment_metadata(cfg,group='E-frozen'):
    if group not in ('E-frozen','E-mean'):raise ValueError('Expected an immutable predictor experiment')
    kind=predictor_mode(group)
    path=ROOT/f'configs/pace_{kind}_experiment.json';spec=json.loads(path.read_text())
    schema='airbot-d-e-frozen-paired-v1' if kind=='frozen' else 'airbot-e-mean-learned-paired-v1'
    if (spec.get('schema')!=schema or cfg!=spec['common_cfg']
        or spec['seeds']!=[7101,7102,7103] or spec['num_envs']!=16
        or spec['iterations']!=16 or spec['rollout_steps']!=24
        or spec['transitions_per_seed']!=6144 or spec['checkpoint_selection']!='last'):
        raise ValueError('Frozen experiment PPO/control/reward/config contract mismatch')
    if kind=='mean':
        if (spec['groups']!=['E-frozen','E-mean'] or spec['evaluation']['seed']!=7531
            or spec['evaluation']['serves_per_policy']!=32 or spec['evaluation']['total_scored_episodes']!=416):
            raise ValueError('Mean paired evaluation contract mismatch')
        references=[spec['mean_artifact'],*spec['baseline_reports'].values()]
        references.extend(row[name] for row in spec['fixed_learned'].values() for name in ('initial','final','result'))
        for item in references:
            if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Mean experiment baseline/artifact changed: '+item['path'])
        baseline=json.loads((ROOT/spec['baseline_reports']['frozen_experiment']['path']).read_text())
        if cfg!=baseline['common_cfg']:raise ValueError('Mean experiment differs from learned baseline config')
    return dict(experiment_spec_path=str(path.relative_to(ROOT)),
        experiment_spec_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        effective_cfg_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def pair_mean_initialization(policy,seed,cfg):
    """Require the one normal policy construction to reproduce the old initial.

    No second network is constructed, and loading exact tensors consumes no RNG.
    The baseline has no saved initial RNG; its CPU state was independently
    reconstructed from the same official constructor and is pinned in the spec.
    """
    frozen_experiment_metadata(cfg,'E-mean')
    spec=json.loads((ROOT/'configs/pace_mean_experiment.json').read_text())
    row=spec['fixed_learned'][str(seed)]
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    cpu_sha=hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest()
    if cpu_sha!=row['post_actor_cpu_rng_sha256']:
        raise ValueError('Mean policy construction RNG differs from learned initialization')
    checkpoint=torch.load(ROOT/row['initial']['path'],map_location='cpu',weights_only=False)
    validate_predictor_checkpoint(checkpoint,'E-frozen')
    if checkpoint['seed']!=seed or checkpoint['cfg']!=cfg or checkpoint['critic_dim']!=200:
        raise ValueError('Paired learned initialization seed/config/schema mismatch')
    expected=checkpoint['policy'];actual=policy.state_dict()
    if actual.keys()!=expected.keys() or any(not torch.equal(v.detach().cpu(),expected[k]) for k,v in actual.items()):
        raise ValueError('Mean Actor/Critic/std initialization differs from paired learned initial')
    policy.load_state_dict(expected,strict=True)
    if (not torch.equal(cpu_rng,torch.get_rng_state())
        or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))):
        raise RuntimeError('Loading paired initialization changed policy sampling RNG')
    return dict(baseline_group='E-frozen',seed=seed,initial=row['initial'],all_policy_tensors_equal=True,
                post_actor_cpu_rng_sha256=cpu_sha,cuda_rng_preserved=True,
                old_initial_rng_saved=False)


def validate_predictor_checkpoint(checkpoint,group):
    """Reject mode/schema confusion before constructing or loading any module.

    Only historical E checkpoints may infer the online mode; new explicit group
    names must carry the mode. A 135-column actor alone does not identify E.
    """
    expected=predictor_mode(group)
    legacy='online' if group=='E' else 'none'
    actual=checkpoint.get('predictor_mode',legacy)
    if actual!=expected or checkpoint.get('group',group)!=group:
        raise ValueError('Checkpoint predictor mode/group mismatch')
    if expected=='none':
        if 'predictor' in checkpoint:raise ValueError('Non-predictor checkpoint contains predictor state')
        return
    if checkpoint.get('actor_dim')!=135 or 'predictor' not in checkpoint:
        raise ValueError('Predictor checkpoint requires actor135 and predictor state')
    state_mode=checkpoint['predictor'].get('mode','online')
    if state_mode!=expected:raise ValueError('Checkpoint predictor payload mode mismatch')
    if expected in ('frozen','mean') and not checkpoint.get('predictor_provenance'):
        raise ValueError('Frozen checkpoint requires explicit artifact provenance')


def set_predictor(env,group,artifact=None,checkpoint=None):
    """Keep the environment's common module and mode-specific owners aligned."""
    mode=predictor_mode(group)
    if checkpoint is not None:validate_predictor_checkpoint(checkpoint,group)
    if mode in ('frozen','mean'):
        if artifact is None:raise ValueError(group+' requires an explicit predictor artifact')
        from airbot_tt.pace_predictor import FrozenPredictor,FixedMeanPredictor
        provider=FixedMeanPredictor if mode=='mean' else FrozenPredictor
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if mode=='mean' else []
        module=provider(env.n,artifact=artifact)
        if mode=='mean' and (not torch.equal(cpu_rng,torch.get_rng_state())
            or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))):
            raise RuntimeError('Mean predictor construction changed policy sampling RNG')
        if checkpoint is not None and checkpoint['predictor_provenance']!=module.provenance:
            raise ValueError('Frozen checkpoint artifact provenance mismatch')
    elif mode=='online':
        if artifact is not None:raise ValueError('Online predictor cannot load a frozen artifact')
        from airbot_tt.pace_predictor import OnlinePredictor
        module=OnlinePredictor(env.n)
    else:
        if artifact is not None:raise ValueError('Group without predictor cannot load a frozen artifact')
        module=None
    if checkpoint is not None and module is not None:module.load_state_dict(checkpoint['predictor'])
    env.group=group;env.predictor_mode=mode
    env.online=module if mode=='online' else None
    env.frozen=module if mode in ('frozen','mean') else None
    return module


def predictor_metadata(env):
    data=dict(predictor_mode=env.predictor_mode,actor_dim=env.actor_width*5,critic_dim=200)
    if env.frozen is not None:data['predictor_provenance']=env.frozen.provenance
    return data


def update_ppo(env,policy,algo):
    """Run the official PPO update, then only the independent online MSE path."""
    module=env.predictor_module
    before={k:v.clone() for k,v in module.predictor.model.state_dict().items()} if module else None
    losses=algo.update()
    if module is not None:
        if not all(torch.equal(v,before[k]) for k,v in module.predictor.model.state_dict().items()):
            raise RuntimeError('PPO changed predictor weights')
        if any(p.grad is not None for p in module.predictor.model.parameters()):
            raise RuntimeError('PPO populated predictor gradients')
    auxiliary=None;refreshed=None
    if env.online is not None:
        policy_before={k:v.clone() for k,v in policy.state_dict().items()}
        auxiliary=env.online.update()
        refreshed=env.refresh_after_predictor_update()
        if not all(torch.equal(v,policy_before[k]) for k,v in policy.state_dict().items()):
            raise RuntimeError('Predictor MSE changed policy weights')
    elif env.frozen is not None:
        auxiliary=dict(status=env.predictor_mode,optimizer_steps=0,weights_unchanged=True)
    return losses,auxiliary,refreshed


def evaluate_initial(env,policy,evaluator,seed):
    """Keep pre-PPO evaluation out of policy sampling and training serve RNG."""
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
    try:
        evaluation=evaluator(policy)
    finally:
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
    env.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(env.n)]
    env.serve_id.zero_();env.reset(env.ids)
    return evaluation,env.observations()


def evaluate(env,policy,seed,episodes):
    collecting=env.prediction_collect;env.prediction_collect=False
    rows=[]
    for batch in range((episodes+env.n-1)//env.n):
        env.rng=[np.random.default_rng(np.random.SeedSequence([seed,batch,i])) for i in range(env.n)]
        env.reset(env.ids);obs,_=env.observations();selected={}
        with torch.no_grad():
            for _ in range(2000):
                obs,_,_,_,info=env.step(policy.act_inference(obs))
                for row in info['episodes']:
                    # Predeclared first serve of each env; fast failures cannot
                    # replace another env's slower, still-running trial.
                    selected.setdefault(row['env'],row)
                if len(selected)==env.n:break
        if len(selected)!=env.n:raise RuntimeError('Incomplete policy evaluation')
        rows.extend(selected[i] for i in range(env.n))
    env.prediction_collect=collecting
    return dict(event_rules=env.events.rules,total=len(rows),hit_rate=sum(r['hit'] for r in rows)/len(rows),
                legal_return_rate=sum(r['outcome']=='legal_return' for r in rows)/len(rows),
                target_success_rate=sum(r['target_success'] for r in rows)/len(rows),
                outcomes={name:sum(r['outcome']==name for r in rows) for name in sorted(set(r['outcome'] for r in rows))},episodes=rows)


def run(device,num_envs,mode,destination,learning_rate=None,serve_y_half_range_m=None,groups=None,serve_z_m=None,serve_bank=None,gate_directory=None,actor_init=None,frozen_predictor=None,mean_predictor=None):
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    destination=Path(destination)
    explicit_gate_directory=gate_directory is not None
    gate_directory=Path(gate_directory) if gate_directory else ROOT/'.agents/runs/foundation/m7'
    before_hashes=source_hashes()
    groups=list(groups) if groups is not None else (['A'] if mode=='smoke' else ['A','B'])
    if not groups or len(set(groups))!=len(groups):raise ValueError('Expected unique PACE groups')
    modes=[predictor_mode(g) for g in groups]
    if any(m!='none' for m in modes) and len(groups)!=1:
        raise ValueError('Each predictor group has a distinct protocol; run it separately')
    if (frozen_predictor is not None)!=('frozen' in modes):
        raise ValueError('An explicit frozen predictor artifact is required only for E-frozen')
    if (mean_predictor is not None)!=('mean' in modes):
        raise ValueError('An explicit mean predictor artifact is required only for E-mean')
    artifact=mean_predictor if 'mean' in modes else frozen_predictor
    immutable_kind=modes[0] if modes[0] in ('frozen','mean') else None
    frozen_provenance=None
    if immutable_kind:
        if num_envs!=16 or mode not in ('smoke','ab') or learning_rate is not None:
            raise ValueError(groups[0]+' protocol requires 16 environments and the unchanged PPO learning rate')
        from airbot_tt.pace_predictor import FrozenPredictor,FixedMeanPredictor
        provider=FixedMeanPredictor if immutable_kind=='mean' else FrozenPredictor
        frozen_provenance=provider(num_envs,artifact=artifact).provenance
        if serve_z_m!=2.3 or serve_y_half_range_m!=.015:
            raise ValueError('Frozen predictor validation covers Z=2.3 and Y half range .015 only')
        gate=json.loads((gate_directory/f'{immutable_kind}.json').read_text())
        if gate['status']!='PASS' or gate.get('source_sha256')!=before_hashes or gate.get('predictor_provenance')!=frozen_provenance:
            raise RuntimeError(groups[0]+' requires a current interface gate for the same artifact')
        if immutable_kind=='mean':
            compatible=json.loads((gate_directory/'mean_compat.json').read_text())
            spec_sha=hashlib.sha256((ROOT/'configs/pace_mean_experiment.json').read_bytes()).hexdigest()
            if (compatible['status']!='PASS' or compatible.get('source_sha256')!=before_hashes
                or compatible.get('predictor_provenance')!=frozen_provenance or compatible.get('experiment_spec_sha256')!=spec_sha):
                raise RuntimeError('E-mean requires a current learned-baseline compatibility replay')
        if mode!='smoke':
            smoke=json.loads((gate_directory/f'{immutable_kind}_smoke.json').read_text())
            if (smoke['status']!='PASS' or smoke.get('source_sha256')!=before_hashes
                or smoke.get('runner_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or smoke.get('predictor_provenance')!=frozen_provenance):
                raise RuntimeError(groups[0]+' requires current runner smoke validation for the same artifact')
    if actor_init is not None and (serve_bank is None or mode!='ab' or num_envs!=16 or list(groups or [])!=['A']):
        raise ValueError('BC Actor initialization requires the fixed-five A diagnostic with 16 environments')
    bank=None
    evaluation_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('tests/pace_bank.py','tests/pace_motion.py')} if serve_bank else {}
    if serve_bank is not None:
        from airbot_tt.pace_serves import FixedServeBank
        bank=FixedServeBank(serve_bank)
        if mode!='ab' or num_envs!=16 or list(groups or [])!=['A']:
            raise ValueError('Five-serve diagnostic requires A, 16 environments, full fixed short budget')
        if any(v is not None for v in (learning_rate,serve_y_half_range_m,serve_z_m)):
            raise ValueError('Five-serve diagnostic does not mix other experiment overrides')
        gate=json.loads((gate_directory/'bank.json').read_text())
        if gate['status']!='PASS' or gate.get('source_sha256')!=before_hashes or gate.get('serve_bank_sha256')!=bank.sha256:
            raise RuntimeError('Fixed-serve gate must match current source and manifest')
        if gate.get('gate_sha256')!=evaluation_hashes['tests/pace_bank.py']:
            raise RuntimeError('Fixed-serve evaluator changed after gate')
    for name in ('lifecycle','parity','control'):
        report=json.loads((gate_directory/f'{name}.json').read_text())
        if report['status']!='PASS' or report.get('source_sha256')!=source_hashes():
            raise RuntimeError(f'M7 gate not current/PASS: {name}')
    online_validation=None
    if 'online' in modes:
        online_validation=online_offline_validation()
        if serve_z_m!=2.3:raise ValueError('E validation currently covers Z=2.3 only')
        if mode!='smoke' or online_validation['historical_predictor_source']:
            online=json.loads((gate_directory/'predictor.json').read_text())
            if online['status']!='PASS' or online.get('source_sha256')!=source_hashes():raise RuntimeError('E requires current online interface gate')
        if mode!='smoke':
            smoke_path=(gate_directory/'online_smoke.json') if explicit_gate_directory else ROOT/'.agents/runs/foundation/m7/m74e/smoke.json'
            smoke=json.loads(smoke_path.read_text())
            if smoke['status']!='PASS' or smoke.get('source_sha256')!=source_hashes() or smoke.get('runner_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError('E requires current runner smoke validation')
    if any(g in ('C','D','E','E-online','E-frozen','E-mean') for g in groups):
        gate='combined' if any(g in groups for g in ('D','E','E-online','E-frozen','E-mean')) else 'return'
        report=json.loads((gate_directory/f'{gate}.json').read_text())
        if report['status']!='PASS' or report.get('source_sha256')!=source_hashes():
            raise RuntimeError(f'{groups} requires a current passing {gate} gate')
    ActorCritic,PPO=original_ppo()
    cfg=json.loads((ROOT/'configs/pace_rl.json').read_text())
    if learning_rate is not None:
        if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Learning rate must be finite and positive')
        cfg['ppo']['learning_rate']=learning_rate
    if serve_y_half_range_m is not None:
        if not np.isfinite(serve_y_half_range_m) or serve_y_half_range_m<0:raise ValueError('Invalid serve Y half range')
        cfg['serve_y_half_range_m']=float(serve_y_half_range_m)
    if serve_z_m is not None:
        if not np.isfinite(serve_z_m) or serve_z_m<=0:raise ValueError('Invalid serve Z')
        cfg['serve_z_m']=float(serve_z_m)
    if bank is not None:
        cfg['serve_bank_sha256']=bank.sha256
        cfg['serve_bank_manifest']=bank.manifest
    p=cfg['ppo']
    # Fixed small diagnostic budget; smoke never establishes learning effectiveness.
    iterations=4 if mode=='smoke' else cfg['diagnostic_iterations']
    seeds=cfg['seeds'][:1] if mode=='smoke' else cfg['seeds']
    if bank is not None and (iterations!=16 or p['steps']!=24 or seeds!=[7101,7102,7103]):
        raise ValueError('Fixed-five protocol requires three seeds and exactly 6144 transitions each')
    if immutable_kind and (p['steps']!=24 or iterations!=(4 if mode=='smoke' else 16)
        or seeds!=([7101] if mode=='smoke' else [7101,7102,7103])):
        raise ValueError(groups[0]+' requires 1536 smoke transitions or three fixed 6144-transition trials')
    experiment_metadata=frozen_experiment_metadata(cfg,groups[0]) if immutable_kind else {}
    initialization=None
    if actor_init is not None:
        from airbot_tt.pace_bc import validate_initialization
        # Validate the complete three-seed gate before starting any PPO trial.
        initialization=validate_initialization(actor_init,cfg,before_hashes)
    destination.mkdir(parents=True,exist_ok=False)
    env=PaceEnv(num_envs,device,group=groups[0],serve_y_half_range_m=serve_y_half_range_m,serve_z_m=serve_z_m,serve_bank=serve_bank,frozen_predictor=frozen_predictor,mean_predictor=mean_predictor);torch.set_num_threads(1)
    clock=time.monotonic();runs=[]
    protocol=dict(mode=mode,iterations=iterations,num_envs=num_envs,rollout_steps=p['steps'],seeds=seeds,groups=groups,
                  transitions_per_trial=iterations*num_envs*p['steps'],cfg=cfg,source_sha256=source_hashes(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scientific_conclusion='INCONCLUSIVE',heldout_final_test=False,evaluation_sha256=evaluation_hashes)
    protocol.update(predictor_metadata(env),**experiment_metadata,checkpoint_selection='fixed_last')
    if online_validation is not None:protocol['offline_predictor_validation']=online_validation
    if initialization is not None:
        protocol['actor_initialization']=dict(directory=str(Path(actor_init).resolve()),manifest=initialization,
            manifest_sha256=hashlib.sha256((Path(actor_init)/'initialization.json').read_bytes()).hexdigest())
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def evaluate_trial(actor):
        if bank is not None:
            from pace_bank import evaluate_bank
            return evaluate_bank(env,actor,capture=True)
        return evaluate(env,actor,7201,max(num_envs,16))
    baseline=None
    if mode=='ab':
        class ZeroAction:
            def act_inference(self,obs):return torch.zeros((len(obs),6),device=obs.device)
        baseline=evaluate_trial(ZeroAction())
        (destination/'zero_action.json').write_text(json.dumps(baseline,indent=2)+'\n')
    for seed in seeds:
        for group in groups:
            trial=destination/f'{group}_{seed}';trial.mkdir()
            started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            if predictor_mode(group)!='none':set_predictor(env,group,artifact)
            env.group=group;env.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(env.n)]
            if bank is not None:env.seed_bank(seed)
            env.reset(env.ids);obs,critic=env.observations()
            policy=ActorCritic(env.actor_width*5,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            paired_initialization=pair_mean_initialization(policy,seed,cfg) if env.predictor_mode=='mean' else None
            paired_rng_cpu=torch.get_rng_state().clone() if paired_initialization else None
            paired_rng_cuda=torch.cuda.get_rng_state_all() if paired_initialization else None
            initialization_provenance=None
            if actor_init is not None:
                from airbot_tt.pace_bc import load_actor_initialization
                # Loading and checking BC artifacts must not consume the original
                # post-construction rollout RNG state.
                rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
                try:
                    initialization_provenance=load_actor_initialization(policy,actor_init,seed,cfg,before_hashes)
                finally:
                    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
            if env.predictor_module is not None:
                initial_checkpoint=dict(policy=policy.state_dict(),predictor=env.predictor_module.state_dict(),cfg=cfg,seed=seed,group=group,
                    source_sha256=source_hashes(),**predictor_metadata(env),**experiment_metadata)
                if paired_initialization is not None:initial_checkpoint['paired_initialization']=paired_initialization
                else:torch.save(initial_checkpoint,trial/'initial.pt')
            initial_evaluation=None
            if env.frozen is not None:
                # Evaluate this 135-column actor before any PPO update. Restore
                # policy sampling RNG and the original training serve stream.
                initial_evaluation,(obs,critic)=evaluate_initial(env,policy,evaluate_trial,seed)
                (trial/'initial_evaluation.json').write_text(json.dumps(initial_evaluation,indent=2)+'\n')
            if bank is not None:
                initial_checkpoint=dict(policy=policy.state_dict(),cfg=cfg,seed=seed,group=group,actor_dim=115)
                if initialization_provenance is not None:
                    initial_checkpoint['actor_initialization']=initialization_provenance
                torch.save(initial_checkpoint,trial/'initial.pt')
                rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
                initial_evaluation=evaluate_trial(policy)
                (trial/'initial_evaluation.json').write_text(json.dumps(initial_evaluation,indent=2)+'\n')
                torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
                env.seed_bank(seed);env.serve_id.zero_();env.reset(env.ids);obs,critic=env.observations()
            transitions_by_id={str(r['id']):0 for r in bank.rows} if bank else None
            gamma=.95**(50*env.policy_dt)
            algo=PPO(policy,num_learning_epochs=p['epochs'],num_mini_batches=p['minibatches'],clip_param=p['clip'],gamma=gamma,lam=gamma,
                     entropy_coef=p['entropy'],learning_rate=p['learning_rate'],schedule='adaptive',desired_kl=.01,device=device)
            if initialization_provenance is not None and algo.optimizer.state:
                raise RuntimeError('BC-to-PPO handoff requires a fresh PPO optimizer')
            algo.init_storage('rl',num_envs,p['steps'],[env.actor_width*5],[200],[6])
            pre_rollout_rng={}
            if paired_initialization is not None:
                if (not torch.equal(paired_rng_cpu,torch.get_rng_state())
                    or any(not torch.equal(a,b) for a,b in zip(paired_rng_cuda,torch.cuda.get_rng_state_all(),strict=True))):
                    raise RuntimeError('Mean initial evaluation/reset/PPO construction changed rollout sampling RNG')
                if algo.optimizer.state:raise RuntimeError('Mean PPO must start with a fresh optimizer')
                pre_rollout_rng=dict(pre_rollout_torch_rng=paired_rng_cpu,pre_rollout_cuda_rng=paired_rng_cuda)
                initial_checkpoint.update(pre_rollout_rng)
                torch.save(initial_checkpoint,trial/'initial.pt')
            actor_initial=torch.cat([v.detach().flatten() for v in policy.actor.parameters()]).clone()
            critic_initial=torch.cat([v.detach().flatten() for v in policy.critic.parameters()]).clone()
            history=[];episodes=[];timeout_count=0
            for iteration in range(iterations):
                if time.monotonic()-started>cfg['smoke_max_s']:
                    raise RuntimeError('Trial exceeded 30 minute diagnostic budget; partial logs retained')
                clip_count=0;tracking_error=0.;request_gap=0.;iteration_start=time.monotonic()
                coverage=dict(prehit_steps=0,hint_used_steps=0,hit_guidance_steps=0,release_events=0,return_valid_events=0,return_scored_events=0,hit_reward_sum=0.,return_reward_sum=0.,hint_reference_steps=0,hint_error_sq_sum=0.)
                with torch.no_grad():
                    for _ in range(p['steps']):
                        if bank is not None:
                            for index in env.bank_index:transitions_by_id[str(bank.rows[int(index)]['id'])]+=1
                        actions=algo.act(obs,critic)
                        clip_count+=float((actions.abs()>1).float().mean())
                        obs,critic,reward,done,info=env.step(actions)
                        coverage['prehit_steps']+=int(info['prehit'].sum())
                        coverage['hint_used_steps']+=int(info['prediction_used'].sum())
                        coverage['hint_reference_steps']+=int(info['hint_reference'].sum())
                        coverage['hint_error_sq_sum']+=float(info['hint_error_sq'].sum())
                        coverage['hit_guidance_steps']+=int((info['guidance']!=0).sum())
                        coverage['release_events']+=int(info['return_events'].sum())
                        coverage['return_valid_events']+=int(info['return_valid_events'].sum())
                        coverage['hit_reward_sum']+=float(info['guidance'].sum())
                        coverage['return_reward_sum']+=float(info['return_increment'].sum())
                        coverage['return_scored_events']+=int((info['return_increment']!=0).sum())
                        terminal_value=policy.evaluate(info['terminal_critic'])
                        corrected=bootstrap_timeout(reward,terminal_value,info['truncated'],gamma)
                        timeout_count+=int(info['truncated'].sum())
                        # Do NOT pass time_outs: upstream would add the OLD value a second time.
                        algo.process_env_step(corrected,done,{})
                        episodes.extend(info['episodes'])
                        tracking_error+=float((info['q_measured']-info['q_applied']).square().mean().sqrt())
                        request_gap+=float((info['q_request']-info['q_applied']).abs().mean())
                    algo.compute_returns(critic)
                losses,auxiliary,refreshed=update_ppo(env,policy,algo)
                if refreshed is not None:obs,critic=refreshed
                with torch.no_grad():
                    storage=algo.storage
                    policy.update_distribution(storage.observations.flatten(0,1))
                    new_log=policy.get_actions_log_prob(storage.actions.flatten(0,1))
                    ratio=(new_log-storage.actions_log_prob.flatten()).exp()
                    clip_fraction=float(((ratio-1).abs()>p['clip']).float().mean())
                    mu,sigma=policy.action_mean,policy.action_std
                    old_mu,old_sigma=storage.mu.flatten(0,1),storage.sigma.flatten(0,1)
                    kl=float((torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean())
                if not all(np.isfinite(v) for v in losses.values()):raise RuntimeError('Non-finite PPO losses')
                row=dict(iteration=iteration+1,transitions=(iteration+1)*num_envs*p['steps'],losses=losses,
                         action_clip_fraction=clip_count/p['steps'],learning_rate=algo.learning_rate,completed_episodes=len(episodes),
                         policy_kl=kl,ppo_clip_fraction=clip_fraction,tracking_rmse_rad=tracking_error/p['steps'],request_applied_gap_rad=request_gap/p['steps'],
                         iteration_s=time.monotonic()-iteration_start,training_hit_rate=sum(e['hit'] for e in episodes)/len(episodes) if episodes else None,
                         training_legal_rate=sum(e['outcome']=='legal_return' for e in episodes)/len(episodes) if episodes else None)
                row.update(guidance_coverage=coverage,predictor=auxiliary)
                history.append(row)
                with (trial/'learning.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            actor_delta=float(torch.linalg.vector_norm(torch.cat([v.detach().flatten() for v in policy.actor.parameters()])-actor_initial))
            critic_delta=float(torch.linalg.vector_norm(torch.cat([v.detach().flatten() for v in policy.critic.parameters()])-critic_initial))
            if actor_delta<=0 or critic_delta<=0:raise RuntimeError('Actor or critic did not update')
            checkpoint=dict(policy=policy.state_dict(),optimizer=algo.optimizer.state_dict(),cfg=cfg,seed=seed,group=group,
                            iteration=iterations,source_sha256=source_hashes(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                            **predictor_metadata(env),**experiment_metadata)
            if paired_initialization is not None:
                checkpoint.update(paired_initialization=paired_initialization,**pre_rollout_rng)
            if initialization_provenance is not None:
                checkpoint['actor_initialization']=initialization_provenance
            if env.predictor_module is not None:
                checkpoint.update(predictor=env.predictor_module.state_dict(),initial_sha256=hashlib.sha256((trial/'initial.pt').read_bytes()).hexdigest())
            if bank is not None:
                checkpoint.update(actor_dim=115,initial_sha256=hashlib.sha256((trial/'initial.pt').read_bytes()).hexdigest(),
                    serve_sequence=env.bank_draws,transitions_by_id=transitions_by_id)
                (trial/'serve_sequence.json').write_text(json.dumps(env.bank_draws,indent=2)+'\n')
            torch.save(checkpoint,trial/'policy.pt')
            with torch.no_grad():expected=policy.act_inference(obs).clone()
            loaded=ActorCritic(env.actor_width*5,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            disk_checkpoint=torch.load(trial/'policy.pt',map_location='cpu',weights_only=False)
            validate_predictor_checkpoint(disk_checkpoint,group)
            loaded.load_state_dict(disk_checkpoint['policy'])
            with torch.no_grad():assert torch.equal(expected,loaded.act_inference(obs))
            if env.predictor_module is not None:
                previous=env.predictor_module
                restored=set_predictor(env,group,artifact,disk_checkpoint)
                probe=torch.zeros(2,15)
                assert torch.equal(restored.predictor.predict(probe),previous.predictor.predict(probe))
            replay_evaluation=None
            if bank is not None:
                if disk_checkpoint['cfg']!=cfg or disk_checkpoint['actor_dim']!=115:
                    raise RuntimeError('Checkpoint data/control contract mismatch')
                evaluation=evaluate_trial(policy)
                replay_evaluation=evaluate_trial(loaded)
                for expected_row,actual_row in zip(evaluation['episodes'],replay_evaluation['episodes'],strict=True):
                    for field in ('raw_id','hit','outcome','net_touched','target_success'):
                        assert expected_row[field]==actual_row[field],(field,expected_row,actual_row)
                    if expected_row['landing_xy_m'] is not None:
                        np.testing.assert_allclose(expected_row['landing_xy_m'],actual_row['landing_xy_m'],atol=1e-5,rtol=0)
                assert sum(transitions_by_id.values())==iterations*num_envs*p['steps']
            else:evaluation=evaluate_trial(loaded)
            if env.online is not None:
                final_predictor=env.online.state_dict()
                initial=torch.load(trial/'initial.pt',map_location='cpu',weights_only=False)
                initial_actor=ActorCritic(135,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
                initial_actor.load_state_dict(initial['policy'])
                set_predictor(env,group,checkpoint=initial)
                initial_evaluation=evaluate(env,initial_actor,7201,max(num_envs,16))
                env.online.load_state_dict(final_predictor)
            result=dict(status='PASS',training_seed=seed,group=group,runtime=dict(python=platform.python_version(),torch=torch.__version__,device=device),
                        metrics={k:evaluation[k] for k in ('hit_rate','legal_return_rate','target_success_rate')},evaluation=evaluation,
                        actor_update_norm=actor_delta,critic_update_norm=critic_delta,timeout_count=timeout_count,
                        transitions=iterations*num_envs*p['steps'],seconds=time.monotonic()-started,
                        source_sha256=source_hashes(),checkpoint_sha256=hashlib.sha256((trial/'policy.pt').read_bytes()).hexdigest(),
                        scientific_conclusion='INCONCLUSIVE',history=history,training_episodes=episodes,initial_evaluation=initial_evaluation,
                        replay_evaluation=replay_evaluation,transitions_by_id=transitions_by_id,**predictor_metadata(env),**experiment_metadata)
            if initialization_provenance is not None:
                result.update(actor_initialization=initialization_provenance,ppo_optimizer_started_empty=True,
                              parameter_update_reference='BC initialization; BC fitting delta reported separately')
            if paired_initialization is not None:
                result.update(paired_initialization=paired_initialization,ppo_optimizer_started_empty=True,
                    pre_rollout_cpu_rng_sha256=hashlib.sha256(pre_rollout_rng['pre_rollout_torch_rng'].numpy().tobytes()).hexdigest(),
                    pre_rollout_cuda_rng_sha256=[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in pre_rollout_rng['pre_rollout_cuda_rng']])
            (trial/'result.json').write_text(json.dumps(result,indent=2)+'\n');runs.append(result)
            print(json.dumps({k:result[k] for k in ('group','training_seed','metrics','transitions','seconds')}),flush=True)
    for name,expected in evaluation_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Evaluation helper changed during training'
    if bank is not None:
        assert hashlib.sha256(Path(serve_bank).read_bytes()).hexdigest()==bank.sha256,'Serve manifest changed during training'
    assert before_hashes==source_hashes(),'Source changed during training'
    return dict(status='PASS',scope=mode+' diagnostic, not formal M7 five-group research',runs=runs,seconds=time.monotonic()-clock,
                scientific_conclusion='INCONCLUSIVE',zero_action_baseline=baseline,source_sha256=source_hashes(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                **predictor_metadata(env),**experiment_metadata)


def reevaluate(device,num_envs,source,destination,serve_bank=None,frozen_predictor=None,mean_predictor=None):
    """Re-score frozen policies under current event rules; never construct PPO."""
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    source=Path(source);destination=Path(destination)
    protocol=json.loads((source/'protocol.json').read_text())
    cfg=protocol['cfg']
    modes=[predictor_mode(group) for group in protocol['groups']]
    if any(mode!='none' for mode in modes) and len(modes)!=1:
        raise ValueError('Predictor re-evaluation requires a single explicit group')
    expected_mode=modes[0]
    legacy_mode='online' if protocol['groups']==['E'] else 'none'
    if protocol.get('predictor_mode',legacy_mode)!=expected_mode:
        raise ValueError('Protocol predictor mode/group mismatch')
    if ((frozen_predictor is not None and expected_mode!='frozen')
        or (mean_predictor is not None and expected_mode!='mean')):
        raise ValueError('Predictor artifact flag does not match checkpoint mode')
    artifact=mean_predictor if expected_mode=='mean' else frozen_predictor
    experiment_metadata={}
    if expected_mode in ('frozen','mean'):
        experiment_metadata=frozen_experiment_metadata(cfg,protocol['groups'][0])
        if any(protocol.get(k)!=v for k,v in experiment_metadata.items()):
            raise ValueError('Frozen re-evaluation experiment specification mismatch')
        provenance=protocol.get('predictor_provenance')
        if not provenance:raise ValueError('Frozen protocol requires artifact provenance')
        if artifact is None:
            artifact=provenance.get('artifact_path')
            if artifact is None:raise ValueError('Immutable predictor re-evaluation requires an explicit artifact')
            artifact=ROOT/Path(artifact)
        from airbot_tt.pace_predictor import FrozenPredictor,FixedMeanPredictor
        provider=FixedMeanPredictor if expected_mode=='mean' else FrozenPredictor
        if provider(num_envs,artifact=artifact).provenance!=provenance:
            raise ValueError('Re-evaluation frozen artifact/protocol mismatch')
        if expected_mode=='mean':mean_predictor=artifact
        else:frozen_predictor=artifact
    base_cfg=dict(cfg);serve_y_half_range_m=base_cfg.pop('serve_y_half_range_m',None)
    serve_z_m=base_cfg.pop('serve_z_m',None)
    if expected_mode in ('frozen','mean') and (serve_z_m!=2.3 or serve_y_half_range_m!=.015):
        raise ValueError('Frozen predictor validation covers Z=2.3 and Y half range .015 only')
    bank_sha=base_cfg.pop('serve_bank_sha256',None)
    bank_manifest=base_cfg.pop('serve_bank_manifest',None)
    if bank_sha is not None:
        from airbot_tt.pace_serves import FixedServeBank
        if serve_bank is None:raise ValueError('Fixed-bank checkpoint requires --pace-serve-bank')
        bank=FixedServeBank(serve_bank)
        if bank.sha256!=bank_sha or bank.manifest!=bank_manifest:raise ValueError('Checkpoint serve-bank mismatch')
    elif serve_bank is not None:raise ValueError('Synthetic checkpoint cannot be evaluated as a fixed-bank experiment')
    if protocol['num_envs']!=num_envs or base_cfg!=json.loads((ROOT/'configs/pace_rl.json').read_text()):
        raise ValueError('Re-evaluation requires the same environment count and effective config')
    destination.mkdir(parents=True,exist_ok=False)
    ActorCritic,_=original_ppo();env=PaceEnv(num_envs,device,group=protocol['groups'][0],serve_y_half_range_m=serve_y_half_range_m,serve_z_m=serve_z_m,serve_bank=serve_bank,frozen_predictor=frozen_predictor,mean_predictor=mean_predictor);torch.set_num_threads(1)
    started=time.monotonic();results=[];before_hashes=source_hashes()
    class Zero:
        def act_inference(self,obs):return torch.zeros((len(obs),6),device=obs.device)
    def evaluate_trial(policy):
        if serve_bank is not None:
            from pace_bank import evaluate_bank
            return evaluate_bank(env,policy,capture=True)
        return evaluate(env,policy,7201,16)
    zero=evaluate_trial(Zero())
    for seed in protocol['seeds']:
        for group in protocol['groups']:
            trial=source/f'{group}_{seed}';path=trial/'policy.pt'
            old=json.loads((trial/'result.json').read_text())
            checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            if checkpoint_hash!=old['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint['cfg']!=cfg or checkpoint['seed']!=seed or checkpoint['group']!=group:
                raise ValueError('Checkpoint/protocol mismatch')
            validate_predictor_checkpoint(checkpoint,group)
            if expected_mode in ('frozen','mean') and any(checkpoint.get(k)!=v for k,v in experiment_metadata.items()):
                raise ValueError('Frozen checkpoint experiment specification mismatch')
            p=cfg['ppo']
            if predictor_mode(group)!='none':set_predictor(env,group,artifact,checkpoint)
            policy=ActorCritic(env.actor_width*5,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            policy.load_state_dict(checkpoint['policy']);env.group=group
            evaluation_started=time.monotonic();result=evaluate_trial(policy)
            # No optimizer or state updates; verify tensors stayed identical.
            assert all(torch.equal(value.detach().cpu(),checkpoint['policy'][key]) for key,value in policy.state_dict().items())
            if env.frozen is not None:
                expected=checkpoint['predictor']['predictor']['model']
                if not all(torch.equal(v.cpu(),expected[k]) for k,v in env.frozen.predictor.model.state_dict().items()):
                    raise RuntimeError('Re-evaluation changed frozen predictor weights')
            changed=[dict(env=a['env'],old=a['outcome'],new=b['outcome'],net_touched=b['net_touched'])
                     for a,b in zip(old['evaluation']['episodes'],result['episodes']) if a['outcome']!=b['outcome']]
            row=dict(group=group,training_seed=seed,checkpoint_sha256=checkpoint_hash,
                     old_metrics=old['metrics'],evaluation=result,changed_outcomes=changed,evaluation_seconds=time.monotonic()-evaluation_started,
                     **predictor_metadata(env),**experiment_metadata)
            (destination/f'{group}_{seed}.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
            print('REEVALUATE',group,seed,result['outcomes'],flush=True)
    assert before_hashes==source_hashes(),'Source changed during evaluation'
    report=dict(status='PASS',event_rules=env.events.rules,scope='frozen checkpoint re-evaluation, no training',
                source_directory=str(source),source_sha256=before_hashes,zero_action_baseline=zero,runs=results,seconds=time.monotonic()-started,
                **predictor_metadata(env),**experiment_metadata)
    (destination/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
