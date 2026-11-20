"""Thin initial A/B runner around pinned PACE ActorCritic/PPO; launched by run.sh.

No predictor, no imitation, no hardware, no rewritten PPO. Terminal-value timeout
bootstrap is supplied once before the original storage call.
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


def evaluate(env,policy,seed,episodes):
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
    return dict(event_rules=env.events.rules,total=len(rows),hit_rate=sum(r['hit'] for r in rows)/len(rows),
                legal_return_rate=sum(r['outcome']=='legal_return' for r in rows)/len(rows),
                target_success_rate=sum(r['target_success'] for r in rows)/len(rows),
                outcomes={name:sum(r['outcome']==name for r in rows) for name in sorted(set(r['outcome'] for r in rows))},episodes=rows)


def run(device,num_envs,mode,destination,learning_rate=None,serve_y_half_range_m=None,groups=None,serve_z_m=None):
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    destination=Path(destination)
    for name in ('lifecycle','parity','control'):
        report=json.loads((ROOT/f'.agents/runs/foundation/m7/{name}.json').read_text())
        if report['status']!='PASS' or report.get('source_sha256')!=source_hashes():
            raise RuntimeError(f'M7 gate not current/PASS: {name}')
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
    p=cfg['ppo']
    # Fixed small diagnostic budget; smoke never establishes learning effectiveness.
    iterations=4 if mode=='smoke' else cfg['diagnostic_iterations']
    seeds=cfg['seeds'][:1] if mode=='smoke' else cfg['seeds']
    groups=list(groups) if groups is not None else (['A'] if mode=='smoke' else ['A','B'])
    if not groups or len(set(groups))!=len(groups) or any(g not in ('A','B') for g in groups):raise ValueError('Expected unique A/B groups')
    destination.mkdir(parents=True,exist_ok=False)
    env=PaceEnv(num_envs,device,serve_y_half_range_m=serve_y_half_range_m,serve_z_m=serve_z_m);torch.set_num_threads(1)
    clock=time.monotonic();runs=[]
    baseline=None
    if mode=='ab':
        class ZeroAction:
            def act_inference(self,obs):return torch.zeros((len(obs),6),device=obs.device)
        baseline=evaluate(env,ZeroAction(),7201,max(num_envs,16))
        (destination/'zero_action.json').write_text(json.dumps(baseline,indent=2)+'\n')
    protocol=dict(mode=mode,iterations=iterations,num_envs=num_envs,rollout_steps=p['steps'],seeds=seeds,groups=groups,
                  transitions_per_trial=iterations*num_envs*p['steps'],cfg=cfg,source_sha256=source_hashes(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scientific_conclusion='INCONCLUSIVE',heldout_final_test=False)
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for seed in seeds:
        for group in groups:
            trial=destination/f'{group}_{seed}';trial.mkdir()
            started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.group=group;env.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(env.n)]
            env.reset(env.ids);obs,critic=env.observations()
            policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            gamma=.95**(50*env.policy_dt)
            algo=PPO(policy,num_learning_epochs=p['epochs'],num_mini_batches=p['minibatches'],clip_param=p['clip'],gamma=gamma,lam=gamma,
                     entropy_coef=p['entropy'],learning_rate=p['learning_rate'],schedule='adaptive',desired_kl=.01,device=device)
            algo.init_storage('rl',num_envs,p['steps'],[115],[200],[6])
            actor_initial=torch.cat([v.detach().flatten() for v in policy.actor.parameters()]).clone()
            critic_initial=torch.cat([v.detach().flatten() for v in policy.critic.parameters()]).clone()
            history=[];episodes=[];timeout_count=0
            for iteration in range(iterations):
                if time.monotonic()-started>cfg['smoke_max_s']:
                    raise RuntimeError('Trial exceeded 30 minute diagnostic budget; partial logs retained')
                clip_count=0;tracking_error=0.;request_gap=0.;iteration_start=time.monotonic()
                with torch.no_grad():
                    for _ in range(p['steps']):
                        actions=algo.act(obs,critic)
                        clip_count+=float((actions.abs()>1).float().mean())
                        obs,critic,reward,done,info=env.step(actions)
                        terminal_value=policy.evaluate(info['terminal_critic'])
                        corrected=bootstrap_timeout(reward,terminal_value,info['truncated'],gamma)
                        timeout_count+=int(info['truncated'].sum())
                        # Do NOT pass time_outs: upstream would add the OLD value a second time.
                        algo.process_env_step(corrected,done,{})
                        episodes.extend(info['episodes'])
                        tracking_error+=float((info['q_measured']-info['q_applied']).square().mean().sqrt())
                        request_gap+=float((info['q_request']-info['q_applied']).abs().mean())
                    algo.compute_returns(critic)
                losses=algo.update()
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
                history.append(row)
                with (trial/'learning.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            actor_delta=float(torch.linalg.vector_norm(torch.cat([v.detach().flatten() for v in policy.actor.parameters()])-actor_initial))
            critic_delta=float(torch.linalg.vector_norm(torch.cat([v.detach().flatten() for v in policy.critic.parameters()])-critic_initial))
            if actor_delta<=0 or critic_delta<=0:raise RuntimeError('Actor or critic did not update')
            checkpoint=dict(policy=policy.state_dict(),optimizer=algo.optimizer.state_dict(),cfg=cfg,seed=seed,group=group,
                            iteration=iterations,source_sha256=source_hashes(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(checkpoint,trial/'policy.pt')
            with torch.no_grad():expected=policy.act_inference(obs).clone()
            loaded=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            loaded.load_state_dict(torch.load(trial/'policy.pt',map_location=device,weights_only=False)['policy'])
            with torch.no_grad():assert torch.equal(expected,loaded.act_inference(obs))
            evaluation=evaluate(env,loaded,7201,max(num_envs,16))
            result=dict(status='PASS',training_seed=seed,group=group,runtime=dict(python=platform.python_version(),torch=torch.__version__,device=device),
                        metrics={k:evaluation[k] for k in ('hit_rate','legal_return_rate','target_success_rate')},evaluation=evaluation,
                        actor_update_norm=actor_delta,critic_update_norm=critic_delta,timeout_count=timeout_count,
                        transitions=iterations*num_envs*p['steps'],seconds=time.monotonic()-started,
                        source_sha256=source_hashes(),checkpoint_sha256=hashlib.sha256((trial/'policy.pt').read_bytes()).hexdigest(),
                        scientific_conclusion='INCONCLUSIVE',history=history)
            (trial/'result.json').write_text(json.dumps(result,indent=2)+'\n');runs.append(result)
            print(json.dumps({k:result[k] for k in ('group','training_seed','metrics','transitions','seconds')}),flush=True)
    return dict(status='PASS',scope=mode+' diagnostic, not formal M7 five-group research',runs=runs,seconds=time.monotonic()-clock,
                scientific_conclusion='INCONCLUSIVE',zero_action_baseline=baseline,source_sha256=source_hashes())


def reevaluate(device,num_envs,source,destination):
    """Re-score frozen policies under current event rules; never construct PPO."""
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    source=Path(source);destination=Path(destination)
    protocol=json.loads((source/'protocol.json').read_text())
    cfg=protocol['cfg']
    base_cfg=dict(cfg);serve_y_half_range_m=base_cfg.pop('serve_y_half_range_m',None)
    serve_z_m=base_cfg.pop('serve_z_m',None)
    if protocol['num_envs']!=num_envs or base_cfg!=json.loads((ROOT/'configs/pace_rl.json').read_text()):
        raise ValueError('Re-evaluation requires the same environment count and effective config')
    destination.mkdir(parents=True,exist_ok=False)
    ActorCritic,_=original_ppo();env=PaceEnv(num_envs,device,serve_y_half_range_m=serve_y_half_range_m,serve_z_m=serve_z_m);torch.set_num_threads(1)
    started=time.monotonic();results=[];before_hashes=source_hashes()
    class Zero:
        def act_inference(self,obs):return torch.zeros((len(obs),6),device=obs.device)
    zero=evaluate(env,Zero(),7201,16)
    for seed in protocol['seeds']:
        for group in protocol['groups']:
            trial=source/f'{group}_{seed}';path=trial/'policy.pt'
            old=json.loads((trial/'result.json').read_text())
            checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            if checkpoint_hash!=old['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
            checkpoint=torch.load(path,map_location=device,weights_only=False)
            if checkpoint['cfg']!=cfg or checkpoint['seed']!=seed or checkpoint['group']!=group:
                raise ValueError('Checkpoint/protocol mismatch')
            p=cfg['ppo']
            policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            policy.load_state_dict(checkpoint['policy']);env.group=group
            evaluation_started=time.monotonic();result=evaluate(env,policy,7201,16)
            # No optimizer or state updates; verify tensors stayed identical.
            assert all(torch.equal(value,checkpoint['policy'][key]) for key,value in policy.state_dict().items())
            changed=[dict(env=a['env'],old=a['outcome'],new=b['outcome'],net_touched=b['net_touched'])
                     for a,b in zip(old['evaluation']['episodes'],result['episodes']) if a['outcome']!=b['outcome']]
            row=dict(group=group,training_seed=seed,checkpoint_sha256=checkpoint_hash,
                     old_metrics=old['metrics'],evaluation=result,changed_outcomes=changed,evaluation_seconds=time.monotonic()-evaluation_started)
            (destination/f'{group}_{seed}.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
            print('REEVALUATE',group,seed,result['outcomes'],flush=True)
    assert before_hashes==source_hashes(),'Source changed during evaluation'
    report=dict(status='PASS',event_rules=env.events.rules,scope='frozen checkpoint re-evaluation, no training',
                source_directory=str(source),source_sha256=before_hashes,zero_action_baseline=zero,runs=results,seconds=time.monotonic()-started)
    (destination/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
