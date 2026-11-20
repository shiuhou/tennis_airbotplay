"""First-update diagnostic using the pinned PPO, identical data and RNG per LR.

Run after AppLauncher through pace-check update. No upstream source edits.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from airbot_tt.config import ROOT
from airbot_tt.pace_control import bootstrap_timeout


def distribution_stats(algo):
    s=algo.storage
    with torch.no_grad():
        algo.policy.update_distribution(s.observations.flatten(0,1))
        mu,sigma=algo.policy.action_mean,algo.policy.action_std
        logp=algo.policy.get_actions_log_prob(s.actions.flatten(0,1))
        old_mu,old_sigma=s.mu.flatten(0,1),s.sigma.flatten(0,1)
        kl=(torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1)
        return dict(kl=float(kl.mean()),mu_error=float((mu-old_mu).abs().max()),
                    sigma_error=float((sigma-old_sigma).abs().max()),
                    logp_error=float((logp-s.actions_log_prob.flatten()).abs().max()),
                    clip_fraction=float(((logp-s.actions_log_prob.flatten()).exp().sub(1).abs()>algo.clip_param).float().mean()))


def compare_updates(original,rates):
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    rows=[]
    for lr in rates:
        algo=copy.deepcopy(original)
        algo.learning_rate=lr
        for group in algo.optimizer.param_groups:group['lr']=lr
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        consistency=distribution_stats(algo)
        if max(consistency[k] for k in ('mu_error','sigma_error','logp_error'))>1e-4:
            raise AssertionError(f'Before-update storage mismatch: {consistency}')
        steps=[]
        def before(opt,args,kwargs):
            steps.append(dict(before=distribution_stats(algo),learning_rate=opt.param_groups[0]['lr'],
                actor_grad_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in algo.policy.actor.parameters() if p.grad is not None))),
                critic_grad_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in algo.policy.critic.parameters() if p.grad is not None)))))
        def after(opt,args,kwargs):steps[-1]['after']=distribution_stats(algo)
        pre=algo.optimizer.register_step_pre_hook(before);post=algo.optimizer.register_step_post_hook(after)
        losses=algo.update();pre.remove();post.remove()
        final=distribution_stats(algo)
        row=dict(initial_lr=lr,consistency=consistency,steps=steps,losses=losses,final=final,
                 stability_pass=final['kl']<=.02 and all(np.isfinite(v) for v in losses.values()))
        rows.append(row)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    return rows


def diagnose(device,num_envs,destination):
    from train_pace import original_ppo
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();ActorCritic,PPO=original_ppo()
    cfg=json.loads((ROOT/'configs/pace_rl.json').read_text());p=cfg['ppo']
    env=PaceEnv(num_envs,device);torch.set_num_threads(1)
    results=[]
    for seed in cfg['seeds']:
        for group in ('A','B'):
            if time.monotonic()-started>600:raise RuntimeError('Diagnostic exceeded ten minutes')
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.group=group;env.rng=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(env.n)]
            env.reset(env.ids);obs,critic=env.observations()
            policy=ActorCritic(115,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],activation='elu',init_noise_std=p['init_std']).to(device)
            gamma=.95**(50*env.policy_dt)
            algo=PPO(policy,num_learning_epochs=p['epochs'],num_mini_batches=p['minibatches'],clip_param=p['clip'],gamma=gamma,lam=gamma,
                     entropy_coef=p['entropy'],learning_rate=.001,schedule='adaptive',desired_kl=.01,device=device)
            algo.init_storage('rl',num_envs,p['steps'],[115],[200],[6])
            hits=0;done_count=0;rewards=[]
            with torch.no_grad():
                for _ in range(p['steps']):
                    actions=algo.act(obs,critic)
                    obs,critic,reward,done,info=env.step(actions)
                    corrected=bootstrap_timeout(reward,policy.evaluate(info['terminal_critic']),info['truncated'],gamma)
                    algo.process_env_step(corrected,done,{})
                    done_count+=int(done.sum());hits+=int(env.events.hit.sum());rewards.append(reward.clone())
                algo.compute_returns(critic)
            s=algo.storage
            snapshot={name:value for name,value in vars(s).items() if isinstance(value,torch.Tensor)}
            torch.save(dict(storage=snapshot,policy=policy.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),destination/f'{group}_{seed}_rollout.pt')
            row=dict(seed=seed,group=group,completed_episodes=done_count,hit_steps=hits,
                     reward_min=float(torch.stack(rewards).min()),reward_max=float(torch.stack(rewards).max()),
                     returns_std=float(s.returns.std()),value_std=float(s.values.std()),advantage_std=float(s.advantages.std()),
                     comparisons=compare_updates(algo,[.001,.0001,.00001]))
            results.append(row)
            (destination/f'{group}_{seed}.json').write_text(json.dumps(row,indent=2)+'\n')
            print('UPDATE_DIAGNOSTIC',group,seed,[(x['initial_lr'],x['final']['kl']) for x in row['comparisons']],flush=True)
    return dict(status='PASS',scope='data consistency and paired first-update diagnostic, not task success',
                source_sha256=source_hashes(),seconds=time.monotonic()-started,runs=results)
