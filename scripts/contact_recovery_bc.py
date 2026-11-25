"""One Actor-only fit on original demonstrations and measured recovery suffixes."""
import json
from pathlib import Path
import time

import torch

from airbot_tt.config import ROOT
from airbot_tt.pace_bc import bc_source_hashes, recipe, sha256
from airbot_tt.pace_serves import FIVE_SERVE_IDS
from contact_bc import noisy_evaluation, same_episodes, save_json


def training_data(original, recoveries):
    """Build one shared batch. Ball, source and trajectory each get one vote at their level."""
    if tuple(original['raw_ids']) != FIVE_SERVE_IDS:
        raise ValueError('Unexpected original serve identities')
    by_ball = {raw_id: {} for raw_id in FIVE_SERVE_IDS}
    for row in recoveries:
        raw_id, switch = row['raw_id'], row['switch_tick']
        if raw_id not in by_ball or switch not in (6, 9, 12) or switch in by_ball[raw_id]:
            raise ValueError('Unknown or duplicated recovery trajectory')
        by_ball[raw_id][switch] = row

    observations, actions, weights, trajectories = [], [], [], []
    for index, raw_id in enumerate(FIVE_SERVE_IDS):
        mask = original['trajectory_ids'] == index
        sources = [(None, {k: original[k][mask] for k in ('observations', 'actions', 'ticks')})]
        recoveries_for_ball = by_ball[raw_id]
        sources.extend(sorted(recoveries_for_ball.items()))
        for switch, row in sources:
            obs, action = row['observations'], row['actions']
            count = len(action)
            start = switch or 0
            if count == 0 or obs.shape != (count, 115) or action.shape != (count, 6):
                raise ValueError('Incorrect observation or six-axis action shape')
            if not torch.equal(row['ticks'], torch.arange(start, start + count)):
                raise ValueError('Expected consecutive pre-action ticks starting at the source boundary')
            if not torch.isfinite(obs).all() or not torch.isfinite(action).all() or action.abs().max() > 1 + 1e-7:
                raise ValueError('Non-finite observations or invalid action labels')
            source_weight = .5 if recoveries_for_ball else 1.
            trajectory_weight = source_weight / len(FIVE_SERVE_IDS)
            if switch is not None:
                trajectory_weight /= len(recoveries_for_ball)
            observations.append(obs)
            actions.append(action)
            weights.append(torch.full((count,), trajectory_weight / count, dtype=obs.dtype))
            trajectories.append(dict(raw_id=raw_id, source='original' if switch is None else 'recovery',
                                     switch_tick=switch, first_tick=start, last_tick=start+count-1,
                                     samples=count, weight=trajectory_weight, row_weight=trajectory_weight/count))
    return torch.cat(observations), torch.cat(actions), torch.cat(weights), trajectories


def fit_weighted_actor(policy, observations, actions, weights):
    """Run the original full-batch optimizer with the predeclared per-sample weights."""
    cfg = recipe()
    original = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=cfg['learning_rate'])
    started = time.monotonic()
    history = []
    for step in range(1, cfg['updates'] + 1):
        optimizer.zero_grad(set_to_none=True)
        mse = (policy.actor(observations) - actions).square().mean(1)
        loss = torch.dot(mse, weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite recovery BC loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in policy.actor.parameters()):
            raise RuntimeError('Nonfinite recovery BC gradient')
        optimizer.step()
        if step == 1 or step % cfg['log_interval'] == 0:
            history.append(dict(update=step, loss=float(loss.detach())))
        if time.monotonic() - started > cfg['maximum_fit_seconds']:
            raise RuntimeError('Fixed original BC wall-time guard exceeded')
    current = policy.state_dict()
    if any(not torch.isfinite(v).all() for v in current.values()):
        raise RuntimeError('Nonfinite fitted model')
    if any(not torch.equal(v, original[k]) for k, v in current.items() if not k.startswith('actor.')):
        raise RuntimeError('BC changed original Critic or noise parameters')
    with torch.no_grad():
        error = (policy.actor(observations) - actions).square().mean(1)
        updates = torch.cat([(v-original[k]).flatten() for k, v in current.items() if k.startswith('actor.')])
        return dict(final_loss=float(torch.dot(error, weights)), history=history, optimizer_steps=cfg['updates'],
                    actor_update_norm=float(updates.norm()), critic_unchanged=True, std_unchanged=True,
                    seconds=time.monotonic()-started)


def fit_recovery(env, directory, original_bc, baseline, recoveries):
    """Fit each original initialization once, then score autonomous final and reloaded Actors."""
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from train_pace import original_ppo

    directory, original_bc, baseline = map(Path, (directory, original_bc, baseline))
    stage = json.loads((directory / 'recovery.json').read_text())
    qualifying_balls = {row['raw_id'] for row in stage['cases'] if row['qualifies']}
    keys = {(row['raw_id'], row['switch_tick']) for row in stage['cases'] if row['data_eligible']}
    if stage['status'] != 'PASS' or stage['coverage_gate'] != 'PASS' or len(qualifying_balls) < 3:
        raise ValueError('Fitting requires the completed physical recovery gate')
    if keys != {(row['raw_id'], row['switch_tick']) for row in recoveries}:
        raise ValueError('Recovery labels differ from the complete set of successful takeovers')
    frozen = json.loads((original_bc / 'initialization.json').read_text())
    cfg = json.loads((ROOT / 'configs/pace_rl.json').read_text())
    cfg.update(serve_bank_sha256=env.serve_bank.sha256, serve_bank_manifest=env.serve_bank.manifest)
    original_data = original_bc / 'demonstrations.pt'
    if not (env.n == 16 and env.group == 'A' and env.online is None and cfg == frozen['cfg']
            and frozen['recipe'] == recipe() and frozen['source_sha256'] == source_hashes()
            and frozen['bc_source_sha256'] == bc_source_hashes() and frozen['status'] == 'PASS'
            and sha256(original_data) == frozen['demonstration_sha256']
            and Path(frozen['baseline_directory']).resolve() == baseline.resolve()):
        raise ValueError('Original BC data, recipe, baseline or simulation compatibility failed')
    original = torch.load(original_data, map_location='cpu', weights_only=True)
    if len(original['actions']) != 237:
        raise ValueError('Expected the 237 original demonstration samples')
    obs, actions, weights, trajectories = training_data(original, recoveries)
    obs, actions, weights = (x.to(env.device) for x in (obs, actions, weights))
    inputs = [directory / 'protocol.json', directory / 'recovery.json', directory / 'recoveries.pt',
              original_bc / 'initialization.json', original_data, Path(__file__).resolve()]
    original_results = {}
    for seed in recipe()['seeds']:
        initial = baseline / f'A_{seed}' / 'initial.pt'
        previous = frozen['actors'][str(seed)]
        bc_actor, bc_report = original_bc / previous['path'], original_bc / f'bc_{seed}.json'
        if sha256(initial) != previous['original_initial_sha256'] or sha256(bc_actor) != previous['sha256']:
            raise ValueError('Original paired checkpoint provenance mismatch')
        original_results[seed] = json.loads(bc_report.read_text())
        if original_results[seed]['seed'] != seed or original_results[seed]['contacts'] != previous['contacts']:
            raise ValueError('Original closed-loop outcomes differ from their manifest')
        inputs.extend((initial, bc_actor, bc_report))
    hashes = {str(p.resolve()): sha256(p) for p in inputs}
    fit_dir = directory / 'fit'
    fit_dir.mkdir(exist_ok=True)
    fit_recipe = dict(recipe(), trajectory_weighting='equal_ball_then_source_then_trajectory')
    protocol = dict(schema='airbot-recovery-bc-v1', recipe=fit_recipe, input_sha256=hashes,
                    cfg=cfg, source_sha256=source_hashes(), bc_source_sha256=bc_source_hashes(),
                    trajectories=trajectories, original_samples=237, recovery_samples=len(obs)-237)
    protocol_path = fit_dir / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Cannot resume a recovery fit with different inputs')
    else:
        save_json(protocol_path, protocol)
    protocol_sha = sha256(protocol_path)
    torch.set_num_threads(1)
    actor_critic, _ = original_ppo()
    p = cfg['ppo']

    def new_policy(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        policy = actor_critic(115, 200, 6, actor_hidden_dims=p['hidden'], critic_hidden_dims=p['hidden'],
                              activation='elu', init_noise_std=p['init_std']).to(env.device)
        initial = torch.load(baseline / f'A_{seed}' / 'initial.pt', map_location='cpu', weights_only=False)
        if initial['cfg'] != cfg or initial['seed'] != seed or not all(
                torch.equal(v.detach().cpu(), initial['policy'][k]) for k, v in policy.state_dict().items()):
            raise ValueError('Original seeded Actor, Critic and noise parameters did not reproduce')
        return policy

    runs = []
    for seed in recipe()['seeds']:
        path = fit_dir / f'actor_{seed}.pt'
        policy = new_policy(seed)
        resumed = path.exists()
        if not resumed:
            fitting = fit_weighted_actor(policy, obs, actions, weights)
            temporary = path.with_suffix('.tmp')
            torch.save(dict(seed=seed, actor={k: v.detach().cpu() for k, v in policy.actor.state_dict().items()}, fitting=fitting,
                            protocol_sha256=protocol_sha), temporary)
            temporary.replace(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['seed'] != seed or checkpoint['protocol_sha256'] != protocol_sha:
            raise ValueError('Fitted Actor provenance mismatch')
        # An interrupted evaluation resumes from the completed fit, without extra optimizer updates.
        if resumed:
            policy.actor.load_state_dict(checkpoint['actor'])
        evaluation = evaluate_bank(env, policy, capture=True)
        restored = new_policy(seed)
        restored.actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['actor'])
        with torch.no_grad():
            assert torch.equal(policy.act_inference(obs), restored.act_inference(obs))
        replay = evaluate_bank(env, restored, capture=True)
        same_episodes(evaluation, replay)
        contacts = sum(row['hit'] for row in evaluation['episodes'])
        row = dict(seed=seed, resumed_completed_fit=resumed, fitting=checkpoint['fitting'], actor_sha256=sha256(path),
                   evaluation=evaluation, replay_evaluation=replay, contacts=contacts,
                   original_contacts=original_results[seed]['contacts'],
                   contact_gate_pass=contacts >= recipe()['minimum_contacts_per_seed'])
        save_json(fit_dir / f'bc_{seed}.json', row)
        runs.append(row)
        print('Recovery BC', seed, 'contacts', contacts, 'loss', checkpoint['fitting']['final_loss'], flush=True)
    passed = all(r['contact_gate_pass'] for r in runs)
    noise = dict(status='NOT_RUN', total=0, runs=[])
    if passed:
        for seed in recipe()['seeds']:
            for kind in ('original', 'recovery_bc'):
                policy = new_policy(seed)
                if kind == 'recovery_bc':
                    checkpoint = torch.load(fit_dir / f'actor_{seed}.pt', map_location='cpu', weights_only=True)
                    policy.actor.load_state_dict(checkpoint['actor'])
                before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                trials = [noisy_evaluation(env, policy, seed, rep) for rep in range(recipe()['noise_repetitions'])]
                assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items())
                noise['runs'].append(dict(seed=seed, kind=kind, trials=trials))
        noise['total'] = sum(trial['total'] for item in noise['runs'] for trial in item['trials'])
        assert noise['total'] == 300
        noise['status'] = 'PASS'
    save_json(fit_dir / 'noise.json', noise)
    if hashes != {str(p.resolve()): sha256(p) for p in inputs}:
        raise ValueError('Recovery fitting inputs changed')
    result = dict(status='PASS', schema='airbot-recovery-bc-v1', contact_gate='PASS' if passed else 'FAIL',
                  noise_status=noise['status'], noise_episodes=noise['total'], ppo_status='READY' if passed else 'NOT_RUN',
                  ppo_updates=0, fit_protocol_sha256=protocol_sha, noise_sha256=sha256(fit_dir / 'noise.json'),
                  original_samples=237, recovery_samples=len(obs)-237, runs=runs)
    save_json(fit_dir / 'results.json', result)
    return result
