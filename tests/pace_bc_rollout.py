"""No-learning replay of BC actions on recorded versus self-generated observations.

These diagnostics run only after the fixed BC experiment. They do not refit,
change the contact gate, or authorize PPO. Error comparisons use the common
prefix before either actual or reference paddle contact.
"""
import argparse
import json
from pathlib import Path
import sys


def intervene_observation(obs, references, tick, active, condition):
    """Replace only selected fields in a fresh Actor input, never environment state.

    References are complete saved five-frame observations indexed by policy tick,
    including the original previous-action lag and reset padding. Exhaustion or
    a completed first episode leaves that environment entirely autonomous.
    """
    import torch
    fields = {'baseline': (), 'previous_action': (slice(12, 18),),
              'velocity': (slice(6, 12),), 'both': (slice(6, 18),)}
    if condition not in fields or tick < 0 or obs.ndim != 2 or obs.shape[1] != 115:
        raise ValueError('Expected a named feedback condition and 115-column observation')
    result = obs.clone()
    applied = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)
    frames = result.reshape(-1, 5, 23)
    for i, reference in enumerate(references):
        if active[i] and tick < len(reference) and fields[condition]:
            for field in fields[condition]:
                frames[i, :, field] = reference[tick].reshape(5, 23)[:, field]
            applied[i] = True
    return result, applied


def feedback_audit(directory, device):
    """Bounded 2x2 intervention on the existing seed7101; no optimizer or PPO."""
    import time
    import numpy as np
    import torch
    from airbot_tt.config import ROOT
    from airbot_tt.pace_bc import sha256, bc_source_hashes
    from airbot_tt.pace_env import PaceEnv
    from contact_bc import same_episodes
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from train_pace import original_ppo

    directory = Path(directory)
    started = time.monotonic()
    seed = 7101
    original = json.loads((directory / 'protocol.json').read_text())
    initialization = json.loads((directory / 'initialization.json').read_text())
    assert original['source_sha256'] == source_hashes()
    assert original['bc_source_sha256'] == bc_source_hashes()
    baseline = Path(original['baseline_directory'])
    manifest = baseline.parent / 'manifest.json'
    artifact = directory / initialization['actors'][str(seed)]['path']
    assert sha256(artifact) == initialization['actors'][str(seed)]['sha256']
    assert sha256(manifest) == initialization['serve_bank_sha256']
    assert sha256(directory / 'demonstrations.pt') == initialization['demonstration_sha256']
    data = torch.load(directory / 'demonstrations.pt', map_location='cpu', weights_only=False)
    assert data['raw_ids'] == [9378, 10818, 5255, 12250, 14852]
    collection = json.loads((directory / 'collection.json').read_text())['evaluation']
    inputs = {str(p): sha256(p) for p in (artifact, manifest, directory / 'demonstrations.pt',
              directory / 'protocol.json', directory / 'initialization.json', directory / 'collection.json',
              directory / 'bc_7101.json', directory / 'rollout_diagnostic.npz', baseline / 'A_7101/initial.pt')}
    for name, expected in original['script_input_sha256'].items():
        assert sha256(ROOT / name) == expected
        inputs[str(ROOT / name)] = expected
    conditions = ['baseline', 'previous_action', 'velocity', 'both']
    protocol = dict(schema='bc-feedback-v1', seed=seed, raw_ids=data['raw_ids'], conditions=conditions,
        physics_hz=960, policy_hz=48, scored_episodes=20, num_envs=16,
        changed='Only the Actor-input copy: whole five-frame qd and/or previous-action fields',
        tail_rule='Use actual observations after reference exhaustion; never repeat the last demo frame',
        reset_rule='Intervene and record only the first episode of canonical envs 0..4; padding stays autonomous',
        early_windows_s=[[0, .05], [.05, .2], [0, .2]], windows_right_exclusive=True,
        timing='Integer policy ticks align references; measured elapsed time is saved separately',
        comparison='Errors use actual measured state, never substituted fields; late comparisons stop before either contact',
        interpretation='Descriptive same-five-train-ball diagnosis, not an independent test or deployable policy',
        bc_updates=0, ppo_updates=0, source_sha256=source_hashes(), bc_source_sha256=bc_source_hashes(),
        script_sha256=sha256(__file__), input_sha256=inputs)
    references, targets = [], []
    for i in range(5):
        mask = data['trajectory_ids'] == i
        ticks = data['ticks'][mask]
        assert torch.equal(ticks, torch.arange(len(ticks), dtype=ticks.dtype))
        np.testing.assert_allclose(data['times'][mask].numpy(), ticks.numpy()/48, atol=2e-5, rtol=0)
        reference = data['observations'][mask].to(device)
        target = data['actions'][mask].to(device)
        assert reference.shape == (len(ticks), 115) and target.shape == (len(ticks), 6)
        # Check every saved frame against its preceding label, including t=0 padding.
        for tick in range(len(ticks)):
            for frame in range(5):
                history_tick = max(0, tick - 4 + frame)
                previous = target[history_tick-1] if history_tick else torch.zeros(6, device=device)
                assert torch.equal(reference[tick].reshape(5, 23)[frame, 12:18], previous.clamp(-10, 10))
        assert reference.reshape(-1, 5, 23)[:, :, :12].abs().max() < 10
        references.append(reference)
        targets.append(target)
    # Freeze protocol before constructing/running the physical environment.
    (directory / 'feedback_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(1)
    env = PaceEnv(16, device, group='A', serve_bank=manifest)
    assert env.policy_dt == 1/48 and env.decimation == 20
    ActorCritic, _ = original_ppo()
    cfg = original['cfg']['ppo']
    torch.manual_seed(seed)
    policy = ActorCritic(115, 200, 6, actor_hidden_dims=cfg['hidden'], critic_hidden_dims=cfg['hidden'],
                         activation='elu', init_noise_std=cfg['init_std']).to(device)
    policy.load_state_dict(torch.load(baseline / 'A_7101/initial.pt', map_location='cpu', weights_only=False)['policy'])
    policy.actor.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=False)['actor'])
    policy.eval()
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    command_fields = ('q', 'v', 'request', 'previous_request', 'start_v', 'c2', 'c3', 't')
    arrays, runs = {}, []
    inference_checks = 0
    for condition in conditions:
        traces = [[] for _ in range(5)]

        class Intervention:
            def __init__(self):
                self.first_ids = None
                self.last_action = None
                self.last_ids = None
                self.tick = 0

            def act_inference(self, obs):
                nonlocal inference_checks
                if self.first_ids is None:
                    self.first_ids = env.serve_id[:5].clone()
                active = env.serve_id[:5] == self.first_ids
                assert (env.policy_index[:5][active] == self.tick).all()
                if self.last_action is not None:
                    unchanged_episode = env.serve_id == self.last_ids
                    assert torch.equal(env.previous_action[unchanged_episode], self.last_action[unchanged_episode])
                assert torch.equal(obs, env.actor_history.flatten(1))
                assert torch.equal(obs.reshape(-1, 5, 23)[:, -1, 12:18], env.previous_action.clamp(-10, 10))
                before = dict(observation=obs.clone(), history=env.actor_history.clone(),
                    previous_action=env.previous_action.clone(),
                    measured_q=env.robot.data.joint_pos[:, env.joints].clone(),
                    measured_qd=env.robot.data.joint_vel[:, env.joints].clone(),
                    **{name: getattr(env.command, name).clone() for name in command_fields})
                actor_input, applied = intervene_observation(obs, references, self.tick, active, condition)
                actions = policy.act_inference(actor_input)
                # Check ownership after inference, not only after replacement.
                current = dict(observation=obs, history=env.actor_history, previous_action=env.previous_action,
                    measured_q=env.robot.data.joint_pos[:, env.joints], measured_qd=env.robot.data.joint_vel[:, env.joints],
                    **{name: getattr(env.command, name) for name in command_fields})
                assert all(torch.equal(before[key], value) for key, value in current.items())
                assert torch.equal(actor_input[5:], obs[5:])
                assert not applied[5:].any()
                inference_checks += 1
                for i in range(5):
                    if not active[i]:
                        assert not applied[i] and torch.equal(actor_input[i], obs[i])
                        continue
                    available = self.tick < len(references[i])
                    reference = references[i][self.tick] if available else torch.full_like(obs[i], float('nan'))
                    target = targets[i][self.tick] if available else torch.full_like(actions[i], float('nan'))
                    row = dict(tick=self.tick, time_s=self.tick/48, elapsed_s=float(env.elapsed[i]),
                        reference_available=available, applied=bool(applied[i]),
                        actual_observation=obs[i], actor_input=actor_input[i], reference_observation=reference,
                        action=actions[i], reference_action=target,
                        measured_q_minus_nominal=before['measured_q'][i]-env.nominal,
                        measured_qd=before['measured_qd'][i],
                        **{'command_'+name: before[name][i] for name in command_fields})
                    traces[i].append({key: value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value
                                      for key, value in row.items()})
                self.last_action = actions.clone()
                self.last_ids = env.serve_id.clone()
                self.tick += 1
                return actions

        evaluation = evaluate_bank(env, Intervention(), capture=True)
        evaluation['scope'] = 'Same five inspected train states; deterministic BC with diagnostic Actor-input intervention: ' + condition
        if condition == 'baseline':
            same_episodes(evaluation, json.loads((directory / 'bc_7101.json').read_text())['evaluation'])
        rows = []
        for i, raw_id in enumerate(data['raw_ids']):
            trace = {key: np.asarray([r[key] for r in traces[i]]) for key in traces[i][0]}
            reference_frames = trace['reference_observation'].reshape(-1, 5, 23)
            trace['q_rmse_rad'] = np.sqrt(np.mean((trace['measured_q_minus_nominal']-reference_frames[:, -1, :6])**2, axis=1))
            trace['qd_rmse_rad_s'] = np.sqrt(np.mean((trace['measured_qd']-reference_frames[:, -1, 6:12])**2, axis=1))
            trace['action_rmse'] = np.sqrt(np.mean((trace['action']-trace['reference_action'])**2, axis=1))
            actual_frames = trace['actual_observation'].reshape(-1, 5, 23)
            trace['previous_action_history_rmse'] = np.sqrt(np.mean((actual_frames[:, :, 12:18]-reference_frames[:, :, 12:18])**2, axis=(1, 2)))
            if condition == 'baseline':
                with np.load(directory / 'rollout_diagnostic.npz') as prior:
                    previous = prior[f'{seed}_{raw_id}']
                size = len(previous)
                np.testing.assert_allclose(trace['time_s'][:size], previous[:, 0], atol=1e-7, rtol=0)
                np.testing.assert_allclose(trace['action'][:size], previous[:, 5:11], atol=1e-6, rtol=0)
                np.testing.assert_allclose(trace['measured_q_minus_nominal'][:size], previous[:, 17:23], atol=1e-6, rtol=0)
            for key, value in trace.items():
                arrays[f'{condition}_{raw_id}_{key}'] = value
            episode = evaluation['episodes'][i]
            actual_impact = episode['motion']['impact']
            actual_contact = actual_impact['time_s'] if actual_impact else None
            reference_contact = collection['episodes'][i]['motion']['impact']['time_s']
            prefix_end = min(reference_contact, actual_contact if actual_contact is not None else float('inf'))
            windows = {}
            for name, start, end in (('hold', 0, .05), ('released_early', .05, .2), ('early', 0, .2),
                                     ('common_precontact', 0, prefix_end)):
                selected = trace['reference_available'] & (trace['time_s'] >= start) & (trace['time_s'] < end)
                windows[name] = dict(samples=int(selected.sum()), start_s=start, end_exclusive_s=end,
                    **{metric: dict(rms=float(np.sqrt(np.mean(trace[metric][selected]**2))),
                                   maximum=float(trace[metric][selected].max()),
                                   last=float(trace[metric][selected][-1]))
                       for metric in ('q_rmse_rad', 'qd_rmse_rad_s', 'action_rmse', 'previous_action_history_rmse')})
            assert windows['hold']['samples'] == 3 and windows['released_early']['samples'] == 7
            missing = ~trace['reference_available']
            rows.append(dict(raw_id=raw_id, hit=episode['hit'], outcome=episode['outcome'],
                reference_contact_s=reference_contact, actual_contact_s=actual_contact,
                intervention_steps=int(trace['applied'].sum()), reference_samples=len(references[i]),
                recorded_steps=len(trace['time_s']), reference_exhausted_steps=int(missing.sum()),
                first_reference_exhausted_s=float(trace['time_s'][missing][0]) if missing.any() else None,
                reference_available_through_actual_contact=(actual_contact < len(references[i])/48) if actual_contact else None,
                windows=windows))
        runs.append(dict(name=condition, evaluation=evaluation, rows=rows))
        print('BC_FEEDBACK', condition, 'hits', sum(r['hit'] for r in rows), 'outcomes', evaluation['outcomes'], flush=True)
    assert all(torch.equal(value, weights[key]) for key, value in policy.state_dict().items())
    assert protocol['source_sha256'] == source_hashes() and protocol['bc_source_sha256'] == bc_source_hashes()
    assert all(sha256(path) == expected for path, expected in inputs.items())
    np.savez_compressed(directory / 'feedback_traces.npz', **arrays)
    report = dict(status='PASS', seed=seed, episodes=20, bc_updates=0, ppo_updates=0, conditions=runs,
        policy_unchanged=True, baseline_reproduced=True, input_and_environment_isolation_checks=inference_checks,
        protocol_sha256=sha256(directory / 'feedback_protocol.json'), protocol=protocol,
        traces_sha256=sha256(directory / 'feedback_traces.npz'),
        trace_keys='{condition}_{raw_id}_{field}; full original/reference/input observations, measured q/qd, actions and pre-inference controller memory',
        reference_missing_encoding='NaN arrays after reference ends; metrics exclude these steps and report counts',
        wall_time_s=time.monotonic()-started)
    (directory / 'feedback.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def verify(directory, device):
    import numpy as np
    import torch
    from airbot_tt.config import ROOT
    from airbot_tt.pace_bc import sha256
    from airbot_tt.pace_env import PaceEnv
    from contact_bc import same_episodes
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from train_pace import original_ppo

    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert protocol['source_sha256'] == source_hashes()
    initialization = json.loads((directory / 'initialization.json').read_text())
    assert sha256(directory / 'demonstrations.pt') == initialization['demonstration_sha256']
    data = torch.load(directory / 'demonstrations.pt', map_location='cpu', weights_only=False)
    collection = json.loads((directory / 'collection.json').read_text())
    baseline = Path(protocol['baseline_directory'])
    manifest = baseline.parent / 'manifest.json'
    assert sha256(manifest) == initialization['serve_bank_sha256']
    env = PaceEnv(16, device, group='A', serve_bank=manifest)
    torch.set_num_threads(1)
    ActorCritic, _ = original_ppo()
    config = protocol['cfg']['ppo']
    arrays, results = {}, []
    for seed in protocol['recipe']['seeds']:
        artifact = directory / initialization['actors'][str(seed)]['path']
        assert sha256(artifact) == initialization['actors'][str(seed)]['sha256']
        torch.manual_seed(seed)
        policy = ActorCritic(115, 200, 6, actor_hidden_dims=config['hidden'], critic_hidden_dims=config['hidden'],
                             activation='elu', init_noise_std=config['init_std']).to(device)
        original = torch.load(baseline / f'A_{seed}' / 'initial.pt', map_location='cpu', weights_only=False)
        policy.load_state_dict(original['policy'])
        policy.actor.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=False)['actor'])
        reference_obs = [data['observations'][data['trajectory_ids'] == i].to(device) for i in range(5)]
        reference_actions = [data['actions'][data['trajectory_ids'] == i].to(device) for i in range(5)]
        with torch.no_grad():
            offline_predictions = [policy.act_inference(obs) for obs in reference_obs]
        traces = [[] for _ in range(5)]

        class Trace:
            def __init__(self):
                self.serve_ids = None
                self.tick = 0

            def act_inference(self, obs):
                actions = policy.act_inference(obs)
                if self.serve_ids is None:
                    self.serve_ids = env.serve_id[:5].clone()
                for i in range(5):
                    if env.serve_id[i] != self.serve_ids[i] or self.tick >= len(reference_obs[i]):
                        continue
                    reference = reference_obs[i][self.tick]
                    target = reference_actions[i][self.tick]
                    # Last frame: q-nominal, qd, previous requested action, ball xyz, valid, age.
                    q_error = (obs[i, -23:-17] - reference[-23:-17]).square().mean().sqrt()
                    qd_error = (obs[i, -17:-11] - reference[-17:-11]).square().mean().sqrt()
                    action_error = (actions[i] - target).square().mean().sqrt()
                    offline_error = (offline_predictions[i][self.tick] - target).square().mean().sqrt()
                    traces[i].append(torch.cat((torch.tensor([self.tick / 48], device=device),
                        torch.stack((q_error, qd_error, action_error, offline_error)),
                        actions[i], target, obs[i, -23:-17], reference[-23:-17], env.tcp()[0][i])).cpu().numpy())
                self.tick += 1
                return actions

        evaluation = evaluate_bank(env, Trace(), capture=True)
        same_episodes(evaluation, json.loads((directory / f'bc_{seed}.json').read_text())['evaluation'])
        for i, raw_id in enumerate(data['raw_ids']):
            trace = np.asarray(traces[i])
            arrays[f'{seed}_{raw_id}'] = trace
            contact_time = collection['evaluation']['episodes'][i]['motion']['pre_contact']['t']
            prefix = trace[trace[:, 0] < contact_time]
            first_action = prefix[prefix[:, 3] > .05]
            first_q = prefix[prefix[:, 1] > .01]
            results.append(dict(seed=seed, raw_id=raw_id, reference_contact_time_s=contact_time,
                compared_precontact_steps=len(prefix), initial_action_rmse=float(trace[0, 3]),
                maximum_precontact_action_rmse=float(prefix[:, 3].max()),
                maximum_precontact_offline_action_rmse=float(prefix[:, 4].max()),
                maximum_precontact_q_rmse_rad=float(prefix[:, 1].max()),
                first_action_rmse_over_005_s=float(first_action[0, 0]) if len(first_action) else None,
                first_q_rmse_over_001_rad_s=float(first_q[0, 0]) if len(first_q) else None,
                outcome=evaluation['episodes'][i]['outcome']))
    assert protocol['source_sha256'] == source_hashes()
    np.savez_compressed(directory / 'rollout_diagnostic.npz', **arrays)
    report = dict(status='PASS', ppo_updates=0, bc_updates=0, episodes=15,
                  scope='Same five inspected states; additional replay, not independent samples',
                  columns=['time_s', 'q_rmse_rad', 'qd_rmse_rad_s', 'closed_loop_action_rmse',
                           'recorded_observation_action_rmse', 'action[6]', 'script_action[6]',
                           'q_minus_nominal[6]', 'reference_q_minus_nominal[6]', 'tcp_xyz[3]'],
                  thresholds_are_descriptive_not_acceptance=True, rows=results,
                  source_sha256=source_hashes(), script_sha256=sha256(__file__),
                  input_sha256={name: sha256(directory / name) for name in
                      ('protocol.json', 'initialization.json', 'demonstrations.pt', 'collection.json')})
    (directory / 'rollout_diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def capture_comparison(directory, device):
    """Frozen first ID/seed: script, precomputed BC sequence, autonomous BC.

The whole five-ball bank is scored for each condition. Only the first ID is
recorded for the requested video, selected before seeing new replay outcomes.
Precomputed BC actions use recorded expert observations, not live observations.
"""
    import numpy as np
    import torch
    from airbot_tt.config import ROOT
    from airbot_tt.pace_bc import sha256, bc_source_hashes
    from airbot_tt.pace_env import PaceEnv
    from contact_bc import same_episodes
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from pace_motion import snapshot, observe
    from train_pace import original_ppo

    directory = Path(directory)
    seed = 7101
    bc_protocol = json.loads((directory / 'protocol.json').read_text())
    initialization = json.loads((directory / 'initialization.json').read_text())
    assert bc_protocol['source_sha256'] == source_hashes()
    assert bc_protocol['bc_source_sha256'] == bc_source_hashes()
    baseline = Path(bc_protocol['baseline_directory'])
    manifest = baseline.parent / 'manifest.json'
    artifact = directory / initialization['actors'][str(seed)]['path']
    assert sha256(artifact) == initialization['actors'][str(seed)]['sha256']
    assert sha256(directory / 'demonstrations.pt') == initialization['demonstration_sha256']
    data = torch.load(directory / 'demonstrations.pt', map_location='cpu', weights_only=False)
    assert data['raw_ids'] == [9378, 10818, 5255, 12250, 14852]
    inputs = {str(path): sha256(path) for path in (artifact, manifest, directory / 'demonstrations.pt',
              directory / 'collection.json', directory / 'bc_7101.json', baseline / 'A_7101/initial.pt')}
    for name, expected in bc_protocol['script_input_sha256'].items():
        assert sha256(ROOT / name) == expected
        inputs[str(ROOT / name)] = expected
    protocol = dict(selected_raw_id=9378, seed=seed, selection='First fixed bank ID and first training seed; selected before comparison',
                    physics_hz=960, capture_hz=120, conditions=['script', 'frozen_bc', 'closed_loop_bc'],
                    scope='No-learning comparison on the same five inspected train states; video shows first ID only',
                    frozen_sequence='BC actor means on saved demonstration observations, followed by last-action hold after demonstration ends; no live observation feedback',
                    source_sha256=source_hashes(), input_sha256=inputs, script_sha256=sha256(__file__))
    (directory / 'replay_comparison_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(1)
    env = PaceEnv(16, device, group='A', serve_bank=manifest)
    ActorCritic, _ = original_ppo()
    cfg = bc_protocol['cfg']['ppo']
    torch.manual_seed(seed)
    policy = ActorCritic(115, 200, 6, actor_hidden_dims=cfg['hidden'], critic_hidden_dims=cfg['hidden'],
                         activation='elu', init_noise_std=cfg['init_std']).to(device)
    initial = torch.load(baseline / 'A_7101/initial.pt', map_location='cpu', weights_only=False)
    policy.load_state_dict(initial['policy'])
    policy.actor.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=False)['actor'])
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    with torch.no_grad():
        precomputed = [policy.act_inference(data['observations'][data['trajectory_ids'] == i].to(device)).clone()
                       for i in range(5)]
    audit = ROOT / '.agents/runs/foundation/deepmind_audit'
    plans = {row['id']: row for name in ('position_search', 'position_remaining')
             for row in json.loads((audit / f'{name}.json').read_text())['rows'] if row['accepted']}
    scripts = [env.tensor(plans[raw_id]['preflight']['actions']) for raw_id in data['raw_ids']]
    original_step = env.physics_step
    runs, evaluations = [], {}
    active = None
    tick = 0
    finished = False

    def snap():
        p, quat, _ = env.tcp()
        return snapshot(env.robot, env.paddle, (p[0].cpu().numpy(), quat[0].cpu().numpy()),
                        env.state()[0].cpu().numpy(), float(env.elapsed[0]),
                        env.origins[0].cpu().numpy(), env.events.contacts[0].cpu().tolist())

    def capture_step():
        nonlocal tick, finished
        if finished:
            return original_step()
        before = snap()
        result = original_step()
        after = snap()
        tick += 1
        finished = bool(env.events.outcome[0])
        observe(active, before, after, env.bc['radius_m'], tick, finished)
        return result

    env.physics_step = capture_step
    try:
        for kind in protocol['conditions']:
            active = dict(name=kind, trace=[], impact=None, release=None,
                          post_hit_table_contact=None, net_contacts=[])
            tick, finished = 0, False

            class Playback:
                def __init__(self):
                    self.step = 0

                def act_inference(self, obs):
                    if self.step == 0:
                        active['trace'].append(snap())
                    if kind == 'closed_loop_bc':
                        result = policy.act_inference(obs)
                    else:
                        sequence = scripts if kind == 'script' else precomputed
                        which = list(range(5)) + [0] * 11
                        result = torch.stack([sequence[i][min(self.step, len(sequence[i])-1)] for i in which])
                    self.step += 1
                    return result

            evaluation = evaluate_bank(env, Playback(), capture=True)
            if kind == 'script':
                same_episodes(evaluation, json.loads((directory / 'collection.json').read_text())['evaluation'])
            elif kind == 'closed_loop_bc':
                same_episodes(evaluation, json.loads((directory / 'bc_7101.json').read_text())['evaluation'])
            assert finished and active['trace']
            np.testing.assert_allclose(active['trace'][0]['ball'], env.serve_bank.states[0], atol=2e-6, rtol=0)
            row = evaluation['episodes'][0]
            active.update(outcome=row['outcome'], actual_hit=row['hit'], duration_s=active['trace'][-1]['t'],
                          landing_xy_m=None if active['post_hit_table_contact'] is None else active['post_hit_table_contact']['ball'][:2])
            runs.append(active)
            evaluations[kind] = evaluation
            print('BC_REPLAY', kind, 'five_ball_hits', sum(e['hit'] for e in evaluation['episodes']),
                  'selected_outcome', row['outcome'], flush=True)
    finally:
        env.physics_step = original_step
    assert all(torch.equal(value, weights[key]) for key, value in policy.state_dict().items())
    assert protocol['source_sha256'] == source_hashes()
    assert all(sha256(path) == expected for path, expected in inputs.items())
    report = dict(status='PASS', selected_raw_id=9378, seed=seed, physics_hz=960, capture_hz=120,
                  episodes=15, runs=runs, bank_evaluations=evaluations, protocol=protocol,
                  bc_updates=0, ppo_updates=0, policy_unchanged=True)
    (directory / 'replay_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--capture-comparison', action='store_true', help='Record fixed first ID/seed script, precomputed BC and autonomous BC')
    mode.add_argument('--feedback-audit', action='store_true', help='No-learning four-condition previous-action/velocity input intervention')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    outputs = (('feedback.json', 'feedback_protocol.json', 'feedback_traces.npz') if args.feedback_audit else
               ('replay_comparison.json', 'replay_comparison_protocol.json') if args.capture_comparison else
               ('rollout_diagnostic.json', 'rollout_diagnostic.npz'))
    for name in outputs:
        if (args.directory / name).exists():
            parser.error('Refusing to overwrite existing diagnostic evidence')
    # Match run.sh's launcher: native Kit teardown can otherwise spin after all
    # measurements are saved. Completion is checked from both report and exit.
    app = AppLauncher(args, fast_shutdown=True).app
    exit_code = 0
    try:
        operation = feedback_audit if args.feedback_audit else capture_comparison if args.capture_comparison else verify
        result = operation(args.directory, args.device)
        print(json.dumps(dict(status=result['status'], episodes=result['episodes'])), flush=True)
    except BaseException:
        exit_code = 1
        raise
    finally:
        # Use the complete run_airbot.py shutdown contract, including removal
        # of Lab's STOP callback before Kit closes its stage.
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        app.app.post_quit(exit_code)
        app.close()
