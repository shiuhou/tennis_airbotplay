"""Fixed five-ball PhysX experiment. Drive a learner prefix and splice a scripted continuation."""
import argparse
import json
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-bc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['collect','fit'], required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.stage == 'collect' and args.output.exists():
        parser.error('Existing study directory; choose an unused output path')
    app = AppLauncher(args, fast_shutdown=True).app

import numpy as np
import torch

from airbot_tt.config import ROOT
from airbot_tt.pace_bc import bc_source_hashes, recipe, sha256

FIVE = (9378, 10818, 5255, 12250, 14852)
SWITCH_TICKS = (6, 9, 12)
Q_THRESHOLD = 0.01
MIN_IDS = 3
COMMAND = ('q', 'v', 'request', 'previous_request', 'start_v', 'c2', 'c3', 't')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def provenance(original_bc):
    from contact_bc import same_episodes
    from pace_m7 import source_hashes
    original = read(original_bc / 'protocol.json')
    feedback = read(original_bc / 'feedback.json')
    assert feedback['status'] == 'PASS' and feedback['process_exit_code'] == 0 and feedback['baseline_reproduced']
    assert feedback['protocol_sha256'] == sha256(original_bc / 'feedback_protocol.json')
    assert feedback['traces_sha256'] == sha256(original_bc / 'feedback_traces.npz')
    prior = feedback['protocol']
    assert prior == read(original_bc / 'feedback_protocol.json')
    assert prior['source_sha256'] == original['source_sha256'] == source_hashes()
    assert prior['bc_source_sha256'] == original['bc_source_sha256'] == bc_source_hashes()
    assert prior['raw_ids'] == list(FIVE) and prior['seed'] == 7101
    assert original['recipe'] == recipe() and original['baseline_protocol_sha256'] == sha256(Path(original['baseline_directory']) / 'protocol.json')
    for name, expected in prior['input_sha256'].items():
        assert sha256(ROOT / name) == expected, name
    assert sha256(ROOT / 'tests/pace_bc_rollout.py') == prior['script_sha256']
    baseline = feedback['conditions'][0]
    assert baseline['name'] == 'baseline'
    same_episodes(baseline['evaluation'], read(original_bc / 'bc_7101.json')['evaluation'])
    assert all(not episode['hit'] for episode in baseline['evaluation']['episodes'])
    initial = read(original_bc / 'initialization.json')
    old_run = read(original_bc.parent / 'bc_run.json')
    assert old_run['status'] == 'PASS' and old_run['process_exit_code'] == 0 and old_run['contact_gate'] == 'FAIL'
    assert initial['contact_gate_pass'] is False and old_run['samples'] == 237
    inputs = dict(prior['input_sha256'])
    for seed in recipe()['seeds']:
        item = initial['actors'][str(seed)]
        for path, expected in ((original_bc / item['path'], item['sha256']),
                               (Path(original['baseline_directory']) / f'A_{seed}/initial.pt', item['original_initial_sha256'])):
            assert sha256(path) == expected
            inputs[str(path)] = expected
        path = original_bc / f'bc_{seed}.json'
        data = read(path)
        assert data['contacts'] == item['contacts'] == 0
        same_episodes(data['evaluation'], data['replay_evaluation'])
        inputs[str(path)] = sha256(path)
    for path in [original_bc / name for name in ('feedback_protocol.json', 'feedback.json', 'feedback_traces.npz')] + [original_bc.parent / 'bc_run.json']:
        inputs[str(path)] = sha256(path)
    return original, inputs, baseline['evaluation']


def establish(original_bc, directory):
    original, inputs, _ = provenance(original_bc)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='airbot-five-recovery-bc-v1', fixed_ids=list(FIVE), seed=7101,
        switch_ticks=list(SWITCH_TICKS), policy_hz=48, physics_hz=960, num_envs=16, scored_cases=15,
        q_rmse_threshold_rad=Q_THRESHOLD, minimum_distinct_recovered_ids=MIN_IDS,
        coverage='Count distinct baseline-missed IDs with a valid continuation contact and switch measured-minus-demo q RMSE >= threshold.',
        enrollment='After coverage passes use every baseline-missed valid continuation contact, including smaller switch deviations.',
        samples='Actual pre-action obs115 and same-time requested action6, inclusive of the step that first causes valid contact. No prefix, post-contact, reset, or padding rows.',
        suffix='Use original time-indexed script including its original final-request hold; normal state and command memory throughout.',
        switch_action_rmse='Counterfactual learner mean on actual switch obs versus original same-tick script request, before action selection.',
        fitting='Original 237 samples plus eligible suffixes. Equal ball mass; original and recovery split per ball when available; trajectories equal within source; rows equal within trajectory.',
        fixed_fit_recipe=recipe(), gate='All three last checkpoints independently reach >=3/5 valid contacts, deterministic current observations after reset.',
        conditional='300 paired noise episodes, nonveto, then original A PPO with 6144 transitions per seed and fresh optimizer. No further variants.',
        historical_cfg=original['cfg'], source_sha256=original['source_sha256'], bc_source_sha256=original['bc_source_sha256'],
        input_sha256=inputs, collection_source_sha256=sha256(__file__),
        clip_selection=[], scope='Same five inspected train balls, one collecting Actor. Descriptive coverage only.')
    save(directory / 'protocol.json', protocol)
    return protocol


def arrays(rows):
    return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}


class SwitchPolicy:
    def __init__(self, env, policy, scripts, switch):
        self.env, self.policy, self.scripts, self.switch = env, policy, scripts, switch
        self.first = None
        self.tick = 0
        self.records = [[] for _ in FIVE]

    def act_inference(self, obs):
        env = self.env
        if self.first is None:
            self.first = env.serve_id[:len(FIVE)].clone()
        assert torch.equal(obs, env.actor_history.flatten(1))
        active = env.serve_id[:len(FIVE)] == self.first
        assert (env.policy_index[:len(FIVE)][active] == self.tick).all()
        learner = self.policy.act_inference(obs)
        actions = learner.clone()
        measured_q = env.robot.data.joint_pos[:, env.joints] - env.nominal
        measured_qd = env.robot.data.joint_vel[:, env.joints]
        for i, script in enumerate(self.scripts):
            if not active[i]:
                continue
            target = script[min(self.tick, len(script)-1)]
            if self.tick >= self.switch:
                actions[i] = target
            record = dict(tick=self.tick, elapsed_s=env.elapsed[i], actual_observation=obs[i],
                          measured_q_minus_nominal=measured_q[i], measured_qd=measured_qd[i],
                          learner_action=learner[i], script_action=target, action=actions[i],
                          prehit=~env.events.hit[i], serve_id=env.serve_id[i], previous_action=env.previous_action[i],
                          **{'command_'+field: getattr(env.command, field)[i] for field in COMMAND})
            self.records[i].append({name: value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value
                                    for name, value in record.items()})
        self.tick += 1
        return actions


def inspect_case(trace, reference, baseline, episode, switch, raw_id, prior):
    """Recover first-contact-causing interval from actual pre-step hit masks."""
    np.testing.assert_array_equal(trace['tick'], np.arange(len(trace['tick'])))
    assert np.unique(trace['serve_id']).size == 1 and episode['raw_id'] == raw_id
    np.testing.assert_allclose(trace['elapsed_s'], trace['tick'] / 48, atol=2e-5, rtol=0)
    if not trace['prehit'][:switch+1].all():
        raise ValueError('Switch must precede first contact')
    prefix = slice(0, switch+1)
    for name in ('actual_observation', 'measured_q_minus_nominal', 'measured_qd', *('command_'+field for field in COMMAND)):
        np.testing.assert_allclose(trace[name][prefix], prior[f'baseline_{raw_id}_{name}'][prefix], atol=1e-6, rtol=0)
    np.testing.assert_allclose(trace['learner_action'][prefix], prior[f'baseline_{raw_id}_action'][prefix], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(trace['action'][:switch], trace['learner_action'][:switch])
    np.testing.assert_array_equal(trace['action'][switch:], trace['script_action'][switch:])
    np.testing.assert_array_equal(trace['previous_action'][1:], trace['action'][:-1])
    rmse = lambda x: float(np.sqrt(np.mean(x**2)))
    q = rmse(trace['measured_q_minus_nominal'][switch] - reference[:6])
    qd = rmse(trace['measured_qd'][switch] - reference[6:12])
    da = rmse(trace['learner_action'][switch] - trace['script_action'][switch])
    hit = episode['hit']
    eligible = hit and not baseline['hit']
    impact = episode['motion']['impact']
    assert hit == bool(impact)
    samples = np.flatnonzero((trace['tick'] >= switch) & trace['prehit']) if eligible else np.empty(0, dtype=int)
    if eligible:
        end = samples[-1]
        assert trace['elapsed_s'][end] < impact['time_s'] <= trace['elapsed_s'][end] + 1/48 + 2e-5
        np.testing.assert_array_equal(samples, np.arange(switch, end+1))
    row = dict(raw_id=raw_id, switch_tick=switch, switch_q_rmse_rad=q, switch_qd_rmse_rad_s=qd,
               switch_action_rmse=da, valid_contact=hit, legal=episode['outcome']=='legal_return',
               fault=episode['outcome'], baseline_hit=baseline['hit'], baseline_fault=baseline['outcome'],
               qualifies=eligible and q >= Q_THRESHOLD, data_eligible=eligible, suffix_samples=len(samples),
               impact_time_s=impact['time_s'] if hit else None)
    return row, samples


def collect(original_bc, directory, device):
    from airbot_tt.pace_env import PaceEnv
    from pace_bank import evaluate_bank
    from train_pace import original_ppo
    started = time.monotonic()
    protocol = establish(original_bc, directory)
    original, _, baseline = provenance(original_bc)
    data = torch.load(original_bc / 'demonstrations.pt', map_location='cpu', weights_only=False)
    assert data['raw_ids'] == list(FIVE) and data['observations'].shape == (237, 115)
    scripts = {}
    for name in ('position_search', 'position_remaining'):
        path = ROOT / f'.agents/runs/foundation/deepmind_audit/{name}.json'
        for row in read(path)['rows']:
            if row['accepted']:
                scripts[row['id']] = torch.tensor(row['preflight']['actions'], dtype=torch.float32, device=device)
    targets = [scripts[raw_id] for raw_id in FIVE]
    references = []
    for i, target in enumerate(targets):
        assert torch.isfinite(target).all() and target.abs().max() <= 1+1e-7
        selected = data['trajectory_ids'] == i
        ticks = data['ticks'][selected]
        assert torch.equal(target[ticks.clamp(max=len(target)-1)].cpu(), data['actions'][selected])
        references.append(data['observations'][selected].numpy().reshape(-1,5,23)[:,-1,:])
    manifest = original_bc.parent / 'manifest.json'
    torch.set_num_threads(1)
    env = PaceEnv(16, device, group='A', serve_bank=manifest)
    assert env.policy_dt == 1/48 and env.decimation == 20 and env.online is None
    ActorCritic, _ = original_ppo()
    cfg = original['cfg']['ppo']
    torch.manual_seed(7101)
    policy = ActorCritic(115, 200, 6, actor_hidden_dims=cfg['hidden'], critic_hidden_dims=cfg['hidden'],
                         activation='elu', init_noise_std=cfg['init_std']).to(device)
    initial = torch.load(Path(original['baseline_directory']) / 'A_7101/initial.pt', map_location='cpu', weights_only=False)
    policy.load_state_dict(initial['policy'])
    policy.actor.load_state_dict(torch.load(original_bc / 'actor_7101.pt', map_location='cpu', weights_only=False)['actor'])
    policy.eval()
    cases, evaluations, recoveries, output = [], [], [], {}
    with np.load(original_bc / 'feedback_traces.npz') as prior:
        for switch in SWITCH_TICKS:
            controller = SwitchPolicy(env, policy, targets, switch)
            evaluation = evaluate_bank(env, controller, capture=True)
            evaluations.append(evaluation)
            for i, raw_id in enumerate(FIVE):
                trace = arrays(controller.records[i])
                case, inds = inspect_case(trace, references[i][switch], baseline['episodes'][i],
                                          evaluation['episodes'][i], switch, raw_id, prior)
                cases.append(case)
                for key, value in trace.items():
                    output[f't{switch}_{raw_id}_{key}'] = value
                if case['data_eligible']:
                    recoveries.append(dict(raw_id=raw_id, switch_tick=switch, ticks=torch.from_numpy(inds),
                                           observations=torch.from_numpy(trace['actual_observation'][inds].copy()),
                                           actions=torch.from_numpy(trace['action'][inds].copy())))
            print('RECOVERY', switch, [(c['raw_id'], c['valid_contact'], c['qualifies']) for c in cases[-5:]], flush=True)
    assert all(sha256(ROOT / name) == expected for name, expected in protocol['input_sha256'].items())
    assert protocol['collection_source_sha256'] == sha256(__file__)
    assert protocol['bc_source_sha256'] == bc_source_hashes()
    ids = [raw_id for raw_id in FIVE if any(c['raw_id']==raw_id and c['qualifies'] for c in cases)]
    passed = len(ids) >= MIN_IDS
    np.savez_compressed(directory / 'recovery_traces.npz', **output)
    if passed:
        torch.save(recoveries, directory / 'recoveries.pt')
    result = dict(status='PASS', coverage_gate='PASS' if passed else 'FAIL', qualifying_ids=ids, cases=cases,
                  evaluations=evaluations, scored_episodes=len(cases), baseline_evaluation=baseline,
                  eligible_trajectories=len(recoveries), eligible_samples=sum(c['suffix_samples'] for c in cases),
                  protocol_sha256=sha256(directory / 'protocol.json'), traces_sha256=sha256(directory / 'recovery_traces.npz'),
                  recoveries_sha256=sha256(directory / 'recoveries.pt') if passed else None,
                  elapsed_s=time.monotonic()-started, bc_updates=0, ppo_transitions=0)
    save(directory / 'recovery.json', result)
    return result


if __name__ == '__main__':
    code = 0
    print('RECOVERY_RUNTIME', torch.__version__, torch.__file__, flush=True)
    try:
        if args.stage == 'collect':
            result = collect(args.original_bc, args.output, args.device)
        else:
            from airbot_tt.pace_env import PaceEnv
            from contact_recovery_bc import fit_recovery
            original, _, _ = provenance(args.original_bc)
            stage_a = read(args.output / 'recovery.json')
            assert stage_a['status']=='PASS' and stage_a['process_exit_code']==0 and stage_a['coverage_gate']=='PASS'
            assert stage_a['protocol_sha256'] == sha256(args.output / 'protocol.json')
            assert stage_a['recoveries_sha256'] == sha256(args.output / 'recoveries.pt')
            assert stage_a['traces_sha256'] == sha256(args.output / 'recovery_traces.npz')
            recoveries = torch.load(args.output / 'recoveries.pt', map_location='cpu', weights_only=False)
            torch.set_num_threads(1)
            env = PaceEnv(16, args.device, group='A', serve_bank=args.original_bc.parent / 'manifest.json')
            result = fit_recovery(env, args.output, args.original_bc, Path(original['baseline_directory']), recoveries)
        print('RECOVERY_STAGE_COMPLETE', args.stage, result['status'], flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
        raise
    finally:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        app.app.post_quit(code)
        app.close()
