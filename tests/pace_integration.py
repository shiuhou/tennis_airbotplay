"""Replay fixed historical BC/PPO evidence under the explicitly integrated source.

Run bank (60 scored episodes) and probe (2) in separate Isaac processes. The
--preflight option performs provenance and strict model loading on CPU only.
Historical training validators are deliberately neither called nor replaced:
their source identity remains historical, while this protocol names both source
versions and authorizes only deterministic inference with immutable weights.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.agents/runs/foundation/m7/deepmind_five'
INTEGRATION = ROOT / '.agents/runs/foundation/m7/e_integration'
IDS = [9378, 10818, 5255, 12250, 14852]
SEEDS = [7101, 7102, 7103]
GROUPS = {
    'legacy_bc': ('bc', [0, 0, 0]),
    'recovery_bc': ('recovery_bc/fit', [2, 4, 4]),
    'candidate2_bc': ('refinement/candidate_2', [3, 4, 3]),
    'candidate2_ppo': ('refinement/ppo', [1, 0, 1]),
}
PATCHES = {
    'src/airbot_tt/pace_bc.py': 'refinement/legacy_pace_bc.patch',
    'tests/pace_recovery.py': 'recovery_bc/collection_source.patch',
    'tests/pace_recovery_probe.py': 'recovery_bc/probe/source_probe.patch',
    'scripts/run_recovery_bc.sh': 'recovery_bc/probe/source_launcher.patch',
    'tests/pace_bc_refine.py': 'refinement/candidate_1/study_source.patch',
}
HASH_MAPS = ('source_sha256', 'bc_source_sha256', 'study_source_sha256',
             'input_sha256', 'script_input_sha256', 'evaluation_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative(path):
    return str((ROOT / path).resolve().relative_to(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


class HistoricalInputs:
    """Exact old bytes or a named reconstruction; never substitute current hashes."""

    def __init__(self, baseline_path, transfer_path):
        self.baseline = read(baseline_path)
        self.transfer = read(transfer_path)
        require(self.baseline['schema'] == 'airbot-e-integration-before-v1', 'Unknown baseline')
        require(self.transfer['schema'] == 'airbot-e-integration-transfer-v1', 'Unknown transfer')
        require(Path(self.baseline['destination']).resolve() == ROOT, 'Baseline workspace differs')
        require(Path(self.transfer['destination']).resolve() == ROOT, 'Transfer workspace differs')
        self.files = self.baseline['files']
        self.protected = self.baseline['protected_evidence']
        self.imported = {item['path']: item for item in self.transfer['code']}
        require(len(self.imported) == len(self.transfer['code']), 'Duplicate imported source')
        self.current_inputs = {}
        self.mapping = {}
        self.receipts = set()
        for name, item in self.protected.items():
            require((ROOT / name).stat().st_size == item['bytes'] and sha(ROOT / name) == item['sha256'],
                    'Protected evidence changed: ' + name)
            self.current_inputs[name] = item['sha256']
        for name, item in self.imported.items():
            require(item['before_sha256'] == self.files.get(name, {}).get('sha256'),
                    'Transfer before-source differs from captured baseline: ' + name)
            require(sha(ROOT / name) == item['after_sha256'], 'Imported source changed: ' + name)
            self.current_inputs[name] = item['after_sha256']
        for path in (baseline_path, transfer_path):
            self.current_inputs[relative(path)] = sha(path)

    def check(self, name, expected):
        name = relative(name)
        key = name + '@' + expected
        if key in self.mapping:
            return
        current = sha(ROOT / name)
        captured = self.files.get(name, self.protected.get(name, {})).get('sha256')
        record = dict(path=name, historical_sha256=expected, captured_before_sha256=captured,
                      current_sha256=current)
        if name in self.protected:
            require(expected == captured == current, 'Historical evidence mismatch: ' + name)
            record['verification'] = 'immutable_captured_evidence'
        elif name in self.files:
            if expected == captured:
                if current != captured:
                    require(name in self.imported and self.imported[name]['after_sha256'] == current,
                            'Unrecorded integration change: ' + name)
                record['verification'] = 'captured_source_with_explicit_transfer'
            else:
                require(name in PATCHES and current == captured,
                        'Historical source has no exact reconstruction: ' + name)
                patch = BASE / PATCHES[name]
                patch_name = relative(patch)
                require(patch_name in self.protected and sha(patch) == self.protected[patch_name]['sha256'],
                        'Uncaptured historical patch: ' + patch_name)
                with tempfile.TemporaryDirectory() as directory:
                    old = Path(directory) / 'original'
                    subprocess.run(['patch', '--batch', '--reverse', '--fuzz=0', '--output', str(old),
                                    str(ROOT / name)], input=patch.read_bytes(), check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    require(sha(old) == expected, 'Historical reconstruction differs: ' + name)
                record.update(verification='exact_inverse_patch', patch=patch_name,
                              patch_sha256=sha(patch))
        else:
            # Historical scripts point to audit inputs outside the captured five-ball tree.
            require(current == expected, 'Historical external input changed: ' + name)
            record['verification'] = 'exact_historical_input'
        self.mapping[key] = record
        self.current_inputs[name] = current

    def receipt(self, path):
        path = (ROOT / path).resolve()
        name = relative(path)
        if name in self.receipts:
            return
        self.receipts.add(name)
        require(name in self.protected or name in self.current_inputs,
                'Receipt is not bound by captured evidence or its parent: ' + name)
        data = read(path)
        if not isinstance(data, dict):
            return
        for key in HASH_MAPS:
            mapping = data.get(key, {})
            if not isinstance(mapping, dict):
                continue
            for source, expected in mapping.items():
                # The bank manifest names three files relative to its audit directory.
                target = ROOT / source
                if data.get('schema') == 'deepmind-five-train-serves-v1' and '/' not in source:
                    audit = Path(data['audit_directory'])
                    target = audit / 'source' / source if source in ('serves.json', 'rallies.json') else audit / source
                self.check(target, expected)
                if key in ('input_sha256', 'script_input_sha256') and target.suffix == '.json':
                    self.receipt(target)
        if 'runner_sha256' in data:
            self.check('scripts/train_pace.py', data['runner_sha256'])
        if 'collection_source_sha256' in data:
            self.check('tests/pace_recovery.py', data['collection_source_sha256'])

    def unchanged(self):
        for name, expected in self.current_inputs.items():
            require(sha(ROOT / name) == expected, 'Input changed during replay: ' + name)


def signature(evaluation):
    return [(r['raw_id'], r['hit'], r['outcome'], r['net_touched'], r['target_success'])
            for r in evaluation['episodes']]


def inspect(evaluation, bank):
    rows = evaluation['episodes']
    require(evaluation['total'] == len(rows) == 5, 'Incomplete canonical five-ball evaluation')
    require([r['raw_id'] for r in rows] == IDS and [r['env'] for r in rows] == list(range(5)),
            'Canonical ball order or spatial index changed')
    require(evaluation['padded_environments_excluded'] == 11 and evaluation['physics_capture'],
            'Wrong padding or missing physical evidence')
    require(evaluation['serve_bank_sha256'] == sha(BASE / 'manifest.json'), 'Bank identity differs')
    for row, source in zip(rows, bank['rows'], strict=True):
        require(row['initial_state'] == source['state'] and type(row['hit']) is bool,
                'Initial state or hit type differs')
        require(bool(row['motion']['impact']) == row['hit'], 'Contact record disagrees with outcome')
        require(row['motion']['terminal']['outcome'] == row['outcome'], 'Terminal outcome differs')
    counts = [sum(r['hit'] for r in rows), sum(r['outcome'] == 'legal_return' for r in rows),
              sum(r['target_success'] for r in rows)]
    require(dict(Counter(r['outcome'] for r in rows)) == evaluation['outcomes'], 'Outcome denominator differs')
    require([c / 5 for c in counts] == [evaluation[k] for k in
            ('hit_rate', 'legal_return_rate', 'target_success_rate')], 'Rate denominator differs')
    return counts


def historical_preflight(baseline_path, transfer_path):
    inputs = HistoricalInputs(baseline_path, transfer_path)
    bank = read(BASE / 'manifest.json')
    require([r['id'] for r in bank['rows']] == IDS, 'Unexpected bank')
    cfg = read(BASE / 'bc/protocol.json')['cfg']
    require(cfg['serve_bank_sha256'] == sha(BASE / 'manifest.json') and cfg['serve_bank_manifest'] == bank,
            'Historical cfg bank does not match captured manifest')
    specs = []
    for group, (directory, contacts) in GROUPS.items():
        path = BASE / directory
        protocol = read(path / 'protocol.json')
        inputs.receipt(path / 'protocol.json')
        require(protocol['cfg'] == cfg, 'Historical configurations differ: ' + group)
        manifest = None
        if group in ('legacy_bc', 'candidate2_bc'):
            manifest = read(path / 'initialization.json')
            inputs.receipt(path / 'initialization.json')
            require(manifest['status'] == 'PASS' and manifest['cfg'] == cfg, 'Invalid BC initialization')
            require(manifest['contact_gate_pass'] == (group == 'candidate2_bc'), 'Historical BC gate changed')
            require(manifest['serve_bank_sha256'] == sha(BASE / 'manifest.json'), 'BC bank changed')
            require(manifest['source_sha256'] == protocol['source_sha256'] and
                    manifest['bc_source_sha256'] == protocol['bc_source_sha256'], 'BC source identity differs')
            data_name = 'demonstrations.pt' if group == 'legacy_bc' else 'dataset.pt'
            require(manifest['demonstration_sha256'] == sha(path / data_name), 'BC data changed')
        result_path = (BASE / 'bc_run.json' if group == 'legacy_bc' else
                       BASE / 'refinement/ppo.json' if group == 'candidate2_ppo' else path / 'results.json')
        result = read(result_path)
        inputs.receipt(result_path)
        require(result['status'] == 'PASS' and result['process_exit_code'] == 0, 'Historical process did not pass')
        if group == 'legacy_bc':
            require(protocol['baseline_protocol_sha256'] == sha(BASE / 'policies/protocol.json'),
                    'Original policy protocol changed')
            initial_protocol = read(BASE / 'policies/protocol.json')
            require(initial_protocol['cfg'] == cfg and initial_protocol['seeds'] == SEEDS and
                    initial_protocol['groups'] == ['A'] and
                    initial_protocol['source_sha256'] == protocol['source_sha256'],
                    'Original initialization environment differs')
            # Only the random initial is reused. Its older PPO runner has no
            # snapshot; load_policy independently reproduces every initial tensor
            # with the pinned official constructor. Do not claim runner replay.
            for name, expected in initial_protocol['evaluation_sha256'].items():
                inputs.check(name, expected)
        elif group == 'recovery_bc':
            require(result['fit_protocol_sha256'] == sha(path / 'protocol.json') and
                    result['contact_gate'] == 'FAIL', 'Recovery receipt chain differs')
        elif group == 'candidate2_bc':
            require(manifest['protocol_sha256'] == result['protocol_sha256'] == sha(path / 'protocol.json') and
                    manifest['evaluation_sha256'] == sha(path / 'results.json') and
                    protocol['dataset_sha256'] == sha(path / 'dataset.pt') and result['contact_gate'] == 'PASS',
                    'Candidate2 receipt chain differs')
        else:
            actor_init = protocol['actor_initialization']
            candidate = BASE / 'refinement/candidate_2'
            require(actor_init['manifest'] == read(candidate / 'initialization.json') and
                    actor_init['manifest_sha256'] == sha(candidate / 'initialization.json') and
                    Path(actor_init['directory']).resolve() == candidate, 'PPO initialization chain differs')
            require((protocol['num_envs'], protocol['rollout_steps'], protocol['iterations'],
                     protocol['transitions_per_trial']) == (16, 24, 16, 6144) and
                    protocol['seeds'] == SEEDS and protocol['groups'] == ['A'], 'Historical PPO budget differs')
        for index, seed in enumerate(SEEDS):
            receipt_path = path / (f'A_{seed}/result.json' if group == 'candidate2_ppo' else f'bc_{seed}.json')
            row = read(receipt_path)
            inputs.receipt(receipt_path)
            require(row == result['runs'][index], 'Historical per-seed result differs from aggregate')
            require(row.get('seed', row.get('training_seed')) == seed, 'Historical seed order changed')
            require(inspect(row['evaluation'], bank) == [contacts[index], 0, 0], 'Historical counts changed')
            require(signature(row['evaluation']) == signature(row['replay_evaluation']), 'Historical reload differs')
            checkpoint = path / (f'A_{seed}/policy.pt' if group == 'candidate2_ppo' else f'actor_{seed}.pt')
            initial = BASE / f'policies/A_{seed}/initial.pt'
            if manifest is not None:
                actor = manifest['actors'][str(seed)]
                require(actor['path'] == checkpoint.name and actor['sha256'] == sha(checkpoint) and
                        actor['original_initial_sha256'] == sha(initial) and actor['contacts'] == contacts[index],
                        'Actor initialization chain differs')
            else:
                field = 'checkpoint_sha256' if group == 'candidate2_ppo' else 'actor_sha256'
                require(row[field] == sha(checkpoint), 'Checkpoint differs from result')
            specs.append(dict(group=group, seed=seed, checkpoint=relative(checkpoint),
                              checkpoint_sha256=sha(checkpoint), original_initial=relative(initial),
                              original_initial_sha256=sha(initial), protocol=relative(path / 'protocol.json'),
                              reference=relative(receipt_path), expected_contacts=contacts[index]))
    probe = BASE / 'recovery_bc/probe'
    inputs.receipt(probe / 'protocol.json')
    old_probe, probe_protocol = read(probe / 'probe.json'), read(probe / 'protocol.json')
    require(old_probe['status'] == 'PASS' and old_probe['process_exit_code'] == 0 and
            old_probe['protocol_sha256'] == sha(probe / 'protocol.json') and
            old_probe['traces_sha256'] == sha(probe / 'traces.npz'), 'Historical probe chain differs')
    require((probe_protocol['raw_id'], probe_protocol['seed'], probe_protocol['switch_tick'],
             probe_protocol['scored_episodes']) == (9378, 7101, 12, 2), 'Historical probe design changed')
    require([(r['episode']['hit'], r['episode']['outcome']) for r in old_probe['cases']] ==
            [(False, 'miss'), (True, 'wrong_bounce')], 'Historical probe outcomes changed')
    inputs.current_inputs[relative(__file__)] = sha(__file__)
    inputs.current_inputs['tests/pace_m7.py'] = sha(ROOT / 'tests/pace_m7.py')
    # Include the newly added hash-map entries without importing Isaac on CPU.
    for name in ('pace_frozen', 'pace_frozen_experiment', 'pace_mean', 'pace_mean_experiment', 'pace_sources'):
        source = f'configs/{name}.json'
        inputs.current_inputs[source] = sha(ROOT / source)
    return inputs, cfg, bank, specs


def load_policy(spec, cfg, device):
    """Strict tensor loading after exact file/provenance validation; no optimizer."""
    import torch
    from train_pace import original_ppo, validate_predictor_checkpoint
    torch.manual_seed(spec['seed'])
    ActorCritic, _ = original_ppo()
    ppo = cfg['ppo']
    policy = ActorCritic(115, 200, 6, actor_hidden_dims=ppo['hidden'], critic_hidden_dims=ppo['hidden'],
                         activation='elu', init_noise_std=ppo['init_std']).to(device)
    original = torch.load(ROOT / spec['original_initial'], map_location='cpu', weights_only=False)
    require(original['cfg'] == cfg and original['seed'] == spec['seed'] and original['group'] == 'A' and
            original['actor_dim'] == 115, 'Original full policy metadata differs')
    validate_predictor_checkpoint(original, 'A')
    require(set(policy.state_dict()) == set(original['policy']) and all(
        torch.equal(value.detach().cpu(), original['policy'][name]) for name, value in policy.state_dict().items()),
        'Seeded official constructor does not reproduce the complete original initial')
    policy.load_state_dict(original['policy'], strict=True)
    checkpoint = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=False)
    require(checkpoint['seed'] == spec['seed'], 'Actor checkpoint seed differs')
    if spec['group'] == 'candidate2_ppo':
        protocol = read(ROOT / spec['protocol'])
        row = read(ROOT / spec['reference'])
        require(checkpoint['cfg'] == cfg and checkpoint['group'] == 'A' and checkpoint['actor_dim'] == 115 and
                checkpoint['source_sha256'] == protocol['source_sha256'] and checkpoint['iteration'] == 16,
                'PPO checkpoint metadata differs')
        validate_predictor_checkpoint(checkpoint, 'A')
        candidate = BASE / 'refinement/candidate_2'
        initialization = read(candidate / 'initialization.json')
        actor = initialization['actors'][str(spec['seed'])]
        provenance = checkpoint['actor_initialization']
        require(provenance == row['actor_initialization'] and
                provenance['initialization_sha256'] == sha(candidate / 'initialization.json') and
                provenance['actor_sha256'] == actor['sha256'] and
                provenance['original_initial_sha256'] == spec['original_initial_sha256'],
                'PPO actor initialization metadata differs')
        paired_path = (ROOT / spec['checkpoint']).parent / 'initial.pt'
        paired = torch.load(paired_path, map_location='cpu', weights_only=False)
        require(checkpoint['initial_sha256'] == sha(paired_path) and paired['cfg'] == cfg and
                paired['actor_initialization'] == provenance, 'PPO paired initial chain differs')
        actor_state = torch.load(candidate / actor['path'], map_location='cpu', weights_only=True)['actor']
        require(set(paired['policy']) == set(original['policy']), 'Paired policy keys differ')
        for name, value in paired['policy'].items():
            expected = actor_state[name.removeprefix('actor.')] if name.startswith('actor.') else original['policy'][name]
            require(torch.equal(value, expected), 'Paired original critic/std or BC actor changed: ' + name)
        policy.load_state_dict(checkpoint['policy'], strict=True)
        expected_policy = checkpoint['policy']
    else:
        if spec['group'] != 'legacy_bc':
            require(checkpoint['protocol_sha256'] == sha(ROOT / spec['protocol']), 'Actor protocol binding differs')
        policy.actor.load_state_dict(checkpoint['actor'], strict=True)
        expected_policy = {name: checkpoint['actor'][name.removeprefix('actor.')] if name.startswith('actor.') else value
                           for name, value in original['policy'].items()}
    policy.eval()
    require(set(policy.state_dict()) == set(expected_policy), 'Full policy state keys differ')
    for name, value in policy.state_dict().items():
        require(torch.equal(value.detach().cpu(), expected_policy[name]) and torch.isfinite(value).all().item(),
                'Strict load differs or has nonfinite values: ' + name)
    policy.requires_grad_(False)
    return policy


def immutable(policy, before):
    import torch
    require(all(torch.equal(value, before[name]) for name, value in policy.state_dict().items()),
            'Policy weights changed during inference')


def bank_replay(cfg, bank, specs, device, output, report):
    import numpy as np
    from airbot_tt.pace_env import PaceEnv
    from pace_bank import evaluate_bank
    env = PaceEnv(16, device, group='A', serve_bank=BASE / 'manifest.json')
    require(env.policy_dt == 1 / 48 and env.decimation == 20 and env.online is None,
            'Current A controller differs')
    for spec in specs:
        policy = load_policy(spec, cfg, device)
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        evaluation = evaluate_bank(env, policy, capture=True)
        row = dict(group=spec['group'], seed=spec['seed'], evaluation=evaluation,
                   checkpoint_sha256=spec['checkpoint_sha256'])
        report['runs'].append(row)
        report['scored_episodes'] += len(evaluation['episodes'])
        save(output / 'result.json', report)
        reference = read(ROOT / spec['reference'])['evaluation']
        row['hit_legal_target'] = inspect(evaluation, bank)
        require(row['hit_legal_target'] == [spec['expected_contacts'], 0, 0], 'Current counts differ from history')
        require(signature(evaluation) == signature(reference), 'Current per-ball outcomes differ from history')
        require(evaluation['event_rules'] == reference['event_rules'], 'Event rule version changed')
        row['landing_max_abs_error_m'] = 0.0
        for current, previous in zip(evaluation['episodes'], reference['episodes'], strict=True):
            require((current['landing_xy_m'] is None) == (previous['landing_xy_m'] is None),
                    'Landing availability differs')
            if current['landing_xy_m'] is not None:
                delta = np.asarray(current['landing_xy_m']) - previous['landing_xy_m']
                row['landing_max_abs_error_m'] = max(row['landing_max_abs_error_m'], float(np.max(np.abs(delta))))
                np.testing.assert_allclose(current['landing_xy_m'], previous['landing_xy_m'], atol=1e-5, rtol=0)
        immutable(policy, before)
        row['weights_unchanged'] = True
        row['historical_outcomes_reproduced'] = True
        print('INTEGRATION_BANK', spec['group'], spec['seed'], row['hit_legal_target'], flush=True)
    require(report['scored_episodes'] == 60, 'Incomplete bank integration denominator')


def probe_replay(cfg, specs, device, output, report):
    import numpy as np
    import torch
    from airbot_tt.pace_env import PaceEnv
    from pace_recovery import COMMAND
    from pace_recovery_probe import episode
    spec = next(s for s in specs if s['group'] == 'recovery_bc' and s['seed'] == 7101)
    policy = load_policy(spec, cfg, device)
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    source = read(ROOT / '.agents/runs/foundation/deepmind_audit/position_search.json')
    rows = [r for r in source['rows'] if r['id'] == 9378 and r['accepted']]
    require(len(rows) == 1, 'Missing unique original contact script')
    script = torch.tensor(rows[0]['preflight']['actions'], dtype=torch.float32, device=device)
    env = PaceEnv(16, device, group='A', serve_bank=BASE / 'manifest.json')
    require(env.policy_dt == 1 / 48 and env.decimation == 20 and env.online is None, 'Probe controller differs')
    traces = {}
    old = read(BASE / 'recovery_bc/probe/probe.json')
    no_switch = int(np.ceil(env.shot['timeout_s'] / env.policy_dt)) + 2
    for index, (name, switch) in enumerate((('autonomous', no_switch), ('takeover', 12))):
        row, trace, checks = episode(env, policy, script, switch)
        traces[name] = trace
        report['runs'].append(dict(condition=name, episode=row, checks=checks))
        report['scored_episodes'] += 1
        np.savez_compressed(output / 'traces.npz', **{f'{n}_{k}': v for n, t in traces.items() for k, v in t.items()})
        save(output / 'result.json', report)
        previous = old['cases'][index]['episode']
        require((row['raw_id'], row['hit'], row['outcome'], row['target_success']) ==
                (previous['raw_id'], previous['hit'], previous['outcome'], previous['target_success']),
                'Probe outcome differs from history')
        print('INTEGRATION_PROBE', name, row['hit'], row['outcome'], flush=True)
    prefix = {}
    for key in ('actual_observation', 'measured_q_minus_nominal', 'measured_qd', 'previous_action',
                'learner_action', *('command_' + field for field in COMMAND)):
        difference = traces['autonomous'][key][:13] - traces['takeover'][key][:13]
        prefix[key] = float(np.max(np.abs(difference)))
        np.testing.assert_allclose(difference, 0, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(traces['autonomous']['action'][:12], traces['takeover']['action'][:12])
    report['prefix_max_abs'] = prefix
    previous = read(ROOT / spec['reference'])['evaluation']['episodes'][0]
    current = report['runs'][0]['episode']
    report['baseline_scalar_abs_error'] = {}
    for key in ('duration_s', 'min_ball_distance_m', 'guidance_valid_s', 'plateau_s', 'guidance_sum', 'reward'):
        report['baseline_scalar_abs_error'][key] = abs(current[key] - previous[key])
        np.testing.assert_allclose(current[key], previous[key], atol=2e-6, rtol=0)
    for key in ('ball', 'tcp', 'quaternion_wxyz', 'normal', 'tcp_velocity', 'omega'):
        np.testing.assert_allclose(current['motion']['terminal'][key], previous['motion']['terminal'][key],
                                   atol=2e-6, rtol=0)
    immutable(policy, before)
    report.update(weights_unchanged=True, historical_outcomes_reproduced=True,
                  traces_sha256=sha(output / 'traces.npz'))
    require(report['scored_episodes'] == 2, 'Incomplete probe denominator')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['bank', 'probe'], required=True)
    parser.add_argument('--baseline', type=Path, default=INTEGRATION / 'baseline.json')
    parser.add_argument('--transfer', type=Path, default=INTEGRATION / 'transfer.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true', help='CPU provenance and strict model load only; zero episodes')
    cpu_only = '--preflight' in sys.argv
    if cpu_only:
        parser.add_argument('--device', default='cpu', choices=['cpu'])
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output directory; preserve completed and failed evidence')
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ('src', 'tests', 'scripts'):
        sys.path.insert(0, str(ROOT / directory))
    started, app, code = time.monotonic(), None, 0
    report = dict(status='RUNNING', stage=args.stage, preflight_only=cpu_only, scored_episodes=0,
                  expected_scored_episodes=0 if cpu_only else 60 if args.stage == 'bank' else 2,
                  runs=[], bc_updates=0, ppo_updates=0, ppo_transitions=0, predictor_updates=0)
    try:
        inputs, cfg, bank, specs = historical_preflight(args.baseline, args.transfer)
        protocol = dict(schema='airbot-e-integration-replay-v1', stage=args.stage, preflight_only=cpu_only,
            baseline_sha256=sha(args.baseline), transfer_sha256=sha(args.transfer), cfg=cfg,
            scope='Current-source compatibility replay of historical training evidence; inference only, no heldout claim.',
            training_permission=False, bc_updates=0, ppo_updates=0, ppo_transitions=0, predictor_updates=0,
            num_envs=16, canonical_ids=IDS, seeds=SEEDS, scored_episodes=report['expected_scored_episodes'],
            execution_order=specs if args.stage == 'bank' else [dict(group='recovery_bc', seed=7101, raw_id=9378,
                conditions=['autonomous', 'takeover'], switch_tick=12, switch_time_s=.25)],
            comparison=dict(bank='Exact ordered per-ball hit/outcome/net/target; full five-ball denominator; original landing tolerance 1e-5m.',
                probe='Exact historical outcomes; original autonomous scalar/terminal tolerance 2e-6 and same-run prefix 1e-6.',
                historical_validators='Unmodified. Their original source checks remain strict and reject changed source.'),
            historical_to_current=list(inputs.mapping.values()), verified_receipts=sorted(inputs.receipts),
            original_initial_verification=dict(protocol_sha256=sha(BASE / 'policies/protocol.json'),
                historical_runner_sha256=read(BASE / 'policies/protocol.json')['runner_sha256'],
                runner_source_reconstructed=False, initial_tensors='Exact seeded official ActorCritic reconstruction'),
            current_input_sha256=inputs.current_inputs)
        save(args.output / 'protocol.json', protocol)
        report['protocol_sha256'] = sha(args.output / 'protocol.json')
        save(args.output / 'result.json', report)
        if not cpu_only:
            app = AppLauncher(args, fast_shutdown=True).app
        import torch
        torch.set_num_threads(1)
        if cpu_only:
            for spec in specs:
                load_policy(spec, cfg, 'cpu')
                report['runs'].append(dict(group=spec['group'], seed=spec['seed'], strict_model_load='PASS'))
            report['strict_model_loads'] = len(specs)
        else:
            from pace_m7 import source_hashes
            report['source_sha256'] = source_hashes()
            require(all(inputs.current_inputs.get(name) == expected for name, expected in report['source_sha256'].items()),
                    'Current runtime source is not frozen in replay protocol')
            run = bank_replay if args.stage == 'bank' else probe_replay
            if args.stage == 'bank':
                run(cfg, bank, specs, args.device, args.output, report)
            else:
                run(cfg, specs, args.device, args.output, report)
        inputs.unchanged()
        report.update(status='PASS', immutable_inputs=True, protected_evidence_files=len(inputs.protected),
                      historical_receipts=len(inputs.receipts), elapsed_s=time.monotonic() - started)
        save(args.output / 'result.json', report)
        print('INTEGRATION_COMPLETE', args.stage, 'PREFLIGHT' if cpu_only else 'REPLAY', 'PASS', flush=True)
    except BaseException as error:
        code = 1
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        save(args.output / 'result.json', report)
        traceback.print_exc()
    finally:
        if app is not None:
            from isaaclab.sim import SimulationContext
            SimulationContext.clear_instance()
            app.app.post_quit(code)
            app.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
