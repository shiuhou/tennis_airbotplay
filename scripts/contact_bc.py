"""Collect five inspected contact scripts, fit Actors, and gate BC initialization.

Launched inside the existing Isaac application by run.sh --mode pace-bc.
No PPO updates occur here. A failing contact gate is a completed negative result.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from airbot_tt.config import ROOT
from airbot_tt.pace_bc import bc_source_hashes, fit_actor, recipe, sha256


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def same_episodes(actual, expected):
    assert actual['total'] == expected['total'] == 5
    for a, b in zip(actual['episodes'], expected['episodes'], strict=True):
        for field in ('raw_id', 'hit', 'outcome', 'net_touched', 'target_success'):
            assert a[field] == b[field], (field, a, b)
        if b['landing_xy_m'] is not None:
            np.testing.assert_allclose(a['landing_xy_m'], b['landing_xy_m'], atol=1e-5, rtol=0)
        else:
            assert a['landing_xy_m'] is None


class Demonstrations:
    """Copy pre-action observations only while each original episode is alive."""
    def __init__(self, env, scripts):
        self.env, self.scripts = env, scripts
        self.initial_serve_ids = None
        self.tick = 0
        self.observations, self.actions, self.ids = [], [], []
        self.ticks, self.times, self.phases = [], [], []

    def act_inference(self, obs):
        env = self.env
        if self.initial_serve_ids is None:
            self.initial_serve_ids = env.serve_id[:5].clone()
        which = list(range(5)) + [0] * (env.n - 5)
        action = env.tensor(np.array([self.scripts[i][min(self.tick, len(self.scripts[i])-1)] for i in which]))
        for i in range(5):
            if env.serve_id[i] != self.initial_serve_ids[i]:
                continue
            self.observations.append(obs[i].detach().cpu().clone())
            self.actions.append(action[i].detach().cpu().clone())
            self.ids.append(i)
            self.ticks.append(self.tick)
            self.times.append(float(env.elapsed[i]))
            self.phases.append(int(env.events.phase[i]))
        self.tick += 1
        return action

    def tensors(self):
        return dict(observations=torch.stack(self.observations), actions=torch.stack(self.actions),
                    trajectory_ids=torch.tensor(self.ids), ticks=torch.tensor(self.ticks),
                    times=torch.tensor(self.times), phases=torch.tensor(self.phases))


def noisy_evaluation(env, policy, seed, repetition):
    """Independent per-ball streams shared by original and BC policies.

The actor mean remains a function of its own closed-loop observations. Only the
standard normal draws are paired, not the resulting state or episode duration.
"""
    from pace_bank import evaluate_bank
    cfg = recipe()
    streams = [np.random.default_rng(np.random.SeedSequence([cfg['noise_seed'], seed, row['id'], repetition]))
               for row in env.serve_bank.rows]

    class NoisyActor:
        def act_inference(self, obs):
            mean = policy.act_inference(obs)
            noise = np.zeros((env.n, 6), dtype=np.float32)
            for i, rng in enumerate(streams):
                noise[i] = rng.standard_normal(6)
            return mean + cfg['noise_std'] * env.tensor(noise)

    result = evaluate_bank(env, NoisyActor(), capture=False)
    result.update(scope='Closed-loop Gaussian action-noise diagnostic on five inspected train states',
                  training_seed=seed, repetition=repetition, noise_std=cfg['noise_std'],
                  noise_seed=cfg['noise_seed'])
    return result


def run(device, num_envs, destination, serve_bank, baseline_directory, gate_directory):
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_serves import FixedServeBank, FIVE_SERVE_IDS
    from pace_bank import evaluate_bank
    from pace_m7 import source_hashes
    from train_pace import original_ppo

    if num_envs != 16 or any(x is None for x in (serve_bank, baseline_directory, gate_directory)):
        raise ValueError('Contact BC requires 16 environments, bank, baseline checkpoints and gates')
    destination, baseline_directory, gate_directory = map(Path, (destination, baseline_directory, gate_directory))
    bank = FixedServeBank(serve_bank)
    assert tuple(r['id'] for r in bank.rows) == FIVE_SERVE_IDS
    cfg = json.loads((ROOT / 'configs/pace_rl.json').read_text())
    cfg.update(serve_bank_sha256=bank.sha256, serve_bank_manifest=bank.manifest)
    baseline_protocol = json.loads((baseline_directory / 'protocol.json').read_text())
    before, helper_before = source_hashes(), bc_source_hashes()
    if baseline_protocol['cfg'] != cfg or baseline_protocol['source_sha256'] != before:
        raise ValueError('Historical baseline no longer matches environment, control or data')
    if baseline_protocol['seeds'] != recipe()['seeds'] or baseline_protocol['groups'] != ['A']:
        raise ValueError('Wrong paired baseline')
    for name in ('lifecycle', 'parity', 'control', 'bank'):
        gate = json.loads((gate_directory / f'{name}.json').read_text())
        if gate['status'] != 'PASS' or gate.get('process_exit_code') != 0 or gate['source_sha256'] != before:
            raise ValueError(f'Missing current passing {name} gate')
        if name == 'bank' and (gate['serve_bank_sha256'] != bank.sha256 or
                                gate['gate_sha256'] != helper_before['tests/pace_bank.py']):
            raise ValueError('Bank gate manifest or evaluator changed')
        if name == 'bank':
            script_inputs = gate['input_sha256']
            if any(sha256(ROOT / name) != expected for name, expected in script_inputs.items()):
                raise ValueError('Approved contact scripts or their physics evidence changed')
    destination.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='airbot-contact-bc-v1', recipe=recipe(), cfg=cfg,
                    source_sha256=before, bc_source_sha256=helper_before,
                    baseline_directory=str(baseline_directory.resolve()),
                    baseline_protocol_sha256=sha256(baseline_directory / 'protocol.json'),
                    script_input_sha256=script_inputs,
                    scope='Five inspected train states; contact demonstrations, no heldout/generalization',
                    actor_only=True, ppo_updates=0, conditional_ppo_transitions_per_seed=6144,
                    noisy_gate_is_diagnostic_only=True)
    save_json(destination / 'protocol.json', protocol)
    started = time.monotonic()
    torch.set_num_threads(1)
    ActorCritic, _ = original_ppo()
    p = cfg['ppo']

    def new_policy(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return ActorCritic(115, 200, 6, actor_hidden_dims=p['hidden'], critic_hidden_dims=p['hidden'],
                           activation='elu', init_noise_std=p['init_std']).to(device)

    env = PaceEnv(num_envs, device, group='A', serve_bank=serve_bank)

    class Zero:
        def act_inference(self, obs):
            return torch.zeros((len(obs), 6), device=obs.device)

    zero = evaluate_bank(env, Zero(), capture=True)
    same_episodes(zero, json.loads((baseline_directory / 'zero_action.json').read_text()))
    baseline_replay = dict(zero=zero, runs=[])
    for seed in recipe()['seeds']:
        trial = baseline_directory / f'A_{seed}'
        initial = torch.load(trial / 'initial.pt', map_location='cpu', weights_only=False)
        final = torch.load(trial / 'policy.pt', map_location='cpu', weights_only=False)
        old = json.loads((trial / 'result.json').read_text())
        assert sha256(trial / 'policy.pt') == old['checkpoint_sha256']
        assert sha256(trial / 'initial.pt') == final['initial_sha256']
        assert initial['cfg'] == final['cfg'] == cfg
        assert initial['seed'] == final['seed'] == seed
        policy = new_policy(seed)
        assert all(torch.equal(value.detach().cpu(), initial['policy'][key].cpu())
                   for key, value in policy.state_dict().items()), 'Original seeded model not reproduced'
        initial_evaluation = evaluate_bank(env, policy, capture=True)
        same_episodes(initial_evaluation, old['initial_evaluation'])
        policy.load_state_dict(final['policy'])
        final_evaluation = evaluate_bank(env, policy, capture=True)
        same_episodes(final_evaluation, old['evaluation'])
        baseline_replay['runs'].append(dict(seed=seed, initial=initial_evaluation, final=final_evaluation))
    baseline_replay.update(status='PASS', total=35)
    save_json(destination / 'baseline_replay.json', baseline_replay)

    audit = ROOT / '.agents/runs/foundation/deepmind_audit'
    plan_files = [audit / f'{name}.json' for name in ('position_search', 'position_remaining')]
    plans = {r['id']: r for path in plan_files for r in json.loads(path.read_text())['rows'] if r['accepted']}
    scripts = [np.asarray(plans[raw_id]['preflight']['actions']) for raw_id in FIVE_SERVE_IDS]
    assert all(np.isfinite(a).all() and np.abs(a).max() <= 1 + 1e-7 for a in scripts)
    recorder = Demonstrations(env, scripts)
    demonstrations_evaluation = evaluate_bank(env, recorder, capture=True)
    same_episodes(demonstrations_evaluation, json.loads((gate_directory / 'bank.json').read_text())['contact_script_evaluation'])
    assert all(row['hit'] for row in demonstrations_evaluation['episodes'])
    data = recorder.tensors()
    counts = []
    for i in range(5):
        mask = data['trajectory_ids'] == i
        ticks = data['ticks'][mask]
        count = int(mask.sum()); counts.append(count)
        assert torch.equal(ticks, torch.arange(count))
        expected_actions = torch.tensor(np.asarray([scripts[i][min(t, len(scripts[i])-1)] for t in range(count)]), dtype=torch.float32)
        assert torch.equal(data['actions'][mask], expected_actions)
        np.testing.assert_allclose(data['times'][mask].numpy(), ticks.numpy() / 48, atol=1e-5, rtol=0)
        assert abs(count / 48 - demonstrations_evaluation['episodes'][i]['duration_s']) < 2e-5
    assert data['observations'].shape == (sum(counts), 115)
    data.update(raw_ids=list(FIVE_SERVE_IDS), source_sha256=before,
                input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in plan_files},
                serve_bank_sha256=bank.sha256, scope='First complete episodes, requested actions, no reset/padding samples')
    torch.save(data, destination / 'demonstrations.pt')
    save_json(destination / 'collection.json', dict(status='PASS', samples=sum(counts), samples_by_id=dict(zip(map(str, FIVE_SERVE_IDS), counts)),
              evaluation=demonstrations_evaluation, source_sha256=before, input_sha256=data['input_sha256']))

    runs, actors = [], {}
    observations, actions, ids = [data[name].to(device) for name in ('observations', 'actions', 'trajectory_ids')]
    for seed in recipe()['seeds']:
        policy = new_policy(seed)
        original = torch.load(baseline_directory / f'A_{seed}' / 'initial.pt', map_location='cpu', weights_only=False)
        assert all(torch.equal(value.detach().cpu(), original['policy'][key].cpu()) for key, value in policy.state_dict().items())
        fitted = fit_actor(policy, observations, actions, ids)
        path = destination / f'actor_{seed}.pt'
        torch.save(dict(actor={k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}, seed=seed), path)
        evaluation = evaluate_bank(env, policy, capture=True)
        restored = new_policy(seed)
        restored.actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['actor'])
        with torch.no_grad():
            assert torch.equal(policy.act_inference(observations), restored.act_inference(observations))
        replay = evaluate_bank(env, restored, capture=True)
        same_episodes(evaluation, replay)
        contacts = sum(row['hit'] for row in evaluation['episodes'])
        row = dict(seed=seed, fitting=fitted, evaluation=evaluation, replay_evaluation=replay,
                   contacts=contacts, contact_gate_pass=contacts >= recipe()['minimum_contacts_per_seed'])
        runs.append(row)
        actors[str(seed)] = dict(path=path.name, sha256=sha256(path), contacts=contacts,
                                original_initial_sha256=sha256(baseline_directory / f'A_{seed}' / 'initial.pt'))
        save_json(destination / f'bc_{seed}.json', row)
        print('BC', seed, 'contacts', contacts, 'legal', evaluation['legal_return_rate'], 'loss', fitted['final_loss'], flush=True)

    contact_pass = all(row['contact_gate_pass'] for row in runs)
    noisy = dict(status='NOT_RUN', reason='Requires all three deterministic contact gates', runs=[])
    if contact_pass:
        noisy = dict(status='PASS', repetitions_per_ball=recipe()['noise_repetitions'],
                     scope='Paired action-noise repetitions on the same five train states, not independent incoming balls', runs=[])
        for seed in recipe()['seeds']:
            original, cloned = new_policy(seed), new_policy(seed)
            cloned.actor.load_state_dict(torch.load(destination / actors[str(seed)]['path'], map_location='cpu', weights_only=False)['actor'])
            for label, policy in (('original', original), ('bc', cloned)):
                snapshot = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                trials = [noisy_evaluation(env, policy, seed, rep) for rep in range(recipe()['noise_repetitions'])]
                assert all(torch.equal(value, snapshot[key]) for key, value in policy.state_dict().items())
                noisy['runs'].append(dict(seed=seed, kind=label, trials=trials))
        noisy['total'] = sum(trial['total'] for row in noisy['runs'] for trial in row['trials'])
    save_json(destination / 'noise.json', noisy)
    assert before == source_hashes() and helper_before == bc_source_hashes(), 'Source changed during BC experiment'
    assert all(sha256(ROOT / name) == expected for name, expected in script_inputs.items()), 'Contact scripts changed during BC'
    initialization = dict(schema='airbot-contact-bc-v1', status='PASS', contact_gate_pass=contact_pass,
                          recipe=recipe(), cfg=cfg, source_sha256=before, bc_source_sha256=helper_before,
                          serve_bank_sha256=bank.sha256, baseline_directory=str(baseline_directory.resolve()),
                          demonstration_sha256=sha256(destination / 'demonstrations.pt'), actors=actors)
    save_json(destination / 'initialization.json', initialization)
    result = dict(status='PASS', scientific_conclusion='INCONCLUSIVE', contact_gate='PASS' if contact_pass else 'FAIL',
                  ppo_status='READY' if contact_pass else 'NOT_RUN', ppo_updates=0, samples=sum(counts),
                  runs=runs, noise_status=noisy['status'], seconds=time.monotonic()-started,
                  source_sha256=before, bc_source_sha256=helper_before, serve_bank_sha256=bank.sha256)
    save_json(destination / 'results.json', result)
    return result
