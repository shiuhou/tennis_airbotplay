"""Actor-only contact initialization, balanced supervision and artifact guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from airbot_tt.pace_bc import (
    bc_source_hashes, fit_actor, load_actor_initialization, recipe,
    trajectory_balanced_loss, validate_initialization,
)


class SmallPolicy(nn.Module):
    """Use the same actor/critic/std ownership as the pinned ActorCritic."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(115, 8), nn.ELU(), nn.Linear(8, 6))
        self.critic = nn.Sequential(nn.Linear(200, 8), nn.ELU(), nn.Linear(8, 1))
        self.std = nn.Parameter(torch.full((6,), .3))


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class ContactBCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_longer_trajectory_does_not_outvote_shorter_trajectory(self):
        prediction = torch.tensor([[1.] * 6] * 3 + [[3.] * 6], requires_grad=True)
        target = torch.zeros_like(prediction)
        ids = torch.tensor([9378, 9378, 9378, 10818])
        loss = trajectory_balanced_loss(prediction, target, ids)
        self.assertAlmostEqual(float(loss), 5.)
        # Duplicating the longer trajectory must not change the relative weight.
        expanded = torch.tensor([0, 1, 2, 0, 1, 2, 3])
        duplicated = trajectory_balanced_loss(prediction[expanded], target[expanded], ids[expanded])
        self.assertTrue(torch.equal(loss, duplicated))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad[-1].sum()), float(prediction.grad[:3].sum()))

    def test_real_bc_fit_updates_actor_without_training_critic_or_noise(self):
        torch.manual_seed(7101)
        policy = SmallPolicy()
        observations = torch.randn(10, 115)
        targets = .2 * torch.tanh(observations[:, :6])
        ids = torch.tensor([9378, 10818, 5255, 12250, 14852] * 2)
        before = snapshot(policy)
        inputs_before = observations.clone(), targets.clone(), ids.clone()
        initial_loss = float(trajectory_balanced_loss(policy.actor(observations), targets, ids))
        result = fit_actor(policy, observations, targets, ids)
        final_loss = float(trajectory_balanced_loss(policy.actor(observations), targets, ids))
        self.assertLess(final_loss, initial_loss * .1)
        self.assertEqual(result['optimizer_steps'], 2000)
        self.assertGreater(result['actor_update_norm'], 0.)
        self.assertTrue(result['critic_unchanged'])
        self.assertTrue(result['std_unchanged'])
        self.assertTrue(any(not torch.equal(value, before[key])
                            for key, value in policy.state_dict().items() if key.startswith('actor.')))
        for key, value in policy.state_dict().items():
            if not key.startswith('actor.'):
                self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(all(parameter.grad is None for parameter in policy.critic.parameters()))
        self.assertIsNone(policy.std.grad)
        for actual, original in zip((observations, targets, ids), inputs_before):
            self.assertTrue(torch.equal(actual, original))

    def test_invalid_demonstrations_cannot_partially_train_actor(self):
        policy = SmallPolicy()
        observations = torch.zeros(5, 115)
        ids = torch.arange(5)
        before = snapshot(policy)
        for bad_value in (float('nan'), 1.1):
            actions = torch.zeros(5, 6)
            actions[0, 0] = bad_value
            with self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                fit_actor(policy, observations, actions, ids)
            self.assertTrue(all(torch.equal(value, before[key])
                                for key, value in policy.state_dict().items()))

    def test_demonstrations_copy_pre_action_inputs_and_exclude_reset_and_padding(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/contact_bc.py'
        spec = importlib.util.spec_from_file_location('contact_bc_unit_subject', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = SimpleNamespace(n=7, serve_id=torch.ones(7, dtype=torch.long), elapsed=torch.zeros(7),
                              events=SimpleNamespace(phase=torch.zeros(7, dtype=torch.long)),
                              tensor=lambda array: torch.tensor(array, dtype=torch.float32))
        scripts = [np.full((2, 6), .1 * i, dtype=np.float32) for i in range(5)]
        recorder = module.Demonstrations(env, scripts)
        obs = torch.arange(7 * 115, dtype=torch.float32).reshape(7, 115)
        original_obs = obs.clone()
        action = recorder.act_inference(obs)
        self.assertEqual(action.shape, (7, 6))
        # Mutate the caller-owned buffers as a simulator step would do.
        obs.add_(1000)
        action.fill_(999)
        env.serve_id[0] += 1
        env.elapsed += 1 / 48
        env.events.phase += 1
        next_action = recorder.act_inference(obs)
        obs.zero_()
        next_action.fill_(999)
        env.serve_id += 1
        recorder.act_inference(obs)
        data = recorder.tensors()
        self.assertEqual(data['observations'].shape, (9, 115))
        self.assertTrue(torch.equal(data['observations'][:5], original_obs[:5]))
        self.assertTrue(torch.equal(data['observations'][5:], original_obs[1:5] + 1000))
        expected_actions = torch.tensor(np.asarray([scripts[i][0] for i in range(5)] +
                                                   [scripts[i][1] for i in range(1, 5)]))
        self.assertTrue(torch.equal(data['actions'], expected_actions))
        self.assertEqual(data['trajectory_ids'].tolist(), [0, 1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(data['ticks'].tolist(), [0] * 5 + [1] * 4)
        self.assertEqual(data['phases'].tolist(), [0] * 5 + [1] * 4)


class FeedbackInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('pace_bc_rollout.py')
        spec = importlib.util.spec_from_file_location('bc_feedback_unit_subject', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.intervene = staticmethod(module.intervene_observation)

    def test_whole_history_changes_only_requested_fields_in_an_independent_copy(self):
        obs = torch.arange(7 * 115, dtype=torch.float32).reshape(7, 115)
        references = [torch.arange(3 * 115, dtype=torch.float32).reshape(3, 115) + 10000 * (i + 1)
                      for i in range(5)]
        original = obs.clone()
        saved_references = [reference.clone() for reference in references]
        active = torch.ones(5, dtype=torch.bool)
        for condition in ('baseline', 'previous_action', 'velocity', 'both'):
            with self.subTest(condition=condition):
                result, applied = self.intervene(obs, references, 1, active, condition)
                actual = result.reshape(7, 5, 23)
                initial = obs.reshape(7, 5, 23)
                for i in range(5):
                    reference = references[i][1].reshape(5, 23)
                    self.assertTrue(torch.equal(actual[i, :, :6], initial[i, :, :6]))
                    self.assertTrue(torch.equal(actual[i, :, 18:], initial[i, :, 18:]))
                    expected_velocity = reference[:, 6:12] if condition in ('velocity', 'both') else initial[i, :, 6:12]
                    expected_previous = reference[:, 12:18] if condition in ('previous_action', 'both') else initial[i, :, 12:18]
                    self.assertTrue(torch.equal(actual[i, :, 6:12], expected_velocity))
                    self.assertTrue(torch.equal(actual[i, :, 12:18], expected_previous))
                self.assertEqual(applied.tolist(), [condition != 'baseline'] * 5 + [False, False])
                self.assertTrue(torch.equal(result[5:], obs[5:]))
                # Even baseline is a separate input buffer. Actor-side edits
                # cannot mutate the caller's observation or stored examples.
                result.fill_(-999)
                self.assertTrue(torch.equal(obs, original))
                for reference, saved in zip(references, saved_references):
                    self.assertTrue(torch.equal(reference, saved))

    def test_saved_previous_action_lag_and_reset_padding_are_preserved(self):
        actions = torch.arange(5 * 6, dtype=torch.float32).reshape(5, 6) + 1
        reference = torch.zeros(5, 5, 23)
        # At tick 4, the frames are times 0..4: previous actions are
        # reset-zero, a0, a1, a2, a3. The current target a4 is distinct.
        reference[4, 1:, 12:18] = actions[:4]
        obs = torch.full((2, 115), -1.)
        at_reset, _ = self.intervene(obs, [reference.flatten(1)], 0, [True], 'previous_action')
        at_four, _ = self.intervene(obs, [reference.flatten(1)], 4, [True], 'previous_action')
        self.assertTrue(torch.equal(at_reset[0].reshape(5, 23)[:, 12:18], torch.zeros(5, 6)))
        previous = at_four[0].reshape(5, 23)[:, 12:18]
        self.assertTrue(torch.equal(previous, torch.cat((torch.zeros(1, 6), actions[:4]))))
        self.assertFalse(torch.equal(previous[-1], actions[4]))
        self.assertTrue(torch.equal(at_four[1], obs[1]))

    def test_reset_exhausted_and_padding_rows_remain_autonomous(self):
        obs = torch.arange(7 * 115, dtype=torch.float32).reshape(7, 115)
        references = [torch.full((length, 115), 9000. + i) for i, length in enumerate((3, 4, 2, 0, 4))]
        # Row 1 has reset, row 2 has just exhausted its demonstration,
        # row 3 has no reference; rows 5 and 6 are unscored padding.
        result, applied = self.intervene(obs, references, 2, [True, False, True, True, True], 'both')
        self.assertEqual(applied.tolist(), [True, False, False, False, True, False, False])
        for i in (1, 2, 3, 5, 6):
            self.assertTrue(torch.equal(result[i], obs[i]), i)
        for i in (0, 4):
            actual = result[i].reshape(5, 23)
            self.assertTrue(torch.equal(actual[:, 6:18], references[i][2].reshape(5, 23)[:, 6:18]))
        # After every reference has ended, even previously active rows use
        # their current actual observations instead of a stale final frame.
        tail, tail_applied = self.intervene(obs, references, 4, [True] * 5, 'both')
        self.assertTrue(torch.equal(tail, obs))
        self.assertFalse(tail_applied.any())


class InitializationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original = self.root / 'original'
        self.artifacts = self.root / 'bc'
        self.artifacts.mkdir()
        self.cfg = {'serve_bank_sha256': 'fixed-five-bank', 'ppo': {'init_std': .3}}
        self.sources = {'environment': 'unchanged-source'}
        self.policies = {}
        rows = {}
        for seed in (7101, 7102, 7103):
            torch.manual_seed(seed)
            policy = SmallPolicy()
            self.policies[seed] = policy
            initial_dir = self.original / f'A_{seed}'
            initial_dir.mkdir(parents=True)
            initial_path = initial_dir / 'initial.pt'
            torch.save({'seed': seed, 'cfg': self.cfg, 'policy': snapshot(policy)}, initial_path)
            actor_path = self.artifacts / f'actor_{seed}.pt'
            actor_state = {key: value + .1 for key, value in snapshot(policy.actor).items()}
            torch.save({'seed': seed, 'actor': actor_state}, actor_path)
            rows[str(seed)] = dict(path=actor_path.name, sha256=self.digest(actor_path),
                                  original_initial_sha256=self.digest(initial_path), contacts=3)
        demonstrations = self.artifacts / 'demonstrations.pt'
        torch.save({'observations': torch.zeros(5, 115), 'actions': torch.zeros(5, 6),
                    'trajectory_ids': torch.tensor([9378, 10818, 5255, 12250, 14852])}, demonstrations)
        self.manifest = dict(schema='airbot-contact-bc-v1', status='PASS', contact_gate_pass=True,
                             recipe=recipe(), cfg=self.cfg, source_sha256=self.sources,
                             bc_source_sha256=bc_source_hashes(), serve_bank_sha256='fixed-five-bank',
                             demonstration_sha256=self.digest(demonstrations),
                             baseline_directory=str(self.original), actors=rows)
        self.save_manifest()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_manifest(self):
        (self.artifacts / 'initialization.json').write_text(json.dumps(self.manifest))

    def assert_rejected_without_mutation(self, seed=7101, policy=None):
        policy = self.policies[7101] if policy is None else policy
        before = snapshot(policy)
        with self.assertRaises(ValueError):
            load_actor_initialization(policy, self.artifacts, seed, self.cfg, self.sources)
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_loads_only_actor_and_preserves_original_noise_critic_and_rng(self):
        policy = self.policies[7101]
        before = snapshot(policy)
        rng = torch.get_rng_state().clone()
        provenance = load_actor_initialization(policy, self.artifacts, 7101, self.cfg, self.sources)
        self.assertEqual(provenance['seed'], 7101)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in policy.state_dict().items():
            expected = before[key] + .1 if key.startswith('actor.') else before[key]
            self.assertTrue(torch.equal(value, expected), key)

    def test_failed_other_seed_blocks_entire_paired_experiment(self):
        self.manifest['actors']['7103']['contacts'] = 2
        self.save_manifest()
        self.assert_rejected_without_mutation()
        with self.assertRaises(ValueError):
            validate_initialization(self.artifacts, self.cfg, self.sources)

    def test_requires_exact_original_initialization_and_known_seed(self):
        self.assert_rejected_without_mutation(seed=7104)
        with torch.no_grad():
            self.policies[7101].std[0] += .01
        self.assert_rejected_without_mutation()

    def test_changed_artifact_is_rejected_before_loading(self):
        path = self.artifacts / 'actor_7101.pt'
        path.write_bytes(path.read_bytes() + b'changed')
        self.assert_rejected_without_mutation()

    def test_matching_hash_does_not_allow_nonfinite_actor_parameters(self):
        path = self.artifacts / 'actor_7101.pt'
        artifact = torch.load(path, weights_only=False)
        next(iter(artifact['actor'].values())).flatten()[0] = float('nan')
        torch.save(artifact, path)
        self.manifest['actors']['7101']['sha256'] = self.digest(path)
        self.save_manifest()
        self.assert_rejected_without_mutation()

    def test_changed_demonstrations_or_original_checkpoint_are_rejected(self):
        paths = [self.artifacts / 'demonstrations.pt', self.original / 'A_7102' / 'initial.pt']
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                try:
                    self.assert_rejected_without_mutation()
                finally:
                    path.write_bytes(original)

    def make_refinement(self):
        repo = Path(__file__).resolve().parents[1]
        for folder in ('src', 'configs', 'scripts'):
            (self.root / folder).symlink_to(repo / folder, target_is_directory=True)
        (self.root / 'tests').mkdir()
        for name in ('pace_bank.py', 'pace_motion.py'):
            (self.root / 'tests' / name).symlink_to(repo / 'tests' / name)
        source = self.root / 'tests/pace_bc_refine.py'
        source.write_text('# fixture study source\n')
        root_patch = patch('airbot_tt.pace_bc.ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

        dataset = self.artifacts / 'dataset.pt'
        (self.artifacts / 'demonstrations.pt').rename(dataset)
        self.manifest.update(schema='airbot-contact-refinement-v1',
            recipe=dict(recipe(), candidate=1, trajectory_weighting='equal_ball_then_source_then_trajectory'),
            study_source_sha256={'tests/pace_bc_refine.py': self.digest(source)})
        protocol = {k: self.manifest[k] for k in ('cfg', 'recipe', 'source_sha256', 'bc_source_sha256', 'study_source_sha256')}
        protocol.update(input_sha256={str(dataset): self.digest(dataset)}, dataset_sha256=self.digest(dataset))
        path = self.artifacts / 'protocol.json'
        path.write_text(json.dumps(protocol))
        self.manifest['protocol_sha256'] = self.digest(path)

        from airbot_tt.pace_serves import FIVE_SERVE_IDS
        episodes = [dict(raw_id=raw_id, hit=i < 3, outcome='out' if i < 3 else 'miss')
                    for i, raw_id in enumerate(FIVE_SERVE_IDS)]
        evaluation = dict(total=5, episodes=episodes)
        runs = []
        for seed in recipe()['seeds']:
            row = self.manifest['actors'][str(seed)]
            path = self.artifacts / row['path']
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['protocol_sha256'] = self.manifest['protocol_sha256']
            torch.save(checkpoint, path)
            row['sha256'] = self.digest(path)
            runs.append(dict(seed=seed, contacts=3, actor_sha256=row['sha256'],
                             fitting={'optimizer_steps': recipe()['updates']},
                             evaluation=evaluation, replay_evaluation=evaluation))
        self.save_refinement_results(dict(status='PASS', process_exit_code=0, contact_gate='PASS', runs=runs))
        return source

    def save_refinement_results(self, result):
        path = self.artifacts / 'results.json'
        path.write_text(json.dumps(result))
        self.manifest['evaluation_sha256'] = self.digest(path)
        self.save_manifest()

    def test_refinement_loads_verified_candidate_actor_only(self):
        self.make_refinement()
        policy = self.policies[7101]
        before = snapshot(policy)
        info = load_actor_initialization(policy, self.artifacts, 7101, self.cfg, self.sources)
        self.assertEqual(info['schema'], 'airbot-contact-refinement-v1')
        self.assertEqual(info['demonstration_sha256'], self.digest(self.artifacts / 'dataset.pt'))
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, before[key] + .1 if key.startswith('actor.') else before[key]))

    def test_refinement_recomputes_all_seeds_and_reload_before_loading(self):
        self.make_refinement()
        results = (self.artifacts / 'results.json').read_text()
        for failure in ('insufficient_hits', 'changed_reload', 'duplicate_id'):
            with self.subTest(failure=failure):
                result = json.loads(results)
                run = result['runs'][-1]
                if failure == 'insufficient_hits':
                    for name in ('evaluation', 'replay_evaluation'):
                        run[name]['episodes'][2]['hit'] = False
                elif failure == 'changed_reload':
                    run['replay_evaluation']['episodes'][0]['outcome'] = 'wrong_bounce'
                else:
                    run['evaluation']['episodes'][1]['raw_id'] = run['evaluation']['episodes'][0]['raw_id']
                self.save_refinement_results(result)
                self.assert_rejected_without_mutation()

    def test_refinement_rejects_changed_inputs_and_actor_before_loading(self):
        source = self.make_refinement()
        for path in (source, *(self.artifacts / name for name in ('dataset.pt','protocol.json','results.json','actor_7103.pt'))):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b'changed')
                    self.assert_rejected_without_mutation()
                finally:
                    path.write_bytes(before)


if __name__ == '__main__':
    unittest.main()
