"""Corrective demonstration weighting and row boundaries without Isaac startup."""
from pathlib import Path
import sys
import unittest

import torch

from airbot_tt.pace_serves import FIVE_SERVE_IDS

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from contact_recovery_bc import fit_weighted_actor, training_data
    from pace_bc_refine import assemble, fit_scheduled_actor
finally:
    sys.path.pop(0)


def example(count, start, value):
    return dict(ticks=torch.arange(start, start + count), observations=torch.full((count, 115), value),
                actions=torch.full((count, 6), value / 10))


class RecoveryFittingTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(raw_ids=list(FIVE_SERVE_IDS), trajectory_ids=torch.arange(5),
                             ticks=torch.zeros(5, dtype=torch.long), observations=torch.zeros(5, 115),
                             actions=torch.zeros(5, 6))
        self.recoveries = [
            dict(raw_id=9378, switch_tick=6, **example(3, 6, 1.)),
            dict(raw_id=9378, switch_tick=9, **example(5, 9, 2.)),
            dict(raw_id=10818, switch_tick=12, **example(2, 12, 3.))]

    def test_weights_equalize_balls_sources_and_unequal_length_suffixes(self):
        obs, actions, weights, trajectories = training_data(self.original, self.recoveries)
        self.assertEqual(len(obs), 15)
        self.assertEqual(len(actions), len(weights))
        actual = {}
        pos = 0
        for row in trajectories:
            n = row['samples']
            chunk = weights[pos:pos+n]
            actual[row['raw_id'], row['switch_tick']] = float(chunk.sum())
            self.assertAlmostEqual(float(chunk.min()), float(chunk.max()))
            pos += n
        expected = {(9378, None): .1, (9378, 6): .05, (9378, 9): .05,
                    (10818, None): .1, (10818, 12): .1,
                    (5255, None): .2, (12250, None): .2, (14852, None): .2}
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in actual.items():
            self.assertAlmostEqual(value, expected[key])
        # The last two suffixes have different row counts and errors. Check the actual row alignment too.
        measured = torch.dot(weights, actions.square().mean(1))
        self.assertAlmostEqual(float(measured), .05*.1**2 + .05*.2**2 + .1*.3**2)

    def test_invalid_or_repeated_takeovers_do_not_enter_dataset(self):
        bad = dict(self.recoveries[0], ticks=torch.tensor([5, 6, 7]))
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            training_data(self.original, [bad])
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            training_data(self.original, self.recoveries + self.recoveries[:1])
        with self.assertRaisesRegex(ValueError, 'invalid action'):
            training_data(self.original, [dict(self.recoveries[0], actions=torch.full((3, 6), 1.1))])

    def test_new_collector_suffix_preserves_ball_and_source_mass(self):
        # Different lengths and another collector at an existing switch are valid.
        addition = dict(raw_id=9378, switch_tick=6, collector_sha256='new-model', **example(7, 6, 4.))
        batch = assemble(self.original, self.recoveries, [addition])
        position = 0
        totals = {}
        for row in batch['trajectories']:
            count = row['samples']
            mass = float(batch['weights'][position:position+count].sum())
            totals[row['raw_id'], row['source']] = totals.get((row['raw_id'], row['source']), 0) + mass
            self.assertAlmostEqual(mass, row['weight'])
            if row['raw_id'] == 9378 and row['source'] == 'recovery':
                self.assertAlmostEqual(mass, .1/3)
            position += count
        self.assertAlmostEqual(totals[9378, 'original'], .1)
        self.assertAlmostEqual(totals[9378, 'recovery'], .1)
        self.assertEqual(position, 22)
        self.assertTrue(torch.equal(batch['observations'][-7:], addition['observations']))
        self.assertTrue(torch.equal(batch['actions'][-7:], addition['actions']))
        with self.assertRaisesRegex(ValueError, 'repeated'):
            assemble(self.original, self.recoveries, [addition, addition])

    def test_fixed_recipe_fits_only_actor_and_reports_last_weighted_loss(self):
        for fit in (fit_weighted_actor, fit_scheduled_actor):
            with self.subTest(recipe=fit.__name__):
                torch.manual_seed(9)
                policy = torch.nn.Module()
                policy.actor = torch.nn.Linear(115, 6)
                policy.critic = torch.nn.Linear(200, 1)
                policy.std = torch.nn.Parameter(torch.full((6,), .3))
                previous = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                obs, target, w, _ = training_data(self.original, self.recoveries)
                initial_loss = float(torch.dot((policy.actor(obs)-target).square().mean(1), w).detach())
                result = fit(policy, obs, target, w)
                self.assertEqual(result['optimizer_steps'], 2000)
                self.assertLess(result['final_loss'], initial_loss * .1)
                with torch.no_grad():
                    self.assertAlmostEqual(result['final_loss'], float(torch.dot((policy.actor(obs)-target).square().mean(1), w)))
                for key, value in policy.state_dict().items():
                    if not key.startswith('actor.'):
                        self.assertTrue(torch.equal(value, previous[key]))


if __name__ == '__main__':
    unittest.main()
