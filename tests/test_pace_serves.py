"""Fixed train question provenance and balanced independent episode schedules."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from airbot_tt.pace_serves import (
    AUDIT_DIR, FIVE_SERVE_IDS, FixedServeBank, ShuffledServeCycles, prepare_five_serve_bank,
)


class ServeBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.audit = cls.root / 'audit'
        (cls.audit / 'source').mkdir(parents=True)
        for name in ('serves.json', 'rallies.json'):
            shutil.copyfile(AUDIT_DIR / 'source' / name, cls.audit / 'source' / name)
        shutil.copyfile(AUDIT_DIR / 'contact_cases.json', cls.audit / 'contact_cases.json')
        cls.manifest = cls.root / 'bank.json'
        prepare_five_serve_bank(cls.manifest, cls.audit)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def changed_manifest(self, change):
        manifest = json.loads(self.manifest.read_text())
        change(manifest)
        path = self.root / 'changed.json'
        path.write_text(json.dumps(manifest))
        return path

    def test_original_identity_spin_and_idempotent_preparation(self):
        bank = FixedServeBank(self.manifest)
        self.assertEqual(tuple(r['id'] for r in bank.rows), FIVE_SERVE_IDS)
        self.assertEqual(bank.states.shape, (5, 6))
        self.assertFalse(bank.states.flags.writeable)
        self.assertEqual(bank.sha256, hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertTrue(all(np.linalg.norm(r['omega']) > 0 for r in bank.rows))
        self.assertTrue(all(r['simulated_omega'] == [0, 0, 0] for r in bank.rows))
        before = self.manifest.read_bytes()
        self.assertEqual(prepare_five_serve_bank(self.manifest, self.audit), self.manifest)
        self.assertEqual(before, self.manifest.read_bytes())

    def test_reject_validation_or_duplicate_identity(self):
        for change in (
            lambda m: m['rows'][0].update(split='validation'),
            lambda m: m['rows'][0].update(id=m['rows'][1]['id']),
        ):
            with self.assertRaises(ValueError):
                FixedServeBank(self.changed_manifest(change))

    def test_reject_nonfinite_or_altered_state(self):
        for value in (float('nan'), float('inf'), 123.0):
            path = self.changed_manifest(lambda m: m['rows'][0]['state'].__setitem__(0, value))
            with self.assertRaises(ValueError):
                FixedServeBank(path)

    def test_reject_rotated_spin_or_nonzero_simulated_spin(self):
        for field in ('omega', 'simulated_omega'):
            path = self.changed_manifest(lambda m: m['rows'][0].update({field: [0., 0., 1.]}))
            with self.assertRaises(ValueError):
                FixedServeBank(path)

    def test_source_hash_is_rechecked_and_existing_bank_not_replaced(self):
        original = self.audit / 'source/rallies.json'
        content = original.read_bytes()
        try:
            original.write_bytes(content + b'\n')
            with self.assertRaisesRegex(ValueError, 'Official DeepMind bytes differ'):
                FixedServeBank(self.manifest)
        finally:
            original.write_bytes(content)
        changed = self.changed_manifest(lambda m: m.update(scope='different'))
        before = changed.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_five_serve_bank(changed, self.audit)
        self.assertEqual(before, changed.read_bytes())


class ServeCycleTests(unittest.TestCase):
    def test_every_cycle_contains_all_five_and_reseed_reproduces(self):
        a = ShuffledServeCycles(5, 16, 7101)
        expected = [[a.next(i) for _ in range(20)] for i in range(16)]
        for row in expected:
            for k in range(0, 20, 5):
                self.assertEqual(sorted(row[k:k + 5]), list(range(5)))
        a.reseed(7101)
        self.assertEqual(expected, [[a.next(i) for _ in range(20)] for i in range(16)])

    def test_partial_resets_do_not_advance_other_envs(self):
        interleaved = ShuffledServeCycles(5, 3, 7102)
        independent = ShuffledServeCycles(5, 3, 7102)
        observed = []
        for k in range(17):
            for _ in range(k % 4):
                interleaved.next(0)
            interleaved.next(2)
            observed.append(interleaved.next(1))
        self.assertEqual(observed, [independent.next(1) for _ in range(17)])
        with self.assertRaises(IndexError):
            interleaved.next(3)

    def test_training_seed_changes_sequence(self):
        a = ShuffledServeCycles(5, 2, 7101)
        b = ShuffledServeCycles(5, 2, 7103)
        self.assertNotEqual([a.next(0) for _ in range(20)], [b.next(0) for _ in range(20)])


if __name__ == '__main__':
    unittest.main()
