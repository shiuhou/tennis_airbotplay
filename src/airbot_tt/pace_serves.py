"""The approved five-row DeepMind training question and independent serve cycles.

This is a finite memorization/learning diagnostic, not a held-out dataset or a
claim that the rows already have successful-return witnesses. Original spin is
retained; simulation deliberately starts with zero spin.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ROOT

FIVE_SERVE_IDS = (9378, 10818, 5255, 12250, 14852)
SOURCE_COMMIT = '54cf18e41a8f134d98ad38710001ed50aabb8ebc'
AUDIT_DIR = ROOT / '.agents/runs/foundation/deepmind_audit'
SCHEMA = 'deepmind-five-train-serves-v1'
FIELDS = ('pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
          'w_vel_x', 'w_vel_y', 'w_vel_z')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_rows(audit_dir):
    """Rebuild selected states from pinned official bytes; audit split/contact is checked."""
    audit_dir = Path(audit_dir)
    source = json.loads((ROOT / 'configs/deepmind_sources.json').read_text())
    if source['commit'] != SOURCE_COMMIT:
        raise ValueError('The five-serve experiment requires the audited source commit')
    hashes = {'configs/deepmind_sources.json': _sha(ROOT / 'configs/deepmind_sources.json')}
    official = {}
    for filename in ('serves.json', 'rallies.json'):
        item = next(x for x in source['files'] if x['path'] == filename)
        path = audit_dir / 'source' / filename
        if _sha(path) != item['sha256']:
            raise ValueError('Official DeepMind bytes differ: ' + filename)
        hashes[filename] = item['sha256']
        for row in json.loads(path.read_text()):
            if row['id'] in FIVE_SERVE_IDS:
                if row['id'] in official:
                    raise ValueError('Duplicate official row identity')
                official[row['id']] = (filename.removesuffix('.json'), row)
    contacts_path = audit_dir / 'contact_cases.json'
    contacts = json.loads(contacts_path.read_text())
    if contacts['source_commit'] != SOURCE_COMMIT:
        raise ValueError('Contact evidence source differs')
    selected = [r for r in contacts['rows'] if r['id'] in FIVE_SERVE_IDS]
    if len(selected) != len(FIVE_SERVE_IDS) or len({r['id'] for r in selected}) != len(FIVE_SERVE_IDS):
        raise ValueError('Missing or duplicate audited train contacts')
    audited = {r['id']: r for r in selected}
    hashes['contact_cases.json'] = _sha(contacts_path)
    rows = []
    for raw_id in FIVE_SERVE_IDS:
        if raw_id not in official:
            raise ValueError('Missing pinned source row')
        kind, raw = official[raw_id]
        original = np.array([raw[field] for field in FIELDS], dtype=float)
        if not np.isfinite(original).all() or original[1] * original[4] >= 0:
            raise ValueError('Invalid source state')
        sign = 1 if original[1] > 0 else -1
        rotation = np.array([[0, sign, 0], [-sign, 0, 0], [0, 0, 1]])
        state = np.r_[rotation @ original[:3] + [0, 0, .76], rotation @ original[3:6]]
        omega = rotation @ original[6:]
        evidence = audited[raw_id]
        if evidence['split'] != 'train' or not evidence.get('low_spin') or np.linalg.norm(omega) > 10:
            raise ValueError('Selected row is not an audited low-spin train contact')
        # Frozen raw-ID split from the existing audit, independently recomputed.
        bucket = int.from_bytes(hashlib.sha256(f'8121:{raw_id}'.encode()).digest()[:8], 'big') / 2**64
        if bucket >= .6:
            raise ValueError('Selected identity is outside the frozen train split')
        if (not np.array_equal(state, np.array(evidence['state']))
                or not np.array_equal(omega, np.array(evidence['omega']))
                or not np.array_equal(original, np.array(evidence['source_state']))):
            raise ValueError('Audited transform differs from pinned original state')
        rows.append(dict(id=raw_id, kind=kind, split='train', source_state=original.tolist(),
                         state=state.tolist(), omega=omega.tolist(), simulated_omega=[0., 0., 0.]))
    return rows, hashes


def prepare_five_serve_bank(destination, audit_dir=AUDIT_DIR):
    """Write a small deterministic manifest; identical existing manifests are verified."""
    destination = Path(destination)
    rows, hashes = _verified_rows(audit_dir)
    manifest = dict(schema=SCHEMA, source_commit=SOURCE_COMMIT, source_license='CC-BY-4.0',
                    attribution='Google LLC, Competitive Robot Table Tennis: Initial Ball States (2025)',
                    audit_directory=str(Path(audit_dir).resolve()), source_sha256=hashes,
                    scope='Five inspected train contacts, no perturbations; not heldout or certified return data',
                    spin_approximation='Original omega retained; simulated angular velocity is zero',
                    rows=rows)
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise FileExistsError('Existing serve bank differs; it will not be overwritten')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + '\n')
    FixedServeBank(destination)
    return destination


class FixedServeBank:
    """Validated immutable states for the predeclared five-serve experiment."""

    def __init__(self, path):
        path = Path(path)
        self.manifest = json.loads(path.read_text())
        manifest = self.manifest
        if manifest.get('schema') != SCHEMA or manifest.get('source_commit') != SOURCE_COMMIT:
            raise ValueError('Unsupported serve-bank contract')
        rows = manifest.get('rows', [])
        if tuple(r.get('id') for r in rows) != FIVE_SERVE_IDS:
            raise ValueError('Expected exactly the five ordered training identities')
        states = np.asarray([r.get('state') for r in rows], dtype=float)
        if states.shape != (5, 6) or not np.isfinite(states).all():
            raise ValueError('Serve states must be finite 5 by 6 SI values')
        if any(r.get('split') != 'train' or r.get('simulated_omega') != [0., 0., 0.] for r in rows):
            raise ValueError('Only zero-spin approximations of train rows are allowed')
        # Verify official source and transform on every load, before opening a simulator.
        expected, hashes = _verified_rows(manifest.get('audit_directory', AUDIT_DIR))
        if rows != expected or manifest.get('source_sha256') != hashes:
            raise ValueError('Serve bank no longer matches audited source states/hashes')
        self.rows = rows
        self.states = states
        self.states.setflags(write=False)
        self.sha256 = _sha(path)


class ShuffledServeCycles:
    """Each environment sees every row once per shuffled cycle, independent of others."""

    def __init__(self, bank_size, num_envs, seed):
        if bank_size <= 0 or num_envs <= 0:
            raise ValueError('Positive bank/environment sizes required')
        self.bank_size = bank_size
        self.num_envs = num_envs
        self.reseed(seed)

    def reseed(self, seed):
        self.rng = [np.random.default_rng(np.random.SeedSequence([seed, i])) for i in range(self.num_envs)]
        self.pending = [[] for _ in range(self.num_envs)]

    def next(self, env_id):
        if not 0 <= env_id < self.num_envs:
            raise IndexError('Unknown environment')
        if not self.pending[env_id]:
            self.pending[env_id] = self.rng[env_id].permutation(self.bank_size).tolist()
        return self.pending[env_id].pop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Local experiment manifest path')
    args = parser.parse_args()
    destination = prepare_five_serve_bank(args.output)
    print('FIVE_SERVE_BANK', destination, FixedServeBank(destination).sha256)
