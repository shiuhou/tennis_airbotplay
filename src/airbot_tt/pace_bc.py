"""Actor-only contact imitation; an isolated experiment, not a PACE replacement.

Reuse the pinned PACE ActorCritic. The upstream Distillation class requires a
StudentTeacher policy, so this small offline fit uses PyTorch Adam/MSE directly.
"""
import hashlib
import json
from pathlib import Path
import time

import torch

from airbot_tt.config import ROOT


def recipe():
    return json.loads((ROOT / 'configs/pace_bc.json').read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bc_source_hashes():
    return {name: sha256(ROOT / name) for name in (
        'configs/pace_bc.json', 'src/airbot_tt/pace_bc.py',
        'scripts/contact_bc.py', 'scripts/train_pace.py',
        'scripts/run_airbot.py', 'tests/pace_bank.py', 'tests/pace_motion.py')}


def trajectory_balanced_loss(prediction, target, trajectory_ids):
    """Each trajectory has equal influence regardless of its duration."""
    error = (prediction - target).square().mean(dim=1)
    return torch.stack([error[trajectory_ids == i].mean()
                        for i in torch.unique(trajectory_ids)]).mean()


def fit_actor(policy, observations, actions, trajectory_ids):
    cfg = recipe()
    if observations.ndim != 2 or actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError('Expected observation matrix and six requested action labels')
    if len(observations) != len(actions) or trajectory_ids.shape != (len(actions),) or not len(actions):
        raise ValueError('Unpaired or empty demonstration data')
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError('Non-finite demonstrations')
    if actions.abs().max() > 1 + 1e-7:
        raise ValueError('Contact demonstrations must contain bounded requested actions')
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=cfg['learning_rate'])
    started = time.monotonic()
    history = []
    for update in range(1, cfg['updates'] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = trajectory_balanced_loss(policy.actor(observations), actions, trajectory_ids)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite BC loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.actor.parameters()):
            raise RuntimeError('Non-finite BC gradients')
        optimizer.step()
        if update == 1 or update % cfg['log_interval'] == 0:
            history.append(dict(update=update, loss=float(loss.detach())))
        if time.monotonic() - started > cfg['maximum_fit_seconds']:
            raise RuntimeError('Fixed BC fit exceeded wall-time guard; do not extend budget')
    current = policy.state_dict()
    for key, value in current.items():
        if not torch.isfinite(value).all():
            raise RuntimeError('Non-finite BC model')
        if not key.startswith('actor.') and not torch.equal(value, initial[key]):
            raise RuntimeError('BC modified a non-Actor parameter')
    with torch.no_grad():
        error = policy.actor(observations) - actions
        per_trajectory = {str(int(i)): float(error[trajectory_ids == i].square().mean().sqrt())
                          for i in torch.unique(trajectory_ids)}
        joint_rmse = {str(int(i)): error[trajectory_ids == i].square().mean(0).sqrt().cpu().tolist()
                      for i in torch.unique(trajectory_ids)}
        delta = torch.cat([(value - initial[key]).flatten() for key, value in current.items()
                           if key.startswith('actor.')]).norm()
    return dict(history=history, per_trajectory_rmse=per_trajectory,
                per_trajectory_joint_rmse=joint_rmse, actor_update_norm=float(delta),
                final_loss=float(trajectory_balanced_loss(policy.actor(observations), actions, trajectory_ids).detach()),
                optimizer_steps=cfg['updates'], critic_unchanged=True, std_unchanged=True,
                seconds=time.monotonic() - started)


def _refinement_outcomes(evaluation):
    from airbot_tt.pace_serves import FIVE_SERVE_IDS

    episodes = evaluation['episodes']
    if evaluation['total'] != len(FIVE_SERVE_IDS) or tuple(e['raw_id'] for e in episodes) != FIVE_SERVE_IDS:
        raise ValueError('Refinement evaluation must contain each canonical ball in order')
    if any(type(e['hit']) is not bool or not isinstance(e['outcome'], str) or not e['outcome'] for e in episodes):
        raise ValueError('Expected a measured contact flag and terminal outcome per ball')
    return [(e['hit'], e['outcome']) for e in episodes]


def _validate_refinement(directory, manifest, cfg, source_hashes):
    base = recipe()
    fit = manifest['recipe']
    candidate, updates = fit.get('candidate'), fit.get('updates')
    if (not base.keys() <= fit.keys() or type(candidate) is not int or candidate not in (1, 2, 3)
            or type(updates) is not int or not 1 <= updates <= base['updates']):
        raise ValueError('Invalid BC refinement candidate or update budget')
    # Fitting choices may vary across candidates. The evaluation, seed and checkpoint contracts are fixed.
    invariant = ('seeds', 'minimum_contacts_per_seed', 'checkpoint_selection', 'noise_repetitions',
                 'noise_seed', 'noise_std', 'noise_is_diagnostic_only')
    if any(fit[k] != base[k] for k in invariant):
        raise ValueError('Refinement changed the fixed evaluation or checkpoint contract')
    if candidate == 1 and (fit['trajectory_weighting'] != 'equal_ball_then_source_then_trajectory'
                           or fit['optimizer'] != base['optimizer'] or fit['learning_rate'] != base['learning_rate']):
        raise ValueError('First BC refinement requires its declared data-only fit')
    if manifest['cfg'] != cfg or manifest['source_sha256'] != source_hashes or manifest['bc_source_sha256'] != bc_source_hashes():
        raise ValueError('Refinement environment, control or BC source changed')
    sources = manifest['study_source_sha256']
    if 'tests/pace_bc_refine.py' not in sources or any(sha256(ROOT / k) != v for k, v in sources.items()):
        raise ValueError('Refinement study source changed')
    for name, key in [('dataset.pt', 'demonstration_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('results.json', 'evaluation_sha256')]:
        if sha256(directory / name) != manifest[key]:
            raise ValueError(f'Refinement artifact changed: {name}')
    protocol = json.loads((directory / 'protocol.json').read_text())
    if any(protocol[k] != manifest[k] for k in ('cfg', 'recipe', 'source_sha256', 'bc_source_sha256', 'study_source_sha256')):
        raise ValueError('Refinement protocol and initialization disagree')
    inputs = protocol['input_sha256']
    if not inputs or any(sha256(ROOT / k) != v for k, v in inputs.items()):
        raise ValueError('Refinement protocol input changed')
    dataset = (directory / 'dataset.pt').resolve()
    dataset_hash = protocol.get('dataset_sha256')
    if dataset_hash is None:
        dataset_hash = next((v for k, v in inputs.items() if (ROOT / k).resolve() == dataset), None)
    if dataset_hash != manifest['demonstration_sha256']:
        raise ValueError('Refinement protocol does not identify the fitted dataset')
    result = json.loads((directory / 'results.json').read_text())
    if result['status'] != 'PASS' or result['process_exit_code'] != 0 or result['contact_gate'] != 'PASS':
        raise ValueError('Refinement requires a completed passing three-seed evaluation')
    seeds = base['seeds']
    if [r['seed'] for r in result['runs']] != seeds or set(manifest['actors']) != {str(s) for s in seeds}:
        raise ValueError('Refinement must contain the three paired seeds')
    for run in result['runs']:
        seed = run['seed']
        actor = manifest['actors'][str(seed)]
        if actor['path'] != f'actor_{seed}.pt' or run['actor_sha256'] != actor['sha256']:
            raise ValueError('Refinement evaluation must identify the same fitted Actor')
        if run['fitting']['optimizer_steps'] != updates:
            raise ValueError('Refinement evaluation requires the declared last update')
        outcomes = _refinement_outcomes(run['evaluation'])
        if _refinement_outcomes(run['replay_evaluation']) != outcomes:
            raise ValueError('Refinement reload changed a canonical outcome')
        contacts = sum(hit for hit, _ in outcomes)
        if contacts != run['contacts'] or contacts != actor['contacts'] or contacts < base['minimum_contacts_per_seed']:
            raise ValueError('Refinement contact gate failed on the recorded five balls')


def validate_initialization(directory, cfg, source_hashes):
    """Validate the whole three-seed experiment before any PPO trial starts."""
    directory = Path(directory)
    manifest = json.loads((directory / 'initialization.json').read_text())
    refinement = manifest.get('schema') == 'airbot-contact-refinement-v1'
    if not refinement and manifest.get('schema') != 'airbot-contact-bc-v1':
        raise ValueError('Wrong BC initialization schema')
    if manifest.get('status') != 'PASS' or not manifest.get('contact_gate_pass'):
        raise ValueError('All three BC Actors must pass the closed-loop contact gate')
    if refinement:
        _validate_refinement(directory, manifest, cfg, source_hashes)
    else:
        if manifest['recipe'] != recipe() or manifest['cfg'] != cfg:
            raise ValueError('BC recipe or environment/PPO configuration mismatch')
        if manifest['source_sha256'] != source_hashes or manifest['bc_source_sha256'] != bc_source_hashes():
            raise ValueError('BC initialization source changed after validation')
        if manifest['serve_bank_sha256'] != cfg.get('serve_bank_sha256'):
            raise ValueError('BC serve bank mismatch')
        if sha256(directory / 'demonstrations.pt') != manifest['demonstration_sha256']:
            raise ValueError('BC demonstrations changed')
    seeds = recipe()['seeds']
    if set(manifest['actors']) != {str(seed) for seed in seeds}:
        raise ValueError('BC initialization must contain all three paired seeds')
    for seed in seeds:
        row = manifest['actors'][str(seed)]
        if row['contacts'] < recipe()['minimum_contacts_per_seed']:
            raise ValueError('A BC Actor failed the contact gate')
        if sha256(directory / row['path']) != row['sha256']:
            raise ValueError('BC Actor artifact changed')
        if refinement:
            checkpoint = torch.load(directory / row['path'], map_location='cpu', weights_only=True)
            if checkpoint['seed'] != seed or checkpoint.get('protocol_sha256') != manifest['protocol_sha256']:
                raise ValueError('BC Actor was fitted under a different refinement protocol or seed')
        original = Path(manifest['baseline_directory']) / f'A_{seed}' / 'initial.pt'
        if sha256(original) != row['original_initial_sha256']:
            raise ValueError('Original paired initialization changed')
    return manifest


def load_actor_initialization(policy, directory, seed, cfg, source_hashes):
    manifest = validate_initialization(directory, cfg, source_hashes)
    if str(seed) not in manifest['actors']:
        raise ValueError('Unknown BC seed')
    row = manifest['actors'][str(seed)]
    original = torch.load(Path(manifest['baseline_directory']) / f'A_{seed}' / 'initial.pt',
                          map_location='cpu', weights_only=False)
    current = policy.state_dict()
    if original['seed'] != seed or original['cfg'] != cfg or set(current) != set(original['policy']):
        raise ValueError('Original initialization contract mismatch')
    if any(not torch.equal(value.detach().cpu(), original['policy'][key].cpu())
           for key, value in current.items()):
        raise ValueError('Policy is not the exact original seeded initialization')
    artifact = torch.load(Path(directory) / row['path'], map_location='cpu', weights_only=False)
    state = artifact['actor']
    expected = policy.actor.state_dict()
    if artifact['seed'] != seed or set(state) != set(expected):
        raise ValueError('Wrong BC Actor artifact')
    if any(value.shape != expected[key].shape or value.dtype != expected[key].dtype
           or not torch.isfinite(value).all() for key, value in state.items()):
        raise ValueError('Invalid BC Actor parameters')
    # All validation precedes the only policy mutation.
    policy.actor.load_state_dict(state)
    assert all(torch.equal(value.detach().cpu(), original['policy'][key].cpu())
               for key, value in policy.state_dict().items() if not key.startswith('actor.'))
    return dict(schema=manifest['schema'], directory=str(Path(directory).resolve()), seed=seed,
                actor_sha256=row['sha256'], original_initial_sha256=row['original_initial_sha256'],
                demonstration_sha256=manifest['demonstration_sha256'], recipe=manifest['recipe'],
                initialization_sha256=sha256(Path(directory) / 'initialization.json'))
