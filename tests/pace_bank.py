"""Actual PhysX gate and deterministic evaluation for the five inspected train balls.

No PPO update is performed here. Padded environments and post-reset episodes are
excluded from evaluation. Motion snapshots are measurement only.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from airbot_tt.config import ROOT


class _MotionCapture:
    """Observe the first episode of the five scored environments at physical dt."""

    def __init__(self, env, count):
        self.env = env
        self.count = count
        self.finished = [False] * count
        self.rows = [dict(pre_contact=None, impact=None, first_separation=None,
                          landing=None, net_contacts=[], terminal=None,
                          contact_point_source='Ball-centre projection onto measured TCP plane; geometric proxy, not solver contact point')
                     for _ in range(count)]
        self.original = env.physics_step

    def snapshot(self):
        from isaaclab.utils.math import quat_rotate
        env = self.env
        p, quat, velocity = env.tcp()
        axis = torch.zeros_like(p); axis[:, 2] = 1
        normal = quat_rotate(quat, axis)
        # One batched transfer for all kinematics used by this measurement.
        value = torch.cat((env.elapsed[:, None], env.state(), p, quat, normal,
                           velocity, env.robot.data.body_ang_vel_w[:, env.paddle]), 1)[:self.count].cpu().numpy()
        contacts = env.events.contacts[:self.count].cpu().numpy()
        return [dict(t=float(v[0]), ball=v[1:7].tolist(), tcp=v[7:10].tolist(),
                     quaternion_wxyz=v[10:14].tolist(), normal=v[14:17].tolist(),
                     tcp_velocity=v[17:20].tolist(), omega=v[20:23].tolist(),
                     contacts=contacts[i].tolist()) for i, v in enumerate(value)]

    def step(self):
        from pace_motion import impact
        env = self.env
        before = self.snapshot()
        previous_hit = env.events.hit[:self.count].cpu().tolist()
        result = self.original()
        after = self.snapshot()
        hits = env.events.hit[:self.count].cpu().tolist()
        outcomes = env.events.outcome[:self.count].cpu().tolist()
        for i, row in enumerate(self.rows):
            if self.finished[i]:
                continue
            old, new = before[i], after[i]
            if hits[i] and not previous_hit[i] and row['impact'] is None:
                row['pre_contact'] = old
                row['impact'] = impact(old, new, env.bc['radius_m'])
            if row['impact'] is not None:
                if not new['contacts'][2] and row['first_separation'] is None:
                    normal = np.asarray(new['normal'])
                    offset = np.asarray(new['ball'][:3]) - new['tcp']
                    separation = float(offset @ normal)
                    point = np.asarray(new['ball'][:3]) - separation * normal
                    surface = np.asarray(new['tcp_velocity']) + np.cross(new['omega'], point - new['tcp'])
                    relative = float((np.asarray(new['ball'][3:]) - surface) @ normal)
                    if separation * relative > 0:
                        row['first_separation'] = dict(new, signed_face_distance_m=separation,
                            relative_normal_velocity_estimate_m_s=relative,
                            surface_velocity_estimate_m_s=surface.tolist())
                if new['contacts'][0] and not old['contacts'][0] and row['landing'] is None:
                    row['landing'] = new
            if new['contacts'][1] and not old['contacts'][1]:
                row['net_contacts'].append(new)
            if outcomes[i]:
                self.finished[i] = True
                row['terminal'] = dict(new, outcome=env.events.names[outcomes[i]])
        return result


def evaluate_bank(env, policy, capture=False):
    """Score each canonical row once, using deterministic policy.act_inference.

    The caller owns policy/RNG isolation. This helper neither samples actor
    actions nor updates weights. It replaces env.physics_step only while needed.
    """
    if env.serve_bank is None or env.n < len(env.serve_bank.rows):
        raise ValueError('Five-serve evaluation requires a fixed bank and at least five environments')
    count = len(env.serve_bank.rows)
    indices = np.zeros(env.n, dtype=np.int64)
    indices[:count] = np.arange(count)
    env.reset(env.ids, bank_indices=indices)
    np.testing.assert_allclose(env.state().cpu().numpy(), env.serve_bank.states[indices], atol=2e-6, rtol=0)
    assert not env.ball.data.root_ang_vel_w.any()
    obs, _ = env.observations()
    selected = {}
    recorder = _MotionCapture(env, count) if capture else None
    original_step = env.physics_step
    collecting = env.prediction_collect
    env.prediction_collect = False
    if recorder is not None:
        env.physics_step = recorder.step
    try:
        with torch.no_grad():
            for _ in range(int(np.ceil(env.shot['timeout_s'] / env.policy_dt)) + 2):
                obs, _, _, _, info = env.step(policy.act_inference(obs))
                for row in info['episodes']:
                    selected.setdefault(row['env'], row)
                # All first episodes must finish, including unscored padding.
                if len(selected) == env.n:
                    break
        if len(selected) != env.n:
            raise RuntimeError('Incomplete fixed-bank evaluation')
    finally:
        env.physics_step = original_step
        env.prediction_collect = collecting
    rows = []
    for i, source in enumerate(env.serve_bank.rows):
        row = dict(selected[i])
        assert row['raw_id'] == source['id']
        np.testing.assert_array_equal(row['initial_state'], source['state'])
        if recorder is not None:
            assert recorder.finished[i]
            row['motion'] = recorder.rows[i]
            row['measured_post_hit_landing_xy_m'] = None if row['motion']['landing'] is None else row['motion']['landing']['ball'][:2]
        rows.append(row)
    return dict(event_rules=env.events.rules, total=count,
        serve_bank_sha256=env.serve_bank.sha256,
        scope='Five inspected training states, deterministic actor means; not heldout/generalization',
        hit_rate=sum(r['hit'] for r in rows) / count,
        legal_return_rate=sum(r['outcome'] == 'legal_return' for r in rows) / count,
        target_success_rate=sum(r['target_success'] for r in rows) / count,
        outcomes=dict(Counter(r['outcome'] for r in rows)), episodes=rows,
        padded_environments_excluded=env.n-count, physics_capture=capture)


def verify(device='cuda:0', num_envs=16, manifest=None):
    """Validate real bank resets, origins, isolation and previous contact scripts."""
    from airbot_tt.pace_env import PaceEnv
    from airbot_tt.pace_serves import FIVE_SERVE_IDS
    from pace_m7 import source_hashes
    if num_envs != 16 or manifest is None:
        raise ValueError('Five-serve gate requires 16 environments and an explicit manifest')
    started = time.monotonic()
    before = source_hashes()
    torch.set_num_threads(1)
    env = PaceEnv(num_envs, device, group='A', serve_bank=manifest)
    assert tuple(r['id'] for r in env.serve_bank.rows) == FIVE_SERVE_IDS
    checks = {}
    indices = np.arange(num_envs) % len(FIVE_SERVE_IDS)
    env.reset(env.ids, bank_indices=indices)
    np.testing.assert_allclose(env.state().cpu().numpy(), env.serve_bank.states[indices], atol=2e-6, rtol=0)
    assert not env.ball.data.root_ang_vel_w.any()
    assert torch.any(env.origins[1:] != env.origins[0])
    obs, critic = env.observations()
    assert obs.shape == (16, 115) and critic.shape == (16, 200)
    assert (obs[0]-obs[5]).abs().max() < 2e-5
    assert (critic[0]-critic[5]).abs().max() < 2e-5
    checks['exact_source_states_zero_spin_nonzero_origins'] = True
    checks['translation_invariant_observations'] = True
    writes = env.state_writes
    with torch.no_grad():
        for _ in range(3):
            _, _, _, done, _ = env.step(torch.zeros((env.n, 6), device=device), auto_reset=False)
            assert not done.any()
    assert env.state_writes == writes
    untouched = (env.state()[1:].clone(), env.actor_history[1:].clone(), env.critic_history[1:].clone(),
                 env.robot.data.joint_pos[1:].clone(), env.serve_id[1:].clone(), env.bank_index[1:].copy())
    env.reset([0], bank_indices=[4])
    current = (env.state()[1:], env.actor_history[1:], env.critic_history[1:],
               env.robot.data.joint_pos[1:], env.serve_id[1:], env.bank_index[1:])
    assert all(torch.equal(a, b) for a, b in zip(untouched[:-1], current[:-1]))
    np.testing.assert_array_equal(untouched[-1], current[-1])
    assert env.bank_index[0] == 4
    checks['partial_reset_isolation'] = True
    checks['normal_step_no_direct_state_writes'] = True

    sequences = []
    for interleave in (False, True):
        env.seed_bank(8341)
        seq = []
        for _ in range(10):
            env.reset([0]); seq.append(int(env.bank_index[0]))
            if interleave:
                env.reset([1, 2])
        sequences.append(seq)
        assert sorted(seq[:5]) == list(range(5)) and sorted(seq[5:]) == list(range(5))
    assert sequences[0] == sequences[1]
    checks['seeded_cycles_and_environment_independence'] = True

    # Final deterministic script replay exercises auto-reset and episode attribution.
    audit = ROOT / '.agents/runs/foundation/deepmind_audit'
    plan_files = [audit / f'{name}.json' for name in ('position_search', 'position_remaining')]
    plans = {r['id']:r for path in plan_files for r in json.loads(path.read_text())['rows'] if r['accepted']}
    old = json.loads((audit / 'physics.json').read_text())
    previous = {r['id']:r['episode'] for r in old['diagonal']}
    actions = [np.array(plans[raw_id]['preflight']['actions']) for raw_id in FIVE_SERVE_IDS]
    assert all(np.isfinite(x).all() and abs(x).max() <= 1 + 1e-7 for x in actions)

    class ContactScripts:
        def __init__(self):
            self.tick = 0
        def act_inference(self, obs):
            which = list(range(5)) + [0] * (env.n-5)
            result = np.array([actions[i][min(self.tick, len(actions[i])-1)] for i in which])
            self.tick += 1
            return env.tensor(result)

    env.seed_bank(8342)
    evaluation = evaluate_bank(env, ContactScripts(), capture=True)
    for row in evaluation['episodes']:
        expected = previous[row['raw_id']]
        assert row['hit'] and row['hit'] == expected['hit']
        assert row['outcome'] == expected['outcome']
        assert row['outcome'] == ('out' if row['raw_id'] == 10818 else 'wrong_bounce')
        assert row['motion']['impact'] is not None
    assert evaluation['total'] == 5
    assert len(env.bank_draws) > env.n
    assert all(d['raw_id'] in FIVE_SERVE_IDS for d in env.bank_draws)
    assert all(0 <= i < 5 for i in env.bank_index)
    checks['previous_five_contact_scripts_reproduced'] = True
    checks['auto_reset_stays_in_bank_and_terminal_ids_preserved'] = True
    checks['padding_and_post_reset_episodes_excluded'] = True
    assert before == source_hashes(), 'Production source changed during bank gate'
    return dict(status='PASS', checks=checks, source_sha256=before,
        serve_bank_sha256=env.serve_bank.sha256,
        gate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in plan_files + [audit/'physics.json']},
        contact_script_evaluation=evaluation, bank_draws=env.bank_draws,
        cycle_env0_indices=sequences[0], seconds=time.monotonic()-started,
        ppo_updates=0, normal_motion_direct_state_writes=0)
