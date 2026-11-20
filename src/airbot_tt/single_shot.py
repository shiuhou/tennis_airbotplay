"""Single-shot event rules and deadline-aware positioning feasibility (SI).

Feasibility certifies a collision-checked rest-to-rest positioning trajectory,
not a useful striking velocity or a successful return; those belong to M5.
"""

import numpy as np

from .trajectory import Trajectory


EVENT_RULES = "rally_net_contact_v2"


class ShotEvents:
    rules = EVENT_RULES
    def __init__(self, ball_config, timeout_s):
        self.config = ball_config
        self.timeout = timeout_s
        self.reset()

    def reset(self):
        self.phase = 'incoming'
        self.outcome = None
        self.events = []
        self.previous = None
        self.contacts = set()
        self.net_touched = False

    def _event(self, kind, time, position):
        self.events.append(dict(kind=kind, time_s=float(time), position_m=list(map(float, position))))

    def update(self, time, state, contacts=(), arm_fault=None):
        if self.outcome is not None:
            return self.outcome
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('Expected finite ball position and velocity.')
        p, v = state[:3], state[3:]
        contacts = set(contacts)
        new = contacts - self.contacts
        self.contacts = contacts
        if 'net' in new:
            self.net_touched = True
            self._event('net_contact', time, p)
        if arm_fault or contacts.intersection({'arm', 'fixer', 'handle'}):
            self.outcome = 'unexpected_contact' if not arm_fault else 'arm_fault'
        elif 'paddle' in new:
            self._event('paddle_contact', time, p)
            if self.phase != 'own_bounce':
                self.outcome = 'illegal_hit'
            else:
                self.phase = 'hit'
        if self.outcome is None and self.previous is not None:
            old_t, old = self.previous
            if self.phase == 'hit' and old[0] < 0 <= p[0] and v[0] > 0:
                fraction = -old[0] / (p[0] - old[0])
                crossing = old[:3] + fraction * (p - old[:3])
                self._event('net_crossing', old_t + fraction * (time-old_t), crossing)
                c = self.config
                if abs(crossing[1]) > c['table_size_m'][1]/2:
                    self.outcome = 'out'
                else:
                    self.phase = 'crossed'
        if self.outcome is None and 'table' in new:
            self._event('table_contact', time, p)
            if self.phase == 'incoming' and p[0] < 0:
                self.phase = 'own_bounce'
            elif self.phase == 'crossed' and p[0] > 0:
                self.outcome = 'legal_return'
            else:
                self.outcome = 'wrong_bounce'
        if self.outcome is None:
            if p[2] < self.config['table_top_m'] - .05:
                self.outcome = 'miss' if self.phase in ('incoming', 'own_bounce') else 'out'
            elif abs(p[0]) > 2 or abs(p[1]) > 1.2:
                self.outcome = 'out'
            elif time >= self.timeout:
                self.outcome = 'timeout'
        self.previous = (time, state.copy())
        if self.outcome is not None:
            self._event(self.outcome, time, p)
        return self.outcome


def positioning_candidate(model, checker, start_q, target, arrival_s, config):
    available = arrival_s - config['reaction_delay_s']
    if available <= 0:
        return dict(accepted=False, reason='deadline')
    solution = model.ik(target, start_q)
    if not solution['success']:
        return dict(accepted=False, reason='unreachable')
    try:
        path = Trajectory(model, [start_q, solution['q_rad']],
                          vmax=config['position_speed_rad_s'], amax=config['position_acceleration_rad_s2'],
                          require_clear=checker.require_clear)
    except ValueError as error:
        return dict(accepted=False, reason='collision_or_limits', detail=str(error))
    if path.duration > available:
        return dict(accepted=False, reason='deadline', required_s=path.duration, available_s=available)
    return dict(accepted=True, q_rad=solution['q_rad'], duration_s=path.duration,
                available_s=available, checked_samples=path.checked_samples,
                scope='positioning only; striking velocity and return not certified')
