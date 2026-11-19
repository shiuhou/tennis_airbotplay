"""M6 CPU reference teacher and PACE-shaped rewards, SI / local table frame.

Formula provenance: PACE fdb192f8 legged_lab/mdp/rewards.py and T1 config.
Physics, target, invalid masks and physical events are AIRBOT adaptations.
No robot commands or PPO state are handled here.
"""
import json
import numpy as np
from .config import ROOT
from .prediction import predict_ball


def settings():
    return json.loads((ROOT/'configs/pace.json').read_text())


def apex_teacher(state, *, bounced=False, hit=False, config=None):
    """Predict an upcoming *post own-table-bounce* apex, never a fallback point."""
    s = np.asarray(state, dtype=float)
    if s.shape != (6,) or not np.isfinite(s).all():
        raise ValueError('Expected finite six-dimensional ball state.')
    invalid = dict(point=[0., 0., 0.], time_s=0., valid=False)
    if hit or s[3] >= 0 or (bounced and s[5] <= 0):
        return {**invalid, 'reason': 'hit_or_outgoing_or_past_apex'}
    from .ball import load_ball_config
    c = load_ball_config() if config is None else config
    if s[2] < c['table_top_m'] + c['radius_m']:
        return {**invalid, 'reason': 'below_surface'}
    forecast = predict_ball(s, config=c, first_table_event='landing' if bounced else 'bounce')
    after_bounce = bounced
    for event in forecast['events']:
        if event['kind'] == 'bounce':
            if event['position_m'][0] >= 0:
                return {**invalid, 'reason': 'opponent_bounce'}
            after_bounce = True
        if event['kind'] == 'apex' and after_bounce:
            return dict(point=event['position_m'], time_s=event['time_s'], valid=True, reason='valid')
    return {**invalid, 'reason': forecast['termination']}


def return_teacher(state, *, config=None):
    """Call on measured release state, not on first overlap/contact state."""
    s = np.asarray(state, dtype=float)
    if s.shape != (6,) or not np.isfinite(s).all():
        raise ValueError('Expected finite release state.')
    invalid = dict(valid=False, landing=[0., 0., 0.], net_z=0.)
    if s[3] <= 0 or s[0] >= 0:
        return {**invalid, 'reason': 'not_forward_before_net'}
    from .ball import load_ball_config
    c = load_ball_config() if config is None else config
    if s[2] < c['table_top_m']+c['radius_m']:
        return {**invalid, 'reason': 'below_surface'}
    f = predict_ball(s, config=c, first_table_event='landing')
    net = next((e for e in f['events'] if e['kind']=='net_crossing'), None)
    landing = next((e for e in f['events'] if e['kind']=='landing'), None)
    if net is None or landing is None or landing['position_m'][0] <= 0:
        return {**invalid, 'reason': f['termination']}
    return dict(valid=True, landing=landing['position_m'], net_z=net['position_m'][2], reason='valid')


def guidance_rewards(tcp, teacher, returned=None, *, dt=1/48, cfg=None):
    """Raw, weighted and total components; event caller must enforce once-only.

    Sparse components retain source weight*0.02 per event at any policy rate.
    Dense hit guidance uses actual dt. Sparse task success/regularizers are M7.
    """
    c = settings()['reward'] if cfg is None else cfg
    tcp = np.asarray(tcp, dtype=float)
    if tcp.shape != (3,) or not np.isfinite(tcp).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid TCP or dt.')
    hit = (np.exp(-max(np.linalg.norm(tcp-np.asarray(teacher['point'])), c['hit_floor_m']) /
                  c['hit_std_m']**2) if teacher['valid'] else 0.)
    landing = net = 0.
    if returned is not None and returned['valid']:
        landing = c['landing_intercept']-np.linalg.norm(np.asarray(returned['landing'])[:2]-c['goal_xy_m'])
        net = np.exp(-abs(returned['net_z']-c['net_target_z_m'])/c['net_std_m'])
    raw = dict(hit=float(hit), landing=float(landing), net=float(net))
    weighted = dict(hit=raw['hit']*c['hit_weight']*dt,
                    landing=raw['landing']*c['landing_weight']*c['source_event_dt_s'],
                    net=raw['net']*c['net_weight']*c['source_event_dt_s'])
    return dict(raw=raw, weighted=weighted, total=sum(weighted.values()))


class ReleaseLatch:
    """One scoring release per serve; receives legal phase from ShotEvents.

    Physical contact must end and ball separate along actual TCP normal.
    Timed-out contact is invalid and never retried until reset.
    """
    def __init__(self, max_contact_s=.05):
        self.max_contact_s = max_contact_s
        self.reset()

    def reset(self):
        self.started = None
        self.done = False
        self.reason = 'waiting'

    def update(self, time_s, state, touching, legal_hit, tcp, normal, tcp_velocity):
        if self.done:
            return None
        if touching and legal_hit and self.started is None:
            self.started = time_s
            self.reason = 'contact'
        if self.started is None:
            return None
        if not legal_hit:
            self.done, self.reason = True, 'invalidated'
            return None
        if time_s-self.started > self.max_contact_s:
            self.done, self.reason = True, 'contact_timeout'
            return None
        s = np.asarray(state)
        n = np.asarray(normal)
        separation = np.dot(s[:3]-tcp, n)
        speed = np.dot(s[3:]-tcp_velocity, n)
        if not touching and separation*speed > 0:
            self.done, self.reason = True, 'released'
            return return_teacher(s)
        return None
