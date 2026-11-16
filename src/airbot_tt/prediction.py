"""Continuous low-spin forecasts, independent of PhysX stepping (SI).

Predicts free flight and frictionless table restitution, not robot/paddle motion.
Table-edge grazing and net-top grazing are explicitly unsupported.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .ball import drag_force, load_ball_config


def predict_ball(state, horizon_s=2.0, *, config=None, first_table_event='bounce'):
    """Return chronological events relative to the supplied measured state.

    `first_table_event='landing'` is appropriate after a measured paddle hit.
    A bounce forecast continues to its next landing. No history survives calls.
    Positions describe the ball centre, including at table contact.
    """
    c = load_ball_config() if config is None else config
    y = np.asarray(state, dtype=float).copy()
    if y.shape != (6,) or not np.isfinite(y).all():
        raise ValueError('Expected finite [x,y,z,vx,vy,vz].')
    if not np.isfinite(horizon_s) or horizon_s <= 0 or horizon_s > 10:
        raise ValueError('Forecast horizon must be in (0,10] seconds.')
    if first_table_event not in ('bounce', 'landing'):
        raise ValueError('First table event must be bounce or landing.')
    if c['friction'] != 0:
        raise ValueError('Forecast currently requires frictionless contacts.')
    surface = c['table_top_m'] + c['radius_m']
    if y[2] < surface:
        raise ValueError('Initial ball centre must be above the table contact plane.')
    result = dict(events=[], termination='horizon', horizon_s=float(horizon_s),
                  scope='zero-spin flight and table; no future paddle/arm collision')
    t0, table_kind = 0., first_table_event

    def rhs(t, s):
        return np.r_[s[3:], drag_force(s[3:], c)/c['mass_kg'] + [0,0,-c['gravity_m_s2']]]

    def table(t, s):
        return s[2]-surface
    table.direction, table.terminal = -1, True

    def apex(t, s):
        return s[5]
    apex.direction = -1

    def net(t, s):
        return s[0]

    def net_front(t, s):
        return s[0] - np.sign(y[0])*(c['net_thickness_m']/2+c['radius_m'])

    def event(kind, time, s):
        return dict(kind=kind, time_s=float(time), position_m=s[:3].tolist(), velocity_m_s=s[3:].tolist())

    for _ in range(2):
        net_front.direction = -np.sign(y[0])
        sol = solve_ivp(rhs, (t0,horizon_s), y, method='DOP853', rtol=1e-9, atol=1e-11,
                        max_step=.02, events=(table,apex,net,net_front))
        if not sol.success:
            raise RuntimeError(sol.message)
        pending = []
        for index, kind in [(1,'apex'), (2,'net_crossing')]:
            for time, s in zip(sol.t_events[index],sol.y_events[index]):
                if time > t0+1e-9:
                    pending.append(event(kind,time,s))
        stop = None
        for time, s in zip(sol.t_events[3],sol.y_events[3]):
            if abs(s[1]) <= c['table_size_m'][1]/2+c['radius_m']:
                top = c['table_top_m']+c['net_height_m']
                if s[2] <= top+c['radius_m']:
                    kind = ('net_hit' if s[2] <= top and abs(s[1]) < c['table_size_m'][1]/2-c['radius_m']
                            else 'unsupported_net_edge')
                    stop = event(kind,time,s)
                    break
        if stop is not None:
            result['events'].extend(sorted([e for e in pending if e['time_s'] < stop['time_s']]+[stop],key=lambda e:e['time_s']))
            result['termination'] = stop['kind']
            return result
        result['events'].extend(sorted(pending,key=lambda e:e['time_s']))
        if not len(sol.t_events[0]):
            return result
        time, s = sol.t_events[0][0], sol.y_events[0][0]
        edge_dist = np.array(c['table_size_m'][:2])/2 - abs(s[:2])
        if np.any(edge_dist < -c['radius_m']):
            kind = 'out_of_table'
        elif np.any(edge_dist < c['radius_m']):
            kind = 'unsupported_table_edge'
        else:
            kind = table_kind
        result['events'].append(event(kind,time,s))
        if kind != 'bounce':
            result['termination'] = kind
            return result
        y = s.copy()
        y[5] *= -c['table_restitution']
        t0, table_kind = float(time), 'landing'
        if t0 >= horizon_s:
            return result
    return result
