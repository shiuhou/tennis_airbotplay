"""Zero-spin model-based return candidates (world SI), with explicit rejection.

This is a bounded local search, not a global feasibility proof. Frictionless
impact constrains only normal contact velocity; tangential/rotational motion
is chosen through the minimum-norm joint solution and checked as a trajectory.
"""
import json
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from .config import ROOT
from .ball import drag_force
from .prediction import predict_ball
from .trajectory import StrikeTrajectory


def load_strike_config():
    return json.loads((ROOT/'configs/strike.json').read_text())


def flight_state(state, duration, config):
    """Free flight only; caller must split at physical contacts."""
    def rhs(t,s):
        return np.r_[s[3:],drag_force(s[3:],config)/config['mass_kg']+[0,0,-config['gravity_m_s2']]]
    sol=solve_ivp(rhs,(0,duration),state,method='DOP853',rtol=1e-8,atol=1e-10)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1]


def return_velocity(position, goal_xy, duration, config):
    """Shoot a drag-affected flight to the table, then check net and first landing."""
    goal=np.r_[goal_xy,config['table_top_m']+config['radius_m']]
    guess=(goal-position)/duration+[0,0,.5*config['gravity_m_s2']*duration]
    solution=least_squares(lambda v:flight_state(np.r_[position,v],duration,config)[:3]-goal,
                           guess,ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=30)
    if np.linalg.norm(solution.fun)>.001:raise ValueError('outgoing_flight_solve')
    forecast=predict_ball(np.r_[position,solution.x],duration+.1,config=config,first_table_event='landing')
    if forecast['termination']!='landing':raise ValueError('outgoing_'+forecast['termination'])
    landing=forecast['events'][-1]
    if np.linalg.norm(np.array(landing['position_m'])[:2]-goal_xy)>.002:raise ValueError('early_landing')
    if not any(e['kind']=='net_crossing' for e in forecast['events']):raise ValueError('no_net_crossing')
    return solution.x


def plan_return(model, checker, start_q, state, goal_xy, base_position, ball_config, config=None):
    """Plan at measured pre-bounce state; times are relative to that measurement.

    Returns the trajectory and requested strike state, or a rejection histogram.
    Wall-clock planning cost is reported separately; callers must account for it
    before executing against a live ball. Offline scripted replay is not realtime.
    """
    c=load_strike_config() if config is None else config
    started=perf_counter();rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    forecast=predict_ball(state,config=ball_config)
    bounce=next((e for e in forecast['events'] if e['kind']=='bounce'),None)
    if bounce is None:return dict(accepted=False,reason='no_own_bounce',planning_s=perf_counter()-started)
    if bounce['position_m'][0]>=0:return dict(accepted=False,reason='wrong_side_bounce',planning_s=perf_counter()-started)
    after=np.r_[bounce['position_m'],bounce['velocity_m_s']];after[5]*=-ball_config['table_restitution']
    for offset in c['candidate_after_bounce_s']:
        incoming=flight_state(after,offset,ball_config);position=incoming[:3];vin=incoming[3:]
        hit_time=bounce['time_s']+offset
        for duration in c['return_flight_s']:
            try:outgoing=return_velocity(position,goal_xy,duration,ball_config)
            except ValueError as e:reject(str(e));continue
            normal=outgoing-vin;normal/=np.linalg.norm(normal)
            if normal[0]<=0:reject('back_face');continue
            pose=np.eye(4);x=np.array([-normal[2],0,normal[0]]);x/=np.linalg.norm(x)
            # Reproject X orthogonal to n for lateral goals; Y completes a right-handed frame.
            pose[:3,:3]=np.array([x,np.cross(normal,x),normal]).T
            pose[:3,3]=position-ball_config['radius_m']*normal-base_position
            ik=model.ik(pose,start_q)
            if not ik['success']:reject('unreachable');continue
            q=np.array(ik['q_rad']);row=normal@model.jacobian(q)[:3]
            e=ball_config['paddle_restitution'];normal_speed=np.dot(outgoing+e*vin,normal)/(1+e)
            if np.dot(row,row)<1e-10:reject('normal_velocity_singular');continue
            qd=row*normal_speed/np.dot(row,row)
            end=q+.5*c['recovery_s']*qd
            try:
                path=StrikeTrajectory(model,start_q,q,qd,end,hit_time-c['reaction_delay_s'],c['recovery_s'],
                      vmax=c['velocity_limit_rad_s'],amax=c['acceleration_limit_rad_s2'],require_clear=checker.require_clear)
            except ValueError as e:reject(str(e));continue
            return dict(accepted=True,path=path,hit_time_s=hit_time,strike_q=q,strike_qd=qd,end_q=end,
                        desired_pose=pose,incoming=incoming,outgoing=outgoing,normal=normal,
                        requested_twist=model.jacobian(q)@qd,planning_s=perf_counter()-started,rejections=rejected)
    return dict(accepted=False,reason='no_candidate',rejections=rejected,planning_s=perf_counter()-started)


def serves(split, config=None):
    c=load_strike_config() if config is None else config
    if split not in ('development','validation'):raise ValueError('Unknown serve split.')
    rng=np.random.default_rng(c[split+'_seed'])
    return [dict(id=f'{split}_{i}',position_m=(np.array(c['serve_position_m'])+rng.uniform(-1,1,3)*c['position_half_range_m']).tolist(),
                 velocity_m_s=(np.array(c['serve_velocity_m_s'])+rng.uniform(-1,1,3)*c['velocity_half_range_m_s']).tolist())
            for i in range(c[split+'_count'])]
