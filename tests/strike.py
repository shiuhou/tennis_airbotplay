"""M5 bounded dry-swing acceptance using measured PhysX TCP poses."""
import hashlib
import numpy as np
from contextlib import ExitStack
from unittest.mock import patch
from scipy.spatial.transform import Rotation
from airbot_tt.shot_scene import SingleShot
from airbot_tt.config import ROOT
from airbot_tt.kinematics import pose_matrix, pose_error
from foundation import _check


def verify(device='cpu'):
    task=SingleShot(device, speed_limit=2.0);c=task.control
    report=dict(checks={},trials=[],scope='M5 dry-swing gate only; not moving impact or return acceptance',source_sha256={})
    for f in ['src/airbot_tt/trajectory.py','src/airbot_tt/runtime.py','tests/strike.py','configs/single_shot.json']:
        report['source_sha256'][f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    cases = [(f'j1_{s}', [s,0,0,0,0,0]) for s in [.1,.4,.8]]
    cases += [('forward', [0,-.4,.6,0,-.2,0]), ('reverse', [0,.3,-.5,0,.2,0]),
              ('mixed', [.3,-.3,.4,.2,-.2,.1])]
    for name, velocity in cases:
        with _check(report,f'passage_{name}'):
            task.reset(dict(position_m=[3,0,2],velocity_m_s=[0,0,0]));task.running=False
            q=c.q();v=np.array(velocity);hit=q+.15*v;end=q+.3*v
            c.start_strike(hit,v,end,.3,.3,vmax=1.,amax=8.)
            rows=[]
            base = c.robot.data.root_pos_w[0].cpu().numpy().copy()
            with ExitStack() as guard:
                guard.enter_context(patch.object(c.robot,'write_joint_state_to_sim',side_effect=AssertionError('state write')))
                for _ in range(round(1.1/c.dt)):
                    c.step()
                    state=c.state();pose=pose_matrix([*state['tcp_base']['position_m'],*state['tcp_base']['quaternion_wxyz']])
                    rows.append((pose,c.q(),c.qd()))
                    assert c.fault is None,c.fault
            j=round(.3/c.dt)-1
            actual=rows[j][0];target=c.model.fk(hit)
            # Differentiate measured physical TCP poses, not model FK(q) feedback.
            linear=(rows[j+1][0][:3,3]-rows[j-1][0][:3,3])/(2*c.dt)
            angular=Rotation.from_matrix(rows[j+1][0][:3,:3]@rows[j-1][0][:3,:3].T).as_rotvec()/(2*c.dt)
            desired=c.model.jacobian(hit)@v
            error=pose_error(actual,target)
            # Crossing the joint-space plane through the requested strike state.
            progress = np.array([np.dot(row[1]-hit,v) for row in rows])
            crossings = [k for k in range(max(0,j-8),min(len(rows)-1,j+8)) if progress[k]<=0<progress[k+1]]
            assert crossings, 'No measured strike-plane crossing near deadline'
            k = min(crossings,key=lambda k:abs(k-j))
            crossing = (k+1-progress[k]/(progress[k+1]-progress[k]))*c.dt
            metric=dict(case=name,joint_velocity_rad_s=v.tolist(),timing_error_s=float(abs(crossing-.3)),position_error_m=float(np.linalg.norm(error[:3])),
                        orientation_error_rad=float(np.linalg.norm(error[3:])),
                        linear_velocity_error_m_s=float(np.linalg.norm(linear-desired[:3])),
                        angular_velocity_error_rad_s=float(np.linalg.norm(angular-desired[3:])),
                        desired_twist=desired.tolist(),measured_twist=np.r_[linear,angular].tolist(),
                        final_speed_rad_s=float(max(abs(rows[-1][2]))))
            report['trials'].append(metric)
            assert metric['position_error_m']<.002 and metric['orientation_error_rad']<.02,metric
            assert metric['linear_velocity_error_m_s']<.03,metric
            assert metric['angular_velocity_error_rad_s']<.08,metric
            assert metric['timing_error_s']<2*c.dt,metric
            np.testing.assert_allclose(c.robot.data.root_pos_w[0].cpu().numpy(),base,atol=1e-7,rtol=0)
            assert metric['final_speed_rad_s']<.01,metric
    report['status']='PASS'
    return report


def verify_impact(device='cpu', hz=480):
    """Driven racket impact, measured contact velocity, independent flight after hit."""
    from scipy.integrate import solve_ivp
    from airbot_tt.ball import drag_force
    from airbot_tt.prediction import predict_ball
    from airbot_tt.planning import load_strike_config, plan_return
    cfg=load_strike_config()
    task=SingleShot(device, speed_limit=2., physics_hz=hz, initial=cfg['ready_q_rad']);c=task.control;bc=task.ball_config
    report=dict(checks={},trials=[],physics_hz=hz,scope='moving face impact; not legal return',source_sha256={})
    for f in ['tests/strike.py','src/airbot_tt/runtime.py','src/airbot_tt/trajectory.py','src/airbot_tt/shot_scene.py','src/airbot_tt/planning.py','configs/strike.json','configs/ball.json']:
        report['source_sha256'][f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    def rhs(t,s):
        return np.r_[s[3:],drag_force(s[3:],bc)/bc['mass_kg']+[0,0,-bc['gravity_m_s2']]]
    def tcp():
        state=c.state()['tcp_base'];T=pose_matrix([*state['position_m'],*state['quaternion_wxyz']]);T[:3,3]+=np.array(task.config['base_position_m'])
        return T
    for goal in cfg['goals_xy_m']:
        name=f"goal_{goal[1]}"
        with _check(report,name):
            task.reset(dict(position_m=[3,0,2],velocity_m_s=[0,0,0]));task.running=False
            plan=plan_return(c.model,c.checker,cfg['ready_q_rad'],np.r_[cfg['serve_position_m'],cfg['serve_velocity_m_s']],goal,
                             np.array(task.config['base_position_m']),bc,cfg)
            assert plan['accepted'],plan
            target=plan['desired_pose'].copy();target[:3,3]+=task.config['base_position_m'];n=target[:3,2]
            hit_time=plan['path'].hit_time
            state_hit=np.r_[target[:3,3]+bc['radius_m']*n,-3*n]
            launch_time=round((hit_time-.05)/c.dt)*c.dt
            launch=solve_ivp(rhs,(hit_time,launch_time),state_hit,rtol=1e-10,atol=1e-12).y[:,-1]
            c.start_prepared_strike(plan['path'])
            for _ in range(round(launch_time/c.dt)):c.step()
            task.world.reset(launch[:3],launch[3:])
            previous=launch.copy();old_tcp=tcp();contact=None;post=None;landing=None
            with ExitStack() as guard:
                guard.enter_context(patch.object(c.robot,'write_joint_state_to_sim',side_effect=AssertionError('joint state write')))
                guard.enter_context(patch.object(task.world.ball,'write_root_state_to_sim',side_effect=AssertionError('ball state write')))
                for step in range(round(.8/c.dt)):
                    c.step();state=task.state.copy();T=tcp();time=launch_time+(step+1)*c.dt
                    assert c.fault is None,c.fault
                    if contact is None and 'paddle' in task.last_contacts:
                        normal=old_tcp[:3,2]
                        omega=Rotation.from_matrix(T[:3,:3]@old_tcp[:3,:3].T).as_rotvec()/c.dt
                        linear=(T[:3,3]-old_tcp[:3,3])/c.dt
                        point=previous[:3]-bc['radius_m']*normal
                        surface=linear+np.cross(omega,point-old_tcp[:3,3])
                        relative=previous[3:]-surface
                        expected=previous[3:]-(1+bc['paddle_restitution'])*np.dot(relative,normal)*normal
                        contact=dict(time_s=time,normal=normal.tolist(),surface_velocity_m_s=surface.tolist(),
                                     incoming_velocity_m_s=previous[3:].tolist(),expected_velocity_m_s=expected.tolist())
                        # Compare immediately post impulse, compensating one gravity/drag step.
                        expected+=rhs(0,previous)[3:]*c.dt
                        contact['outgoing_velocity_m_s']=state[3:].tolist()
                        contact['relative_velocity_error']=float(np.linalg.norm(state[3:]-expected)/np.linalg.norm(expected))
                        post=state.copy()
                    if contact and 'table' in task.last_contacts:
                        landing=state[:3].tolist();break
                    previous=state;old_tcp=T
            assert contact is not None,'No face contact'
            assert landing is not None,'No table landing'
            contact.update(case=name,landing_m=landing,
                           forecast=predict_ball(post,first_table_event='landing',config=bc))
            report['trials'].append(contact)
            assert contact['relative_velocity_error']<.05,contact
    report['status']='PASS'
    return report


def verify_return(device='cpu', split='development'):
    """Fixed nominal model-based scripts; held-out serve variations never replan."""
    from time import perf_counter
    from airbot_tt.planning import load_strike_config, plan_return, serves
    cfg=load_strike_config();task=SingleShot(device,speed_limit=cfg['simulation_speed_limit_rad_s'],initial=cfg['ready_q_rad'],physics_hz=cfg['physics_hz']);c=task.control
    report=dict(checks={},trials=[],dry_swings=[],preparation=[],split=split,config=cfg,
                scope='three nominal precomputed scripts; no online IK or per-serve planning',source_sha256={})
    for f in ['tests/strike.py','src/airbot_tt/planning.py','src/airbot_tt/trajectory.py','src/airbot_tt/runtime.py','src/airbot_tt/shot_scene.py','configs/strike.json','configs/ball.json']:
        report['source_sha256'][f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    plans=[]
    with _check(report,'nominal_preparation'):
        for goal in cfg['goals_xy_m']:
            plan=plan_return(c.model,c.checker,cfg['ready_q_rad'],np.r_[cfg['serve_position_m'],cfg['serve_velocity_m_s']],goal,
                             np.array(task.config['base_position_m']),task.ball_config,cfg)
            report['preparation'].append(dict(goal=goal,accepted=plan['accepted'],planning_s=plan['planning_s'],rejections=plan.get('rejections',{})))
            assert plan['accepted'],plan
            plans.append(plan)
    # Certify the actual three scripts before trying any return outcome.
    for goal,plan in zip(cfg['goals_xy_m'],plans):
        with _check(report,f'dry_goal_{goal[1]}'):
            task.reset(dict(position_m=[3,0,2],velocity_m_s=[0,0,0]));task.running=False
            c.start_prepared_strike(plan['path']);rows=[];joint_rows=[]
            with patch.object(c.robot,'write_joint_state_to_sim',side_effect=AssertionError('state write')):
                for _ in range(round((plan['path'].duration+.5)/c.dt)):
                    c.step();s=c.state()['tcp_base'];rows.append(pose_matrix([*s['position_m'],*s['quaternion_wxyz']]));joint_rows.append(c.q())
                    assert c.fault is None,c.fault
            j=round(plan['path'].hit_time/c.dt)-1;target=c.model.fk(plan['path'].sample((j+1)*c.dt)[0])
            linear=(rows[j+1][:3,3]-rows[j-1][:3,3])/(2*c.dt)
            angular=Rotation.from_matrix(rows[j+1][:3,:3]@rows[j-1][:3,:3].T).as_rotvec()/(2*c.dt)
            error=pose_error(rows[j],target);desired=plan['requested_twist']
            progress=np.array([np.dot(q-plan['strike_q'],plan['strike_qd']) for q in joint_rows])
            roots=[k for k in range(max(0,j-8),min(len(rows)-1,j+8)) if progress[k]<=0<progress[k+1]]
            assert roots,'No measured strike passage'
            k=min(roots,key=lambda k:abs(k-j))
            crossing=(k+1-progress[k]/(progress[k+1]-progress[k]))*c.dt
            metric=dict(goal=goal,timing_error_s=abs(crossing-plan['path'].hit_time),position_error_m=float(np.linalg.norm(error[:3])),orientation_error_rad=float(np.linalg.norm(error[3:])),
                        linear_velocity_error_m_s=float(np.linalg.norm(linear-desired[:3])),angular_velocity_error_rad_s=float(np.linalg.norm(angular-desired[3:])))
            report['dry_swings'].append(metric)
            assert metric['position_error_m']<.002 and metric['orientation_error_rad']<.02,metric
            assert metric['linear_velocity_error_m_s']<.03 and metric['angular_velocity_error_rad_s']<.08,metric
            assert metric['timing_error_s']<2*c.dt,metric
            assert max(abs(c.qd()))<.01
    with _check(report,'all_serves'):
        for serve in serves(split,cfg):
            for goal,plan in zip(cfg['goals_xy_m'],plans):
                task.reset(serve);started=False;selection_time=0.
                base=c.robot.data.root_pos_w[0].cpu().numpy().copy()
                with ExitStack() as guard:
                    guard.enter_context(patch.object(c.robot,'write_joint_state_to_sim',side_effect=AssertionError('joint write')))
                    guard.enter_context(patch.object(task.world.ball,'write_root_state_to_sim',side_effect=AssertionError('ball write')))
                    while task.events.outcome is None:
                        if not started and task.time+1e-10>=cfg['reaction_delay_s']:
                            clock=perf_counter();c.start_prepared_strike(plan['path']);selection_time=perf_counter()-clock;started=True
                        task.step()
                np.testing.assert_allclose(c.robot.data.root_pos_w[0].cpu().numpy(),base,atol=1e-7,rtol=0)
                landing=next((e['position_m'] for e in task.events.events if e['kind']=='legal_return'),None)
                error=None if landing is None else float(np.linalg.norm(np.array(landing[:2])-goal))
                row=dict(serve=serve['id'],goal=goal,outcome=task.events.outcome,events=task.events.events,landing_error_m=error,
                         target_success=landing is not None and bool(np.all(abs(np.array(landing[:2])-goal)<=cfg['goal_half_width_m'])),selection_time_s=selection_time)
                report['trials'].append(row)
        count=len(report['trials']);legal=sum(t['outcome']=='legal_return' for t in report['trials'])
        report.update(legal_returns=legal,total_serves=count,legal_return_rate=legal/count)
        report['by_goal']=[dict(goal=g,total=sum(t['goal']==g for t in report['trials']),
                               legal=sum(t['goal']==g and t['outcome']=='legal_return' for t in report['trials']),
                               target_hits=sum(t['goal']==g and t['target_success'] for t in report['trials'])) for g in cfg['goals_xy_m']]
        assert legal/count>=.8,report['by_goal']
    report['status']='PASS'
    return report
