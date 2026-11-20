"""M3 live single-shot reset, contacts, outcomes and positioning feasibility."""
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from airbot_tt.shot_scene import SingleShot
from airbot_tt.single_shot import positioning_candidate
from airbot_tt.config import ROOT
from foundation import _check


def verify(device='cpu'):
    task=SingleShot(device)
    control=task.control
    report=dict(event_rules=task.events.rules,checks={},episodes=[],scope='M3 single-shot infrastructure; no learned or planned hitting policy',
                config=task.config,source_sha256={})
    for path in ['configs/single_shot.json','configs/ball.json','src/airbot_tt/shot_scene.py',
                 'src/airbot_tt/single_shot.py','src/airbot_tt/runtime.py','src/airbot_tt/ball_scene.py',
                 'src/airbot_tt/ball.py','src/airbot_tt/tool.py','src/airbot_tt/collision.py',
                 'src/airbot_tt/trajectory.py','src/airbot_tt/prediction.py','tests/prediction.py','scripts/run_airbot.py','tests/single_shot.py',
                 'assets/derived/racket/geometry.json']:
        report['source_sha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    with _check(report, 'fixed_table_mount'):
        from pxr import UsdPhysics
        from airbot_tt.runtime import ROBOT_PRIM
        relation = UsdPhysics.FilteredPairsAPI(control.sim.stage.GetPrimAtPath(ROBOT_PRIM+'/base_link')).GetFilteredPairsRel()
        assert list(map(str, relation.GetTargets())) == ['/World/Table']
        control.checker.require_clear(control.initial)
        assert control.robot.is_fixed_base
    for serve in task.config['serves']:
        with _check(report,serve['id']):
            trials=[]
            outcomes=[]
            sequences=[]
            for repeat in range(2):
                task.reset(serve)
                initial_forecast=task.forecast()
                initial_ball=task.state.copy()
                start=control.q().copy()
                writes=control.state_writes
                rows=[]
                with ExitStack() as guard:
                    for obj,methods in [(task.world.ball,['write_root_state_to_sim','write_root_pose_to_sim','write_root_velocity_to_sim']),
                                        (control.robot,['write_joint_state_to_sim'])]:
                        for name in methods:
                            guard.enter_context(patch.object(obj,name,side_effect=AssertionError('State write during episode')))
                    while task.events.outcome is None:
                        row=task.step(); rows.append(row)
                assert control.state_writes==writes
                if repeat==0:
                    from prediction import observed_events, compare_events
                    from airbot_tt.prediction import predict_ball
                    states=np.vstack([initial_ball,*[r['ball_state'] for r in rows]])
                    forces=np.zeros((len(states),3))
                    for j,row in enumerate(rows,1):
                        forces[j,2]=float('table' in row['contacts'])
                        forces[j,0]=float('net' in row['contacts'])
                    hit=next((j for j,row in enumerate(rows,1) if 'paddle' in row['contacts']),None)
                    end=hit if hit is not None else len(states)
                    measured=observed_events(states[:end],forces[:end],control.dt,task.ball_config)
                    predicted=[e for e in initial_forecast['events'] if e['time_s']<=(end-1)*control.dt]
                    forecast_errors=compare_events(predicted,measured)
                    if hit is not None:
                        post=predict_ball(states[hit],config=task.ball_config,first_table_event='landing')
                        measured=observed_events(states[hit:],forces[hit:],control.dt,task.ball_config,'landing')
                        forecast_errors += compare_events(post['events'],measured)
                    assert forecast_errors
                    assert max(e['position_error_m'] for e in forecast_errors)<=.02, forecast_errors
                    assert max(e['time_error_s'] for e in forecast_errors)<=.01, forecast_errors
                    events=json.loads(json.dumps(task.events.events))
                    candidates=[]
                    reasons=Counter()
                    rejected_details=[]
                    for row in rows[::task.config['candidate_stride_steps']]:
                        if row['phase']=='hit' and 'paddle' in row['contacts']:
                            # A measured hit certifies the already-positioned pose;
                            # a ball centre sampled before impact need not be
                            # exactly one radius from the face at a discrete step.
                            result=positioning_candidate(control.model,control.checker,start,
                                    control.model.fk(start),row['time_s'],task.config)
                            if result['accepted']:
                                candidates.append(dict(time_s=row['time_s'],position_m=row['ball_state'][:3].tolist(),
                                                       evidence='measured contact at prepositioned pose',**result))
                        if row['phase']!='own_bounce' or row['ball_state'][0]>control.model.fk(start)[0,3]+task.config['base_position_m'][0]+.2:
                            continue
                        target=control.model.fk(start)
                        target[:3,3]=row['ball_state'][:3]-np.array(task.config['base_position_m'])-task.ball_config['radius_m']*target[:3,2]
                        result=positioning_candidate(control.model,control.checker,start,target,row['time_s'],task.config)
                        if result['accepted']:
                            candidates.append(dict(time_s=row['time_s'],position_m=row['ball_state'][:3].tolist(),**result))
                        else:
                            reasons[result['reason']]+=1
                            if result['reason']=='deadline':
                                rejected_details.append(dict(time_s=row['time_s'],target=target[:3,3].tolist(),**result))
                    episode=dict(serve_id=serve['id'],outcome=task.events.outcome,events=events,
                                 forecast_errors=forecast_errors,
                                 feasible_candidates=candidates,rejected_candidates=dict(reasons),
                                 deadline_examples=rejected_details[-3:],
                                 normal_joint_state_writes=0,normal_ball_state_writes=0)
                    report['episodes'].append(episode)
                outcomes.append(task.events.outcome)
                sequences.append([e['kind'] for e in task.events.events])
                np.testing.assert_allclose(control.state()['base_world']['position_m'], task.config['base_position_m'], atol=1e-6, rtol=0)
                trials.append(np.array([r['ball_state'] for r in rows]))
            assert outcomes[0] == outcomes[1] and sequences[0] == sequences[1], 'Reset changes event sequence'
            assert trials[0].shape==trials[1].shape, 'Reset changes episode length'
            position_error=float(np.max(np.abs(trials[0][:,:3]-trials[1][:,:3])))
            velocity_error=float(np.max(np.abs(trials[0][:,3:]-trials[1][:,3:])))
            assert position_error<1e-5 and velocity_error<1e-4, f'Nonrepeatable reset: {position_error}m, {velocity_error}m/s'
            report['episodes'][-1].update(repeat_position_error_m=position_error,repeat_velocity_error_m_s=velocity_error)
    with _check(report,'feasible_and_rejected_serve_set'):
        assert any(e['feasible_candidates'] for e in report['episodes']), 'No deadline-feasible serves'
        assert not next(e for e in report['episodes'] if e['serve_id']=='too_wide')['feasible_candidates']
    with _check(report,'physical_net_contact_then_resolution'):
        task.reset(dict(position_m=[-.15,.5,.87],velocity_m_s=[3,0,0]))
        while task.events.outcome is None:
            task.step()
        assert task.events.net_touched, 'Net contact was not observed'
        assert any(e['kind']=='net_contact' for e in task.events.events)
        assert task.events.outcome in ('wrong_bounce','miss','out','timeout'), task.events.outcome
    with _check(report,'physical_fixer_failure'):
        centre=control.model.tool_geometry['components']['fixer']['center_of_mass_link6_m']
        position=(control.model.frames(control.initial)[6] @ [*centre,1])[:3]+task.config['base_position_m']
        task.reset(dict(position_m=position.tolist(),velocity_m_s=[0,0,0]))
        while task.events.outcome is None:
            task.step()
        assert task.events.outcome in ('unexpected_contact','arm_fault'), task.events.outcome
    with _check(report,'reset_clears_failure'):
        task.reset(task.config['serves'][0])
        assert task.events.outcome is None and not task.events.events and control.fault is None
    report['status']='PASS'
    report['outcomes']=dict(Counter(e['outcome'] for e in report['episodes']))
    report['all_serves_denominator']=len(report['episodes'])
    return report
