"""M4: continuous forecast against independently stepped PhysX observations."""
import hashlib
import json
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from airbot_tt.ball_scene import BallTestbed
from airbot_tt.config import ROOT
from airbot_tt.prediction import predict_ball
from foundation import _check


def observed_events(states, forces, dt, config, first_table_event='bounce'):
    """Extract reference events from measured states/contact, never predictor RHS."""
    events=[];kind=first_table_event
    def add(name,time,p):
        events.append(dict(kind=name,time_s=float(time),position_m=np.asarray(p).tolist()))
    for j in range(1,len(states)):
        a,b=states[j-1],states[j];time=j*dt
        table=forces[j,2]>1e-5 and forces[j-1,2]<=1e-5
        if abs(forces[j,0])>1e-5:
            add('net_hit',time,b[:3]);break
        if table:
            add(kind,time,b[:3])
            if kind=='landing':break
            kind='landing'
            continue
        if a[5]>0>=b[5]:
            f=a[5]/(a[5]-b[5]);add('apex',(j-1+f)*dt,a[:3]+f*(b[:3]-a[:3]))
        if a[0]*b[0]<0:
            f=-a[0]/(b[0]-a[0]);add('net_crossing',(j-1+f)*dt,a[:3]+f*(b[:3]-a[:3]))
        h=config['table_top_m']+config['radius_m']
        if a[2]>h>=b[2] and np.any(abs(b[:2])>np.array(config['table_size_m'][:2])/2+config['radius_m']):
            f=(h-a[2])/(b[2]-a[2]);add('out_of_table',(j-1+f)*dt,a[:3]+f*(b[:3]-a[:3]));break
    return sorted(events,key=lambda e:e['time_s'])


def compare_events(predicted, measured):
    assert [e['kind'] for e in predicted]==[e['kind'] for e in measured], (predicted,measured)
    return [dict(kind=p['kind'], position_error_m=float(np.linalg.norm(np.array(p['position_m'])-m['position_m'])),
                 time_error_s=abs(p['time_s']-m['time_s'])) for p,m in zip(predicted,measured)]


def verify(device='cpu'):
    dt=1/json.loads((ROOT/'configs/single_shot.json').read_text())['physics_hz']
    bed=BallTestbed(dt,device,static_paddle=False)
    report=dict(checks={},cases=[],dt_s=dt,dataset='m4-events-v1',
                scope='zero-spin, frictionless table; net face and clear crossings; no edge grazing or future paddle contact',
                source_sha256={})
    for path in ['src/airbot_tt/prediction.py','src/airbot_tt/ball.py','src/airbot_tt/ball_scene.py',
                 'configs/ball.json','configs/single_shot.json','tests/prediction.py','scripts/run_airbot.py']:
        report['source_sha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    # Fixed acceptance fixtures, not a trained/test-seed distribution. No fitting.
    cases=[]
    for i,y in enumerate(np.linspace(-.35,.35,8)):
        cases.append((f'bounce_{i}',[-.2,y,1.2+.01*i],[-1.1,.03,-.4], 'bounce'))
        cases.append((f'return_{i}',[-.8,y,1.15+.01*i],[2.8,.05,1.5], 'landing'))
    for i,y in enumerate([-.3,-.1,.1,.3]):
        cases.append((f'net_{i}',[-.15,y,.87],[3,0,0], 'landing'))
        cases.append((f'out_{i}',[1.5,y,1.1],[0,0,-1], 'landing'))
    for name,p,v,first in cases:
        with _check(report,name):
            prediction=predict_ball([*p,*v],config=bed.config,first_table_event=first)
            bed.reset(p,v);states=[np.r_[p,v]];forces=[np.zeros(3)]
            with ExitStack() as guard:
                for method in ('write_root_state_to_sim','write_root_pose_to_sim','write_root_velocity_to_sim'):
                    guard.enter_context(patch.object(bed.ball,method,side_effect=AssertionError('State write during rollout')))
                for _ in range(round(2/dt)):
                    state,force=bed.step();states.append(state);forces.append(force)
            measured=observed_events(np.array(states),np.array(forces),dt,bed.config,first)
            errors=compare_events(prediction['events'],measured)
            report['cases'].append(dict(id=name,initial_state=[*p,*v],first_table_event=first,
                                        predicted=prediction,measured=measured,errors=errors))
    with _check(report,'per_event_p95_gate'):
        report['metrics']={}
        for kind in ('bounce','apex','net_crossing','landing','net_hit','out_of_table'):
            rows=[e for c in report['cases'] for e in c['errors'] if e['kind']==kind]
            assert rows,kind
            pos=[e['position_error_m'] for e in rows];time=[e['time_error_s'] for e in rows]
            metric=dict(count=len(rows),p95_position_m=float(np.percentile(pos,95)),
                        p95_time_s=float(np.percentile(time,95)),max_position_m=max(pos),max_time_s=max(time))
            report['metrics'][kind]=metric
            assert metric['p95_position_m']<=.02 and metric['p95_time_s']<=.01,(kind,metric)
    report.update(status='PASS',all_cases_denominator=len(cases),normal_state_writes=0)
    return report
