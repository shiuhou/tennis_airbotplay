#!/usr/bin/env python3
"""Train/evaluate the official PACE MLP on isolated PhysX M6 datasets (no PPO)."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from airbot_tt.config import ROOT
from airbot_tt.pace_guidance import apex_teacher, guidance_rewards, settings
from airbot_tt.pace_predictor import Predictor, PredictionHook, windows


def metric(prediction, truth):
    error = np.asarray(prediction)-truth
    distance = np.linalg.norm(error,axis=1)
    return dict(n=len(error), rmse_m=float(np.sqrt(np.mean(distance**2))),
                xyz_rmse_m=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                p95_m=float(np.percentile(distance,95)))


def dataset(split, dest):
    d = dict(np.load(dest/f'{split}.npz'))
    x,y,ends = windows(d['state'][:,:3],d['time'],d['serve'],np.ones(len(d['time']),bool),d['target'],d['valid'])
    return d,x,y,ends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'evaluate', 'all'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'.agents/runs/foundation/m6')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New experiment directory; separate from historical evidence')
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    c=settings(); dest=args.data_dir; output=args.output_dir
    if args.stage in ('train', 'all'):
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError('Evaluation requires a completed training directory')
    collection=json.loads((dest/'dataset.json').read_text())
    for name, row in collection['splits'].items():
        assert hashlib.sha256((dest/f'{name}.npz').read_bytes()).hexdigest()==row['sha256']
        assert row['seed']==c['splits'][name]['seed'] and row['all_serves']==c['splits'][name]['serves']
        assert all(row['collection_config'][k]==c[k] for k in ('schema','physics_hz','decimation','history'))
    for name, sha in collection['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, f'Stale dataset: {name}'
    train,x,y,ends=dataset('train',dest)
    val,vx,vy,ve=dataset('validation',dest)
    predictor=Predictor()
    training_source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__).resolve(), ROOT/'src/airbot_tt/pace_predictor.py', ROOT/'src/airbot_tt/pace_guidance.py']}
    receipt_path=output/'training.json'
    if args.stage in ('train', 'all'):
        curve=[]; best=float('inf')
        for epoch in range(c['predictor']['epochs']):
            loss=predictor.fit_epoch(x,y)
            score=metric(predictor.predict(vx).numpy(),val['truth'][ve])['rmse_m']
            curve.append(dict(epoch=epoch+1,train_mse=loss,validation_rmse_m=score))
            if score < best:
                best=score; predictor.save(output/'predictor.pt')
            print(f'epoch {epoch+1}: train MSE {loss:.6f}, validation {score*1000:.2f} mm',flush=True)
        receipt=dict(source_sha256=training_source, status='PASS' if best<=c['gate']['predictor_rmse_m'] else 'FAIL',
                     config=c, learning_curve=curve, best_validation_rmse_m=best,
                     training_seconds=time.monotonic()-started,
                     dataset_sha256=hashlib.sha256((dest/'dataset.json').read_bytes()).hexdigest(),
                     checkpoint_sha256=hashlib.sha256((output/'predictor.pt').read_bytes()).hexdigest())
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        if receipt['status']!='PASS':
            raise SystemExit('Development gate failed; held-out remains unopened.')
        if args.stage=='train':
            return
    receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='PASS' or receipt['config']!=c or receipt['source_sha256']!=training_source:
        raise ValueError('Training gate/config mismatch')
    for path,key in ((dest/'dataset.json','dataset_sha256'),(output/'predictor.pt','checkpoint_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt[key]:
            raise ValueError(f'Training receipt mismatch: {path}')
    if (output/'validation.json').exists():
        raise ValueError('Evaluation already exists; preserve original held-out evidence')
    curve=receipt['learning_curve']
    predictor.load(output/'predictor.pt')
    # Held-out is opened only after checkpoint selection, never used for fitting.
    held,hx,hy,he=dataset('heldout',dest)
    learned=predictor.predict(hx).numpy()
    estimated=[]; constant=[]; common=[]
    dt=c['decimation']/c['physics_hz']
    times=np.arange(-4,1)*dt
    for i,history in enumerate(hx.reshape(-1,5,3)):
        # Causal least-squares local acceleration fit; no true velocity access.
        coefficients=np.polyfit(times,history,2)
        v=coefficients[1]
        forecast=apex_teacher(np.r_[history[-1],v],bounced=bool(v[2]>0))
        if forecast['valid']:
            common.append(i); estimated.append(forecast['point'])
            constant.append(history[-1]+v*forecast['time_s'])
    common=np.array(common,dtype=int)
    truth=held['truth'][he]
    report=dict(scope=collection['scope'], config=c, checkpoint_epoch=predictor.updates,
                training_seconds=receipt['training_seconds'], learning_curve=curve,
                collection=collection, checkpoint_sha256=hashlib.sha256((output/'predictor.pt').read_bytes()).hexdigest(),
                samples=dict(train=len(x), validation=len(vx), heldout=len(hx), common=len(common),
                             rejected_heldout_samples=len(held['time'])-len(hx)),
                learned_all=metric(learned,truth), learned_common=metric(learned[common],truth[common]),
                constant_velocity=metric(constant,truth[common]), estimated_physics=metric(estimated,truth[common]),
                teacher=metric(hy,truth), training_mean=metric(np.tile(y.mean(0),(len(truth),1)),truth),
                coverage_common=float(len(common)/len(hx)),
                per_serve_rmse_m={str(int(s)):metric(learned[held['serve'][he]==s],truth[held['serve'][he]==s])['rmse_m']
                                  for s in np.unique(held['serve'][he])})
    # Diagnostics only, never used to select/refit the checkpoint.
    report['synthetic_observation_diagnostics']={}
    rng=np.random.default_rng(6190)
    for sigma in (.002,.007):
        noisy=held['state'][:,:3]+rng.normal(0,sigma,held['state'][:,:3].shape)
        nx,_,ne=windows(noisy,held['time'],held['serve'],np.ones(len(noisy),bool),held['target'],held['valid'])
        report['synthetic_observation_diagnostics'][f'noise_std_{sigma}_m']=metric(predictor.predict(nx).numpy(),held['truth'][ne])
    delayed=np.roll(held['state'][:,:3],1,axis=0)
    delay_valid=np.r_[False,held['serve'][1:]==held['serve'][:-1]]
    dx,_,de=windows(delayed,held['time'],held['serve'],delay_valid,held['target'],held['valid'])
    report['synthetic_observation_diagnostics']['one_policy_step_delay']=metric(predictor.predict(dx).numpy(),held['truth'][de])
    # Truth is used ONLY to label diagnostic strata, never in baseline inputs.
    cross_bounce=np.array([np.any((held['state'][e-4:e,5]<0)&(held['state'][e-3:e+1,5]>0)) for e in he])
    phases=np.where(cross_bounce,'across_bounce',np.where(held['state'][he,5]>0,'post_bounce','pre_bounce'))
    report['by_phase']={}
    for phase in np.unique(phases):
        mask=phases[common]==phase
        report['by_phase'][phase]=dict(learned=metric(learned[common][mask],truth[common][mask]),
                                      estimated_physics=metric(np.array(estimated)[mask],truth[common][mask]),
                                      constant_velocity=metric(np.array(constant)[mask],truth[common][mask]))
    valid=held['valid']
    teacher_error=np.linalg.norm(held['target'][valid]-held['truth'][valid],axis=1)
    teacher_time_error=abs(held['teacher_time'][valid]-held['truth_time'][valid])
    report['teacher_all_valid']=dict(samples=int(valid.sum()),p95_position_m=float(np.percentile(teacher_error,95)),
                                    p95_time_s=float(np.percentile(teacher_time_error,95)))
    # Measured heldout history drives the actual hook, before obs is returned.
    hook=PredictionHook(predictor,1)
    hook_predictions=[]; hook_valid=[]
    for p,t,s in zip(held['state'][:,:3],held['time'],held['serve']):
        out=hook.observe(p[None], [t], [s], [True])
        hook_predictions.append(out[0,:3].numpy()); hook_valid.append(bool(out[0,3]))
    hook_predictions=np.array(hook_predictions)
    hook_error=float(np.max(abs(hook_predictions[he]-learned)))
    # Necessary geometric bound, not IK or a certificate of reachability inside.
    xml=ET.parse(ROOT/'assets/airbot_play/airbot_play.urdf').getroot()
    link_lengths=[np.linalg.norm(np.fromstring(j.find('origin').get('xyz'),sep=' ')) for j in xml.findall('joint')]
    geometry=json.loads((ROOT/'assets/derived/racket/geometry.json').read_text())
    tcp=np.array(geometry['racket_tcp_candidate']['position_link6_m'])
    bound=float(sum(link_lengths)+np.linalg.norm(tcp))
    base=np.array(json.loads((ROOT/'configs/single_shot.json').read_text())['base_position_m'])
    _,unique=np.unique(held['serve'][he],return_index=True)
    distance=np.linalg.norm(truth[unique]-base,axis=1)
    report['apex_workspace_diagnostic']=dict(serves=len(unique),base_to_tcp_upper_bound_m=bound,
        min_base_to_apex_m=float(distance.min()),max_base_to_apex_m=float(distance.max()),
        definitely_outside_count=int((distance>bound).sum()),
        meaning='Triangle-inequality necessary bound; no target replacement or serve exclusion.')
    report['hook_max_difference_m']=hook_error
    gates=dict(teacher_position=report['teacher_all_valid']['p95_position_m']<=c['gate']['teacher_p95_position_m'],
               teacher_time=report['teacher_all_valid']['p95_time_s']<=c['gate']['teacher_p95_time_s'],
               predictor_30mm=report['learned_all']['rmse_m']<=c['gate']['predictor_rmse_m'],
               better_than_constant=report['learned_common']['rmse_m']<report['constant_velocity']['rmse_m'],
               all_serves=all(len(np.unique(dataset(s,dest)[0]['serve']))==c['splits'][s]['serves'] for s in c['splits']),
               hook=hook_error<1e-6 and bool(np.all(np.array(hook_valid)[he])))
    report['checks']=gates
    report['status']='PASS' if all(gates.values()) else 'FAIL'
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__),ROOT/'src/airbot_tt/pace_predictor.py',ROOT/'configs/pace_sources.json']}
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','checks','learned_all','constant_velocity','training_mean','coverage_common')},indent=2))
    if report['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
