"""Two fixed physical episodes: new BC 7101 / ball 9378, autonomous vs tick12 takeover.

Reuses the frozen recovery recorder and original physics. No fitting or state injection.
"""
import argparse
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-bc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['probe', 'remaining'], required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output directory; existing evidence is protected')
    app = AppLauncher(args, fast_shutdown=True).app

import numpy as np
import torch

from airbot_tt.config import ROOT
from airbot_tt.pace_bc import bc_source_hashes, sha256
from pace_recovery import COMMAND, SwitchPolicy, arrays, provenance, read, save

SWITCH = 12
RAW_ID = 9378
REMAINING = ((7101, 5255), (7101, 12250), (7102, 14852), (7103, 10818))
FIELDS = {'q_rad': 'measured_q_minus_nominal', 'qd_rad_s': 'measured_qd',
          'previous_action': 'previous_action'}


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def episode(env, policy, script, switch, bank_index=0):
    from pace_bank import _MotionCapture
    # Retain each ball's historical spatial index; padding repeats this ball only.
    scored = bank_index
    raw_id = env.serve_bank.rows[bank_index]['id']
    env.reset(env.ids, bank_indices=np.full(env.n, bank_index, dtype=np.int64))
    np.testing.assert_allclose(env.state()[scored].cpu().numpy(), env.serve_bank.states[bank_index], atol=2e-6, rtol=0)
    assert not env.ball.data.root_ang_vel_w.any()
    controller = SwitchPolicy(env, policy, [script]*(scored+1), switch)
    capture = _MotionCapture(env, scored+1)
    original_step, collecting = env.physics_step, env.prediction_collect
    env.physics_step, env.prediction_collect = capture.step, False
    obs, _ = env.observations()
    row = None
    normal_writes, padding_resets = 0, 0
    switch_snapshot = None
    try:
        with torch.no_grad():
            for _ in range(int(np.ceil(env.shot['timeout_s'] / env.policy_dt)) + 2):
                if controller.tick == SWITCH:
                    switch_snapshot = capture.snapshot()[scored]
                writes = env.state_writes
                obs, _, _, done, info = env.step(controller.act_inference(obs), auto_reset=False)
                normal_writes += env.state_writes - writes
                assert normal_writes == 0
                if done[scored]:
                    row = next(r for r in info['episodes'] if r['env'] == scored)
                    break
                # Finished padding may reset; the scored arm and its history never do.
                ids = done.nonzero().flatten()
                if ids.numel():
                    assert (ids != scored).all()
                    env.reset(ids, bank_indices=np.full(len(ids), bank_index, dtype=np.int64))
                    padding_resets += len(ids)
                    obs, _ = env.observations()
    finally:
        env.physics_step, env.prediction_collect = original_step, collecting
    assert row is not None and capture.finished[scored] and row['raw_id'] == raw_id
    row = dict(row, motion=capture.rows[scored])
    trace = arrays(controller.records[scored])
    np.testing.assert_array_equal(trace['tick'], np.arange(len(trace['tick'])))
    np.testing.assert_allclose(trace['elapsed_s'], trace['tick']/48, atol=2e-5, rtol=0)
    assert np.unique(trace['serve_id']).size == 1 and trace['prehit'][:SWITCH+1].all()
    np.testing.assert_array_equal(trace['previous_action'][1:], trace['action'][:-1])
    np.testing.assert_array_equal(trace['action'][:switch], trace['learner_action'][:switch])
    np.testing.assert_array_equal(trace['action'][switch:], trace['script_action'][switch:])
    assert row['hit'] == bool(row['motion']['impact'])
    if row['hit']:
        last = np.flatnonzero(trace['prehit'])[-1]
        assert trace['elapsed_s'][last] < row['motion']['impact']['time_s'] <= trace['elapsed_s'][last]+1/48+2e-5
    return row, trace, dict(normal_motion_state_writes=normal_writes, padding_resets=padding_resets,
                            switch_snapshot=switch_snapshot)


def compare(traces, references, policy, device):
    """Same-time distances over actual shared pre-contact, labeled suffix support only."""
    rows, reference_predictions = [], {}
    for switch, ref in references.items():
        with torch.no_grad():
            predicted = policy.actor(torch.as_tensor(ref['actual_observation'], device=device)).cpu().numpy()
        reference_predictions[f'reference_t{switch}_new_actor_action'] = predicted
        for name, trace in traces.items():
            n = min(len(trace['tick']), len(ref['tick']))
            valid = (trace['prehit'][:n] & ref['prehit'][:n] & (trace['tick'][:n] >= SWITCH))
            ticks = np.flatnonzero(valid)
            assert len(ticks) and ticks[0] == SWITCH
            np.testing.assert_array_equal(ticks, np.arange(SWITCH, ticks[-1]+1))
            row = dict(condition=name, reference_switch_tick=switch, first_tick=int(ticks[0]),
                       last_tick=int(ticks[-1]), samples=len(ticks), switch={}, common_precontact={})
            for label, key in FIELDS.items():
                delta = trace[key][:n] - ref[key][:n]
                row['switch'][label] = rmse(delta[SWITCH])
                row['common_precontact'][label] = dict(rmse=rmse(delta[ticks]),
                    final_rmse=rmse(delta[ticks[-1]]), max_rmse=float(np.sqrt(np.mean(delta[ticks].astype(float)**2, axis=1)).max()))
            row['switch']['new_actor_request_vs_script_rmse'] = rmse(trace['learner_action'][SWITCH]-trace['script_action'][SWITCH])
            row['switch']['reference_new_actor_vs_label_rmse'] = rmse(predicted[SWITCH]-ref['script_action'][SWITCH])
            row['common_precontact']['new_actor_request_vs_script_rmse'] = rmse(trace['learner_action'][ticks]-trace['script_action'][ticks])
            row['common_precontact']['reference_new_actor_vs_label_rmse'] = rmse(predicted[ticks]-ref['script_action'][ticks])
            rows.append(row)
    return rows, reference_predictions


def probe(original_bc, output, device):
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    from train_pace import original_ppo

    started = time.monotonic()
    original, inputs, _ = provenance(original_bc)
    recovery = original_bc.parent / 'recovery_bc'
    fitted = recovery / 'fit'
    fit = read(fitted/'results.json')
    fit_protocol = read(fitted/'protocol.json')
    prior = read(fitted/'bc_7101.json')
    stage = read(recovery/'recovery.json')
    assert fit['status'] == stage['status'] == 'PASS' and fit['process_exit_code'] == stage['process_exit_code'] == 0
    assert fit['contact_gate'] == 'FAIL' and [r['contacts'] for r in fit['runs']] == [2,4,4]
    assert fit['fit_protocol_sha256'] == sha256(fitted/'protocol.json')
    assert fit_protocol['cfg'] == original['cfg'] and fit_protocol['source_sha256'] == source_hashes()
    assert fit_protocol['bc_source_sha256'] == bc_source_hashes()
    assert stage['protocol_sha256'] == sha256(recovery/'protocol.json')
    assert stage['traces_sha256'] == sha256(recovery/'recovery_traces.npz')
    assert stage['recoveries_sha256'] == sha256(recovery/'recoveries.pt')
    assert prior == fit['runs'][0] and prior['actor_sha256'] == sha256(fitted/'actor_7101.pt')
    assert all(sha256(ROOT/name) == expected for name, expected in fit_protocol['input_sha256'].items())
    inputs.update(fit_protocol['input_sha256'])
    for path in (recovery/'recovery.json', recovery/'recovery_traces.npz', recovery/'recoveries.pt',
                 fitted/'results.json', fitted/'protocol.json', fitted/'bc_7101.json', fitted/'actor_7101.pt',
                 ROOT/'tests/pace_recovery.py', Path(__file__), ROOT/'scripts/run_recovery_bc.sh'):
        inputs[str(path.resolve())] = sha256(path)
    assert all(sha256(ROOT/name) == expected for name, expected in inputs.items())
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='airbot-recovery-probe-v1', raw_id=RAW_ID, seed=7101, switch_tick=SWITCH,
        scored_episodes=2, conditions=['autonomous','takeover'], num_envs=16, padding_scored=False,
        control='Original A, six residual +/-0.6rad, 48Hz policy / 960Hz physics; actor mean, current real history.',
        takeover='Original same-time script, retaining real q/qd/history/previous raw action/controller memory.',
        comparison='All three same-ball recovery suffixes; per-field six-joint RMSE, matched tick, common precontact only.',
        decision='Contact rescue permits proposing new-policy recovery data; no rescue motivates timing/range diagnosis. Neither proves a unique cause.',
        bc_updates=0, ppo_transitions=0, input_sha256=inputs, source_sha256=source_hashes(),
        bc_source_sha256=bc_source_hashes(), selected_videos=[], scope='Two episodes on one inspected training ball. Original three-seed BC gate unchanged.')
    save(output/'protocol.json', protocol)
    checkpoint = torch.load(fitted/'actor_7101.pt', map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == 7101 and checkpoint['protocol_sha256'] == sha256(fitted/'protocol.json')
    source = read(ROOT/'.agents/runs/foundation/deepmind_audit/position_search.json')
    script_rows = [r for r in source['rows'] if r['id'] == RAW_ID and r['accepted']]
    assert len(script_rows) == 1
    script = torch.tensor(script_rows[0]['preflight']['actions'], dtype=torch.float32, device=device)
    labels = torch.load(recovery/'recoveries.pt', map_location='cpu', weights_only=True)
    references = {}
    with np.load(recovery/'recovery_traces.npz') as archive:
        for switch in (6,9,12):
            prefix = f't{switch}_{RAW_ID}_'
            ref = {k[len(prefix):]:archive[k].copy() for k in archive.files if k.startswith(prefix)}
            labeled = next(r for r in labels if r['raw_id']==RAW_ID and r['switch_tick']==switch)
            ticks = labeled['ticks'].numpy()
            np.testing.assert_array_equal(labeled['observations'], ref['actual_observation'][ticks])
            np.testing.assert_array_equal(labeled['actions'], ref['action'][ticks])
            np.testing.assert_array_equal(ref['script_action'], script[torch.as_tensor(ref['tick'],device=device).clamp(max=len(script)-1)].cpu())
            np.testing.assert_array_equal(ticks, np.flatnonzero((ref['tick']>=switch) & ref['prehit']))
            references[switch] = ref
    torch.set_num_threads(1)
    torch.manual_seed(7101)
    ActorCritic, _ = original_ppo()
    cfg = original['cfg']['ppo']
    policy = ActorCritic(115,200,6,actor_hidden_dims=cfg['hidden'],critic_hidden_dims=cfg['hidden'],
                         activation='elu',init_noise_std=cfg['init_std']).to(device)
    initial = torch.load(Path(original['baseline_directory'])/'A_7101/initial.pt', map_location='cpu',weights_only=False)
    policy.load_state_dict(initial['policy'])
    policy.actor.load_state_dict(checkpoint['actor'])
    policy.eval()
    before = {k:v.detach().clone() for k,v in policy.state_dict().items()}
    env = PaceEnv(16,device,group='A',serve_bank=original_bc.parent/'manifest.json')
    assert env.policy_dt==1/48 and env.decimation==20 and env.online is None
    cases, traces, checks = [], {}, {}
    no_switch = int(np.ceil(env.shot['timeout_s']/env.policy_dt))+2
    for name, switch in [('autonomous',no_switch),('takeover',SWITCH)]:
        row, trace, check = episode(env,policy,script,switch)
        cases.append(dict(condition=name,episode=row))
        traces[name], checks[name] = trace, check
        print('PROBE',name,row['hit'],row['outcome'],flush=True)
    # States at tick12 are measured before selecting either continuation.
    prefix_errors = {}
    for key in ('actual_observation','measured_q_minus_nominal','measured_qd','previous_action','learner_action',
                *('command_'+field for field in COMMAND)):
        delta = traces['autonomous'][key][:SWITCH+1]-traces['takeover'][key][:SWITCH+1]
        prefix_errors[key] = float(np.max(np.abs(delta)))
        np.testing.assert_allclose(delta,0,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(traces['autonomous']['action'][:SWITCH],traces['takeover']['action'][:SWITCH])
    old, new = prior['evaluation']['episodes'][0], cases[0]['episode']
    assert old['hit']==new['hit']==False and old['outcome']==new['outcome']=='miss'
    baseline_errors = {}
    for key in ('duration_s','min_ball_distance_m','guidance_valid_s','plateau_s','guidance_sum','reward'):
        baseline_errors[key] = abs(new[key]-old[key])
        np.testing.assert_allclose(new[key],old[key],atol=2e-6,rtol=0)
    for key in ('ball','tcp','quaternion_wxyz','normal','tcp_velocity','omega'):
        np.testing.assert_allclose(new['motion']['terminal'][key],old['motion']['terminal'][key],atol=2e-6,rtol=0)
    comparison, extra = compare(traces,references,policy,device)
    data = {f'{name}_{key}':value for name,trace in traces.items() for key,value in trace.items()}
    data.update(extra)
    np.savez_compressed(output/'traces.npz',**data)
    assert all(torch.equal(v,before[k]) for k,v in policy.state_dict().items())
    assert all(sha256(ROOT/name)==expected for name,expected in inputs.items())
    assert protocol['source_sha256']==source_hashes() and protocol['bc_source_sha256']==bc_source_hashes()
    result = dict(status='PASS',scored_episodes=2,cases=cases,comparison=comparison,checks=checks,
        prefix_max_abs=prefix_errors,baseline_reproduced=True,baseline_scalar_abs_error=baseline_errors,
        weights_unchanged=True,bc_updates=0,ppo_transitions=0,bc_gate='FAIL',noise_status='NOT_RUN',ppo_status='NOT_RUN',
        recommendation='NEW_POLICY_RECOVERY_DATA_CANDIDATE' if cases[1]['episode']['hit'] else 'INVESTIGATE_RECOVERY_TIMING_RANGE',
        protocol_sha256=sha256(output/'protocol.json'),traces_sha256=sha256(output/'traces.npz'),elapsed_s=time.monotonic()-started)
    save(output/'probe.json',result)
    return result


def prior_probe_inputs(recovery):
    """Verify unchanged historical evidence, reconstructing only two extended tools.

    Original receipts and their expected digests remain authoritative. The inverse
    patches reproduce their exact old tool bytes; physics/model inputs must still
    match live files directly.
    """
    import hashlib
    import subprocess
    import tempfile
    previous = recovery/'probe'
    report, protocol = read(previous/'probe.json'), read(previous/'protocol.json')
    assert report['status']=='PASS' and report['process_exit_code']==0
    assert report['protocol_sha256']==sha256(previous/'protocol.json')
    assert report['traces_sha256']==sha256(previous/'traces.npz')
    patches = {'tests/pace_recovery_probe.py':'source_probe.patch',
               'scripts/run_recovery_bc.sh':'source_launcher.patch'}
    reconstructed = {}
    for name, expected in protocol['input_sha256'].items():
        path = (ROOT/name).resolve()
        relative = str(path.relative_to(ROOT))
        if relative in patches:
            patch = previous/patches[relative]
            with tempfile.TemporaryDirectory() as tmp:
                old = Path(tmp)/'original'
                subprocess.run(['patch','--batch','--fuzz=0','--reverse','--output',str(old),str(path)],
                               input=patch.read_bytes(),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                digest = hashlib.sha256(old.read_bytes()).hexdigest()
                assert digest==expected, relative
            reconstructed[relative] = dict(historical_sha256=expected,current_sha256=sha256(path),
                                          inverse_patch=str(patch.resolve()),patch_sha256=sha256(patch))
        else:
            assert sha256(path)==expected, name
    assert len(reconstructed)==2
    assert [(c['episode']['hit'],c['episode']['outcome']) for c in report['cases']]==[(False,'miss'),(True,'wrong_bounce')]
    return report, reconstructed


def remaining(original_bc, output, device):
    """Eight scored episodes, four fixed missed cases; historical 9378 is read only."""
    from airbot_tt.pace_env import PaceEnv
    from pace_m7 import source_hashes
    from pace_recovery import FIVE
    from train_pace import original_ppo
    started = time.monotonic()
    original, inputs, _ = provenance(original_bc)
    recovery = original_bc.parent/'recovery_bc'
    fitted = recovery/'fit'
    old_probe, reconstruction = prior_probe_inputs(recovery)
    fit, fit_protocol = read(fitted/'results.json'), read(fitted/'protocol.json')
    stage = read(recovery/'recovery.json')
    assert fit['status']==stage['status']=='PASS' and fit['process_exit_code']==stage['process_exit_code']==0
    assert fit['contact_gate']=='FAIL' and [r['contacts'] for r in fit['runs']]==[2,4,4]
    assert fit['fit_protocol_sha256']==sha256(fitted/'protocol.json')
    assert fit_protocol['cfg']==original['cfg'] and fit_protocol['source_sha256']==source_hashes()
    assert fit_protocol['bc_source_sha256']==bc_source_hashes()
    assert all(sha256(ROOT/k)==v for k,v in fit_protocol['input_sha256'].items())
    assert stage['protocol_sha256']==sha256(recovery/'protocol.json')
    assert stage['traces_sha256']==sha256(recovery/'recovery_traces.npz')
    assert stage['recoveries_sha256']==sha256(recovery/'recoveries.pt')
    inputs.update(fit_protocol['input_sha256'])
    paths = [recovery/n for n in ('recovery.json','protocol.json','recovery_traces.npz','recoveries.pt')]
    paths += [fitted/n for n in ('protocol.json','results.json')]
    paths += [recovery/'probe'/n for n in ('probe.json','protocol.json','traces.npz','source_probe.patch','source_launcher.patch')]
    paths += [ROOT/n for n in ('tests/pace_recovery.py','tests/pace_recovery_probe.py','scripts/run_recovery_bc.sh')]
    script_reports = [ROOT/f'.agents/runs/foundation/deepmind_audit/{n}.json'
                      for n in ('position_search','position_remaining')]
    paths += script_reports + [original_bc/'replay_comparison.json']
    for seed in (7101,7102,7103):
        prior=read(fitted/f'bc_{seed}.json')
        assert prior==fit['runs'][seed-7101] and prior['actor_sha256']==sha256(fitted/f'actor_{seed}.pt')
        paths += [fitted/f'actor_{seed}.pt',fitted/f'bc_{seed}.json']
    inputs.update({str(p.resolve()):sha256(p) for p in paths})
    script_eval=read(original_bc/'replay_comparison.json')['bank_evaluations']['script']
    reference_impacts={e['raw_id']:e['motion']['impact']['time_s'] for e in script_eval['episodes']}
    assert script_eval['hit_rate']==1 and script_eval['legal_return_rate']==0
    declared=[dict(seed=seed,raw_id=raw_id,canonical_env=FIVE.index(raw_id),
                   reference_script_impact_s=reference_impacts[raw_id],
                   reference_remaining_s=reference_impacts[raw_id]-SWITCH/48) for seed,raw_id in REMAINING]
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema='airbot-recovery-remaining-v1',switch_tick=SWITCH,cases=declared,
        scored_episodes=8,num_envs=16,conditions=['autonomous','takeover'],
        history='Prior 7101/9378 is linked and not replayed, including padding; original receipts remain unchanged.',
        padding='Every environment uses the current case ball; score only its historical canonical spatial index.',
        timing='Reference remaining time = original script first impact minus exact tick12/48, not a feasibility deadline. Actual takeover delay uses measured switch time and exists only for hits.',
        candidates='Only physically successful takeover pre-step obs/action suffixes, inclusive of first valid contact interval; no prefixes, padding, resets or post-hit samples. No fitting.',
        comparison='All three old same-ball recovery suffixes; same-tick six-joint q/qd/raw previous action RMSE. Full source fitting errors also reported.',
        stop='Only these eight episodes. No earlier/later switches, BC, noise, PPO or E integration.',
        input_sha256=inputs,historical_source_reconstruction=reconstruction,
        source_sha256=source_hashes(),bc_source_sha256=bc_source_hashes(),
        prior_probe=str((recovery/'probe/probe.json').resolve()),bc_updates=0,ppo_transitions=0)
    save(output/'protocol.json',protocol)
    labels=torch.load(recovery/'recoveries.pt',map_location='cpu',weights_only=True)
    demonstrations=torch.load(original_bc/'demonstrations.pt',map_location='cpu',weights_only=True)
    scripts={r['id']:torch.tensor(r['preflight']['actions'],dtype=torch.float32,device=device)
             for p in script_reports for r in read(p)['rows'] if r['accepted']}
    torch.set_num_threads(1)
    env=PaceEnv(16,device,group='A',serve_bank=original_bc.parent/'manifest.json')
    assert env.policy_dt==1/48 and env.decimation==20 and env.online is None
    ActorCritic,_=original_ppo()
    cfg=original['cfg']['ppo']
    runs,all_traces,candidates=[],{},[]
    with np.load(recovery/'recovery_traces.npz') as archive, torch.no_grad():
        for spec in declared:
            seed,raw_id,index=spec['seed'],spec['raw_id'],spec['canonical_env']
            checkpoint=torch.load(fitted/f'actor_{seed}.pt',map_location='cpu',weights_only=True)
            assert checkpoint['seed']==seed and checkpoint['protocol_sha256']==sha256(fitted/'protocol.json')
            torch.manual_seed(seed)
            policy=ActorCritic(115,200,6,actor_hidden_dims=cfg['hidden'],critic_hidden_dims=cfg['hidden'],
                              activation='elu',init_noise_std=cfg['init_std']).to(device)
            initial=torch.load(Path(original['baseline_directory'])/f'A_{seed}/initial.pt',map_location='cpu',weights_only=False)
            policy.load_state_dict(initial['policy']);policy.actor.load_state_dict(checkpoint['actor']);policy.eval()
            before={k:v.clone() for k,v in policy.state_dict().items()}
            reference={}
            fitting=[]
            for tick in (6,9,12):
                prefix=f't{tick}_{raw_id}_'
                ref={k[len(prefix):]:archive[k].copy() for k in archive.files if k.startswith(prefix)}
                label=next(r for r in labels if r['raw_id']==raw_id and r['switch_tick']==tick)
                ticks=label['ticks'].numpy()
                np.testing.assert_array_equal(ticks,np.flatnonzero((ref['tick']>=tick)&ref['prehit']))
                np.testing.assert_array_equal(label['observations'],ref['actual_observation'][ticks])
                np.testing.assert_array_equal(label['actions'],ref['action'][ticks])
                np.testing.assert_array_equal(ref['script_action'],scripts[raw_id][torch.as_tensor(ref['tick'],device=device).clamp(max=len(scripts[raw_id])-1)].cpu())
                prediction=policy.actor(label['observations'].to(device)).cpu().numpy()
                fitting.append(dict(source='recovery',switch_tick=tick,samples=len(ticks),
                                    action_rmse=rmse(prediction-label['actions'].numpy())))
                reference[tick]=ref
            selected=demonstrations['trajectory_ids']==index
            original_prediction=policy.actor(demonstrations['observations'][selected].to(device)).cpu().numpy()
            fitting.insert(0,dict(source='original',switch_tick=None,samples=int(selected.sum()),
                                 action_rmse=rmse(original_prediction-demonstrations['actions'][selected].numpy())))
            traces,cases,checks={},{},{}
            no_switch=int(np.ceil(env.shot['timeout_s']/env.policy_dt))+2
            for name,switch in [('autonomous',no_switch),('takeover',SWITCH)]:
                row,trace,check=episode(env,policy,scripts[raw_id],switch,bank_index=index)
                cases[name],traces[name],checks[name]=row,trace,check
                print('REMAINING',seed,raw_id,name,row['hit'],row['outcome'],flush=True)
            old=fit['runs'][seed-7101]['evaluation']['episodes'][index]
            new=cases['autonomous']
            assert old['raw_id']==raw_id and old['hit']==new['hit']==False and old['outcome']==new['outcome']
            baseline_errors={}
            for key in ('duration_s','min_ball_distance_m','guidance_valid_s','plateau_s','guidance_sum','reward'):
                baseline_errors[key]=abs(new[key]-old[key])
                np.testing.assert_allclose(new[key],old[key],atol=2e-6,rtol=0)
            for key in ('ball','tcp','quaternion_wxyz','normal','tcp_velocity','omega'):
                np.testing.assert_allclose(new['motion']['terminal'][key],old['motion']['terminal'][key],atol=2e-6,rtol=0)
            prefix_errors={}
            for key in ('actual_observation','measured_q_minus_nominal','measured_qd','previous_action','learner_action',
                        *('command_'+f for f in COMMAND)):
                delta=traces['autonomous'][key][:SWITCH+1]-traces['takeover'][key][:SWITCH+1]
                prefix_errors[key]=float(np.abs(delta).max())
                np.testing.assert_allclose(delta,0,atol=1e-6,rtol=0)
            np.testing.assert_array_equal(traces['autonomous']['action'][:SWITCH],traces['takeover']['action'][:SWITCH])
            assert checks['autonomous']['switch_snapshot']==checks['takeover']['switch_snapshot']
            comparison,extra=compare(traces,reference,policy,device)
            prefix=f's{seed}_{raw_id}_'
            all_traces.update({prefix+name+'_'+k:v for name,t in traces.items() for k,v in t.items()})
            all_traces.update({prefix+k:v for k,v in extra.items()})
            takeover=cases['takeover'];impact=takeover['motion']['impact']
            candidate_ticks=np.flatnonzero((traces['takeover']['tick']>=SWITCH)&traces['takeover']['prehit']) if takeover['hit'] else np.array([],dtype=int)
            if len(candidate_ticks):
                np.testing.assert_array_equal(candidate_ticks,np.arange(SWITCH,candidate_ticks[-1]+1))
                candidates.append(dict(seed=seed,raw_id=raw_id,switch_tick=SWITCH,
                    ticks=torch.from_numpy(candidate_ticks.copy()),
                    observations=torch.from_numpy(traces['takeover']['actual_observation'][candidate_ticks].copy()),
                    actions=torch.from_numpy(traces['takeover']['action'][candidate_ticks].copy())))
            assert all(torch.equal(v,before[k]) for k,v in policy.state_dict().items())
            runs.append(dict(**spec,episodes=cases,checks=checks,prefix_max_abs=prefix_errors,
                baseline_scalar_abs_error=baseline_errors,baseline_reproduced=True,comparison=comparison,
                old_support_fitting=fitting,candidate_samples=len(candidate_ticks),
                takeover_impact_s=impact['time_s'] if impact else None,
                actual_takeover_delay_s=impact['time_s']-checks['takeover']['switch_snapshot']['t'] if impact else None))
    np.savez_compressed(output/'traces.npz',**all_traces)
    torch.save(candidates,output/'candidates.pt')
    assert all(sha256(ROOT/name)==expected for name,expected in inputs.items())
    assert protocol['source_sha256']==source_hashes() and protocol['bc_source_sha256']==bc_source_hashes()
    prior_probe_inputs(recovery)
    result=dict(status='PASS',schema=protocol['schema'],scored_episodes=8,runs=runs,
        autonomous_contacts=sum(r['episodes']['autonomous']['hit'] for r in runs),
        takeover_contacts=sum(r['episodes']['takeover']['hit'] for r in runs),
        legal_returns=sum(e['outcome']=='legal_return' for r in runs for e in r['episodes'].values()),
        candidate_trajectories=len(candidates),candidate_samples=sum(r['candidate_samples'] for r in runs),
        reused_9378=dict(report=protocol['prior_probe'],report_sha256=sha256(recovery/'probe/probe.json'),
                         scored_episodes=old_probe['scored_episodes'],replayed_this_run=False),
        bc_gate='FAIL',noise_status='NOT_RUN',ppo_status='NOT_RUN',bc_updates=0,ppo_transitions=0,
        protocol_sha256=sha256(output/'protocol.json'),traces_sha256=sha256(output/'traces.npz'),
        candidates_sha256=sha256(output/'candidates.pt'),elapsed_s=time.monotonic()-started)
    save(output/'probe.json',result)
    return result


if __name__ == '__main__':
    code=0
    try:
        run = remaining if args.stage == 'remaining' else probe
        result=run(args.original_bc,args.output,args.device)
        print('RECOVERY_STAGE_COMPLETE',args.stage,result['status'],flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        code=1
        raise
    finally:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        app.app.post_quit(code)
        app.close()
