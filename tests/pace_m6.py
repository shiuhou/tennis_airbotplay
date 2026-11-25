"""Independent 960 Hz PhysX flights for M6; no robot or learned model runs here."""
import hashlib
import json
from pathlib import Path
import numpy as np
from airbot_tt.ball_scene import BallTestbed
from airbot_tt.config import ROOT
from airbot_tt.pace_guidance import settings, apex_teacher


def collect(device='cpu', split_only='all', destination=None, serve_z_m=None):
    c = settings()
    distribution = json.loads((ROOT/'configs/strike.json').read_text())
    if serve_z_m is not None:
        if not np.isfinite(serve_z_m) or serve_z_m<=0:raise ValueError('Invalid serve height')
        distribution['serve_position_m'][2]=float(serve_z_m)
    bed = BallTestbed(1/c['physics_hz'], device, static_paddle=False)
    dest = Path(destination) if destination is not None else ROOT/'.agents/runs/foundation/m6'
    dest.mkdir(parents=True, exist_ok=True)
    reports = (json.loads((dest/'dataset.json').read_text())['splits']
               if split_only != 'all' else {})
    offset=0
    for split, spec in c['splits'].items():
        serve_offset=offset;offset+=spec['serves']
        if split_only != 'all' and split != split_only:
            continue
        if (dest/f'{split}.npz').exists():raise FileExistsError('Preserve existing dataset; choose a new destination')
        rng = np.random.default_rng(spec['seed'])
        rows = []
        truths = []; failures = []
        for serve in range(spec['serves']):
            initial = np.r_[np.array(distribution['serve_position_m']) + rng.uniform(-1,1,3)*distribution['position_half_range_m'],
                            np.array(distribution['serve_velocity_m_s']) + rng.uniform(-1,1,3)*distribution['velocity_half_range_m_s']]
            bed.reset(initial[:3], initial[3:])
            state = initial.copy(); bounced = False; apex = None; samples = []
            for step in range(round(1.5*c['physics_hz'])):
                t = step/c['physics_hz']
                if step % c['decimation'] == 0:
                    teacher = apex_teacher(state, bounced=bounced)
                    samples.append((state.copy(), t, teacher))
                previous = state.copy()
                state, force = bed.step()
                if force[2] > 1e-5:
                    bounced = True
                elif bounced and previous[5] > 0 >= state[5]:
                    fraction = previous[5]/(previous[5]-state[5])
                    apex = previous[:3]+fraction*(state[:3]-previous[:3])
                    apex_time = t+fraction/c['physics_hz']
                    break
            if apex is None:
                failures.append(serve)
                continue
            truths.append(apex.tolist())
            for measured, t, teacher in samples:
                rows.append((measured, t, serve_offset+serve, teacher, apex, apex_time))
            if (serve+1)%32==0:
                print(f'M6 PhysX {split}: {serve+1}/{spec["serves"]}', flush=True)
        path = dest/f'{split}.npz'
        np.savez_compressed(path,
            state=np.array([r[0] for r in rows]), time=np.array([r[1] for r in rows]),
            serve=np.array([r[2] for r in rows]), target=np.array([r[3]['point'] for r in rows]),
            valid=np.array([r[3]['valid'] for r in rows]), truth=np.array([r[4] for r in rows]),
            truth_time=np.array([r[5] for r in rows]), teacher_time=np.array([r[1]+r[3]['time_s'] for r in rows]))
        reports[split] = dict(collection_config=c, seed=spec['seed'], all_serves=spec['serves'], observed_apex=len(truths),
                             failed_serves=failures, samples=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if failures:
            raise AssertionError(f'No apex for serves {split}: {failures}; no silent exclusion allowed.')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'configs/pace.json',ROOT/'configs/ball.json',ROOT/'configs/strike.json',Path(__file__),
             ROOT/'src/airbot_tt/pace_guidance.py',ROOT/'src/airbot_tt/ball_scene.py']}
    result=dict(status='PASS', splits=reports, source_sha256=hashes, distribution=distribution,
                physics_hz=c['physics_hz'], policy_hz=c['physics_hz']/c['decimation'],
                scope=f'Clean synthetic Z={distribution["serve_position_m"][2]} m; serve-level split before windows; independent PhysX flights without paddle')
    (dest/'dataset.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
