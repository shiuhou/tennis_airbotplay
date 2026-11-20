"""Compare completed independent-process ball acceptance reports."""

import hashlib
import json
import math
from pathlib import Path


def compare(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    result = {"status": "FAIL", "reports_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "convergence": []}
    try:
        assert len(reports) == 3, "Need all three timestep reports"
        assert all(r["status"] == "PASS" and r["process_exit_code"] == 0 for r in reports), "A PhysX run did not pass"
        assert len({r["run_token"] for r in reports}) == 3, "Expected three independent runs"
        for key in ("config", "source_sha256", "commit", "paddle_mesh_sha256", "paddle_collision", "device"):
            assert all(r[key] == reports[0][key] for r in reports), f"Inconsistent {key} across runs"
        steps = [r["timesteps"][0] for r in reports]
        assert [s["dt_s"] for s in steps] == [1 / 240, 1 / 480, 1 / 960], "Wrong timestep sequence"
        for coarse, fine in zip(steps, steps[1:]):
            errors = {"coarse_dt_s": coarse["dt_s"], "fine_dt_s": fine["dt_s"]}
            for prefix in ("table", "paddle"):
                speed = abs(coarse[f"{prefix}_outgoing_m_s"] - fine[f"{prefix}_outgoing_m_s"]) / abs(fine[f"{prefix}_outgoing_m_s"])
                timing = abs(coarse[f"{prefix}_contact_s"] - fine[f"{prefix}_contact_s"])
                errors[prefix + "_relative_speed_change"] = speed
                errors[prefix + "_contact_time_change_s"] = timing
                assert speed <= .02 and timing <= 2 * coarse["dt_s"], f"{prefix} not converged: {speed}, {timing}"
            landing = math.dist(coarse["landing_xy_m"], fine["landing_xy_m"])
            errors["landing_change_m"] = landing
            result["convergence"].append(errors)
            assert landing <= .01, f"Landing not converged: {landing}"
        result.update(status="PASS", accepted_dt_s=steps[0]["dt_s"], config=reports[0]["config"],
                      source_sha256=reports[0]["source_sha256"], scope=reports[0]["scope"])
    except (AssertionError, KeyError, ValueError, ZeroDivisionError) as error:
        result["error"] = str(error)
    return result


def compare_impact(paths):
    """Same convergence thresholds, but for a driven articulated racket."""
    reports = [json.loads(path.read_text()) for path in paths]
    result = dict(status='FAIL', reports_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, convergence=[])
    try:
        frequencies=[r['physics_hz'] for r in reports]
        assert frequencies in ([240,480,960],[240,480,960,1920])
        assert all(r['status']=='PASS' and r['process_exit_code']==0 for r in reports)
        assert len({r['run_token'] for r in reports})==len(reports)
        assert all(r['source_sha256']==reports[0]['source_sha256'] for r in reports)
        result['relative_normal_energy'] = []
        for report in reports:
            for trial in report['trials']:
                normal, surface = trial['normal'], trial['surface_velocity_m_s']
                def normal_speed(velocity):
                    return sum((v-u)*n for v,u,n in zip(velocity,surface,normal))
                incoming = normal_speed(trial['incoming_velocity_m_s'])
                outgoing = normal_speed(trial['outgoing_velocity_m_s'])
                expected = normal_speed(trial['expected_velocity_m_s'])
                error = abs(outgoing*outgoing-expected*expected)/(expected*expected)
                result['relative_normal_energy'].append(dict(hz=report['physics_hz'],case=trial['case'],
                    measured_ratio=(outgoing/incoming)**2,expected_ratio=(expected/incoming)**2,relative_error=error))
                # Squared velocity: 5% velocity error permits about 10% energy error.
                assert error <= .1025, 'Relative normal impact energy mismatch'
        pair_pass = {}
        for coarse,fine in zip(reports,reports[1:]):
            pair_pass[coarse['physics_hz']] = True
            assert [t['case'] for t in coarse['trials']]==[t['case'] for t in fine['trials']]
            for a,b in zip(coarse['trials'],fine['trials']):
                error=dict(case=a['case'],coarse_hz=coarse['physics_hz'],
                           relative_velocity_change=math.dist(a['outgoing_velocity_m_s'],b['outgoing_velocity_m_s'])/math.dist(b['outgoing_velocity_m_s'],[0,0,0]),
                           landing_change_m=math.dist(a['landing_m'][:2],b['landing_m'][:2]),
                           contact_time_change_s=abs(a['time_s']-b['time_s']))
                result['convergence'].append(error)
                passed = (error['relative_velocity_change']<=.02 and error['landing_change_m']<=.01
                          and error['contact_time_change_s']<=2/coarse['physics_hz'])
                error['status'] = 'PASS' if passed else 'FAIL'
                pair_pass[coarse['physics_hz']] &= passed
        assert pair_pass[frequencies[-2]], 'Finest compared pair does not converge'
        accepted=frequencies[-2]
        for hz in reversed(frequencies[:-2]):
            if not pair_pass[hz]:break
            accepted=hz
        result.update(status='PASS', accepted_hz=accepted,
                      scope='accepted frequency only; failed coarse cases retained')
    except (AssertionError,KeyError,ValueError,ZeroDivisionError) as error:
        result['error']=str(error)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--impact", action="store_true")
    parser.add_argument("--finest-hz", type=int, choices=[960,1920], default=960)
    args = parser.parse_args()
    report = (compare_impact([args.directory / f"impact_{hz}.json" for hz in ([240,480,960,1920] if args.finest_hz==1920 else [240,480,960])]) if args.impact
              else compare([args.directory / f"dt_{hz}.json" for hz in (240, 480, 960)]))
    (args.directory / ("impact_convergence.json" if args.impact else "convergence.json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], "ball timestep convergence", report.get("error", ""))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
