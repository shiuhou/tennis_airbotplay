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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = compare([args.directory / f"dt_{hz}.json" for hz in (240, 480, 960)])
    (args.directory / "convergence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], "ball timestep convergence", report.get("error", ""))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
