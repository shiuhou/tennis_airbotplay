"""Command line interface for the six-axis AIRBOT Play simulator."""

import argparse
import json
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import traceback

RUNS = Path(".agents/runs/foundation")


def report_path(args):
    return Path(args.report) if args.report else RUNS / f"{args.mode}.json"


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def interactive(control, app):
    from airbot_tt.kinematics import pose_dict, pose_matrix

    def print_json(value):
        print(json.dumps(value, allow_nan=False), flush=True)

    commands = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            commands.put(line.split())
        commands.put(["quit"])

    threading.Thread(target=read_commands, daemon=True).start()
    print("Commands: state | fk [q1 .. q6] | jacobian [q1 .. q6] | "
          "joint q1 .. q6 | velocity seconds v1 .. v6 | ik/move_pose x y z qw qx qy qz | "
          "trajectory FILE | trace FILE | collision | stop | reset | quit", flush=True)
    while app.is_running():
        try:
            words = commands.get_nowait()
        except queue.Empty:
            words = []
        if words == ["quit"]:
            return
        if words:
            command, *values = words
            try:
                if command == "state" and not values:
                    print_json(control.state())
                elif command in {"fk", "jacobian"} and len(values) in (0, 6):
                    q = [float(value) for value in values] if values else control.q()
                    print_json(pose_dict(control.model.fk(q)) if command == "fk" else control.model.jacobian(q).tolist())
                elif command == "joint" and len(values) == 6:
                    print_json(control.move_joint([float(value) for value in values]))
                elif command == "velocity" and len(values) == 7:
                    print_json(control.move_velocity([float(value) for value in values[1:]], float(values[0])))
                elif command in {"ik", "move_pose"} and len(values) == 7:
                    goal = pose_matrix([float(value) for value in values])
                    print_json(control.solve_pose(goal, execute=command == "move_pose"))
                elif command == "trajectory" and len(values) == 1:
                    data = json.loads(Path(values[0]).read_text())
                    if not isinstance(data, dict) or "waypoints_rad" not in data or set(data) - {"waypoints_rad", "durations_s"}:
                        raise ValueError("Expected waypoints_rad and optional durations_s.")
                    print_json(control.start_trajectory(data["waypoints_rad"], data.get("durations_s")))
                elif command == "trace" and len(values) == 1:
                    save_report(Path(values[0]), {"active": control.active, "samples": control.last_trace})
                    print_json({"samples_written": len(control.last_trace)})
                elif command == "collision" and not values:
                    print_json(control.checker.check(control.q()))
                elif command == "stop" and not values:
                    control.stop()
                    print_json(control.state())
                elif command == "reset" and not values:
                    control.reset()
                    print_json(control.state())
                else:
                    raise ValueError("Unknown command or incorrect number of arguments.")
            except (TypeError, ValueError, OSError, OverflowError) as exc:
                print_json({"error": str(exc)})
        control.step(render=control.sim.has_gui())


def dispatch(args, app):
    if args.mode == "pace-eval":
        from train_pace import reevaluate
        return reevaluate(args.device, args.pace_envs, args.pace_checkpoints, args.pace_output)
    if args.mode in ('pace-smoke', 'pace-ab'):
        from train_pace import run
        return run(args.device, args.pace_envs, 'smoke' if args.mode=='pace-smoke' else 'ab', args.pace_output, args.pace_learning_rate, args.pace_serve_y_half_range, args.pace_groups, args.pace_serve_z)
    if args.mode == "pace-check":
        if args.pace_check == "update":
            from pace_update import diagnose
            return diagnose(args.device, args.pace_envs, args.pace_output)
        from pace_m7 import verify
        return verify(args.device, args.pace_envs, args.pace_check)
    if args.mode == "return-demo":
        from airbot_tt.recording import record_returns
        return record_returns(args.device or "cpu", Path(args.video))
    if args.mode == "return":
        from strike import verify_return
        return verify_return(args.device or "cpu", split=args.return_split)
    if args.mode == "impact":
        from strike import verify_impact
        return verify_impact(args.device or "cpu", hz=args.ball_hz or 480)
    if args.mode == "strike":
        from strike import verify
        return verify(args.device or "cpu")
    if args.mode == "pace-data":
        from pace_m6 import collect
        return collect(args.device or "cpu", args.pace_split)
    if args.mode == "prediction":
        from prediction import verify
        return verify(args.device or "cpu")
    if args.mode == "shot":
        from single_shot import verify
        return verify(args.device or "cpu")
    if args.mode == "ball":
        from ball_physics import verify
        result = verify(args.device or "cpu", dt=1 / (args.ball_hz or 240), paddle_collision=args.ball_collision)
        result.update(isaac_lab=Path("/workspace/isaaclab/VERSION").read_text().strip(),
                      isaac_sim=Path("/isaac-sim/VERSION").read_text().strip(), device=args.device)
        return result
    import numpy
    import scipy
    from airbot_tt.kinematics import Kinematics
    from airbot_tt.runtime import AirbotSimulation, URDF_IMPORTER_VERSION

    if args.tool_config:
        from airbot_tt.tool import racket_model
        model = racket_model(args.tool_config)
    else:
        model = Kinematics()
    obstacles = json.loads(Path(args.obstacles).read_text()) if args.obstacles else ()
    if args.mode == "verify":
        from foundation import verify_obstacles
        obstacles = verify_obstacles(model)
    if args.mode == "contact":
        from foundation import contact_obstacles
        obstacles = contact_obstacles(model, args.contact_body)
    control = AirbotSimulation(device=args.device or "cpu", obstacles=obstacles, render=args.mode == "demo",
                               allow_initial_contact=args.mode == "contact", model=model)
    if args.mode == "verify":
        from foundation import verify
        result = verify(control)
    elif args.mode == "contact":
        from foundation import contact
        result = contact(control, args.contact_body)
    elif args.mode == "demo":
        from airbot_tt.recording import record
        result = record(control, Path(args.video))
    elif args.mode == "state":
        for _ in range(24):
            control.step()
        if control.fault:
            raise RuntimeError(control.fault)
        result = {"state": control.state()}
        print(json.dumps(result["state"], indent=2), flush=True)
    else:
        interactive(control, app)
        result = {}
    result.update({
        "isaac_lab": Path("/workspace/isaaclab/VERSION").read_text().strip(), "isaac_sim": Path("/isaac-sim/VERSION").read_text().strip(),
        "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
        "urdf_importer": URDF_IMPORTER_VERSION, "dt_s": control.dt, "device": str(control.robot.device),
    })
    if model.has_racket:
        result["tool_profile"] = model.tool_profile
        result["tool_geometry"] = model.tool_geometry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=["verify", "contact", "demo", "state", "interactive", "ball", "shot", "prediction", "strike", "impact", "return", "return-demo", "pace-data", "pace-check", "pace-smoke", "pace-ab", "pace-eval"], default="interactive")
    parser.add_argument("--report")
    parser.add_argument("--pace-output", default=".agents/runs/foundation/m7/smoke")
    parser.add_argument("--pace-checkpoints", help="Existing A/B run directory for frozen policy re-evaluation")
    parser.add_argument("--pace-envs", type=int, default=2)
    parser.add_argument("--pace-serve-z", type=float, help="Initial serve height in metres; recorded experiment override, not a teacher change")
    parser.add_argument("--pace-serve-y-half-range", type=float, help="Single-factor serve Y range override in metres; leaves M5 config unchanged")
    parser.add_argument("--pace-groups", nargs="+", choices=["A","B"], help="Explicit groups for a controlled short experiment")
    parser.add_argument("--pace-learning-rate", type=float, help="Explicit PPO initial-rate experiment; recorded in protocol/checkpoint")
    parser.add_argument("--pace-check", choices=["lifecycle", "control", "reference", "parity", "update"], default="lifecycle")
    parser.add_argument("--pace-split", choices=["all", "heldout"], default="all")
    parser.add_argument("--return-split", choices=["development", "validation"], default="development")
    parser.add_argument("--ball-hz", type=int, choices=[240, 480, 960, 1920], help="Physics frequency for isolated ball acceptance")
    parser.add_argument("--ball-collision", choices=['triangles','compound'], default='triangles')
    parser.add_argument("--obstacles")
    parser.add_argument("--tool-config", help="Confirmed racket mass/distribution profile JSON")
    parser.add_argument("--contact-body", choices=["fixer", "paddle"], help="Tool body for isolated contact acceptance")
    parser.add_argument("--video", default=str(RUNS / "airbot_play.mp4"))
    token = os.environ.get("AIRBOT_RUN_TOKEN")
    if sys.argv[1:2] == ["--check-completion"]:
        expected, process_exit = sys.argv[2], int(sys.argv[3])
        args, _ = parser.parse_known_args(sys.argv[4:])
        try:
            report = json.loads(report_path(args).read_text())
            if report.get("run_token") == expected:
                report["process_exit_code"] = process_exit
                if process_exit and report["status"] != "FAIL":
                    report.update(status="FAIL", error=f"Kit exited with code {process_exit}.")
                save_report(report_path(args), report)
                if report["status"] == "PASS" and process_exit == 0:
                    return 0
        except (OSError, ValueError):
            pass
        print(f"Airbot run did not produce a completed PASS report at {report_path(args)}.", file=sys.stderr)
        return process_exit or 1

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if (args.pace_serve_z is not None or args.pace_learning_rate is not None or args.pace_serve_y_half_range is not None or args.pace_groups is not None) and args.mode not in ("pace-smoke", "pace-ab"):
        parser.error("PPO experiment overrides are only valid for pace-smoke / pace-ab")
    if args.mode == "pace-eval" and not args.pace_checkpoints:
        parser.error("pace-eval requires --pace-checkpoints")
    args.enable_cameras = args.enable_cameras or args.mode in ("demo", "return-demo")
    if args.mode in ("verify", "contact") and args.obstacles:
        parser.error("Acceptance modes supply their own test obstacles.")
    if args.contact_body and (args.mode != "contact" or not args.tool_config):
        parser.error("--contact-body requires --mode contact and --tool-config.")
    if args.mode in ("ball", "shot", "prediction", "strike", "impact", "return", "return-demo", "pace-data") and (args.tool_config or args.obstacles):
        parser.error("Ball and shot acceptance supply their own tool and scene configuration.")
    if args.ball_hz is not None and args.mode not in ("ball", "impact"):
        parser.error("--ball-hz requires --mode ball or impact.")
    if args.ball_hz == 1920 and args.mode != "impact":
        parser.error("1920 Hz is reserved for the moving-impact convergence diagnostic.")
    if args.ball_collision != "triangles" and args.mode != "ball":
        parser.error("--ball-collision requires --mode ball.")
    destination = report_path(args)
    report = {"run_token": token, "mode": args.mode, "status": "STARTING"}
    save_report(destination, report)
    app = AppLauncher(args, fast_shutdown=True).app
    exit_code = 0
    try:
        report.update(dispatch(args, app))
        if report.get("status") in ("FAIL", "NOT_RUN", "BLOCKED"):
            exit_code = 1
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        report.update(getattr(exc, "report", {}))
        report.update(status="FAIL", error=str(exc))
        exit_code = 1
    finally:
        # Kit's native fast shutdown can exit directly. The launcher also
        # checks the real process exit code before accepting this report.
        report.update(run_token=token, mode=args.mode, status=report.get("status") if exit_code else "PASS")
        save_report(destination, report)
        from isaaclab.sim import SimulationContext

        # Remove Lab's timeline STOP handler before Kit closes its stage.
        SimulationContext.clear_instance()
        app.app.post_quit(exit_code)
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
