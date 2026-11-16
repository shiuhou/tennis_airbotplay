"""Reproducible acceptance scenarios against live Isaac Lab and PhysX."""

from contextlib import contextmanager
from copy import deepcopy
import math

import numpy as np

from airbot_tt.config import BASE_POSITION, COMMAND_ACCELERATION, COMMAND_SPEED, CONTACT_TRIP_N, HOME
from airbot_tt.kinematics import origin, pose_error, pose_matrix

GROUND_POSE = [0, -2.0, 0.5, 0, 0, 0]
SELF_POSE = [-0.89100636, -2.01577966, -0.06824791, 1.54435886, -1.48929816, -0.06011115]
CHECKS = (
    "fixed_base_and_limits",
    "urdf_mass_and_inertia",
    "six_axis_position",
    "inverse_kinematics_and_cartesian",
    "timed_waypoints_and_tracking",
    "joint_velocity",
    "invalid_unreachable_and_collisions",
    "stop_and_reset",
    "forward_kinematics_vs_physx",
)


class VerificationFailure(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["error"])


@contextmanager
def _check(report, name):
    try:
        yield
    except Exception as error:
        report["checks"][name] = "FAIL"
        report["status"] = "FAIL"
        report["error"] = f"{name}: {type(error).__name__}: {error}"
        raise VerificationFailure(report) from error
    else:
        report["checks"][name] = "PASS"


def _overlap_box(model, angle, name):
    q = np.array(HOME, dtype=float)
    q[0] += angle
    # Center the box in the link6 collision envelope, 14 mm behind the flange.
    center = (model.flange(q) @ [0.0, 0.0, -0.014, 1.0])[:3] + BASE_POSITION
    return [{"name": name, "center_m": center.tolist(), "size_m": [0.025] * 3}]


def verify_obstacles(model):
    return _overlap_box(model, 1.35, "path_block")


def contact_obstacles(model, body=None):
    if model.has_racket:
        center = (model.frames(HOME)[6] @ [0.021, 0.0, 0.110667, 1.0])[:3] + BASE_POSITION if body == "fixer" else model.fk(HOME)[:3, 3] + BASE_POSITION
        return [{"name": "contact_fixture", "center_m": center.tolist(), "size_m": [0.025]*3}]
    return _overlap_box(model, 0.0, "contact_fixture")


def _pose(state):
    return pose_matrix(list(state["position_m"]) + list(state["quaternion_wxyz"]))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _rejected(action, expected):
    try:
        action()
    except ValueError as error:
        message = str(error)
        _require(expected in message.lower(), f"Unexpected rejection: {message}")
        return message
    raise AssertionError("The invalid command was accepted.")


def _verify_inertia(model, robot, report):
    names = list(model.links)
    order = [robot.body_names.index(name) for name in names]
    view = robot.root_physx_view
    masses = view.get_masses()[0, order].cpu().numpy()
    coms = view.get_coms()[0, order, :3].cpu().numpy()
    tensors = view.get_inertias()[0, order].cpu().numpy().reshape(len(names), 3, 3)
    moments = np.linalg.eigvalsh(tensors)
    expected_masses, expected_coms, expected_moments = [], [], []
    dynamics = {}
    for index, name in enumerate(names):
        inertial = model.xml.find(f"link[@name='{name}']/inertial")
        mass = float(inertial.find("mass").attrib["value"])
        com = np.fromstring(inertial.find("origin").attrib["xyz"], sep=" ")
        values = inertial.find("inertia").attrib
        ixx, ixy, ixz, iyy, iyz, izz = (
            float(values[key]) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        )
        moment = np.linalg.eigvalsh([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        # This pinned articulation API returns a full tensor in link axes.
        # get_coms also carries the principal-axis quaternion; applying it here
        # would rotate the tensor a second time.
        actual_tensor = tensors[index]
        urdf_rotation = origin(inertial.find("origin"))[:3, :3]
        expected_tensor = urdf_rotation @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ urdf_rotation.T
        np.testing.assert_allclose(actual_tensor, expected_tensor, rtol=3e-4, atol=1e-8)
        expected_masses.append(mass)
        expected_coms.append(com)
        expected_moments.append(moment)
        dynamics[name] = {
            "mass_kg": float(masses[index]),
            "urdf_mass_kg": mass,
            "center_of_mass_link_m": coms[index].tolist(),
            "urdf_center_of_mass_link_m": com.tolist(),
            "principal_moments_kg_m2": moments[index].tolist(),
            "urdf_principal_moments_kg_m2": moment.tolist(),
            "inertia_tensor_link_kg_m2": actual_tensor.tolist(),
        }
    report["physx_link_dynamics"] = dynamics
    np.testing.assert_allclose(masses, expected_masses, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(coms, expected_coms, atol=1e-6)
    np.testing.assert_allclose(moments, expected_moments, rtol=2e-4, atol=1e-8)


def verify(control):
    model = control.model
    home = control.initial.copy()
    report = {
        "status": "RUNNING",
        "checks": dict.fromkeys(CHECKS, "NOT_RUN"),
        "physics_dt_s": control.dt,
        "joint_names": list(model.joint_names),
        "tracking_samples": [],
        "metrics": dict.fromkeys(
            (
                "max_fk_position_error_m",
                "max_fk_orientation_error_rad",
                "max_tracking_error_rad",
                "max_measured_velocity_rad_s",
                "max_base_position_drift_m",
                "max_base_orientation_drift_rad",
                "max_contact_force_N",
            ),
            0.0,
        ),
    }
    metrics = report["metrics"]
    base_ref = None
    step_count = 0
    phase_samples = []

    def advance(seconds, phase, allow_fault=False):
        nonlocal step_count, base_ref
        phase_samples.clear()
        for _ in range(math.ceil(seconds / control.dt)):
            control.step()
            step_count += 1
            _require(allow_fault or not control.fault, f"Unexpected motion fault: {control.fault}")
            q = control.q()
            rates = control.qd()
            target = control.target
            _require(np.all(q >= model.lower - 1e-6), f"Below joint limit: {q}")
            _require(np.all(q <= model.upper + 1e-6), f"Above joint limit: {q}")
            _require(np.isfinite(rates).all() and np.isfinite(target).all(), "Nonfinite feedback.")
            metrics["max_tracking_error_rad"] = max(
                metrics["max_tracking_error_rad"], float(np.max(np.abs(q - target)))
            )
            metrics["max_measured_velocity_rad_s"] = max(
                metrics["max_measured_velocity_rad_s"], float(np.max(np.abs(rates)))
            )
            if step_count % 12 != 0:
                continue
            state = control.state()
            metrics["max_contact_force_N"] = max(
                metrics["max_contact_force_N"], float(state["contact_max_N"])
            )
            raw_links = control.robot.data.body_link_state_w[0, [base_id, wrist_id], :7]
            base, wrist = (pose_matrix(p) for p in raw_links.cpu().numpy())
            if base_ref is None:
                base_ref = base
            before = pose_error(base, base_ref)
            metrics["max_base_position_drift_m"] = max(
                metrics["max_base_position_drift_m"], float(np.linalg.norm(before[:3]))
            )
            metrics["max_base_orientation_drift_rad"] = max(
                metrics["max_base_orientation_drift_rad"], float(np.linalg.norm(before[3:]))
            )
            measured_wrist = np.linalg.solve(base, wrist)
            observed_flange = measured_wrist @ model.flange_transform
            np.testing.assert_allclose(_pose(state["flange_base"]), observed_flange, atol=2e-6)
            observed_tcp = observed_flange @ model.tcp_transform
            if model.has_racket:
                paddle_id = control.robot.body_names.index("paddle")
                paddle_state = control.robot.data.body_link_state_w[0, paddle_id, :7].cpu().numpy()
                observed_tcp = np.linalg.solve(base, pose_matrix(paddle_state)) @ model.link6_tcp
            np.testing.assert_allclose(_pose(state["tcp_base"]), observed_tcp, atol=2e-6)
            error = pose_error(model.fk(q), observed_tcp)
            metrics["max_fk_position_error_m"] = max(
                metrics["max_fk_position_error_m"], float(np.linalg.norm(error[:3]))
            )
            metrics["max_fk_orientation_error_rad"] = max(
                metrics["max_fk_orientation_error_rad"], float(np.linalg.norm(error[3:]))
            )
            phase_samples.append(rates)
            report["tracking_samples"].append(
                {
                    "phase": phase, "time_s": step_count * control.dt,
                    "q_rad": q.tolist(), "target_rad": target.tolist(), "qd_rad_s": rates.tolist(),
                    "tcp_base": state["tcp_base"],
                }
            )

    start_writes = control.state_writes
    with _check(report, "fixed_base_and_limits"):
        robot = control.robot
        base_id = robot.body_names.index(model.links[0])
        wrist_id = robot.body_names.index(model.tip_link)
        _require(robot.is_fixed_base, "Articulation base is not fixed.")
        _require(set(robot.body_names) == set(model.links), f"Articulation links: {robot.body_names}")
        _require(set(robot.joint_names) == set(model.joint_names), f"Articulation joints: {robot.joint_names}")
        _require(
            robot.cfg.spawn.articulation_props.enabled_self_collisions,
            "PhysX self-collision is disabled.",
        )
        order = [robot.joint_names.index(name) for name in model.joint_names]
        limits = robot.data.joint_pos_limits[0, order].cpu().numpy()
        np.testing.assert_allclose(limits[:, 0], model.lower, atol=1e-5)
        np.testing.assert_allclose(limits[:, 1], model.upper, atol=1e-5)
        np.testing.assert_allclose(
            control.state()["base_world"]["position_m"], BASE_POSITION, atol=1e-6
        )
        advance(0.35, "hold")

    with _check(report, "urdf_mass_and_inertia"):
        _verify_inertia(model, robot, report)

    if model.has_racket:
        with _check(report, "tool_collision_and_tcp_frames"):
            import omni.usd
            from pxr import PhysxSchema, UsdGeom, UsdPhysics
            from airbot_tt.config import RACKET_CONTACT_OFFSET_M
            from airbot_tt.runtime import ROBOT_PRIM
            stage = omni.usd.get_context().get_stage()
            names, _ = control.sim_contact()
            _require({"fixer", "paddle"} <= set(names), "Missing tool contact sensors.")
            for body in ("fixer", "paddle"):
                prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/{body}")
                _require(prim.HasAPI(UsdPhysics.RigidBodyAPI), f"Missing rigid body {body}.")
            for path in control.tool_colliders:
                prim = stage.GetPrimAtPath(path)
                _require(prim.HasAPI(UsdPhysics.CollisionAPI), f"Missing collider {path}.")
                api = PhysxSchema.PhysxCollisionAPI(prim)
                _require(abs(api.GetContactOffsetAttr().Get()-RACKET_CONTACT_OFFSET_M) < 1e-8,
                         f"Wrong contact offset {path}.")
                _require(api.GetRestOffsetAttr().Get() == 0.0, f"Wrong rest offset {path}.")
            marker = UsdGeom.Xformable(stage.GetPrimAtPath(f"{ROBOT_PRIM}/paddle/RacketTCP"))
            np.testing.assert_allclose(np.array(marker.GetLocalTransformation()).T, model.link6_tcp, atol=1e-10)
            report["tool_collision_prims"] = control.tool_colliders
            report["tool_mass_kg"] = sum(report["physx_link_dynamics"][body]["mass_kg"] for body in ("fixer", "paddle"))

    with _check(report, "six_axis_position"):
        report["axes"] = []
        for index, name in enumerate(model.joint_names):
            excursions = []
            for delta in (0.09, -0.06):
                goal = home + delta * np.eye(6)[index]
                plan = control.move_joint(goal)
                advance(plan["duration_s"] + 0.35, f"{name} {delta:+.2f}")
                actual = control.q()
                np.testing.assert_allclose(actual, goal, atol=0.015)
                _require(np.sign(delta) * (actual[index] - home[index]) > 0.045, "Wrong joint motion.")
                excursions.append(
                    {
                        "goal_rad": goal.tolist(),
                        "actual_rad": actual.tolist(),
                        "max_error_rad": float(np.max(np.abs(actual - goal))),
                    }
                )
            plan = control.move_joint(home)
            advance(plan["duration_s"] + 0.25, f"{name} return")
            report["axes"].append({"joint": name, "excursions": excursions})

    with _check(report, "inverse_kinematics_and_cartesian"):
        seed = home + [0.04, -0.10, 0.12, 0.08, 0.17, -0.10]
        plan = control.move_joint(seed)
        advance(plan["duration_s"] + 0.25, "IK seed")
        desired = model.fk(seed + [0.05, -0.05, 0.08, 0.05, 0.06, -0.04])
        result = control.solve_pose(desired)
        report["ik"] = result
        _require(result["success"], f"IK failed: {result}")
        residual = pose_error(model.fk(result["q_rad"]), desired)
        metrics["ik_fk_position_error_m"] = float(np.linalg.norm(residual[:3]))
        metrics["ik_fk_orientation_error_rad"] = float(np.linalg.norm(residual[3:]))
        _require(metrics["ik_fk_position_error_m"] < 2e-4, "Invalid IK position solution.")
        _require(metrics["ik_fk_orientation_error_rad"] < 2e-3, "Invalid IK orientation solution.")
        plan = control.move_cartesian(desired)
        advance(plan["duration_s"] + 0.55, "Cartesian target")
        residual = pose_error(_pose(control.state()["tcp_base"]), desired)
        metrics["executed_ik_position_error_m"] = float(np.linalg.norm(residual[:3]))
        metrics["executed_ik_orientation_error_rad"] = float(np.linalg.norm(residual[3:]))
        _require(metrics["executed_ik_position_error_m"] < 0.003, "Cartesian position tracking error.")
        _require(metrics["executed_ik_orientation_error_rad"] < 0.015, "Cartesian orientation tracking error.")

    with _check(report, "timed_waypoints_and_tracking"):
        points = [
            home + [0.12, -0.13, 0.20, 0.12, 0.23, -0.13],
            home + [-0.13, -0.18, 0.15, -0.12, -0.14, 0.17],
            home,
        ]
        durations = [1.8, 5.0, 2.3]
        plan = control.start_trajectory(points, durations=durations)
        report["timed_trajectory"] = plan
        np.testing.assert_allclose(plan["duration_s"], sum(durations), atol=1e-8)
        for index, (seconds, point) in enumerate(zip(durations, points, strict=True)):
            advance(seconds, f"waypoint {index + 1}")
            np.testing.assert_allclose(control.q(), point, atol=0.015)
        advance(0.35, "waypoint settle")
        _require(control.trajectory is None, "Completed path remained active.")
        _require(bool(control.last_trace), "Controller trajectory trace is empty.")
        report["controller_trace_samples"] = len(control.last_trace)
        report["timed_trajectory_trace"] = [*control.last_trace[:-1:12], control.last_trace[-1]]
        for field, metric, limit in (
            ("target_velocity_rad_s", "max_commanded_velocity_rad_s", COMMAND_SPEED),
            ("target_acceleration_rad_s2", "max_commanded_acceleration_rad_s2", COMMAND_ACCELERATION),
        ):
            metrics[metric] = float(np.max(np.abs([sample[field] for sample in control.last_trace])))
            _require(metrics[metric] <= limit + 1e-6, f"Profile exceeded limit for {field}.")
        np.testing.assert_allclose(control.q(), home, atol=0.015)

    with _check(report, "joint_velocity"):
        start = control.q().copy()
        command = np.array([0.08, -0.05, 0.04, 0.06, 0.05, -0.035])
        duration = 1.4
        plan = control.move_velocity(command, duration)
        advance(duration + 0.4, "joint velocity")
        velocities = np.array(phase_samples)
        ramp = float(np.abs(command).max() / 0.3)
        expected = start + command * (duration - ramp)
        np.testing.assert_allclose(control.q(), expected, atol=0.015)
        observed = np.sign(command) * (velocities * np.sign(command)).max(axis=0)
        _require(np.all(observed * command > 0.0), f"Missing velocity feedback: {observed}")
        _require(np.all(np.abs(observed) > 0.5 * np.abs(command)), f"Weak velocity feedback: {observed}")
        report["velocity_command"] = {
            "plan": plan, "command_rad_s": command.tolist(), "observed_peak_rad_s": observed.tolist(),
            "expected_end_rad": expected.tolist(), "actual_end_rad": control.q().tolist(),
        }

    with _check(report, "invalid_unreachable_and_collisions"):
        before = deepcopy(control.state()["target_rad"])
        unreachable = pose_matrix([10, 0, 0, 1, 0, 0, 0])
        solve = control.solve_pose(unreachable)
        _require(not solve["success"], f"Unreachable pose solved: {solve}")
        excessive = model.upper.copy()
        excessive[0] += 0.01
        _require(("link5", "ground") in control.checker.check(GROUND_POSE), "Ground fixture missed.")
        self_pairs = control.checker.check(SELF_POSE)
        _require(
            ("base_link", "link4") in self_pairs and all(b != "ground" for _, b in self_pairs),
            f"Self-collision fixture failed: {self_pairs}",
        )
        clear_a = home + [0.65, 0, 0, 0, 0, 0]
        clear_b = home + [2.0, 0, 0, 0, 0, 0]
        middle = home + [1.35, 0, 0, 0, 0, 0]
        checker = control.checker
        _require(not checker.check(clear_a) and not checker.check(clear_b), "Obstacle endpoints not both clear.")
        _require(("link6", "path_block") in checker.check(middle), "Static obstacle fixture missed.")
        report["rejections"] = {
            "unreachable_pose": _rejected(lambda: control.move_cartesian(unreachable), "ik"),
            "out_of_bounds": _rejected(lambda: control.move_joint(excessive), "limit"),
            "nonfinite": _rejected(lambda: control.move_joint([np.nan] * 6), "finite"),
            "ground": _rejected(lambda: control.move_joint(GROUND_POSE), "ground"),
            # A mounted paddle can hit ground before reaching the self-colliding endpoint.
            # The endpoint's explicit base_link/link4 overlap is checked above.
            "self": _rejected(lambda: control.move_joint(SELF_POSE), "collision"),
            "box_between_clear_waypoints": _rejected(
                lambda: control.start_trajectory([clear_a, clear_b]), "path_block"
            ),
        }
        _require(control.trajectory is None and not control.fault, "Rejection changed motion state.")
        np.testing.assert_array_equal(control.state()["target_rad"], before)

    with _check(report, "stop_and_reset"):
        goal = home + [0.25, -0.10, 0.15, 0.08, 0.05, -0.08]
        plan = control.move_joint(goal)
        advance(min(plan["duration_s"] / 3, 0.65), "before stop")
        stop_q = control.q().copy()
        control.stop()
        _require(control.fault and control.trajectory is None, "Stop did not clear and latch the active path.")
        _require(control.state_writes == start_writes, "A normal motion wrote directly to the state.")
        np.testing.assert_allclose(control.state()["target_rad"], stop_q, atol=1e-6)
        _rejected(lambda: control.move_joint(goal), "reset")
        advance(0.55, "after stop", allow_fault=True)
        np.testing.assert_allclose(control.q(), stop_q, atol=0.015)
        control.reset()
        _require(not control.fault and control.trajectory is None, "Reset did not clear motion state.")
        np.testing.assert_allclose(control.q(), home, atol=1e-6)
        np.testing.assert_allclose(control.state()["target_rad"], home, atol=1e-6)
        np.testing.assert_allclose(control.state()["qd_rad_s"], 0.0, atol=1e-6)
        _require(control.state_writes == start_writes + 1, "Reset did not account for its state write.")
        report["direct_state_writes_during_motion"] = 0
        advance(0.4, "reset")

    with _check(report, "forward_kinematics_vs_physx"):
        _require(metrics["max_fk_position_error_m"] < 2e-5, f"FK position error: {metrics}")
        _require(metrics["max_fk_orientation_error_rad"] < 2e-4, f"FK rotation error: {metrics}")
        _require(metrics["max_base_position_drift_m"] < 1e-5, f"Base position drift: {metrics}")
        _require(metrics["max_base_orientation_drift_rad"] < 1e-5, f"Base rotation drift: {metrics}")
        _require(metrics["max_tracking_error_rad"] < 0.015, f"Tracking error: {metrics}")
        _require(metrics["max_measured_velocity_rad_s"] < 0.42, f"Measured velocity limit: {metrics}")

    report["status"] = "PASS"
    return report


def contact(control, body=None):
    """In an isolated scene, measure box overlap through real PhysX contacts."""
    report = {
        "status": "RUNNING",
        "checks": {"physx_contact_and_fault_latch": "NOT_RUN"},
        "physics_dt_s": control.dt,
        "scope": "Intentional tool/box overlap" if control.model.has_racket else "Intentional wrist/box overlap at initialization",
    }
    with _check(report, "physx_contact_and_fault_latch"):
        contact_body = (body or "paddle") if control.model.has_racket else "link6"
        _require(
            (contact_body, "contact_fixture") in control.checker.check(control.q()),
            "Contact test scene missed the initial wrist overlap.",
        )
        initial_writes = control.state_writes
        maximum = 0.0
        wrist_force = 0.0
        for _ in range(24):
            control.step()
            names, forces = control.sim_contact()
            forces = np.asarray(forces)
            _require(np.isfinite(forces).all(), f"Nonfinite contact forces: {forces}")
            magnitudes = np.linalg.norm(forces, axis=1)
            wrist_force = max(wrist_force, float(magnitudes[list(names).index(contact_body)]))
            if float(magnitudes.max()) > maximum:
                maximum = float(magnitudes.max())
                report["peak_forces_world_N"] = dict(zip(names, forces.tolist(), strict=True))
        report["peak_contact_force_N"] = maximum
        report[f"peak_{contact_body}_contact_force_N"] = wrist_force
        report["fault"] = control.fault
        _require(maximum > CONTACT_TRIP_N, f"No significant physical contact measured: {maximum} N")
        _require(wrist_force > CONTACT_TRIP_N, f"No significant wrist contact measured: {wrist_force} N")
        _require(bool(control.fault) and control.trajectory is None, "Physical contact did not latch a fault.")
        _require(control.state_writes == initial_writes, "Fault response wrote the dynamic state.")
        _rejected(lambda: control.move_joint(control.initial), "reset")
    report["status"] = "PASS"
    return report
