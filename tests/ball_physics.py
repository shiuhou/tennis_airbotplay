"""Live PhysX flight, contact and timestep-convergence acceptance.

Flight reference uses independent SciPy integration, never the production force
function. Impact checks use momentum/restitution and measured PhysX contact.
"""

import hashlib
import subprocess
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from airbot_tt.config import ROOT
from airbot_tt.ball_scene import BallTestbed
from foundation import _check, _require


def verify(device="cpu", dt=1 / 240, paddle_collision='triangles'):
    bed = BallTestbed(dt, device, static_paddle=paddle_collision)
    c = bed.config
    report = {"checks": {}, "config": c, "timesteps": [],
              "calibration": "NOT_RUN", "scope": "isolated ball and static CAD paddle; no driven-arm hit",
              "paddle_collision": paddle_collision,
              "paddle_mesh_sha256": bed.paddle_mesh_sha256,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                ("src/airbot_tt/ball.py", "src/airbot_tt/ball_scene.py", "src/airbot_tt/collision.py",
                                 "tests/ball_physics.py", "configs/ball.json")}}

    def trial(position, velocity, duration, drag=False, quaternion=(1, 0, 0, 0)):
        bed.reset(position, velocity, quaternion)
        rows, forces = [np.r_[position, velocity]], [np.zeros(3)]
        # Fail if normal stepping bypasses dynamics through a public state writer.
        with ExitStack() as guards:
            for method in ("write_root_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim"):
                guards.enter_context(patch.object(bed.ball, method, side_effect=AssertionError("State write outside reset")))
            for _ in range(round(duration / bed.dt)):
                state, force = bed.step(drag=drag)
                rows.append(state)
                forces.append(force)
        return np.array(rows), np.array(forces)

    with _check(report, "physical_ball_mass"):
        mass = float(bed.ball.root_physx_view.get_masses()[0, 0])
        _require(abs(mass - c["mass_kg"]) < 1e-9, f"Wrong PhysX ball mass: {mass}")

    result = {"dt_s": dt}
    report["timesteps"].append(result)
    with _check(report, f"flight_dt_{dt}"):
        p, v, duration = np.array([.5, .3, 2.0]), np.array([2., .3, .5]), .2
        states, forces = trial(p, v, duration)
        times = np.arange(len(states)) * dt
        gravity = np.array([0., 0., -c["gravity_m_s2"]])
        expected_p = p + times[:, None] * v + .5 * times[:, None] ** 2 * gravity
        expected_v = v + times[:, None] * gravity
        pe = float(np.max(np.linalg.norm(states[:, :3] - expected_p, axis=1)))
        ve = float(np.max(np.linalg.norm(states[:, 3:] - expected_v, axis=1)))
        _require(pe <= c["gravity_m_s2"] * dt * duration, f"Ballistic position error {pe}")
        _require(ve < 2e-4 and np.max(np.abs(forces)) == 0, f"Ballistic velocity/contact error {ve}")
        result.update(ballistic_position_error_m=pe, ballistic_velocity_error_m_s=ve)
        # Independent coupled 3-D quadratic-drag IVP, DOP853 tolerances well
        # below the simulator tolerance. No imported production RHS.
        k = c["air_density_kg_m3"] * np.pi * c["radius_m"] ** 2 * c["drag_coefficient"] / (2 * c["mass_kg"])
        def rhs(t, y):
            return np.r_[y[3:], gravity - k * np.sqrt(np.dot(y[3:], y[3:])) * y[3:]]
        reference = solve_ivp(rhs, (0, duration), np.r_[p, v], t_eval=times,
                              method="DOP853", rtol=1e-11, atol=1e-12).y.T
        states, forces = trial(p, v, duration, drag=True)
        pe = float(np.max(np.linalg.norm(states[:, :3] - reference[:, :3], axis=1)))
        ve = float(np.max(np.linalg.norm(states[:, 3:] - reference[:, 3:], axis=1)))
        _require(pe < .006 and ve < .02 and np.max(np.abs(forces)) == 0, f"Drag rollout error {pe}, {ve}")
        result.update(drag_position_error_m=pe, drag_velocity_error_m_s=ve)
        rotated, _ = trial(p, v, duration, drag=True, quaternion=(.5, .5, .5, .5))
        _require(np.max(np.abs(rotated - states)) < 1e-5, "World drag changes when the ball frame rotates")

    with _check(report, f"table_bounce_dt_{dt}"):
        states, forces = trial([.5, .3, 1.2], [0, 0, -2], .4)
        contacts = np.flatnonzero(forces[:, 2] > 1e-5)
        _require(len(contacts) > 0, "No table contact measured")
        j = int(contacts[0])
        outgoing = float(states[j, 5])
        ratio = outgoing / -float(states[j - 1, 5])
        _require(outgoing > 0 and abs(ratio - c["table_restitution"]) < .04, f"Table restitution {ratio}")
        _require(np.max(np.abs(states[:, 3:5])) < 1e-5, "Normal bounce creates tangential velocity")
        result.update(table_contact_s=j * dt, table_outgoing_m_s=outgoing, table_restitution=ratio)

    with _check(report, f"cad_paddle_impact_dt_{dt}"):
        states, forces = trial([-.6, 0, 1.16], [-3, 0, 0], .6)
        hits = np.flatnonzero(forces[:, 0] > 1e-5)
        _require(len(hits) > 0, "No CAD paddle contact measured")
        j = int(hits[0])
        ratio = float(states[j, 3] / -states[j - 1, 3])
        result.update(paddle_pre_contact_state=states[j - 1].tolist(),
                      paddle_post_contact_state=states[j].tolist(), paddle_contact_force_n=forces[j].tolist())
        _require(abs(ratio - c["paddle_restitution"]) < .04, f"Paddle restitution {ratio}")
        _require(abs(states[j, 4]) < .01, f"Normal paddle impact creates lateral speed: {states[j, 4]}")
        _require(np.linalg.norm(states[j, 3:]) <= np.linalg.norm(states[j - 1, 3:]) + .02,
                 "Stationary frictionless paddle adds kinetic energy")
        landings = np.flatnonzero((forces[:, 2] > 1e-5) & (np.arange(len(states)) > j + 1))
        _require(len(landings) > 0, "No table landing after paddle impact")
        landing = int(landings[0])
        result.update(paddle_contact_s=j * dt, paddle_outgoing_m_s=float(states[j, 3]),
                      paddle_restitution=ratio, landing_xy_m=states[landing, :2].tolist())

    with _check(report, f"net_and_table_edge_dt_{dt}"):
        states, forces = trial([-.15, .5, .87], [3, 0, 0], .08)
        _require(np.min(forces[:, 0]) < -1e-5 and states[-1, 3] < .1, "Ball tunnels through solid net")
        states, forces = trial([-.15, .5, 1.2], [3, 0, 0], .1)
        _require(states[-1, 0] > 0 and np.max(np.abs(forces)) == 0, "Clear net crossing collides")
        states, forces = trial([1.42, 0, 1.1], [0, 0, -1], .3)
        _require(states[-1, 2] < c["table_top_m"] and np.max(np.abs(forces)) == 0, "Outside-edge ball bounces on phantom table")

    with _check(report, f"repeatability_dt_{dt}"):
        a, af = trial([-.6, 0, 1.16], [-3, 0, 0], .6)
        b, bf = trial([-.6, 0, 1.16], [-3, 0, 0], .6)
        delta = float(np.max(np.abs(a - b)))
        _require(delta < 1e-6 and np.max(np.abs(af - bf)) < 1e-4, f"Nonrepeatable trial: {delta}")
        result["repeat_max_state_difference"] = delta

    report["reset_state_writes"] = bed.reset_count
    report["normal_step_state_writes"] = 0  # Guarded public root-state writers throughout all trials.
    report["status"] = "PASS"
    return report
