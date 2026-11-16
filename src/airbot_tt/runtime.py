"""Isaac Lab articulation, contact monitoring and driven motion."""


import math

import numpy as np
import omni.kit.app
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from .collision import CollisionChecker
from .config import (
    BASE_POSITION, CONTACT_TRIP_N, DRIVE_DAMPING,
    DRIVE_STIFFNESS, DT, HOME, ROOT, SIM_SPEED, URDF,
)
from .kinematics import Kinematics, pose_dict, pose_error, rigid_transform
from .trajectory import Trajectory, VelocityTrajectory

ROBOT_PRIM = "/World/AIRBOT"
URDF_IMPORTER = "isaacsim.asset.importer.urdf"
URDF_IMPORTER_VERSION = "2.3.14"
MAX_TRACE_SAMPLES = 14_400


class _InertialUrdfConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        # Lab 2.1 does not expose this native importer option in its config.
        config.set_import_inertia_tensor(True)
        return config


def _articulation_config(model):
    manager = omni.kit.app.get_app().get_extension_manager()
    pinned = f"{URDF_IMPORTER}-{URDF_IMPORTER_VERSION}"
    enabled = manager.get_enabled_extension_id(URDF_IMPORTER)
    if not enabled:
        manager.set_extension_enabled_immediate(pinned, True)
        enabled = manager.get_enabled_extension_id(URDF_IMPORTER)
    if enabled != pinned and not (enabled or "").startswith(pinned + "+"):
        raise RuntimeError(f"Expected Kit extension {pinned}, found {enabled!r}.")
    # Conversion and application configuration both enforce the fixed base.
    converted = _InertialUrdfConverter(UrdfConverterCfg(
        asset_path=str(model.path),
        usd_dir=str(ROOT / ".agents/runs/foundation/usd"),
        usd_file_name="airbot_play.usd",
        make_instanceable=False,
        fix_base=True,
        root_link_name=model.links[0],
        merge_fixed_joints=False,
        self_collision=True,
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ))
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=converted.usd_path, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, fix_root_link=True,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=BASE_POSITION, joint_pos=dict(zip(model.joint_names, HOME, strict=True)),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={"arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=dict(zip(model.joint_names, model.effort, strict=True)),
            velocity_limit_sim={name: min(SIM_SPEED, velocity) for name, velocity in
                                zip(model.joint_names, model.velocity, strict=True)},
            stiffness=dict(zip(model.joint_names, DRIVE_STIFFNESS, strict=True)),
            damping=dict(zip(model.joint_names, DRIVE_DAMPING, strict=True)),
        )},
    )


class AirbotSimulation:
    """A single scene. Call from the application thread after AppLauncher starts."""

    dt = DT

    def __init__(self, device="cpu", obstacles=(), *, render=False, allow_initial_contact=False, model=None):
        self.model = Kinematics(URDF) if model is None else model
        self.checker = CollisionChecker(self.model, obstacles)
        self.initial = self.model.validate(HOME).copy()
        self._allow_initial_contact = allow_initial_contact
        if not allow_initial_contact:
            self.checker.require_clear(self.initial)

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=device))
        self.sim.set_camera_view((1.10, 0.88, 0.70), (0.0, 0.0, 0.36))
        ground = sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.10), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.25, 0.28)),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
        light = sim_utils.DomeLightCfg(intensity=1600.0)
        light.func("/World/Light", light)
        for index, box in enumerate(self.checker.obstacles):
            cfg = sim_utils.CuboidCfg(
                size=box.size_m, collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.25, 0.08)),
            )
            cfg.func(f"/World/Obstacle{index}", cfg, translation=box.center_m)
        self.robot = Articulation(_articulation_config(self.model))
        self.sensor = ContactSensor(ContactSensorCfg(
            prim_path=f"{ROBOT_PRIM}/(base_link|link[1-6])", update_period=0.0,
        ))
        self.sim.reset()
        self.indices, names = self.robot.find_joints(list(self.model.joint_names), preserve_order=True)
        if self.robot.num_joints != 6 or tuple(names) != tuple(self.model.joint_names) or not self.robot.is_fixed_base:
            raise RuntimeError(f"Expected six fixed-base arm joints, found {self.robot.joint_names}.")
        self.base = self.robot.body_names.index(self.model.links[0])
        self.tip = self.robot.body_names.index(self.model.links[-1])
        self._data = self.robot.data
        self._dtype = self._data.joint_pos.dtype
        self._device = self._data.joint_pos.device
        self._limit = self._tensor(np.column_stack((self.model.lower, self.model.upper)))
        self._effort_limit = self._tensor(self.model.effort.copy())
        self.gravity_effort = torch.zeros_like(self._effort_limit)
        if not torch.allclose(self._data.joint_pos_limits[0, self.indices], self._limit, rtol=0, atol=2e-5):
            raise RuntimeError("PhysX position limits differ from the selected URDF limits.")
        self._diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=str(self._device),
        )
        self._flange_p, self._flange_r = self._pose_tensors(self.model.flange_transform)
        self._tcp_p, self._tcp_r = self._pose_tensors(self.model.tcp_transform)
        self.tick = 0
        self.state_writes = 0
        self.last_trace = []
        self.reset()
        if render:
            self.sim.render()

    def _tensor(self, value):
        return torch.as_tensor(value, device=self._device, dtype=self._dtype)

    def _pose_tensors(self, matrix):
        pose = pose_dict(matrix)
        return self._tensor(pose["position_m"])[None], self._tensor(pose["quaternion_wxyz"])[None]

    def q(self):
        return self._data.joint_pos[0, self.indices].detach().cpu().numpy().astype(float)

    def qd(self):
        return self._data.joint_vel[0, self.indices].detach().cpu().numpy().astype(float)

    @property
    def active(self):
        return self.trajectory is not None

    def _drive(self, q, velocity):
        # PhysX supplies the feed-forward torque from the live rigid-body model.
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[0, self.indices]
        if not torch.isfinite(gravity).all():
            self.trajectory = None
            self.fault = "non-finite gravity compensation"
            raise RuntimeError(self.fault)
        self.gravity_effort = torch.clamp(gravity, -self._effort_limit, self._effort_limit)
        self.robot.set_joint_effort_target(self.gravity_effort[None], joint_ids=self.indices)
        self.robot.set_joint_position_target(self._tensor(q)[None], joint_ids=self.indices)
        self.robot.set_joint_velocity_target(self._tensor(velocity)[None], joint_ids=self.indices)
        self.robot.write_data_to_sim()

    def reset(self):
        if not self._allow_initial_contact:
            self.checker.require_clear(self.initial)
        self.trajectory = None
        self.elapsed = 0.0
        self.fault = None
        self.last_trace = []
        self.target = self.initial.copy()
        self.target_velocity = np.zeros(6)
        self.target_acceleration = np.zeros(6)
        self.robot.write_joint_state_to_sim(self._tensor(self.initial)[None], self._tensor(self.target_velocity)[None],
                                            joint_ids=self.indices)
        self.state_writes += 1
        self.robot.reset()
        self.sensor.reset()
        self._drive(self.target, self.target_velocity)
        self.robot.update(DT)

    def stop(self, reason="operator stop"):
        self.trajectory = None
        self.fault = reason
        self.target = np.clip(self.q(), self.model.lower, self.model.upper)
        self.target_velocity = np.zeros(6)
        self.target_acceleration = np.zeros(6)
        self._drive(self.target, self.target_velocity)

    def _ready_for_motion(self):
        if self.fault is not None:
            raise ValueError(f"Motion fault latched ({self.fault}). Reset before commanding motion.")
        if self.active:
            raise ValueError("Motion is active. Wait for completion, stop, or reset.")
        # A replacement profile begins at rest. This also prevents a target
        # discontinuity when a caller submits a new command immediately.
        if np.max(np.abs(self.qd())) > 0.04:
            raise ValueError("Wait for the measured arm to settle before commanding motion.")
        return self._motion_seed()

    def _start(self, path):
        self.trajectory = path
        self.elapsed = 0.0
        self._trace_stride = max(1, math.ceil(path.duration / (DT * (MAX_TRACE_SAMPLES - 1))))
        self._trace_step = 0
        self.last_trace = []
        return {"duration_s": path.duration, "checked_samples": path.checked_samples}

    def _motion_seed(self):
        q = self.q()
        # Only absorb float32 rounding of the physical limit at its endpoint.
        clipped = np.clip(q, self.model.lower, self.model.upper)
        if not np.isfinite(q).all() or np.max(np.abs(q - clipped)) > 2e-6:
            raise ValueError("Measured joints are outside their position limits.")
        return clipped

    def start_trajectory(self, waypoints, durations=None):
        if not isinstance(waypoints, (list, tuple, np.ndarray)) or len(waypoints) == 0:
            raise ValueError("A trajectory requires at least one six-axis waypoint.")
        path = Trajectory(self.model, [self._ready_for_motion(), *waypoints],
                          durations=durations, require_clear=self.checker.require_clear)
        return self._start(path)

    def move_joint(self, q):
        return self.start_trajectory([q])

    def move_velocity(self, velocity, duration):
        path = VelocityTrajectory(self.model, self._ready_for_motion(), velocity, duration,
                                  require_clear=self.checker.require_clear)
        return self._start(path)

    def solve_pose(self, desired, execute=False):
        seed = self._ready_for_motion() if execute else self._motion_seed()
        # Use NVIDIA's pose DLS step in base coordinates. Evaluating it with the
        # reference FK/Jacobian allows a preflight solution before any drive moves.
        desired = rigid_transform(desired)
        goal_p, goal_r = self._pose_tensors(desired)
        if not torch.isfinite(goal_p).all():
            raise ValueError("TCP position exceeds the simulation numeric range.")
        self._diff_ik.set_command(torch.cat((goal_p, goal_r), dim=-1))
        q = seed.copy()
        for step in range(80):
            current = self.model.fk(q)
            error = pose_error(current, desired)
            pe = float(np.linalg.norm(error[:3]))
            re = float(np.linalg.norm(error[3:]))
            if pe <= 2e-4 and re <= 2e-3:
                result = {"success": True, "q_rad": q.tolist(), "position_error_m": pe,
                          "orientation_error_rad": re, "evaluations": step + 1, "solver": "isaaclab_dls"}
                break
            pos, quat = self._pose_tensors(current)
            next_q = self._diff_ik.compute(pos, quat, self._tensor(self.model.jacobian(q))[None], self._tensor(q)[None])
            delta = np.clip(next_q[0].cpu().numpy() - q, -0.20, 0.20)
            candidate = np.clip(q + delta, self.model.lower, self.model.upper)
            if not np.isfinite(candidate).all() or np.max(np.abs(candidate - q)) < 1e-7:
                result = None
                break
            q = candidate
        else:
            result = None
        if result is None:
            result = self.model.ik(desired, seed)
            result["solver"] = "scipy_bounded"
        if result["success"]:
            pairs = self.checker.check(result["q_rad"])
            result.update(collision_pairs=pairs, collision_free=not pairs)
        if execute:
            if not result["success"]:
                raise ValueError(f"Unreachable TCP pose from the configured IK seeds: {result}")
            result.update(self.move_joint(result["q_rad"]))
        return result

    def move_cartesian(self, desired):
        return self.solve_pose(desired, execute=True)

    def sim_contact(self):
        return self.sensor.body_names, self.sensor.data.net_forces_w[0].detach().cpu().numpy().copy()

    def step(self, render=False):
        reference = self.trajectory
        if reference is not None:
            self._trace_step += 1
            self.elapsed = min(self.elapsed + DT, reference.duration)
            self.target, self.target_velocity, self.target_acceleration = reference.sample(self.elapsed)
            if self.elapsed >= reference.duration:
                self.trajectory = None
        self._drive(self.target, self.target_velocity)
        self.sim.step(render=render)
        self.robot.update(DT)
        self.sensor.update(DT)
        self.tick += 1
        q, qd = self.q(), self.qd()
        if not np.isfinite([q, qd]).all():
            self.trajectory = None
            self.fault = "non-finite joint feedback"
            raise RuntimeError(self.fault)
        _, forces = self.sim_contact()
        if not np.isfinite(forces).all():
            self.stop("non-finite contact feedback")
            raise RuntimeError(self.fault)
        if reference is not None and (self._trace_step % self._trace_stride == 0 or not self.active):
            self.last_trace.append({
                "time_s": self.elapsed, "q_rad": q.tolist(), "qd_rad_s": qd.tolist(),
                "target_rad": self.target.tolist(), "target_velocity_rad_s": self.target_velocity.tolist(),
                "target_acceleration_rad_s2": self.target_acceleration.tolist(),
                "error_rad": (q - self.target).tolist(),
            })
        if self.fault is None:
            force = float(np.linalg.norm(forces, axis=1).max())
            pairs = self.checker.check(q) if self.tick % 8 == 0 else []
            if pairs or force > CONTACT_TRIP_N:
                self.stop(f"collision guard: {pairs}, peak contact {force:.3f} N")
            elif np.any(q < self.model.lower - 0.004) or np.any(q > self.model.upper + 0.004):
                self.stop("measured joint exceeded its position limit")

    def state(self):
        base_p = self._data.body_link_pos_w[:, self.base]
        base_r = self._data.body_link_quat_w[:, self.base]
        tip_p = self._data.body_link_pos_w[:, self.tip]
        tip_r = self._data.body_link_quat_w[:, self.tip]
        flange_p, flange_r = combine_frame_transforms(tip_p, tip_r, self._flange_p, self._flange_r)
        tcp_p, tcp_r = combine_frame_transforms(flange_p, flange_r, self._tcp_p, self._tcp_r)
        flange_base = subtract_frame_transforms(base_p, base_r, flange_p, flange_r)
        tcp_base = subtract_frame_transforms(base_p, base_r, tcp_p, tcp_r)

        def pose(p, r):
            return {"position_m": p[0].tolist(), "quaternion_wxyz": r[0].tolist()}

        names, forces = self.sim_contact()
        return {
            "q_rad": self.q().tolist(), "qd_rad_s": self.qd().tolist(), "target_rad": self.target.tolist(),
            "target_velocity_rad_s": self.target_velocity.tolist(), "joint_names": list(self.model.joint_names),
            "gravity_feedforward_Nm": self.gravity_effort.tolist(),
            "limits_rad": self._limit.tolist(), "documented_velocity_rad_s": self.model.velocity.tolist(),
            "sim_velocity_limit_rad_s": np.minimum(self.model.velocity, SIM_SPEED).tolist(),
            "base_world": pose(base_p, base_r), "link6_world": pose(tip_p, tip_r),
            "flange_world": pose(flange_p, flange_r), "flange_base": pose(*flange_base),
            "tcp_world": pose(tcp_p, tcp_r), "tcp_base": pose(*tcp_base),
            "contact_body_names": names, "contact_forces_N": forces.tolist(),
            "contact_max_N": float(np.max(np.linalg.norm(forces, axis=1))),
            "active": self.active, "fault": self.fault, "state_writes": self.state_writes,
        }
