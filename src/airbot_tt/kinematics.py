"""AIRBOT Play serial-chain kinematics in base_link coordinates (m, rad, wxyz)."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import URDF

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
LINK_NAMES = ("base_link", *(f"link{i}" for i in range(1, 7)))

# Origin of the official MoveIt description's arm_connnect_joint, which mounts
# the removable end-effector adapter. The derived bare arm omits that adapter.
FLANGE_OFFSET_M = (0.0, 0.0, 0.0865)
IK_POSITION_TOLERANCE_M = 2e-4
IK_ORIENTATION_TOLERANCE_RAD = 2e-3


def vector(text, default="0 0 0"):
    values = np.array((default if text is None else text).split(), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite scalars in the URDF.")
    return values


def origin(element):
    """Convert a URDF xyz/rpy origin to a parent-from-child transform."""
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = Rotation.from_euler("xyz", vector(element.get("rpy"))).as_matrix()
        transform[:3, 3] = vector(element.get("xyz"))
    return transform


def pose_matrix(values):
    """Convert [x, y, z, qw, qx, qy, qz] to a homogeneous transform."""
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Pose requires seven finite values: x y z qw qx qy qz.")
    scale = np.max(np.abs(values[3:]))
    if scale == 0:
        raise ValueError("Pose quaternion must be nonzero.")
    transform = np.eye(4)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]] / scale).as_matrix()
    return transform


def pose_dict(transform):
    # SciPy 1.15.2's Rotation.from_matrix requires a writable source buffer.
    quaternion = Rotation.from_matrix(transform[:3, :3].copy()).as_quat()[[3, 0, 1, 2]]
    return {
        "position_m": transform[:3, 3].tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def pose_error(actual, desired):
    """Translation and rotation-vector errors expressed in the common parent frame."""
    rotation_delta = desired[:3, :3] @ actual[:3, :3].T
    return np.r_[desired[:3, 3] - actual[:3, 3], Rotation.from_matrix(rotation_delta).as_rotvec()]


def rigid_transform(value):
    """Copy and validate a finite homogeneous rigid transform."""
    transform = np.array(value, dtype=float, copy=True)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 rigid transform.")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("Expected a rigid transform with a proper rotation.")
    return transform


class Kinematics:
    """Seven physical links and the nominal flange / configurable TCP frames."""

    joint_names = JOINT_NAMES
    links = LINK_NAMES
    tip_link = LINK_NAMES[-1]

    def __init__(self, path=URDF, *, tcp_transform=None):
        self.path = Path(path)
        self.xml = ET.parse(self.path).getroot()
        joint_elements = self.xml.findall("joint")
        joints = {joint.get("name"): joint for joint in joint_elements}
        link_elements = self.xml.findall("link")
        if (
            set(joints) != set(self.joint_names)
            or len(joint_elements) != 6
            or {link.get("name") for link in link_elements} != set(self.links)
            or len(link_elements) != 7
        ):
            raise ValueError("Expected the six-joint bare AIRBOT Play URDF.")

        origins, axes, limits = [], [], []
        for name, parent, child in zip(self.joint_names, self.links[:-1], self.links[1:]):
            joint = joints[name]
            if (
                joint.get("type") != "revolute"
                or joint.find("parent").get("link") != parent
                or joint.find("child").get("link") != child
            ):
                raise ValueError(f"Expected revolute {name} between {parent} and {child}.")
            axis = vector(joint.find("axis").get("xyz"))
            if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-7, rtol=0):
                raise ValueError(f"Expected a unit axis for {name}.")
            origins.append(origin(joint.find("origin")))
            axes.append(axis)
            limit = joint.find("limit")
            limits.append([float(limit.attrib[key]) for key in ("lower", "upper", "velocity", "effort")])

        self._origins, self._axes = np.array(origins), np.array(axes)
        values = np.array(limits).T
        self.lower, self.upper, self.velocity, self.effort = values
        if (
            not np.isfinite(values).all()
            or np.any(self.lower >= self.upper)
            or np.any(self.velocity <= 0)
            or np.any(self.effort < 0)
        ):
            raise ValueError("Invalid joint limits in AIRBOT Play URDF.")

        self.flange_transform = np.eye(4)
        self.flange_transform[:3, 3] = FLANGE_OFFSET_M
        self.tcp_transform = np.eye(4) if tcp_transform is None else rigid_transform(tcp_transform)
        for value in (self.lower, self.upper, self.velocity, self.effort, self.flange_transform, self.tcp_transform):
            value.setflags(write=False)

    @staticmethod
    def _joint_vector(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles in radians.")
        return q

    def validate(self, q):
        """Validate a commanded joint vector against the position limits."""
        q = self._joint_vector(q)
        violations = (q < self.lower) | (q > self.upper)
        if np.any(violations):
            names = ", ".join(name for name, violation in zip(self.joint_names, violations) if violation)
            raise ValueError(f"Joint target exceeds position limits for {names}.")
        return q.copy()

    def frames(self, q):
        """Physical link transforms in base_link, ordered base_link to link6."""
        frames = [np.eye(4)]
        for zero, axis, angle in zip(self._origins, self._axes, self._joint_vector(q)):
            transform = frames[-1] @ zero
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
            frames.append(transform)
        return frames

    def flange(self, q):
        return self.frames(q)[-1] @ self.flange_transform

    def fk(self, q):
        return self.flange(q) @ self.tcp_transform

    def jacobian(self, q):
        """Map joint rates into TCP linear and angular velocity in base_link."""
        frames = self.frames(q)
        tip_position = (frames[-1] @ self.flange_transform @ self.tcp_transform)[:3, 3]
        origins = np.array([transform[:3, 3] for transform in frames[1:]])
        axes = np.array([t[:3, :3] @ axis for t, axis in zip(frames[1:], self._axes)])
        return np.vstack((np.cross(axes, tip_position - origins).T, axes.T))

    def ik(self, desired, seed):
        """Local position-bounded IK with explicit FK residual acceptance."""
        seed = self.validate(seed)
        desired = rigid_transform(desired)

        def residual(q):
            error = pose_error(self.fk(q), desired)
            error[3:] *= 0.2  # Balance angular residuals against this arm's metre-scale reach.
            return error

        solution = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            method="trf",
            max_nfev=250,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        error = pose_error(self.fk(solution.x), desired)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        return {
            "success": position_error <= IK_POSITION_TOLERANCE_M and orientation_error <= IK_ORIENTATION_TOLERANCE_RAD,
            "q_rad": solution.x.tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "evaluations": solution.nfev,
            "message": str(solution.message),
        }
