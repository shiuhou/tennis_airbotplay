"""CPU regression tests for the AIRBOT Play URDF and motion math."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from airbot_tt.kinematics import Kinematics, pose_dict, pose_error, pose_matrix
from airbot_tt.trajectory import Trajectory, VelocityTrajectory


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Kinematics()
        cls.q = np.clip(
            np.array([0.25, -0.24, 0.34, 0.20, 0.35, -0.22]),
            cls.model.lower + 0.1,
            cls.model.upper - 0.1,
        )

    def test_bare_arm_frames_and_tcp_offset(self):
        model = self.model
        self.assertEqual(len(model.joint_names), 6)
        self.assertEqual(len(model.links), 7)
        self.assertEqual(len(set(model.joint_names)), 6)
        self.assertTrue(np.all(model.lower < model.upper))
        self.assertTrue(np.all(model.velocity > 0.0))
        np.testing.assert_allclose(
            np.linalg.solve(model.frames(self.q)[-1], model.flange(self.q)),
            pose_matrix([0, 0, 0.0865, 1, 0, 0, 0]),
            atol=1e-12,
        )
        flange_pose = pose_dict(model.flange_transform)
        np.testing.assert_allclose(flange_pose["position_m"], [0.0, 0.0, 0.0865])
        np.testing.assert_allclose(flange_pose["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(model.fk(self.q), model.flange(self.q), atol=1e-12)
        offset = pose_matrix([0.02, -0.03, 0.12, 0, 1, 0, 0])
        tool_model = Kinematics(tcp_transform=offset)
        np.testing.assert_allclose(
            tool_model.fk(self.q), model.flange(self.q) @ offset, atol=1e-12
        )

    def test_geometric_jacobian_matches_central_difference(self):
        epsilon = 1e-6
        difference = np.empty((6, 6))
        for index, perturbation in enumerate(epsilon * np.eye(6)):
            minus = self.model.fk(self.q - perturbation)
            plus = self.model.fk(self.q + perturbation)
            difference[:3, index] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
            difference[3:, index] = Rotation.from_matrix(
                plus[:3, :3] @ minus[:3, :3].T
            ).as_rotvec() / (2 * epsilon)
        np.testing.assert_allclose(
            self.model.jacobian(self.q), difference, atol=2e-8, rtol=2e-8
        )

    def test_finite_ik_round_trip_and_unreachable_target(self):
        alternate_poses = (
            [0.10, 0.10, -0.15, 0.05, 0.10, 0.10],
            [-0.20, 0.20, -0.30, -0.10, -0.10, 0.20],
            [0.03, -0.05, 0.02, 0.05, 0.05, -0.02],
        )
        for offset in alternate_poses:
            with self.subTest(offset=offset):
                target_q = np.clip(
                    self.q + offset, self.model.lower + 0.02, self.model.upper - 0.02
                )
                target = self.model.fk(target_q)
                result = self.model.ik(target, self.q)
                self.assertTrue(result["success"], result)
                q = self.model.validate(result["q_rad"])
                error = pose_error(self.model.fk(q), target)
                self.assertLess(np.linalg.norm(error[:3]), 2e-4)
                self.assertLess(np.linalg.norm(error[3:]), 2e-3)
        unreachable = pose_matrix([10, 0, 0, 1, 0, 0, 0])
        self.assertFalse(self.model.ik(unreachable, self.q)["success"])

    def test_quaternion_sign_and_invalid_joint_values(self):
        pose = pose_dict(self.model.fk(self.q))
        position, quaternion = pose["position_m"], pose["quaternion_wxyz"]
        np.testing.assert_allclose(
            pose_matrix(position + quaternion),
            pose_matrix(position + [-value for value in quaternion]),
            atol=1e-12,
        )
        outside = self.model.upper.copy()
        outside[0] += 0.0001
        for invalid in ([0] * 5, [np.nan] * 6, [np.inf] * 6, outside):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model.validate(invalid)
        for invalid in ([0] * 7, [0, 0, np.nan, 1, 0, 0, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pose_matrix(invalid)

    def test_timed_trajectory_respects_limits_and_stops_at_waypoints(self):
        waypoints = [
            self.q,
            self.q + [0.075, 0.065, -0.07, 0.03, 0.05, -0.06],
            self.q + [-0.06, -0.035, 0.07, -0.035, 0.04, 0.07],
        ]
        checked = []
        trajectory = Trajectory(
            self.model, waypoints, vmax=0.15, amax=0.3, require_clear=lambda q: checked.append(q.copy())
        )
        samples = [trajectory.sample(t) for t in np.linspace(0.0, trajectory.duration, 2001)]
        self.assertLessEqual(np.max(np.abs([sample[1] for sample in samples])), 0.150001)
        self.assertLessEqual(np.max(np.abs([sample[2] for sample in samples])), 0.300001)
        for time, expected in zip([0, *trajectory.ends], waypoints):
            position, rate, acceleration = trajectory.sample(time)
            np.testing.assert_allclose(position, expected, atol=1e-12)
            np.testing.assert_allclose(rate, 0.0, atol=1e-12)
            np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)
        self.assertEqual(len(checked), trajectory.checked_samples)
        self.assertLessEqual(np.max(np.abs(np.diff(checked, axis=0))), 0.010001)
        np.testing.assert_allclose(checked[0], waypoints[0], atol=1e-12)
        np.testing.assert_allclose(checked[-1], waypoints[-1], atol=1e-12)
        for invalid_times in ([0.001, 2.0], [np.inf, 2.0], [-0.5, 2.0], [3.0], [1e308, 1e308]):
            with self.subTest(times=invalid_times), self.assertRaises(ValueError):
                Trajectory(self.model, waypoints, durations=invalid_times)
        explicit = Trajectory(self.model, waypoints, durations=[2.5, 3.25])
        np.testing.assert_allclose(explicit.durations, [2.5, 3.25])

    def test_velocity_pulse_ramps_and_checks_the_endpoint(self):
        velocity = np.array([0.08, -0.05, 0.04, 0.0, 0.0, 0.05])
        duration = 1.25
        trajectory = VelocityTrajectory(self.model, self.q, velocity, duration)
        ramp = np.max(np.abs(velocity)) / 0.3
        expected = self.q + velocity * (duration - ramp)
        np.testing.assert_allclose(trajectory.sample(0)[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(trajectory.sample(duration / 2)[1], velocity, atol=1e-12)
        np.testing.assert_allclose(trajectory.sample(duration)[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(trajectory.sample(duration)[0], expected, atol=1e-12)
        samples = [trajectory.sample(t) for t in np.linspace(0.0, duration, 1001)]
        self.assertLessEqual(np.max(np.abs([sample[1] for sample in samples])), 0.150001)
        self.assertLessEqual(np.max(np.abs([sample[2] for sample in samples])), 0.300001)
        with self.assertRaises(ValueError):
            VelocityTrajectory(self.model, self.q, velocity, ramp)
        with self.assertRaises(ValueError):
            VelocityTrajectory(self.model, self.q, 2.0 * velocity, duration)
        with self.assertRaises(ValueError):
            VelocityTrajectory(self.model, self.q, np.zeros(6), 1e308)
        with self.assertRaises(ValueError):
            VelocityTrajectory(self.model, self.model.upper, [0.08, 0, 0, 0, 0, 0], duration)


if __name__ == "__main__":
    unittest.main()
