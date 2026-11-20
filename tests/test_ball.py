"""Independent properties of the low-spin aerodynamic force law."""

import unittest

import numpy as np

from airbot_tt.ball import drag_force, load_ball_config


class BallForceTests(unittest.TestCase):
    def test_zero_quadratic_scaling_and_dissipation(self):
        c = load_ball_config()
        velocities = np.array([[0, 0, 0], [3, -4, 1], [-2, 1, -3]], dtype=float)
        forces = drag_force(velocities, c)
        np.testing.assert_array_equal(forces[0], [0, 0, 0])
        np.testing.assert_allclose(drag_force(2 * velocities, c), 4 * forces)
        np.testing.assert_allclose(np.cross(velocities, forces), 0, atol=1e-16)
        self.assertTrue(np.all(np.sum(velocities[1:] * forces[1:], axis=-1) < 0))

    def test_si_force_scale_and_rotation(self):
        c = load_ball_config()
        # Unit projected area and rho=2,Cd=.5 => at 2m/s the drag is exactly 2N.
        c.update(radius_m=np.sqrt(1 / np.pi), air_density_kg_m3=2, drag_coefficient=.5)
        np.testing.assert_allclose(drag_force([2, 0, 0], c), [-2, 0, 0])
        np.testing.assert_allclose(drag_force([0, 0, -2], c), [0, 0, 2])
        with self.assertRaises(ValueError):
            drag_force([0, np.nan, 0], c)
        with self.assertRaises(ValueError):
            drag_force([1, 2], c)
