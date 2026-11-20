"""Racket TCP, fixed-body dynamics and tool-inclusive collision regression."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from airbot_tt.collision import CollisionChecker
from airbot_tt.config import BASE_POSITION, ROOT
from airbot_tt.kinematics import Kinematics, pose_error
from airbot_tt.tool import racket_model
from airbot_tt.trajectory import Trajectory


class RacketTests(unittest.TestCase):
    def test_blade_collision_preserves_striking_plane(self):
        from airbot_tt.collision import stl_triangles
        model = self.model
        parts = model.tool_geometry['components']['paddle']['collision_parts']
        self.assertEqual(len(model.xml.findall("link[@name='paddle']/collision")), 2)
        blade = stl_triangles(ROOT/parts[1]['mesh']).reshape(-1,3)
        heights = (blade-model.link6_tcp[:3,3]) @ model.link6_tcp[:3,2]
        self.assertLess(abs(heights.max()), 1e-7)
        self.assertGreater(heights.min(), -.014)
        self.assertTrue(np.any(abs(heights)<1e-7))

    @classmethod
    def setUpClass(cls):
        cls.model = racket_model(ROOT/'configs/racket.json')
        cls.q = np.array([.25, -.24, .34, .2, .35, -.22])

    def test_captured_link3_link5_lp_is_translation_invariant(self):
        # Actual failing left-serve positioning pose (M7 collision_diagnostic).
        q=np.array([-.004965776538856874,-1.626696610748847,.8330069196512447,
                    1.5684060736171415,1.6664005870412317,-1.5670463286042238])
        checker=CollisionChecker(self.model,base_position=[-1.275,0,.76])
        frames=self.model.frames(q)
        hulls=[]
        for index in (3,5):
            frame=frames[index].copy();frame[:3,3]+=checker.base_position
            hulls.append(checker._shapes[index][0].transform(frame))
        a,b=hulls
        for shift in ([0,0,0],[2,-3,4],[1000,-2000,3000]):
            transform=np.eye(4);transform[:3,3]=shift
            self.assertFalse(a.transform(transform).overlaps(b.transform(transform)))
            self.assertTrue(a.transform(transform).overlaps(a.transform(transform)))

    def test_collision_margin_and_unknown_solver_remain_fail_closed(self):
        from airbot_tt.collision import _Hull,BOX_CORNERS
        from unittest.mock import patch
        from types import SimpleNamespace
        a=_Hull.from_points(BOX_CORNERS*.1)
        self.assertTrue(a.overlaps(_Hull.from_points(BOX_CORNERS*.1+[.10099,0,0])))
        self.assertFalse(a.overlaps(_Hull.from_points(BOX_CORNERS*.1+[.10101,0,0])))
        with patch('airbot_tt.collision.linprog',return_value=SimpleNamespace(status=4,message='numerical failure')):
            with self.assertRaisesRegex(RuntimeError,'Collision feasibility'):
                a.overlaps(a)

    def test_loaded_mass_and_physical_inertia(self):
        model = self.model
        self.assertEqual(len(model.links), 9)
        self.assertEqual(len(model.joint_names), 6)
        mass_sum = 0
        for name, expected in [('fixer', .077), ('paddle', .170)]:
            inertial = model.xml.find(f"link[@name='{name}']/inertial")
            mass = float(inertial.find('mass').get('value'))
            self.assertAlmostEqual(mass, expected)
            mass_sum += mass
            data = inertial.find('inertia').attrib
            tensor = np.array([[float(data['ixx']), float(data['ixy']), float(data['ixz'])],
                               [float(data['ixy']), float(data['iyy']), float(data['iyz'])],
                               [float(data['ixz']), float(data['iyz']), float(data['izz'])]])
            moments = np.linalg.eigvalsh(tensor)
            self.assertGreater(moments.min(), 0)
            self.assertLess(moments.max(), moments.sum()-moments.max())
        self.assertAlmostEqual(mass_sum, .247)
        for frame in model.frames(self.q)[7:]:
            np.testing.assert_allclose(frame, model.frames(self.q)[6], atol=1e-12)

    def test_tcp_axes_and_jacobian(self):
        model = self.model
        np.testing.assert_allclose(model.fk(self.q), model.frames(self.q)[6] @ model.link6_tcp, atol=1e-12)
        rotation = model.link6_tcp[:3, :3]
        self.assertGreater(rotation[1, 2], .999)  # Z is the striking-face normal.
        self.assertGreater(rotation[2, 0], .999)  # X runs from handle toward blade.
        difference = np.empty((6, 6))
        epsilon = 1e-6
        for index, step in enumerate(np.eye(6)*epsilon):
            plus, minus = model.fk(self.q+step), model.fk(self.q-step)
            difference[:3, index] = (plus[:3, 3]-minus[:3, 3])/(2*epsilon)
            difference[3:, index] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec()/(2*epsilon)
        np.testing.assert_allclose(model.jacobian(self.q), difference, atol=2e-8, rtol=2e-8)
        self.assertGreater(np.linalg.norm(model.fk(self.q)[:3, 3]-model.flange(self.q)[:3, 3]), .18)

    def test_ik_and_trajectory_use_racket_tcp(self):
        model = self.model
        target = model.fk(self.q+[.05, -.05, .08, .05, .06, -.04])
        solution = model.ik(target, self.q)
        self.assertTrue(solution['success'], solution)
        residual = pose_error(model.fk(solution['q_rad']), target)
        self.assertLess(np.linalg.norm(residual[:3]), 2e-4)
        self.assertLess(np.linalg.norm(residual[3:]), 2e-3)
        checker = CollisionChecker(model)
        path = Trajectory(model, [self.q, solution['q_rad']], require_clear=checker.require_clear)
        np.testing.assert_allclose(model.fk(path.sample(path.duration)[0]), target, atol=2e-4)

    def test_tool_obstacle_is_not_missed_by_bare_arm(self):
        model = self.model
        q = np.zeros(6)
        box = [{'name': 'blade_only', 'center_m': (model.fk(q)[:3, 3]+BASE_POSITION).tolist(),
                'size_m': [.012]*3}]
        self.assertFalse(CollisionChecker(Kinematics(), box).check(q))
        self.assertIn(('paddle', 'blade_only'), CollisionChecker(model, box).check(q))
        with self.assertRaisesRegex(ValueError, 'blade_only'):
            Trajectory(model, [q, q+[.05, 0, 0, 0, 0, 0]],
                       require_clear=CollisionChecker(model, box).require_clear)

    def test_paddle_ground_collision_with_clear_bare_arm(self):
        q = [1.7362248213, -1.7549656293, 1.2669941930, -1.4052798350, 1.7152322160, .2949753362]
        self.assertFalse(CollisionChecker(Kinematics()).check(q))
        self.assertIn(('paddle', 'ground'), CollisionChecker(self.model).check(q))

    def test_unconfirmed_or_invalid_mass_is_rejected(self):
        profile = json.loads((ROOT/'configs/racket.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profile.json'
            for bad in ('unconfirmed', 'negative', 'nan'):
                data = deepcopy(profile)
                if bad == 'unconfirmed':
                    data['confirmed'] = False
                else:
                    data['mass_kg']['paddle'] = -1 if bad == 'negative' else float('nan')
                path.write_text(json.dumps(data))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    racket_model(path)


if __name__ == '__main__':
    unittest.main()
