"""Boundary states and interior extrema for timed strikes."""
import unittest
import numpy as np
from airbot_tt.kinematics import Kinematics
from airbot_tt.trajectory import StrikeTrajectory


class StrikeTests(unittest.TestCase):
    def setUp(self):
        self.model=Kinematics()
        self.q=np.array([0,-.5,.6,0,0,0])

    def path(self, **kw):
        v=np.array([.05,0,0,0,0,0]);q=self.q
        return StrikeTrajectory(self.model,q,q+v/2,v,q+v,1,1,**kw)

    def test_timed_nonzero_passage_and_rest_endpoints(self):
        p=self.path()
        q,v,a=p.sample(1)
        np.testing.assert_allclose(q,self.q+[.025,0,0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(v,[.05,0,0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(a,0,atol=1e-12)
        for t in [0,2,3]:
            np.testing.assert_allclose(p.sample(t)[1:],0,atol=1e-12)
        np.testing.assert_allclose(p.sample(1-1e-8)[1],p.sample(1+1e-8)[1],atol=1e-8)
        h=1e-5;t=.6
        np.testing.assert_allclose((p.sample(t+h)[0]-p.sample(t-h)[0])/(2*h),p.sample(t)[1],atol=1e-8)

    def test_interior_overshoot_rejected(self):
        # Endpoints within limits but nonzero velocity overshoots position bounds.
        q=self.q.copy();q[0]=self.model.upper[0]-.001
        with self.assertRaisesRegex(ValueError,'derivative 0'):
            StrikeTrajectory(self.model,q,q,[.05,0,0,0,0,0],q,1,1,vmax=1,amax=2)
        # Zero endpoint velocities alone do not bound interior peak speed.
        with self.assertRaisesRegex(ValueError,'derivative 1'):
            StrikeTrajectory(self.model,self.q,self.q+[.1,0,0,0,0,0],np.zeros(6),self.q,1,1,vmax=.15,amax=10)

    def test_collision_samples_actual_path_and_invalid_input(self):
        seen=[];p=self.path(require_clear=lambda q:seen.append(q.copy()))
        self.assertEqual(len(seen),p.checked_samples)
        self.assertGreater(len(seen),400)
        with self.assertRaises(ValueError):self.path(max_joint_step=0)
        with self.assertRaises(ValueError):self.path(vmax=float('nan'))
        with self.assertRaises(ValueError):p.sample(float('nan'))

    def test_acceleration_and_curved_path_rejection(self):
        with self.assertRaisesRegex(ValueError,'derivative 2'):
            self.path(amax=.01)
        # Endpoints all clear, but the nonzero passage makes an interior excursion.
        def clear(q):
            if q[0] < -.002:
                raise ValueError('synthetic obstacle at curved excursion')
        with self.assertRaisesRegex(ValueError,'curved excursion'):
            StrikeTrajectory(self.model,self.q,self.q,[.05,0,0,0,0,0],self.q,1,1,
                             vmax=1,amax=2,require_clear=clear)

    def test_multiaxis_analytic_bounds(self):
        v=np.array([.1,-.2,.3,.1,-.05,.15]);q=self.q
        p=StrikeTrajectory(self.model,q,q+.15*v,v,q+.3*v,.3,.3,vmax=1,amax=8)
        for t in np.linspace(0,p.duration,301):
            position,velocity,acceleration=p.sample(t)
            self.assertTrue(np.all(position>=self.model.lower) and np.all(position<=self.model.upper))
            self.assertTrue(np.all(abs(velocity)<=p.peak_velocity+1e-9))
            self.assertTrue(np.all(abs(acceleration)<=p.peak_acceleration+1e-9))
