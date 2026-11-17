"""Independent ballistic expectations and rejection contracts for return planning."""
import unittest
import numpy as np
from airbot_tt.ball import load_ball_config
from airbot_tt.planning import flight_state, return_velocity, serves, load_strike_config
from airbot_tt.prediction import predict_ball


class PlanningTests(unittest.TestCase):
    def test_inverse_flight_matches_vacuum_analytic(self):
        c=load_ball_config();c['drag_coefficient']=0
        p=np.array([-1.1,0,1.3]);goal=np.array([.75,.45]);time=.6
        velocity=return_velocity(p,goal,time,c)
        expected=(np.r_[goal,c['table_top_m']+c['radius_m']]-p)/time+[0,0,.5*c['gravity_m_s2']*time]
        np.testing.assert_allclose(velocity,expected,atol=1e-7)
        end=flight_state(np.r_[p,velocity],time,c)
        self.assertLess(end[5],0)
        np.testing.assert_allclose(end[:2],goal,atol=1e-7)

    def test_drag_return_and_obstructed_flight(self):
        c=load_ball_config();p=np.array([-1.1,0,1.3]);goal=np.array([.75,-.45])
        v=return_velocity(p,goal,.6,c)
        r=predict_ball(np.r_[p,v],config=c,first_table_event='landing')
        self.assertEqual(r['termination'],'landing')
        np.testing.assert_allclose(r['events'][-1]['position_m'][:2],goal,atol=2e-3)
        with self.assertRaisesRegex(ValueError,'net'):
            return_velocity(np.array([-.3,0,.80]),[.3,0],.10,c)

    def test_frozen_disjoint_reproducible_serve_sets(self):
        self.assertEqual(serves('development'),serves('development'))
        c=load_strike_config()
        self.assertNotEqual(c['development_seed'],c['validation_seed'])
        self.assertEqual(len(serves('validation')),c['validation_count'])
        self.assertNotEqual(serves('development')[0]['position_m'],serves('validation')[0]['position_m'])
        with self.assertRaises(ValueError):serves('final')
