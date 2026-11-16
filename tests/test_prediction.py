"""Analytic and boundary checks for continuous ball forecasts."""
import unittest
import numpy as np
from airbot_tt.ball import load_ball_config
from airbot_tt.prediction import predict_ball


class PredictionTests(unittest.TestCase):
    def test_ballistic_apex_net_and_landing(self):
        c=load_ball_config();c['drag_coefficient']=0
        state=[-.5,.2,1.2,2,0,1]
        r=predict_ball(state,config=c,first_table_event='landing')
        events={e['kind']:e for e in r['events']}
        g=c['gravity_m_s2'];h=c['table_top_m']+c['radius_m']
        t=(1+np.sqrt(1+2*g*(1.2-h)))/g
        self.assertAlmostEqual(events['apex']['time_s'],1/g,places=8)
        self.assertAlmostEqual(events['net_crossing']['time_s'],.25,places=8)
        self.assertAlmostEqual(events['landing']['time_s'],t,places=8)
        np.testing.assert_allclose(events['landing']['position_m'],[-.5+2*t,.2,h],atol=1e-8)

    def test_bounce_continuation_and_call_independence(self):
        c=load_ball_config();c['drag_coefficient']=0
        r=predict_ball([.5,0,1.2,0,0,-1],config=c)
        self.assertEqual([e['kind'] for e in r['events']],['bounce','apex','landing'])
        bounce,apex,landing=r['events'];speed=np.sqrt(1+2*c['gravity_m_s2']*(1.2-.78))*.9
        self.assertAlmostEqual(landing['time_s']-bounce['time_s'],2*speed/c['gravity_m_s2'],places=8)
        self.assertEqual(r,predict_ball([.5,0,1.2,0,0,-1],config=c))

    def test_net_outside_edges_and_invalid_inputs(self):
        self.assertEqual(predict_ball([-.15,0,.87,3,0,0])['termination'],'net_hit')
        self.assertEqual(predict_ball([.15,0,.87,-3,0,0])['termination'],'net_hit')
        self.assertEqual(predict_ball([-.15,.76,.87,3,0,0])['termination'],'unsupported_net_edge')
        self.assertEqual(predict_ball([1.5,0,1.1,0,0,-1])['termination'],'out_of_table')
        self.assertEqual(predict_ball([1.37,0,1.1,0,0,-1])['termination'],'unsupported_table_edge')
        self.assertEqual(predict_ball([.5,0,2,0,0,1],.01)['events'],[])
        for state in ([0,0,float('nan'),0,0,0],[0,0,.5,0,0,0]):
            with self.assertRaises(ValueError):predict_ball(state)
        with self.assertRaises(ValueError):predict_ball([0,0,1,0,0,0],-1)
