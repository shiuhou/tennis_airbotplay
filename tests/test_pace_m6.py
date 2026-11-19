"""AIRBOT M6 contract tests; original code remains independently audited."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from airbot_tt.pace_guidance import apex_teacher, return_teacher, guidance_rewards, ReleaseLatch, settings
from airbot_tt.pace_predictor import Predictor, PredictionHook, windows, actor_frame


class GuidanceTests(unittest.TestCase):
    def test_teacher_phases_and_no_robot_fallback(self):
        self.assertTrue(apex_teacher([.5,0,2.2,-3.3,0,-1.4])['valid'])
        for kw,state in [({'hit':True},[-.4,0,1.1,-1,0,1]),
                         ({'bounced':True},[-.4,0,1.1,-1,0,-1]),
                         ({},[.4,0,.7,-1,0,-1]), ({},[-.4,0,1.1,1,0,1])]:
            t=apex_teacher(state,**kw)
            self.assertFalse(t['valid']);self.assertEqual(t['point'],[0.,0.,0.])

    def test_reward_plateau_and_dt(self):
        teacher=dict(point=[0.,0.,0.],valid=True)
        near=guidance_rewards([.1,0,0],teacher)
        edge=guidance_rewards([.15,0,0],teacher)
        far=guidance_rewards([.3,0,0],teacher)
        self.assertAlmostEqual(near['raw']['hit'],np.exp(-.6))
        self.assertEqual(near,edge);self.assertGreater(near['total'],far['total'])
        returned=dict(valid=True,landing=[.85,0,.78],net_z=1.11)
        a=guidance_rewards([0,0,0],teacher,returned,dt=.02)
        b=guidance_rewards([0,0,0],teacher,returned,dt=1/48)
        self.assertAlmostEqual(a['weighted']['landing'],3.48)
        self.assertEqual(a['weighted']['landing'],b['weighted']['landing'])
        self.assertAlmostEqual(b['weighted']['net'],2.)
        self.assertAlmostEqual(b['weighted']['hit']/a['weighted']['hit'],50/48)
        returned['landing']=[5,0,.78]
        self.assertLess(guidance_rewards([0,0,0],teacher,returned)['raw']['landing'],0)

    def test_return_and_contact_once(self):
        s=np.array([-.8,0,1.15,2.8,0,1.5])
        self.assertTrue(return_teacher(s)['valid'])
        self.assertFalse(return_teacher([-.15,0,.87,3,0,0])['valid'])
        latch=ReleaseLatch()
        args=(s,True,True,np.array([-.82,0,1.15]),np.array([1.,0,0]),np.zeros(3))
        self.assertIsNone(latch.update(0,*args))
        result=latch.update(.005,s,False,True,args[3],args[4],args[5])
        self.assertTrue(result['valid'])
        self.assertIsNone(latch.update(.01,s,False,True,args[3],args[4],args[5]))
        latch.reset();latch.update(0,*args);self.assertIsNone(latch.update(.06,*args))
        self.assertEqual(latch.reason,'contact_timeout')
        latch.reset();latch.update(0,*args)
        self.assertIsNone(latch.update(.005,s,False,False,args[3],args[4],args[5]))
        self.assertEqual(latch.reason,'invalidated')
        latch.reset();self.assertIsNone(latch.update(0,s,True,False,args[3],args[4],args[5]))

    def test_windows_boundary_latest_and_invalid(self):
        p=np.arange(30).reshape(10,3);t=np.arange(10)/48;s=np.repeat([0,1],5)
        valid=np.ones(10,bool)
        x,y,e=windows(p,t,s,valid,p,valid)
        np.testing.assert_array_equal(e,[4,9]);np.testing.assert_array_equal(x[1],p[5:].ravel())
        valid[-1]=False
        self.assertEqual(windows(p,t,s,valid,p,valid)[2].tolist(),[4])
        t[2]+=.01
        self.assertEqual(len(windows(p,t,s,valid,p,valid)[0]),0)


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'.agents/runs/foundation/pace/source/rsl_rl/rsl_rl/runners/on_policy_predictor_regression_runner.py').is_file(),
                     'Run scripts/prepare_pace.py for pinned official PACE sources.')
class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_mlp_and_training(self):
        p=Predictor()
        self.assertEqual(p.model.__class__.__name__,'_MLPPredictor')
        x=np.zeros((32,15),np.float32); y=np.ones((32,3),np.float32)
        before={k:v.clone() for k,v in p.model.state_dict().items()}
        p.fit_epoch(x,y)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in p.model.state_dict().items()))
        self.assertFalse(p.predict(x).requires_grad)
        self.assertTrue(all(v.grad is None for v in p.model.parameters()))

    def test_checkpoint_untrained_trained_and_strict_config(self):
        p=Predictor()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.pt';p.save(path);q=Predictor();q.load(path)
            self.assertFalse(q.trained)
            with self.assertRaises(RuntimeError):q.predict(np.zeros((1,15)))
            p.fit_epoch(np.zeros((10,15)),np.ones((10,3)));p.save(path);q.load(path)
            self.assertTrue(q.trained);self.assertEqual(q.updates,1)
            self.assertTrue(torch.equal(p.predict(np.zeros((1,15))),q.predict(np.zeros((1,15)))))
            q.cfg['schema']='different'
            with self.assertRaises(ValueError):q.load(path)

    def test_predictor_does_not_change_policy_rng_and_stops_exactly(self):
        before=torch.get_rng_state().clone()
        p=Predictor()
        p.fit_epoch(np.zeros((8,15)),np.ones((8,3)))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        p.updates=p.cfg['predictor']['epochs']
        with self.assertRaises(RuntimeError):p.fit_epoch(np.zeros((8,15)),np.ones((8,3)))

    def test_hook_same_step_reset_and_observable_only(self):
        class RecordingPredictor:
            cfg=settings();trained=True
            def predict(self,x):
                # Identity on newest observation exposes a one-step stale hook.
                return torch.tensor(x[:,-3:],dtype=torch.float32)
        h=PredictionHook(RecordingPredictor(),2)
        for i in range(5):out=h.observe([[i,0,1],[100+i,0,1]],[i/48]*2,[0,0],[True,True])
        self.assertEqual(out[:,0].tolist(),[4,104]);self.assertEqual(out[:,3].tolist(),[1,1])
        old=out.clone()
        out=h.observe([[5,0,1],[105,0,1]],[5/48]*2,[1,0],[True,True])
        self.assertEqual(out[:,3].tolist(),[0,1]);self.assertEqual(out[1,0],105)
        self.assertTrue(torch.equal(old,torch.tensor([[4.,0,1,1],[104,0,1,1]])))
        out=h.observe([[6,0,1],[106,0,1]],[6/48]*2,[1,0],[True,False])
        self.assertEqual(out[:,3].tolist(),[0,0])
        self.assertFalse(out.requires_grad)
        fields=[torch.zeros((2,6)) for _ in range(3)]+[torch.zeros((2,3)),torch.ones((2,1)),torch.zeros((2,1))]
        self.assertEqual(actor_frame(*fields).shape,(2,23))
        frame=actor_frame(*fields,prediction=old)
        self.assertEqual(frame.shape,(2,27))
        self.assertTrue(torch.equal(frame[:,23:],old.clamp(-100,100)))
        self.assertFalse(frame.requires_grad)


if __name__=='__main__':unittest.main()
