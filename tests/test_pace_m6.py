"""AIRBOT M6 contract tests; original code remains independently audited."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from airbot_tt.pace_guidance import apex_teacher, return_teacher, guidance_rewards, ReleaseLatch, settings
from airbot_tt.config import ROOT
from airbot_tt.pace_predictor import Predictor, PredictionHook, FrozenPredictor, FixedMeanPredictor, windows, actor_frame


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

    def test_online_buffer_update_cutoff_and_resume(self):
        import copy
        from airbot_tt.pace_predictor import OnlinePredictor
        online=OnlinePredictor(2)
        for j in range(6):
            out=online.observe(np.full((2,3),j/48),[j/48]*2,[1,2],[True,True])
            self.assertFalse(out[:,3].any())
            online.collect(np.ones((2,3)),[True,False])
        rng=torch.get_rng_state().clone();result=online.update()
        self.assertEqual(result['samples'],2);self.assertEqual(result['updates'],1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(online.hook.current()[:,3].all())
        restored=OnlinePredictor(2);restored.load_state_dict(copy.deepcopy(online.state_dict()))
        self.assertEqual(restored.hook.history,[[],[]])
        self.assertEqual(online.update()['mse'],restored.update()['mse'])
        for a,b in zip(online.predictor.model.parameters(),restored.predictor.model.parameters()):
            self.assertTrue(torch.equal(a,b));self.assertIsNone(a.grad)
        online.attempts=20
        self.assertEqual(online.update()['status'],'cutoff')
        self.assertEqual(online.predictor.updates,2)

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


class FixedMeanPredictorTests(unittest.TestCase):
    """Train-only mean control; no fitting and no raw held-out access."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = json.loads((ROOT/'configs/pace_mean.json').read_text())
        with np.load(ROOT/cls.manifest['files']['train']['path']) as data:
            cls.data = {k:data[k].copy() for k in data.files}
        cls.point = torch.tensor(cls.manifest['constant_point_m'], dtype=torch.float32)

    def test_exact_existing_train_only_equal_serve_mean(self):
        d = self.data
        x, target, ends = windows(d['state'][:, :3], d['time'], d['serve'],
                                  np.ones(len(d['time']), bool), d['target'], d['valid'])
        ids = d['serve'][ends]
        actual = np.mean([target[ids == sid].mean(0) for sid in np.unique(ids)], axis=0)
        archived = json.loads((ROOT/self.manifest['files']['cross']['path']).read_text())
        self.assertEqual(len(x), 4868)
        self.assertEqual(len(np.unique(ids)), 128)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.tolist(), archived['train_mean_apex_m'])
        self.assertEqual(actual.tolist(), self.manifest['constant_point_m'])
        mean = FixedMeanPredictor(1)
        self.assertTrue(torch.equal(mean.predictor.predict(x), self.point.expand(len(x), 3)))

    def test_construct_predict_reload_never_build_mlp_optimizer_or_touch_rng(self):
        from unittest.mock import patch
        before = torch.get_rng_state().clone()
        x = torch.arange(30, dtype=torch.float32).reshape(2, 15).requires_grad_()
        with patch('airbot_tt.pace_predictor.original_mlp', side_effect=AssertionError('MLP constructed')), \
             patch('torch.optim.Adam', side_effect=AssertionError('Optimizer constructed')), \
             patch('torch.Generator', side_effect=AssertionError('RNG constructed')), \
             patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA RNG touched')), \
             patch('numpy.load', side_effect=AssertionError('Raw data opened at inference')):
            mean = FixedMeanPredictor(2)
            output = mean.predictor.predict(x)
            mean.load_state_dict(mean.state_dict())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(mean.predictor.cfg, settings())
        self.assertTrue(mean.predictor.trained)
        self.assertEqual(mean.predictor.updates, 0)
        self.assertIsNone(mean.predictor.optimizer)
        self.assertIsNone(mean.predictor.rng)
        self.assertEqual(list(mean.predictor.model.parameters()), [])
        self.assertEqual(list(dict(mean.predictor.model.named_buffers())), ['point'])
        self.assertFalse(mean.predictor.model.training)
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.equal(output, self.point.expand(2, 3)))
        for name in ('buffer', 'collect', 'update'):
            self.assertFalse(hasattr(mean, name))
        output[0, 0] = 0
        self.assertTrue(torch.equal(mean.predictor.predict(x), self.point.expand(2, 3)))
        with self.assertRaisesRegex(RuntimeError, 'cannot train'):
            mean.predictor.fit_epoch(x, output)

    def test_same_five_point_validity_as_frozen_and_partial_reset(self):
        mean, frozen = FixedMeanPredictor(2), FrozenPredictor(2)
        old = None
        for tick in range(9):
            p = np.tile(self.data['state'][tick, :3], (2, 1))
            learned = frozen.observe(p, [tick/48]*2, [1, 2], [True]*2)
            # The mean's content is independent of finite position values.
            actual = mean.observe(p + tick * 10, [tick/48]*2, [1, 2], [True]*2)
            self.assertTrue(torch.equal(actual[:, 3], learned[:, 3]))
            if tick >= 4:
                self.assertTrue(torch.equal(actual[:, :3], self.point.expand(2, 3)))
            else:
                self.assertFalse(actual.any())
            if tick == 4:
                old = actual
        history_other = np.array(mean.hook.history[1])
        for module in (mean, frozen):
            module.hook.reset([0])
            actual = module.observe(np.zeros((2, 3)), [0, 9/48], [3, 2], [True]*2, ids=[0])
            self.assertFalse(actual.any())
            self.assertEqual([len(h) for h in module.hook.history], [1, 5])
        np.testing.assert_array_equal(mean.hook.history[1], history_other)
        self.assertTrue(torch.equal(mean.hook.current()[1], torch.cat((self.point, torch.ones(1)))))
        self.assertTrue(torch.equal(old[:, :3], self.point.expand(2, 3)))
        events = [([1/48, 9/48], [3, 2], [False, True], [[0, 0, 0], [float('nan'), 0, 0]]),
                  ([2/48, float('nan')], [3, 2], [True, True], [[1, 2, 3]]*2),
                  ([4/48, 10/48], [3, 4], [True, True], [[1, 2, 3]]*2)]
        for time, serve, valid, position in events:
            actual = mean.observe(position, time, serve, valid)
            learned = frozen.observe(position, time, serve, valid)
            self.assertTrue(torch.equal(actual[:, 3], learned[:, 3]))
            self.assertEqual(mean.hook.last, frozen.hook.last)
            self.assertEqual([len(h) for h in mean.hook.history], [len(h) for h in frozen.hook.history])
        self.assertEqual([len(h) for h in mean.hook.history], [1, 1])

    def test_checkpoint_exact_constant_and_history_restart(self):
        mean = FixedMeanPredictor(2)
        for tick in range(5):
            mean.observe(np.zeros((2, 3)), [tick/48]*2, [1, 2], [True]*2)
        state = mean.state_dict()
        self.assertEqual(state['mode'], 'mean')
        self.assertEqual(state['predictor']['mode'], 'mean')
        self.assertEqual(state['predictor']['constant_point_m'], self.point.tolist())
        self.assertEqual(state['predictor']['updates'], 0)
        self.assertNotIn('optimizer', state['predictor'])
        self.assertNotIn('rng', state['predictor'])
        self.assertNotIn('buffer', state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mean.pt'
            torch.save(state, path)
            mean.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.assertEqual(mean.hook.history, [[], []])
        self.assertEqual(mean.hook.last, [None, None])
        self.assertFalse(mean.hook.current().any())
        state['predictor']['model']['point'][0] = 0
        self.assertTrue(torch.equal(mean.predictor.predict(torch.zeros(1, 15))[0], self.point))

    def test_rejects_learned_online_changed_constant_and_training_state(self):
        mean = FixedMeanPredictor(1)
        baseline = mean.state_dict()
        changes = [lambda s:s.update(mode='online'), lambda s:s.update(mode='frozen'),
                   lambda s:s.pop('mode'), lambda s:s.update(schema='wrong'),
                   lambda s:s.update(buffer=[]),
                   lambda s:s['provenance'].update(artifact_sha256='0'*64),
                   lambda s:s['predictor'].update(mode='frozen'),
                   lambda s:s['predictor'].update(trained=False),
                   lambda s:s['predictor'].update(updates=1),
                   lambda s:s['predictor'].update(updates=False),
                   lambda s:s['predictor'].update(optimizer={}),
                   lambda s:s['predictor'].update(rng=torch.zeros(1)),
                   lambda s:s['predictor']['config'].update(history=4),
                   lambda s:s['predictor'].update(constant_point_m=[0, 0, 0])]
        for mutate in changes:
            state = copy.deepcopy(baseline); mutate(state)
            with self.assertRaises(ValueError):
                mean.load_state_dict(state)
        for value in (0., float('nan'), float('inf')):
            state = copy.deepcopy(baseline)
            state['predictor']['model']['point'][0] = value
            with self.assertRaises(ValueError):
                mean.load_state_dict(state)
        with self.assertRaises(ValueError):
            mean.load_state_dict(FrozenPredictor(1).state_dict())
        mean.predictor.model.point[0] = 0
        with self.assertRaisesRegex(ValueError, 'constant changed'):
            mean.state_dict()

    def test_rejects_changed_artifact_mean_sources_frames_and_recipe(self):
        changes = [lambda m:m.update(schema='wrong'), lambda m:m.update(mode='frozen'),
                   lambda m:m.update(constant_point_m=[0, 0, 0]),
                   lambda m:m['input'].update(shape=[4, 3]),
                   lambda m:m['input'].update(order='newest_first_xyz'),
                   lambda m:m['input'].update(frame='world'), lambda m:m['output'].update(unit='mm'),
                   lambda m:m['input'].update(policy_hz=50),
                   lambda m:m['settings'].update(history=4),
                   lambda m:m['recipe'].update(reduction='mean_across_windows'),
                   lambda m:m['recipe'].update(validation_used=True),
                   lambda m:m['distribution'].update(serve_position_m=[.5, 0, 2.2]),
                   lambda m:m['runtime_source_sha256'].pop('configs/pace.json'),
                   lambda m:m['runtime_source_sha256'].update({'src/airbot_tt/pace_predictor.py':'0'*64}),
                   lambda m:m['files']['cross'].update(sha256='0'*64),
                   lambda m:m['files']['source_snapshot'].update(sha256='0'*64),
                   lambda m:m['files'].pop('train'), lambda m:m.update(model_state_sha256='0'*64)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'artifact.json'
            for mutate in changes:
                manifest = copy.deepcopy(self.manifest); mutate(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    FixedMeanPredictor(1, path)

    def test_environment_must_match_learned_distribution_and_timing(self):
        mean = FixedMeanPredictor(1)
        distribution = copy.deepcopy(self.manifest['distribution'])
        mean.validate_environment(distribution, 960, 20)
        for field in ('serve_position_m', 'serve_velocity_m_s', 'position_half_range_m', 'velocity_half_range_m_s'):
            changed = copy.deepcopy(distribution); changed[field][-1] += .01
            with self.assertRaises(ValueError):
                mean.validate_environment(changed, 960, 20)
        for hz, decimation in ((480, 10), (960, 19)):
            with self.assertRaises(ValueError):
                mean.validate_environment(distribution, hz, decimation)


class FrozenPredictorTests(unittest.TestCase):
    """CPU-only reuse of the selected artifact; never fits or opens held-out data.

    Run this class explicitly when new predictor training is not authorized.
    Missing local evidence is a failure, never a random model or a skipped pass.
    """
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = json.loads((ROOT/'configs/pace_frozen.json').read_text())
        with np.load(ROOT/cls.manifest['files']['validation']['path']) as data:
            cls.data = {k:data[k].copy() for k in data.files}
        d = cls.data
        cls.x, cls.y, cls.ends = windows(d['state'][:,:3], d['time'], d['serve'],
                                        np.ones(len(d['time']), bool), d['target'], d['valid'])

    def test_real_selected_model_matches_existing_inference(self):
        frozen = FrozenPredictor(1)
        original = Predictor()
        original.load(ROOT/self.manifest['files']['checkpoint']['path'])
        expected = original.predict(self.x)
        actual = frozen.predictor.predict(self.x)
        self.assertEqual(len(self.x), 1223)
        self.assertTrue(torch.equal(actual, expected))
        teacher_rmse = float(np.sqrt(np.mean(np.sum((actual.numpy()-self.y)**2, axis=1))))
        physical_rmse = float(np.sqrt(np.mean(np.sum((actual.numpy()-self.data['truth'][self.ends])**2, axis=1))))
        receipt = json.loads((ROOT/self.manifest['files']['training']['path']).read_text())
        self.assertLess(teacher_rmse, .03)
        self.assertAlmostEqual(physical_rmse, receipt['best_validation_rmse_m'], places=8)
        self.assertEqual(frozen.predictor.updates, 20)

    def test_frozen_constructs_no_optimizer_and_cannot_train(self):
        from unittest.mock import patch
        before = torch.get_rng_state().clone()
        with patch('torch.optim.Adam', side_effect=AssertionError('Optimizer constructed')):
            # Detect accidental CUDA seeding without requiring or touching a GPU.
            with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA RNG touched')) as cuda_seed:
                frozen = FrozenPredictor(2)
            cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIsNone(frozen.predictor.optimizer)
        self.assertIsNone(frozen.predictor.rng)
        self.assertFalse(hasattr(frozen, 'buffer'))
        self.assertFalse(hasattr(frozen, 'collect'))
        self.assertFalse(hasattr(frozen, 'update'))
        result = frozen.predictor.predict(torch.tensor(self.x[:2], requires_grad=True))
        self.assertFalse(result.requires_grad)
        self.assertFalse(frozen.predictor.model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in frozen.predictor.model.parameters()))
        with self.assertRaisesRegex(RuntimeError, 'cannot train'):
            frozen.predictor.fit_epoch(self.x[:1], self.y[:1])

    def test_checkpoint_embeds_fixed_weights_and_restarts_history(self):
        frozen = FrozenPredictor(2)
        for tick, point in enumerate(self.x[0].reshape(5, 3)):
            frozen.observe(np.tile(point, (2, 1)), [tick/48]*2, [1, 2], [True]*2)
        state = frozen.state_dict()
        self.assertEqual(state['mode'], 'frozen')
        self.assertNotIn('optimizer', state['predictor'])
        self.assertNotIn('rng', state['predictor'])
        self.assertNotIn('buffer', state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frozen.pt'
            torch.save(state, path)
            restored = FrozenPredictor(2)
            restored.observe(np.zeros((2, 3)), [0, 0], [1, 2], [True]*2)
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.assertEqual(restored.hook.history, [[], []])
        self.assertEqual(restored.hook.last, [None, None])
        self.assertTrue(torch.equal(frozen.predictor.predict(self.x), restored.predictor.predict(self.x)))
        first = next(iter(state['predictor']['model']))
        state['predictor']['model'][first].add_(1)
        self.assertTrue(torch.equal(frozen.predictor.predict(self.x), restored.predictor.predict(self.x)))

    def test_rejects_online_wrong_weights_untrained_and_provenance(self):
        frozen = FrozenPredictor(1)
        baseline = frozen.state_dict()
        changes = [lambda s:s.update(mode='online'), lambda s:s.update(schema='wrong'),
                   lambda s:s.pop('mode'), lambda s:s['provenance'].update(artifact_sha256='0'*64),
                   lambda s:s['predictor'].update(mode='online'),
                   lambda s:s['predictor'].update(trained=False),
                   lambda s:s['predictor'].update(updates=1),
                   lambda s:s['predictor'].update(optimizer={}),
                   lambda s:s['predictor'].update(schema='wrong')]
        for mutate in changes:
            state = copy.deepcopy(baseline); mutate(state)
            with self.assertRaises(ValueError):
                frozen.load_state_dict(state)
        for value in (1., float('nan'), float('inf')):
            state = copy.deepcopy(baseline)
            first = next(iter(state['predictor']['model']))
            state['predictor']['model'][first].flatten()[0] = value
            with self.assertRaises(ValueError):
                frozen.load_state_dict(state)
        with self.assertRaises(ValueError):
            Predictor().load_state_dict(baseline['predictor'])

    def test_rejects_bad_artifact_contract_and_source(self):
        mutations = [lambda m:m.update(schema='wrong'), lambda m:m.update(mode='online'),
                     lambda m:m['input'].update(shape=[4, 3]),
                     lambda m:m['input'].update(order='newest_first_xyz'),
                     lambda m:m['input'].update(frame='world'),
                     lambda m:m['input'].update(unit='mm'),
                     lambda m:m['input'].update(policy_hz=50),
                     lambda m:m['input'].update(normalization='standard'),
                     lambda m:m['distribution'].update(serve_position_m=[.5, 0, 2.2]),
                     lambda m:m['historical_source_sha256'].update({'src/airbot_tt/pace_predictor.py':'0'*64}),
                     lambda m:m['runtime_source_sha256'].update({'configs/ball.json':'0'*64}),
                     lambda m:m['files']['checkpoint'].update(sha256='0'*64),
                     lambda m:m['files'].pop('training')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'artifact.json'
            for mutate in mutations:
                manifest = copy.deepcopy(self.manifest); mutate(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    FrozenPredictor(1, path)

    def test_rejects_incompatible_environment_distribution(self):
        frozen = FrozenPredictor(1)
        distribution = copy.deepcopy(self.manifest['distribution'])
        frozen.validate_environment(distribution, 960, 20)
        for field in ('serve_position_m', 'serve_velocity_m_s', 'position_half_range_m', 'velocity_half_range_m_s'):
            changed = copy.deepcopy(distribution); changed[field][-1] += .01
            with self.assertRaises(ValueError):
                frozen.validate_environment(changed, 960, 20)
        for hz, decimation in ((480, 10), (960, 19)):
            with self.assertRaises(ValueError):
                frozen.validate_environment(distribution, hz, decimation)

    def test_history_same_step_reset_invalid_and_independent_envs(self):
        frozen = FrozenPredictor(2)
        positions = self.data['state'][:9, :3]
        for tick in range(9):
            point = np.tile(positions[tick], (2, 1))
            result = frozen.observe(point, [tick/48]*2, [1, 2], [True]*2)
            point[:] = 100  # Caller mutation must not rewrite historical inputs.
            if tick < 4:
                self.assertFalse(result[:, 3].any())
            else:
                expected = frozen.predictor.predict(np.tile(positions[tick-4:tick+1].reshape(1, 15), (2, 1)))
                self.assertTrue(torch.equal(result[:, :3], expected))
                self.assertTrue(result[:, 3].all())
        old_output = result
        saved = result.clone()
        result = frozen.observe(np.tile(positions[0], (2, 1)), [0, 9/48], [3, 2], [True]*2, ids=[0])
        self.assertEqual([len(h) for h in frozen.hook.history], [1, 5])
        self.assertFalse(result[:, 3].any())
        np.testing.assert_array_equal(frozen.hook.history[1], positions[4:9])
        self.assertTrue(torch.equal(frozen.hook.current()[1, :3],
                                   frozen.predictor.predict(positions[4:9].reshape(1, 15))[0]))
        frozen.observe([[0, 0, 0], [float('nan'), 0, 1]], [1/48, 9/48], [3, 2], [False, True])
        self.assertEqual(frozen.hook.history, [[], []])
        for tick in range(5):
            frozen.observe(np.tile(positions[tick], (2, 1)), [tick/48]*2, [3, 4], [True]*2)
        result = frozen.observe(np.tile(positions[5], (2, 1)), [7/48, 5/48], [3, 4], [True]*2)
        self.assertEqual(result[:, 3].tolist(), [0, 1])
        self.assertFalse(result.requires_grad)
        result = frozen.observe(np.tile(positions[6], (2, 1)), [8/48, float('nan')], [3, 4], [True]*2)
        self.assertFalse(result[:, 3].any())
        self.assertEqual(frozen.hook.history[1], [])
        self.assertIsNone(frozen.hook.last[1])
        self.assertTrue(torch.equal(old_output, saved))


if __name__=='__main__':unittest.main()
