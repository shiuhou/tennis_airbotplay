"""Execute pinned upstream functions on synthetic tensors, NOT a PACE training run.

Requires local sources listed in configs/pace_sources.json and
PyTorch. Original source files are neither modified nor installed globally.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'.agents/runs/foundation/pace/source'
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and (SOURCE/'rsl_rl/rsl_rl/algorithms/ppo.py').is_file(),
                     'Requires PyTorch and locally verified upstream PACE source; see pace_pipeline.md')
class PaceSourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        manifest = json.loads((ROOT/'configs/pace_sources.json').read_text())
        assert manifest['commit'] == 'fdb192f8fff9f5ca54c4ec4a8441fda5174ca078'
        for item in manifest['files']:
            assert hashlib.sha256((SOURCE/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
        sys.path.insert(0, str(SOURCE/'rsl_rl'))
        from rsl_rl.runners import OnPolicyPredictorRegressionRunner
        assert Path(__import__('rsl_rl').__file__).resolve().is_relative_to(SOURCE)
        cls.Runner = OnPolicyPredictorRegressionRunner
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def original_function(self, path, name, class_name=None, extra=None):
        # Extract the original AST unchanged; bypass only simulator-heavy module imports.
        tree = ast.parse((SOURCE/path).read_text())
        nodes = tree.body
        if class_name:
            nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
        node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        namespace = {'torch': torch, **(extra or {})}
        exec(compile(ast.fix_missing_locations(module), str(SOURCE/path), 'exec'), namespace)
        return namespace[name]

    def env_method(self, name):
        return self.original_function('legged_lab/envs/base/tt_env.py', name, 'TTEnv')

    def runner(self, n=2):
        cfg = dict(policy=dict(class_name='ActorCritic', actor_hidden_dims=[512,512,128],
                               critic_hidden_dims=[512,512,128], activation='elu',
                               init_noise_std=1.,noise_std_type='scalar'),
                   algorithm=dict(class_name='PPO',num_learning_epochs=5,num_mini_batches=4,
                                  clip_param=.2,gamma=.95,lam=.95,entropy_coef=.006,
                                  schedule='adaptive',desired_kl=.01),
                   predictor=dict(history_len=5,traj_max_len=128,hidden_sizes=[64,64],
                                  lr=.0005,epochs_per_update=1,batch_size=1024,train_until_iters=20),
                   num_steps_per_env=24,save_interval=250,empirical_normalization=False)
        env = NS(num_envs=n,num_actions=21,device='cpu',add_noise=False,
                 get_observations=lambda:(torch.zeros(n,405),
                      {'observations':{'critic':torch.zeros(n,490)}}))
        with contextlib.redirect_stdout(io.StringIO()):
            runner = self.Runner(env,copy.deepcopy(cfg),device='cpu')
        return runner

    def test_observation_dimensions_and_prediction_update_timing(self):
        n=2; z=lambda d:torch.zeros(n,d)
        data=NS(root_ang_vel_b=z(3),projected_gravity_b=z(3),heading_w=torch.zeros(n),
                joint_pos=z(21),default_joint_pos=z(21),joint_vel=z(21),
                root_link_pos_w=z(3),root_lin_vel_w=z(3),root_lin_vel_b=z(3))
        scales=NS(**dict.fromkeys(['ang_vel','projected_gravity','joint_pos','joint_vel',
                                  'actions','perception','ball_pos','robot_pos','ball_linvel',
                                  'ball_state','lin_vel'],1.))
        env=NS(robot=NS(data=data),table=NS(data=data),ball=NS(data=data),
               contact_sensor=NS(data=NS(net_forces_w_history=torch.zeros(n,3,2,3))),
               feet_cfg=NS(body_ids=[0,1]),obs_joint_ids=list(range(21)),obs_scales=scales,
               action_buffer=NS(_circular_buffer=NS(buffer=torch.zeros(n,2,21))),
               delayed_perception=z(6),ball_prediction=z(3),ball_future_pose=z(3),
               paddle_touch_point=z(3),robot_future_pos=z(3),ball_future_t=z(1),
               ball_episode_length_buf=torch.zeros(n),episode_length_buf=torch.zeros(n),
               max_ball_episode_length=90,max_episode_length=500,
               has_touch_own_table_prev=torch.zeros(n,dtype=torch.bool),
               has_touch_paddle=torch.zeros(n,dtype=torch.bool),
               num_envs=n,device='cpu',headless=True,scene=NS(env_origins=z(3)))
        obs_fn=self.env_method('compute_current_observations_perception')
        actor,critic=obs_fn(env)
        self.assertEqual(actor.shape,(n,81));self.assertEqual(critic.shape,(n,98))
        self.assertEqual(actor[:,75:78].abs().sum().item(),0.)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env_method('update_prediction')(env,torch.ones(n,3))
        # Hook does not rebuild a previously returned observation.
        self.assertEqual(actor[:,75:78].abs().sum().item(),0.)
        next_actor,_=obs_fn(env)
        self.assertTrue(torch.all(next_actor[:,75:78]==1))
        # Critic paddle point is world-frame; no env-origin subtraction here.
        env.paddle_touch_point[:,0]=10
        _,next_critic=obs_fn(env)
        self.assertTrue(torch.all(next_critic[:,78]==10))

    def test_predictor_dataset_includes_reset_window_and_uses_end_label(self):
        r=self.runner(n=1)
        # One discontinuity represents a ball reset. Original collector ignores reset IDs.
        positions=[0.,1.,2.,3.,100.,101.]
        for i,x in enumerate(positions):
            r.env.ball_pos=torch.tensor([[x,0.,1.]])
            r.env.ball_future_pose=torch.full((1,3),float(i))
            r.env.ball_reset_ids=torch.tensor([0]) if i==4 else torch.tensor([],dtype=torch.long)
            r._record_ball_positions()
        seen=[]
        hook=r._predictor.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        # lr=0 exposes the actual dataset without changing learned parameters.
        for group in r._pred_optim.param_groups:group['lr']=0.
        expected_x=torch.tensor([[0.,0.,1.,1.,0.,1.,2.,0.,1.,3.,0.,1.,100.,0.,1.]])
        with torch.no_grad():
            expected_loss=torch.nn.functional.mse_loss(r._predictor(expected_x),torch.full((1,3),4.)).item()
        seen.clear()
        result=r._train_predictor_offline();hook.remove()
        self.assertEqual(len(seen),1)
        torch.testing.assert_close(seen[0],expected_x)
        self.assertAlmostEqual(result,expected_loss,places=5)
        # range(H,L) omits the most recent complete window ending at index L-1.
        self.assertNotIn(101.,seen[0].tolist()[0])

    def test_predictor_inputs_are_raw_positions_and_history_is_oldest_first(self):
        r=self.runner(n=1);r._pred_trained=True
        for x in range(5):
            r.env.ball_pos=torch.tensor([[float(x),0.,1.]])
            r.env.ball_future_pose=torch.zeros(1,3)
            r._record_ball_positions()
        r.env.delayed_perception=torch.full((1,6),999.)
        captured=[];output=[]
        hook=r._predictor.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
        r.env.update_prediction=lambda pred:output.append(pred)
        with contextlib.redirect_stdout(io.StringIO()):
            r._maybe_predict_and_update_env()
        hook.remove()
        self.assertEqual(captured[0][0,::3].tolist(),[0.,1.,2.,3.,4.])
        self.assertFalse(output[0].requires_grad)

    def test_original_rewards_and_non_latched_success(self):
        def reward(name):return self.original_function('legged_lab/mdp/rewards.py',name)
        env=NS(ball_future_pose=torch.zeros(4,3),paddle_pos=torch.tensor([[.6,0,0],[.3,0,0],[.15,0,0],[.02,0,0]]),
               mask_invalid=torch.zeros(4,dtype=torch.bool))
        r=reward('reward_future_ee_target')(env,std_ee=.5,threshold=.15)
        torch.testing.assert_close(r,torch.exp(-torch.tensor([.6,.3,.15,.15])/.25))
        env=NS(has_touch_paddle=torch.tensor([True]),has_touch_opponent_table_just_now=torch.tensor([True]))
        fn=reward('reward_table_success')
        self.assertEqual(fn(env).item(),1);self.assertEqual(fn(env).item(),1)
        env=NS(predict_x_land=torch.tensor([1.15,5.15]),predict_y_land=torch.zeros(2),
               ball_landing_dis_rew=torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(reward('reward_future_landing_dis')(env,threshold=3.),
                                   torch.tensor([3.,-1.]))

    def test_gae_timeout_bootstrap_and_original_ppo_update(self):
        r=self.runner(n=4);p=r.alg
        # Smaller synthetic batch; this executes original PPO, not a robot task.
        p.init_storage('rl',4,4,[405],[490],[21])
        before=next(p.policy.actor.parameters()).detach().clone()
        pred_before=[v.detach().clone() for v in r._predictor.parameters()]
        for i in range(4):
            obs=torch.randn(4,405);critic=torch.randn(4,490)
            with torch.no_grad():action=p.act(obs,critic)
            self.assertEqual(action.shape,(4,21))
            values=p.transition.values.clone()
            info={'time_outs':torch.tensor([False,False,False,i==3])}
            dones=torch.tensor([False,False,i==3,i==3])
            p.process_env_step(torch.ones(4),dones,info)
            if i==3:
                self.assertAlmostEqual(p.storage.rewards[i,3].item(),1+.95*values[3].item(),places=5)
        p.compute_returns(torch.zeros(4,490))
        self.assertAlmostEqual(p.storage.returns[3,2].item(),1.,places=5)
        self.assertAlmostEqual(p.storage.returns[3,3].item(),p.storage.rewards[3,3].item(),places=5)
        losses=p.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertFalse(torch.equal(before,next(p.policy.actor.parameters())))
        for old,new in zip(pred_before,r._predictor.parameters()):torch.testing.assert_close(old,new)
        self.assertTrue(all(v.grad is None for v in r._predictor.parameters()))

    def test_checkpoint_roundtrip_and_untrained_flag_not_saved(self):
        r=self.runner();r.logger_type='tensorboard'
        self.assertFalse(r._pred_trained)
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'synthetic.pt')
            r.save(path)
            other=self.runner()
            with contextlib.redirect_stdout(io.StringIO()):other.load(path,load_optimizer=False)
        for a,b in zip(r._predictor.parameters(),other._predictor.parameters()):torch.testing.assert_close(a,b)
        self.assertTrue(other._pred_trained)  # load enables inference even for an untrained saved predictor.
        self.assertEqual(other._traj_len,0)

    def test_original_step_order_and_action_scaling(self):
        trace=[];targets=[]
        env=NS(device='cpu',headless=True,clip_actions=100.,action_scale=.25,
               action_joint_ids=list(range(21)),sim_step_counter=0,physics_dt=.002,step_dt=.02,
               cfg=NS(sim=NS(decimation=10)),action_buffer=NS(compute=lambda a:a),
               robot=NS(data=NS(default_joint_pos=torch.full((1,21),.1)),
                        set_joint_position_target=lambda a,ids:targets.append(a.clone())),
               aero=NS(apply_to_rigid_object=lambda b:trace.append('force')),
               ball=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.05]]))),
               scene=NS(write_data_to_sim=lambda:trace.append('write'),
                        update=lambda **kw:trace.append('read')),
               sim=NS(step=lambda **kw:trace.append('physics')),
               compute_perception=lambda:trace.append('perception'),
               compute_paddle_touch=lambda:trace.append('contact'),
               episode_length_buf=torch.zeros(1,dtype=torch.long),
               ball_episode_length_buf=torch.zeros(1,dtype=torch.long),
               max_ball_episode_length=90,command_generator=NS(compute=lambda dt:None),
               event_manager=NS(available_modes=[]),
               compute_intermediate_values=lambda:trace.append('teacher'),
               reset_ball=lambda ids:trace.append('ball_reset'),
               check_reset=lambda:(torch.tensor([False]),torch.tensor([False])),
               curriculum_manager=NS(compute=lambda:None),
               reward_manager=NS(compute=lambda dt:(trace.append('reward') or torch.zeros(1))),
               reset=lambda ids:trace.append('env_reset_dispatch'),
               compute_observations=lambda:(trace.append('obs') or (torch.zeros(1,405),torch.zeros(1,490))),
               extras={'time_outs':torch.tensor([True])})
        reset_fn=self.env_method('reset')
        env.reset=lambda ids:(trace.append('env_reset_dispatch'),reset_fn(env,ids))
        self.env_method('step')(env,torch.full((1,21),2.))
        self.assertEqual(len(targets),10)
        torch.testing.assert_close(targets[0],torch.full((1,21),.6))
        self.assertEqual(trace[:6],['force','write','physics','read','perception','contact'])
        self.assertEqual(trace[-5:],['teacher','ball_reset','reward','env_reset_dispatch','obs'])
        self.assertEqual(env.sim_step_counter,10)
        self.assertFalse(env.time_out_buf.any())
        self.assertTrue(env.extras['time_outs'].any()) # old timeout survives empty reset dispatch.

    def test_teacher_invalid_label_depends_on_robot_not_only_ball(self):
        n=2;z=lambda d:torch.zeros(n,d);b=lambda:torch.zeros(n,dtype=torch.bool)
        env=NS(device='cpu',headless=True,ball_pos=torch.tensor([[-1.,0.,1.],[-1.,0.,1.]]),
               ball=NS(data=NS(root_quat_w=z(4),root_vel_w=z(6),
                              root_lin_vel_w=torch.tensor([[3.,0.,1.],[3.,0.,1.]]),
                              root_ang_vel_w=z(3))),
               robot=NS(data=NS(root_lin_vel_w=z(3))),
               robot_pos=torch.tensor([[-1.8,0.,.7],[-2.,.5,.7]]),
               ball_contact_rew=torch.zeros(n),has_touch_paddle=b(),has_touch_paddle_rew=b(),
               cfg=NS(table=NS(table_opponent_contact_x=(0.,1.35),
                    table_opponent_contact_y=(-.7625,.7625),table_opponent_contact_z=(.76,.83),
                    table_own_contact_x=(-1.37,0.),table_own_contact_y=(-.7625,.7625),
                    table_own_contact_z=(.76,.83))),
               has_touch_own_table_prev=b(),has_touch_opo_table_prev=b(),
               has_return_own_table2_prev=b(),has_first_bounce=b(),has_touch_own_table=b(),
               touch_info=[],ball_contact=torch.zeros(n),
               paddle_touch_point=z(3),scene=NS(env_origins=z(3)),ball_drag_k=.134)
        self.env_method('compute_intermediate_values')(env)
        self.assertTrue(env.mask_invalid.all()) # vx>0 invalidates incoming guidance.
        torch.testing.assert_close(env.ball_future_pose,
                                   torch.tensor([[-1.8,-.6,.89],[-2.,-.1,.89]]))
        self.assertFalse(torch.equal(env.ball_future_pose[0],env.ball_future_pose[1]))

    def test_no_noise_initialization_has_unbound_actor_obs(self):
        # Counterexample to assuming disabling noise is a harmless evaluation switch.
        fn=self.env_method('init_obs_buffer')
        fn.__globals__['CircularBuffer']=lambda **kw:NS()
        env=NS(add_noise=False,num_envs=1,device='cpu',num_perception=6,
               cfg=NS(robot=NS(actor_obs_history_length=5,critic_obs_history_length=5)))
        with self.assertRaises(UnboundLocalError):fn(env)


if __name__=='__main__':
    unittest.main(verbosity=2)
