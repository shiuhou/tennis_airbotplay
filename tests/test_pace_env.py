"""CPU tensor contracts before any PPO; physical integration lives in pace_m7.py."""
import unittest
import numpy as np
import torch
from airbot_tt.ball import load_ball_config
from airbot_tt.pace_control import PositionCommand, TensorShotEvents
from airbot_tt.single_shot import ShotEvents


class TensorEventsTests(unittest.TestCase):
    def test_release_is_once_after_contact_and_partial_reset_isolated(self):
        from airbot_tt.pace_control import TensorReleaseLatch
        latch=TensorReleaseLatch(2,'cpu')
        state=torch.tensor([[-.8,0,1.2,3.,0,1.]]*2)
        tcp=state[:,:3]-torch.tensor([.02,0,0]);normal=torch.tensor([[1.,0,0]]*2);velocity=torch.zeros(2,3)
        def tick(t,touch,legal):
            return latch.update(torch.tensor([t,t]),state,torch.tensor(touch),torch.tensor(legal),tcp,normal,velocity)
        self.assertFalse(tick(.1,[True,True],[True,True]).any())
        self.assertEqual(tick(.101,[False,True],[True,True]).tolist(),[True,False])
        self.assertFalse(tick(.102,[False,True],[True,True]).any())
        self.assertEqual(tick(.103,[False,False],[True,True]).tolist(),[False,True])
        latch.reset([0]);self.assertEqual(latch.reason.tolist(),[0,2])
        self.assertFalse(tick(.2,[False,False],[True,True]).any())
        self.assertFalse(tick(.21,[True,False],[True,True]).any())
        self.assertEqual(tick(.211,[False,False],[True,True]).tolist(),[True,False])

    def test_release_rejects_illegal_timeout_and_motion_towards_paddle(self):
        from airbot_tt.pace_control import TensorReleaseLatch
        latch=TensorReleaseLatch(3,'cpu')
        state=torch.tensor([[-.8,0,1.2,3.,0,1.]]*3);tcp=state[:,:3]-torch.tensor([.02,0,0]);normal=torch.tensor([[1.,0,0]]*3)
        latch.update(torch.zeros(3),state,torch.ones(3,dtype=torch.bool),torch.tensor([False,True,True]),tcp,normal,torch.zeros(3,3))
        state[2,3]=-3.
        result=latch.update(torch.tensor([.01,.051,.01]),state,torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),tcp,normal,torch.zeros(3,3))
        self.assertFalse(result.any());self.assertEqual(latch.reason.tolist(),[0,4,1])
        result=latch.update(torch.full((3,),.02),state,torch.zeros(3,dtype=torch.bool),torch.tensor([False,True,False]),tcp,normal,torch.zeros(3,3))
        self.assertFalse(result.any());self.assertEqual(latch.reason.tolist(),[0,4,3])

    def test_matches_existing_event_machine(self):
        c=load_ball_config()
        cases=[[(.1,[-.5,0,.78,0,0,1],['table']),(.3,[-.9,0,1.1,2,0,0],['paddle']),(.4,[.1,0,1.1,2,0,-1],[]),(.6,[.7,0,.78,2,0,-1],['table'])],
               [(.1,[-.5,0,1,0,0,0],['paddle'])],[(.1,[-.5,0,1,0,0,0],['handle'])],
               [(.1,[-.5,0,.6,0,0,-1],[])],[(3.,[-.5,0,1,0,0,0],[])]]
        for case in cases:
            old=ShotEvents(c,2.);new=TensorShotEvents(1,'cpu',c,2.)
            for t,state,contacts in case:
                old.update(t,state,contacts)
                mask=torch.tensor([[k in contacts for k in ['table','net','paddle','handle']]])
                new.update(torch.tensor([t]),torch.tensor([state]),mask,torch.tensor([False]))
                name=new.names[int(new.outcome[0])]
                self.assertEqual(old.outcome,None if name=='none' else name)
                self.assertEqual(['incoming','own_bounce','hit','crossed'][int(new.phase[0])],old.phase)

    def test_reset_and_contact_latch(self):
        events=TensorShotEvents(2,'cpu',load_ball_config(),2.)
        events.phase[:]=1
        state=torch.tensor([[-.9,0,1,2,0,0]]*2)
        mask=torch.tensor([[False,False,True,False]]*2)
        first=events.update(torch.tensor([.2,.2]),state,mask,torch.zeros(2,dtype=torch.bool))
        second=events.update(torch.tensor([.21,.21]),state,mask,torch.zeros(2,dtype=torch.bool))
        self.assertTrue(first.all());self.assertFalse(second.any())
        events.reset([0]);self.assertEqual(int(events.phase[1]),2);self.assertTrue(events.hit[1])

    def test_rally_net_contact_continues_until_real_outcome_in_both_engines(self):
        prefix=[(.1,[-.4,0,.78,-3,0,2],['table']),(.2,[-.8,0,1,3,0,1],['paddle']),(.25,[-.2,0,1,3,0,0],[])]
        cases=[
            # Touch before crossing, repeated contact, then cross and land in one tick.
            ([(.3,[-.01,0,.93,1,0,-1],['net']),(.31,[-.005,0,.94,1,0,-1],['net']),
              (.4,[.01,0,.78,1,0,1],['table'])],'legal_return'),
            # Centre has crossed before the trailing sphere grazes the net.
            ([(.3,[.001,0,.94,1,0,-1],[]),(.31,[.01,0,.93,1,0,-1],['net']),
              (.4,[.1,0,.78,1,0,1],['table'])],'legal_return'),
            ([(.3,[-.01,0,.9,-1,0,0],['net']),(.4,[-.1,0,.78,-1,0,1],['table'])],'wrong_bounce'),
            ([(.3,[-.01,0,.9,1,0,0],['net']),(.4,[.1,0,.94,1,0,0],[]),
              (.5,[2.1,0,.9,1,0,0],[])],'out'),
            ([(.3,[-.01,0,.9,0,0,0],['net']),(2.,[-.01,0,.9,0,0,0],['net'])],'timeout'),
            ([(.3,[-.01,0,.9,0,0,0],['net','paddle'])],'illegal_hit'),
            ([(.3,[-.01,0,.9,0,0,0],['net','handle'])],'unexpected_contact'),
        ]
        for suffix,expected in cases:
            with self.subTest(expected=expected,suffix=suffix):
                scalar=ShotEvents(load_ball_config(),2.)
                tensor=TensorShotEvents(2,'cpu',load_ball_config(),2.)
                sequence=prefix+suffix
                for j,(t,state,contacts) in enumerate(sequence):
                    scalar.update(t,state,contacts)
                    mask=torch.tensor([[k in contacts for k in ['table','net','paddle','handle']]]*2)
                    tensor.update(torch.tensor([t,t]),torch.tensor([state,state]),mask,torch.zeros(2,dtype=torch.bool))
                    actual=tensor.names[int(tensor.outcome[0])]
                    self.assertEqual(actual,scalar.outcome or 'none')
                    if j<len(sequence)-1:self.assertIsNone(scalar.outcome)
                self.assertEqual(scalar.outcome,expected)
                self.assertTrue(scalar.net_touched);self.assertTrue(tensor.net_touched.all())
                self.assertEqual(sum(e['kind']=='net_contact' for e in scalar.events),1)
                tensor.reset([0]);self.assertFalse(tensor.net_touched[0]);self.assertTrue(tensor.net_touched[1])
                scalar.reset();self.assertFalse(scalar.net_touched)

    def test_command_bounds_and_acceleration(self):
        control=PositionCommand(torch.zeros(6),torch.ones(6)*.6,-torch.ones(6),torch.ones(6),2,1.5,8,1/960)
        control.set_action(torch.ones(2,6)*10)
        self.assertTrue(torch.allclose(control.request,torch.ones(2,6)*.6))
        for _ in range(600):
            old=control.v.clone();q,v=control.step()
            self.assertLessEqual(float((v-old).abs().max()),8/960+1e-6)
            self.assertLessEqual(float(v.abs().max()),1.5+1e-6)
            self.assertLessEqual(float(q.abs().max()),1.+1e-6)
        other=control.q[1].clone();control.reset([0]);self.assertTrue(torch.equal(other,control.q[1]))

    def test_terminal_value_bootstrap_once_in_original_storage(self):
        import sys
        from airbot_tt.config import ROOT
        from airbot_tt.pace_control import bootstrap_timeout
        sys.path.insert(0,str(ROOT/'scripts'))
        from train_pace import original_ppo
        ActorCritic,PPO=original_ppo()
        policy=ActorCritic(2,2,1,actor_hidden_dims=[4],critic_hidden_dims=[4])
        ppo=PPO(policy,gamma=.9,device='cpu')
        ppo.init_storage('rl',2,1,[2],[2],[1])
        ppo.act(torch.zeros(2,2),torch.zeros(2,2))
        corrected=bootstrap_timeout(torch.ones(2),torch.tensor([[10.],[20.]]),torch.tensor([False,True]),.9)
        ppo.process_env_step(corrected,torch.ones(2,dtype=torch.bool),{})
        ppo.compute_returns(torch.zeros(2,2))
        torch.testing.assert_close(ppo.storage.returns[0,:,0],torch.tensor([1.,19.]))


class UpdateDiagnosticTests(unittest.TestCase):
    def test_paired_diagnostic_preserves_original_update_and_rejects_mismatched_logp(self):
        import copy
        import sys
        from airbot_tt.config import ROOT
        sys.path.insert(0,str(ROOT/'scripts'))
        from train_pace import original_ppo
        from pace_update import compare_updates,distribution_stats
        ActorCritic,PPO=original_ppo()
        torch.manual_seed(73)
        policy=ActorCritic(2,2,1,actor_hidden_dims=[4],critic_hidden_dims=[4])
        algo=PPO(policy,num_learning_epochs=2,num_mini_batches=2,learning_rate=1e-4,device='cpu')
        algo.init_storage('rl',2,4,[2],[2],[1])
        with torch.no_grad():
            for _ in range(4):
                obs=torch.randn(2,2)
                algo.act(obs,obs)
                algo.process_env_step(torch.tensor([.1,.2]),torch.zeros(2,dtype=torch.bool),{})
            algo.compute_returns(obs)
        initial=copy.deepcopy(policy.state_dict());rng=torch.get_rng_state().clone()
        paired=compare_updates(algo,[1e-4,1e-5])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in initial.items():torch.testing.assert_close(policy.state_dict()[key],value,rtol=0,atol=0)
        plain=copy.deepcopy(algo);loss=plain.update()
        self.assertEqual(loss,paired[0]['losses'])
        self.assertEqual(distribution_stats(plain),paired[0]['final'])
        algo.storage.actions_log_prob.add_(1)
        with self.assertRaisesRegex(AssertionError,'storage mismatch'):
            compare_updates(algo,[1e-4])




class FrozenObservationTests(unittest.TestCase):
    """Real PaceEnv tensor methods with synthetic state; NOT a PhysX integration test."""

    @staticmethod
    def tensor_env(group):
        import ast
        from types import SimpleNamespace
        from airbot_tt.config import ROOT
        from airbot_tt.pace_predictor import FrozenPredictor, OnlinePredictor, FixedMeanPredictor
        # Importing the real module requires AppLauncher. Compile the unchanged
        # class only; exercise its actual step/frames/observations ordering with
        # an explicit synthetic physics supplier, never a simulator substitute PASS.
        tree=ast.parse((ROOT/'src/airbot_tt/pace_env.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PaceEnv')
        scope=dict(torch=torch,np=np)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'pace_env.py','exec'),scope)
        env=scope['PaceEnv'].__new__(scope['PaceEnv'])
        env.n=2;env.ids=torch.arange(2);env.device='cpu';env.group=group
        env.predictor_mode='mean' if group=='E-mean' else ('frozen' if group=='E-frozen' else ('online' if group in ('E','E-online') else 'none'))
        env.frozen=FixedMeanPredictor(2) if env.predictor_mode=='mean' else (FrozenPredictor(2) if env.predictor_mode=='frozen' else None)
        env.online=OnlinePredictor(2) if env.predictor_mode=='online' else None
        env.prediction_collect=True;env.actor_width=27 if env.predictor_module else 23
        env.cfg=dict(observation_clip=10.,reward=dict(contact=1.,legal_return=5.,action_rate=-.001,fault=-1.))
        env.policy_dt=1/48;env.decimation=1;env.shot=dict(timeout_s=2.)
        env.joints=list(range(6));env.nominal=torch.zeros(6)
        env.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2,6),joint_vel=torch.zeros(2,6)))
        env.previous_action=torch.zeros(2,6);env.prediction=torch.zeros(2,4)
        env.policy_index=torch.zeros(2,dtype=torch.long);env.serve_id=torch.ones(2,dtype=torch.long)
        env.elapsed=torch.zeros(2);env.first_prediction=torch.full((2,),-1.);env.first_hint_used=env.first_prediction.clone()
        env.hint_steps=torch.zeros(2,dtype=torch.long);env.prehit_steps=env.hint_steps.clone();env.hit_reward_steps=env.hint_steps.clone()
        env.hint_missing=torch.zeros(2,3,dtype=torch.long)
        env.hint_reference_steps=torch.zeros(2,dtype=torch.long);env.hint_error_sq_sum=torch.zeros(2)
        env.teacher_valid=torch.ones(2,dtype=torch.bool);env.teacher_time=torch.ones(2);env.teacher_point=torch.tensor([[-1.8,0,1.67]]*2)
        env.episode_guidance=torch.zeros(2);env.episode_return=torch.zeros(2);env.episode_reward=torch.zeros(2)
        env.events=TensorShotEvents(2,'cpu',load_ball_config(),2.)
        env.release=SimpleNamespace(reason=torch.zeros(2,dtype=torch.long))
        env.return_prediction_valid=torch.zeros(2,dtype=torch.bool)
        env.command=PositionCommand(torch.zeros(6),torch.ones(6)*.6,-torch.ones(6),torch.ones(6),2,1.5,8,1/960)
        env.position=torch.tensor([[.5,0,2.3],[.5,.01,2.3]])
        env.state=lambda:torch.cat((env.position,torch.zeros(2,3)),1)
        env.tcp=lambda:(torch.zeros(2,3),torch.zeros(2,4),torch.zeros(2,3))
        def synthetic_physics():
            env.position=env.position+torch.tensor([-.06,0,-.04]);env.elapsed+=env.policy_dt
            env.episode_guidance+=.002;env.episode_return+=.003
            return torch.zeros(2,dtype=torch.bool)
        env.physics_step=synthetic_physics
        if env.predictor_module:env.refresh_prediction(env.ids)
        actor,critic=env.frames()
        env.actor_history=actor[:,None].repeat(1,5,1);env.critic_history=critic[:,None].repeat(1,5,1)
        return env

    def test_same_step_nested_history_critic_contract_and_snapshot(self):
        env=self.tensor_env('E-frozen');d=self.tensor_env('D')
        positions=[env.position.clone()];saved=[]
        for tick in range(1,9):
            obs,critic,reward,_,_=env.step(torch.zeros(2,6),auto_reset=False)
            direct,dc,dr,_,_=d.step(torch.zeros(2,6),auto_reset=False)
            positions.append(env.position.clone())
            self.assertEqual(obs.shape,(2,135));self.assertEqual(critic.shape,(2,200))
            torch.testing.assert_close(obs.reshape(2,5,27)[:,:,:23].reshape(2,115),direct,rtol=0,atol=0)
            torch.testing.assert_close(critic,dc,rtol=0,atol=0);torch.testing.assert_close(reward,dr,rtol=0,atol=0)
            self.assertEqual(bool(env.prediction[:,3].all()),tick>=4)
            saved.append((obs,obs.clone()))
        frames=obs.reshape(2,5,27)
        for frame,end in enumerate(range(4,9)):
            x=torch.stack(positions[end-4:end+1],dim=1).reshape(2,15)
            expected=env.frozen.predictor.predict(x)
            torch.testing.assert_close(frames[:,frame,23:26],expected,rtol=0,atol=0)
        self.assertAlmostEqual(8*env.policy_dt,1/6)
        torch.testing.assert_close(env.first_prediction,torch.full((2,),4/48))
        self.assertEqual(env.hint_steps.tolist(),[4,4])
        for returned,snapshot in saved:torch.testing.assert_close(returned,snapshot,rtol=0,atol=0)
        actor,_=env.frames();env.teacher_point.add_(.1);changed,_=env.frames()
        torch.testing.assert_close(actor,changed,rtol=0,atol=0)
        with self.assertRaisesRegex(RuntimeError,'Only E-online'):env.refresh_after_predictor_update()
        self.assertIsNone(env.online)

    def test_d_and_explicit_online_preserve_legacy_observations_without_updates(self):
        legacy=self.tensor_env('E');online=self.tensor_env('E-online');d=self.tensor_env('D')
        for _ in range(8):
            values=[env.step(torch.zeros(2,6),auto_reset=False) for env in (legacy,online,d)]
            for i in range(4):torch.testing.assert_close(values[0][i],values[1][i],rtol=0,atol=0)
            torch.testing.assert_close(values[1][0].reshape(2,5,27)[:,:,:23].reshape(2,115),values[2][0],rtol=0,atol=0)
            self.assertFalse(values[1][0].reshape(2,5,27)[:,:,26].any())
        self.assertEqual(online.online.predictor.updates,0)
        self.assertEqual(online.online.attempts,0)
        self.assertEqual(len(online.online.buffer),8)
        self.assertIsNotNone(online.online.predictor.optimizer)
        self.assertIsNone(d.predictor_module);self.assertEqual(d.actor_width,23)


    def test_mean_differs_only_in_valid_hint_xyz_and_preserves_d_reward_critic(self):
        mean=self.tensor_env('E-mean');learned=self.tensor_env('E-frozen');d=self.tensor_env('D')
        mu=torch.tensor(mean.frozen.cfg['constant_point_m'])
        for tick in range(1,9):
            m,mc,mr,md,_=mean.step(torch.zeros(2,6),auto_reset=False)
            e,ec,er,ed,_=learned.step(torch.zeros(2,6),auto_reset=False)
            direct,dc,dr,dd,_=d.step(torch.zeros(2,6),auto_reset=False)
            mf=m.reshape(2,5,27);ef=e.reshape(2,5,27)
            torch.testing.assert_close(mf[:,:,:23],ef[:,:,:23],rtol=0,atol=0)
            torch.testing.assert_close(mf[:,:,26],ef[:,:,26],rtol=0,atol=0)
            torch.testing.assert_close(mf[:,:,:23].reshape(2,115),direct,rtol=0,atol=0)
            for a,b,c in [(mc,ec,dc),(mr,er,dr),(md,ed,dd)]:
                torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a,c,rtol=0,atol=0)
            valid=mf[:,:,26].bool()
            torch.testing.assert_close(mf[:,:,23:26][valid],mu.expand(int(valid.sum()),3),rtol=0,atol=0)
            self.assertFalse(mf[:,:,23:26][~valid].any())
        self.assertEqual(mean.hint_steps.tolist(),learned.hint_steps.tolist())
        self.assertIsNone(mean.online);self.assertEqual(mean.predictor_mode,'mean')
        self.assertIsNone(mean.frozen.predictor.optimizer)


if __name__=='__main__': unittest.main()
