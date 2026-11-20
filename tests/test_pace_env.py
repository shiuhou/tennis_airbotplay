"""CPU tensor contracts before any PPO; physical integration lives in pace_m7.py."""
import unittest
import numpy as np
import torch
from airbot_tt.ball import load_ball_config
from airbot_tt.pace_control import PositionCommand, TensorShotEvents
from airbot_tt.single_shot import ShotEvents


class TensorEventsTests(unittest.TestCase):
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


if __name__=='__main__': unittest.main()
