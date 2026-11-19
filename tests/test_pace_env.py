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


if __name__=='__main__': unittest.main()
