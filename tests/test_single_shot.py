"""Single-shot event ordering and explicit positioning deadline rejection."""
import unittest
import numpy as np
from airbot_tt.ball import load_ball_config
from airbot_tt.single_shot import ShotEvents
from airbot_tt.single_shot import positioning_candidate
from airbot_tt.tool import racket_model
from airbot_tt.config import ROOT
from airbot_tt.collision import CollisionChecker
import json


class ShotEventTests(unittest.TestCase):
    def test_table_mount_interface_and_other_contacts(self):
        model = racket_model(ROOT/'configs/racket.json')
        config = json.loads((ROOT/'configs/single_shot.json').read_text())
        ball = load_ball_config()
        base = np.array(config['base_position_m'])
        table = dict(name='table', center_m=[0, 0, ball['table_top_m']-ball['table_size_m'][2]/2],
                     size_m=ball['table_size_m'])
        checker = CollisionChecker(model, [table], base_position=base, base_mount_obstacle='table')
        points = np.concatenate([shape.points for shape in checker._shapes[0]]) + base
        self.assertAlmostEqual(points[:,2].min(), ball['table_top_m'], places=7)
        self.assertEqual(base[1], 0)
        self.assertGreaterEqual(points[:,0].min(), -ball['table_size_m'][0]/2)
        self.assertLess(points[:,0].min()+ball['table_size_m'][0]/2, .02)
        checker.require_clear(config['initial_q_rad'])
        unfiltered = CollisionChecker(model, [table], base_position=base)
        self.assertIn(('base_link', 'table'), unfiltered.check(config['initial_q_rad']))
        # Filtering the fixed seat must not hide a moving link penetrating the table.
        intrusive = dict(table, center_m=[0,0,1.0], size_m=[2.74,1.525,1.0])
        bad = CollisionChecker(model, [intrusive], base_position=base, base_mount_obstacle='table')
        self.assertTrue(any(b == 'table' and a != 'base_link' for a,b in bad.check(config['initial_q_rad'])))

    def test_positioning_limits_and_deadline(self):
        m=racket_model(ROOT/'configs/racket.json')
        c=json.loads((ROOT/'configs/single_shot.json').read_text())
        q=c['initial_q_rad']; checker=CollisionChecker(m,base_position=c['base_position_m'])
        self.assertFalse(positioning_candidate(m,checker,q,m.fk(q),.01,c)['accepted'])
        self.assertTrue(positioning_candidate(m,checker,q,m.fk(q),.5,c)['accepted'])
        target=m.fk(q);target[0,3]=10
        self.assertEqual(positioning_candidate(m,checker,q,target,1,c)['reason'],'unreachable')

    def test_legal_return_and_reset(self):
        s=ShotEvents(load_ball_config(),2)
        s.update(.2,[-.4,0,.78,-3,0,2],['table'])
        s.update(.3,[-.8,0,1,-3,0,1])
        s.update(.4,[-1,0,1,3,0,1],['paddle'])
        s.update(.5,[-.2,0,1.1,3,0,0])
        s.update(.6,[.2,0,1.1,3,0,-1])
        self.assertEqual(s.phase,'crossed')
        self.assertEqual(s.update(.8,[.7,0,.78,3,0,2],['table']),'legal_return')
        s.reset();self.assertEqual(s.events,[]);self.assertIsNone(s.outcome)

    def test_faults_and_contact_debouncing(self):
        for contact,outcome in [('paddle','illegal_hit'),('net',None),('fixer','unexpected_contact'),('arm','unexpected_contact')]:
            s=ShotEvents(load_ball_config(),2)
            self.assertEqual(s.update(.1,[-.5,0,1,1,0,0],[contact]),outcome)
        s=ShotEvents(load_ball_config(),2)
        s.update(.1,[-.4,0,.78,-3,0,2],['table'])
        s.update(.11,[-.42,0,.79,-3,0,2],['table'])
        self.assertIsNone(s.outcome)
        self.assertEqual(len(s.events),1)
        self.assertEqual(s.update(2,[-1,0,1,0,0,0]),'timeout')

    def test_second_own_bounce_and_low_net(self):
        s=ShotEvents(load_ball_config(),2)
        s.update(.1,[-.4,0,.78,-3,0,2],['table']);s.update(.2,[-.5,0,1,-3,0,0])
        self.assertEqual(s.update(.3,[-.6,0,.78,-3,0,2],['table']),'wrong_bounce')
        s.reset();s.update(.1,[-.4,0,.78,-3,0,2],['table'])
        s.update(.2,[-.5,0,1,-3,0,0]);s.update(.3,[-1,0,1,3,0,1],['paddle'])
        s.update(.4,[-.1,0,.85,3,0,0])
        self.assertIsNone(s.update(.5,[.1,0,.85,3,0,0]))
        self.assertEqual(s.phase,'crossed')
