"""CPU official-PPO contracts; synthetic storage only, no policy experiment."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from airbot_tt.config import ROOT
from airbot_tt.pace_predictor import FrozenPredictor, FixedMeanPredictor, OnlinePredictor
from train_pace import (evaluate_initial, frozen_experiment_metadata, online_offline_validation, original_ppo, predictor_mode,
                        pair_mean_initialization,set_predictor, update_ppo, validate_predictor_checkpoint)


ARTIFACT=ROOT/'configs/pace_frozen.json'
MEAN_ARTIFACT=ROOT/'configs/pace_mean.json'


class PredictorEnvironment:
    n=2
    online=None
    frozen=None
    predictor_mode='none'

    @property
    def predictor_module(self):
        return self.frozen if self.predictor_mode in ('frozen','mean') else self.online


class FrozenRunnerTests(unittest.TestCase):
    def test_checkpoint_modes_are_explicit_and_legacy_e_remains_online(self):
        self.assertEqual([predictor_mode(g) for g in ('D','E','E-online','E-frozen')],
                         ['none','online','online','frozen'])
        frozen=FrozenPredictor(2,artifact=ARTIFACT)
        checkpoint=dict(group='E-frozen',predictor_mode='frozen',actor_dim=135,
                        predictor_provenance=frozen.provenance,predictor=frozen.state_dict())
        serialized=io.BytesIO();torch.save(checkpoint,serialized);serialized.seek(0)
        checkpoint=torch.load(serialized,map_location='cpu',weights_only=True)
        validate_predictor_checkpoint(checkpoint,'E-frozen')
        env=PredictorEnvironment()
        restored=set_predictor(env,'E-frozen',ARTIFACT,checkpoint)
        self.assertIs(env.predictor_module,env.frozen)
        self.assertIsNone(env.online)
        probe=torch.tensor([[.3,0,2.3]*5],dtype=torch.float32)
        self.assertTrue(torch.equal(restored.predictor.predict(probe),frozen.predictor.predict(probe)))
        for mutation in ({'predictor_mode':'online'},{'actor_dim':115},{'group':'E-online'},
                         {'predictor':{'mode':'online'}},{'predictor_provenance':None}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_predictor_checkpoint({**checkpoint,**mutation},'E-frozen')
        with self.assertRaises(ValueError):set_predictor(env,'E-frozen',None,checkpoint)
        with self.assertRaises(ValueError):
            set_predictor(env,'E-frozen',ARTIFACT,{**checkpoint,'predictor_provenance':{'wrong':'artifact'}})
        online=OnlinePredictor(2)
        legacy=dict(group='E',actor_dim=135,predictor=online.state_dict())
        validate_predictor_checkpoint(legacy,'E')
        with self.assertRaises(ValueError):validate_predictor_checkpoint({**legacy,'group':'E-online'},'E-online')
        explicit={**legacy,'group':'E-online','predictor_mode':'online'}
        set_predictor(env,'E-online',checkpoint=explicit)
        self.assertIs(env.predictor_module,env.online)
        self.assertIsNone(env.frozen)
        with self.assertRaises(ValueError):set_predictor(env,'E-online',ARTIFACT,explicit)
        with self.assertRaises(ValueError):validate_predictor_checkpoint(checkpoint,'D')

    def test_real_official_ppo_storage_update_cannot_change_frozen_predictor(self):
        self._assert_official_ppo_immutable('E-frozen',ARTIFACT)

    def test_real_official_ppo_storage_update_cannot_change_mean_predictor(self):
        self._assert_official_ppo_immutable('E-mean',MEAN_ARTIFACT)

    def _assert_official_ppo_immutable(self,group,artifact):
        torch.set_num_threads(1);torch.manual_seed(7199)
        env=PredictorEnvironment();frozen=set_predictor(env,group,artifact)
        self.assertFalse(hasattr(frozen,'update'));self.assertFalse(hasattr(frozen,'collect'))
        self.assertIsNone(frozen.predictor.optimizer)
        ActorCritic,PPO=original_ppo()
        cfg=json.loads((ROOT/'configs/pace_rl.json').read_text())['ppo']
        policy=ActorCritic(135,200,6,actor_hidden_dims=cfg['hidden'],critic_hidden_dims=cfg['hidden'],
                          activation='elu',init_noise_std=cfg['init_std'])
        # One bounded synthetic official update verifies integration. These
        # observations are not simulated rollouts or experimental training.
        algo=PPO(policy,num_learning_epochs=1,num_mini_batches=2,learning_rate=cfg['learning_rate'],device='cpu')
        algo.init_storage('rl',2,4,[135],[200],[6])
        before=copy.deepcopy(frozen.state_dict())
        actor_before=[p.detach().clone() for p in policy.actor.parameters()]
        critic_before=[p.detach().clone() for p in policy.critic.parameters()]
        with torch.no_grad():
            for tick in range(4):
                obs=torch.randn(2,5,27)
                obs[:,:,23:26]=frozen.predictor.predict(torch.randn(10,15)).reshape(2,5,3)
                obs[:,:,26]=1.
                critic=torch.randn(2,200)
                algo.act(obs.reshape(2,135),critic)
                algo.process_env_step(torch.tensor([1.+tick,.25+tick]),torch.zeros(2,dtype=torch.bool),{})
                stored_actor=obs.reshape(2,135).clone();last_critic=critic.clone()
                # Isaac reuses mutable tensors. Official storage must retain
                # the already-collected hint values and privileged observations.
                obs.fill_(999);critic.fill_(-999)
                self.assertTrue(torch.equal(algo.storage.observations[tick],stored_actor))
                self.assertTrue(torch.equal(algo.storage.privileged_observations[tick],last_critic))
            algo.compute_returns(last_critic)
        collected_actor=algo.storage.observations.clone()
        collected_critic=algo.storage.privileged_observations.clone()
        losses,status,refreshed=update_ppo(env,policy,algo)
        self.assertTrue(torch.equal(algo.storage.observations,collected_actor))
        self.assertTrue(torch.equal(algo.storage.privileged_observations,collected_critic))
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertEqual(status,dict(status=predictor_mode(group),optimizer_steps=0,weights_unchanged=True))
        self.assertIsNone(refreshed)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(actor_before,policy.actor.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(critic_before,policy.critic.parameters())))
        after=frozen.state_dict()
        self.assertEqual(before['provenance'],after['provenance'])
        self.assertTrue(all(torch.equal(v,after['predictor']['model'][k]) for k,v in before['predictor']['model'].items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in frozen.predictor.model.parameters()))
        self.assertNotIn('optimizer',after['predictor']);self.assertNotIn('rng',after['predictor'])

    def test_mean_pairs_all_initial_tensors_and_rng_for_all_three_seeds(self):
        torch.set_num_threads(1)
        spec=json.loads((ROOT/'configs/pace_mean_experiment.json').read_text());cfg=spec['common_cfg'];p=cfg['ppo']
        ActorCritic,PPO=original_ppo()
        for seed in spec['seeds']:
            with self.subTest(seed=seed):
                torch.manual_seed(seed);before=torch.get_rng_state().clone()
                env=PredictorEnvironment();set_predictor(env,'E-mean',MEAN_ARTIFACT)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                with redirect_stdout(io.StringIO()):
                    policy=ActorCritic(135,200,6,actor_hidden_dims=p['hidden'],critic_hidden_dims=p['hidden'],
                                      activation='elu',init_noise_std=p['init_std'])
                constructed_rng=torch.get_rng_state().clone()
                paired=pair_mean_initialization(policy,seed,cfg)
                self.assertTrue(torch.equal(constructed_rng,torch.get_rng_state()))
                self.assertEqual(paired['post_actor_cpu_rng_sha256'],spec['fixed_learned'][str(seed)]['post_actor_cpu_rng_sha256'])
                baseline=torch.load(ROOT/paired['initial']['path'],map_location='cpu',weights_only=False)
                self.assertEqual(policy.state_dict().keys(),baseline['policy'].keys())
                self.assertTrue(all(torch.equal(v,baseline['policy'][k]) for k,v in policy.state_dict().items()))
                algo=PPO(policy,learning_rate=p['learning_rate'],device='cpu')
                algo.init_storage('rl',2,4,[135],[200],[6])
                self.assertFalse(algo.optimizer.state)
                self.assertTrue(torch.equal(constructed_rng,torch.get_rng_state()))
                # A numerically different Actor, Critic or exploration std must
                # fail rather than silently receiving the baseline weights.
                for name in ('actor.0.weight','critic.0.weight','std'):
                    with torch.no_grad():policy.state_dict()[name].view(-1)[0]+=1.
                    with self.subTest(tensor=name),self.assertRaisesRegex(ValueError,'initialization differs'):
                        pair_mean_initialization(policy,seed,cfg)
                    policy.load_state_dict(baseline['policy'])
                torch.rand(1)
                with self.assertRaisesRegex(ValueError,'construction RNG'):
                    pair_mean_initialization(policy,seed,cfg)

    def test_mean_checkpoint_roundtrip_rejects_learned_mode_and_wrong_constant(self):
        mean=FixedMeanPredictor(2,artifact=MEAN_ARTIFACT)
        checkpoint=dict(group='E-mean',predictor_mode='mean',actor_dim=135,
                        predictor_provenance=mean.provenance,predictor=mean.state_dict())
        serialized=io.BytesIO();torch.save(checkpoint,serialized);serialized.seek(0)
        checkpoint=torch.load(serialized,map_location='cpu',weights_only=True)
        env=PredictorEnvironment();restored=set_predictor(env,'E-mean',MEAN_ARTIFACT,checkpoint)
        self.assertIs(env.predictor_module,env.frozen);self.assertIsNone(env.online)
        self.assertTrue(torch.equal(restored.predictor.predict(torch.zeros(2,15)),mean.predictor.predict(torch.ones(2,15))))
        for group in ('E-frozen','E-online','D'):
            with self.subTest(group=group),self.assertRaises(ValueError):validate_predictor_checkpoint(checkpoint,group)
        for mutation in ({'predictor_mode':'frozen'},{'actor_dim':115},{'predictor':{'mode':'frozen'}},
                         {'predictor_provenance':None}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_predictor_checkpoint({**checkpoint,**mutation},'E-mean')
        with self.assertRaises(ValueError):set_predictor(env,'E-mean',ARTIFACT,checkpoint)
        with self.assertRaises(ValueError):
            set_predictor(env,'E-mean',MEAN_ARTIFACT,{**checkpoint,'predictor_provenance':{'wrong':'artifact'}})
        bad=copy.deepcopy(checkpoint)
        next(iter(bad['predictor']['predictor']['model'].values())).add_(.1)
        with self.assertRaises(ValueError):set_predictor(env,'E-mean',MEAN_ARTIFACT,bad)

    def test_initial_evaluation_restores_policy_rng_and_training_serve_stream(self):
        env=SimpleNamespace(n=2,ids=torch.arange(2),serve_id=torch.full((2,),99),rng=[])
        def reset(ids):
            env.draws=np.array([rng.normal(size=3) for rng in env.rng])
            env.serve_id[ids]+=1
        env.reset=reset;env.observations=lambda:('actor','critic')
        seed=7101;torch.manual_seed(seed);before=torch.get_rng_state().clone()
        def evaluate(policy):
            torch.randn(37)
            env.rng=[np.random.default_rng(7201) for _ in range(2)]
            env.reset(env.ids)
            return {'status':'PASS'}
        result,observations=evaluate_initial(env,None,evaluate,seed)
        self.assertEqual(result,{'status':'PASS'});self.assertEqual(observations,('actor','critic'))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        expected=np.array([np.random.default_rng(np.random.SeedSequence([seed,i])).normal(size=3) for i in range(2)])
        np.testing.assert_array_equal(env.draws,expected)
        self.assertEqual(env.serve_id.tolist(),[1,1])

    def test_frozen_experiment_rejects_control_reward_or_ppo_drift(self):
        cfg=json.loads((ROOT/'configs/pace_frozen_experiment.json').read_text())['common_cfg']
        metadata=frozen_experiment_metadata(cfg)
        self.assertEqual(len(metadata['experiment_spec_sha256']),64)
        self.assertEqual(len(metadata['effective_cfg_sha256']),64)
        for section,key,value in [('ppo','learning_rate',1e-4),('reward','legal_return',6.)]:
            changed=copy.deepcopy(cfg);changed[section][key]=value
            with self.subTest(section=section),self.assertRaises(ValueError):frozen_experiment_metadata(changed)
        changed=copy.deepcopy(cfg);changed['action_scale_rad'][0]=.5
        with self.assertRaises(ValueError):frozen_experiment_metadata(changed)

    def test_mean_experiment_pins_learned_config_artifacts_and_baseline_reports(self):
        spec=json.loads((ROOT/'configs/pace_mean_experiment.json').read_text());cfg=spec['common_cfg']
        metadata=frozen_experiment_metadata(cfg,'E-mean')
        self.assertEqual(metadata['experiment_spec_path'],'configs/pace_mean_experiment.json')
        self.assertEqual(cfg,json.loads((ROOT/'configs/pace_frozen_experiment.json').read_text())['common_cfg'])
        self.assertEqual(metadata['experiment_spec_sha256'],hashlib.sha256((ROOT/'configs/pace_mean_experiment.json').read_bytes()).hexdigest())
        for key,value in [('serve_z_m',2.2),('decimation',24)]:
            changed=copy.deepcopy(cfg);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):frozen_experiment_metadata(changed,'E-mean')
        changed=copy.deepcopy(cfg);changed['ppo']['epochs']+=1
        with self.assertRaises(ValueError):frozen_experiment_metadata(changed,'E-mean')

    def test_online_historical_validation_reads_no_frozen_weights(self):
        # This gate audits actual immutable records. Loading an offline model
        # here would silently change E-online's from-scratch contract.
        with mock.patch('torch.load',side_effect=AssertionError('Unexpected offline model load')):
            evidence=online_offline_validation()
        self.assertTrue(evidence['historical_predictor_source'])
        self.assertEqual(len(evidence['source_snapshot_sha256']),64)


if __name__=='__main__':unittest.main()
