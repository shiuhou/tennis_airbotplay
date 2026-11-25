"""PACE's original MLP with AIRBOT serve-safe data and observation contracts."""
import hashlib
import importlib
import json
import copy
from pathlib import Path
import sys
import numpy as np
import torch
from .config import ROOT
from .pace_guidance import settings

SOURCE = ROOT/'.agents/runs/foundation/pace/source'


def original_mlp():
    """Use official code directly; reject changed/missing pinned dependencies."""
    manifest = json.loads((ROOT/'configs/pace_sources.json').read_text())
    for item in manifest['files']:
        path = SOURCE/item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Missing/changed official source: {path}. Run scripts/prepare_pace.py.')
    existing = sys.modules.get('rsl_rl')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError('Different rsl_rl already imported; use an isolated process.')
    if str(SOURCE/'rsl_rl') not in sys.path:
        sys.path.insert(0, str(SOURCE/'rsl_rl'))
    module = importlib.import_module('rsl_rl.runners.on_policy_predictor_regression_runner')
    return module._MLPPredictor(15, (64, 64), 3)


def windows(position, time, serve, valid, target, target_valid, *, dt=1/48):
    """Flatten oldest first, label at final sample; include last full window."""
    p, time, serve = map(np.asarray, (position, time, serve))
    valid, target, target_valid = map(np.asarray, (valid, target, target_valid))
    if p.shape != (len(time), 3) or target.shape != p.shape:
        raise ValueError('Expected [samples,3] positions and targets.')
    xs, ys, ends = [], [], []
    for end in range(4, len(p)):
        sl = slice(end-4,end+1)
        if (np.all(serve[sl] == serve[end]) and np.all(valid[sl]) and target_valid[end]
            and np.allclose(np.diff(time[sl]), dt, atol=1e-7, rtol=0)
            and np.isfinite(p[sl]).all() and np.isfinite(target[end]).all()):
            xs.append(p[sl].reshape(15)); ys.append(target[end]); ends.append(end)
    return (np.asarray(xs, dtype=np.float32).reshape(-1,15),
            np.asarray(ys, dtype=np.float32).reshape(-1,3), np.asarray(ends,dtype=int))


class Predictor:
    def __init__(self, *, inference_only=False):
        self.cfg = settings()
        self.inference_only = inference_only
        with torch.random.fork_rng(devices=[]):
            if inference_only:
                # The model is constructed on CPU. torch.manual_seed also seeds
                # CUDA, whose generators are not restored by this CPU-only fork.
                torch.random.default_generator.manual_seed(self.cfg['predictor']['seed'])
            else:
                torch.manual_seed(self.cfg['predictor']['seed'])
            self.model = original_mlp()
        self.rng = None if inference_only else torch.Generator().manual_seed(self.cfg['predictor']['seed'])
        self.optimizer = None if inference_only else torch.optim.Adam(self.model.parameters(), lr=self.cfg['predictor']['lr'])
        if inference_only:
            self.model.eval().requires_grad_(False)
        self.trained = False
        self.updates = 0

    def fit_epoch(self, x, y, *, batch_size=None):
        if self.inference_only:
            raise RuntimeError('Frozen predictor cannot train.')
        if self.updates >= self.cfg['predictor']['epochs']:
            raise RuntimeError('Predictor update budget exhausted.')
        if len(x) == 0:
            raise ValueError('No valid training windows.')
        x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Non-finite training data.')
        self.model.train()
        total = 0.
        for ids in torch.randperm(len(x), generator=self.rng).split(batch_size or self.cfg['predictor']['batch_size']):
            loss = torch.nn.functional.mse_loss(self.model(x[ids]), y[ids])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(); self.optimizer.step()
            total += loss.item()*len(ids)
        self.updates += 1
        self.trained = True
        self.optimizer.zero_grad(set_to_none=True)
        return total/len(x)

    @torch.no_grad()
    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Predictor has not been trained.')
        self.model.eval()
        return self.model(torch.as_tensor(x, dtype=torch.float32)).detach()

    def state_dict(self):
        data = dict(schema=self.cfg['schema'], config=copy.deepcopy(self.cfg),
                        source_manifest_sha256=hashlib.sha256((ROOT/'configs/pace_sources.json').read_bytes()).hexdigest(),
                        ball_config_sha256=hashlib.sha256((ROOT/'configs/ball.json').read_bytes()).hexdigest(),
                        model=self.model.state_dict(), trained=self.trained, updates=self.updates)
        if self.inference_only:
            data.update(mode='frozen', model={k:v.detach().clone() for k,v in data['model'].items()})
        else:
            data.update(optimizer=self.optimizer.state_dict(), rng=self.rng.get_state())
        return data

    def save(self, path):
        torch.save(self.state_dict(),path)

    def load(self, path):
        self.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))

    def load_state_dict(self,data):
        sha = hashlib.sha256((ROOT/'configs/pace_sources.json').read_bytes()).hexdigest()
        if (data['schema'] != self.cfg['schema'] or data['config'] != self.cfg or data['source_manifest_sha256'] != sha
            or data['ball_config_sha256'] != hashlib.sha256((ROOT/'configs/ball.json').read_bytes()).hexdigest()):
            raise ValueError('Checkpoint source/config mismatch.')
        if self.inference_only:
            if (data.get('mode') not in (None, 'frozen') or data['trained'] is not True
                or type(data['updates']) is not int or data['updates'] <= 0):
                raise ValueError('Frozen predictor requires a trained compatible checkpoint.')
            if any(not torch.isfinite(v).all() for v in data['model'].values()):
                raise ValueError('Frozen predictor has non-finite weights.')
        elif data.get('mode') == 'frozen':
            raise ValueError('Cannot resume a frozen predictor as a trainable predictor.')
        self.model.load_state_dict(data['model'], strict=True)
        self.trained, self.updates = data['trained'], data['updates']
        if self.inference_only:
            self.model.eval().requires_grad_(False)
        else:
            self.optimizer.load_state_dict(data['optimizer'])
            self.rng.set_state(data['rng'])


class PredictionHook:
    """Policy-rate same-step prediction. Independent per-env history, no truth input.

    Reset/invalid/gaps clear the five-point window. Output owns its storage;
    previously returned observations cannot be mutated by the next call.
    """
    def __init__(self, predictor, num_envs):
        self.predictor = predictor
        self.dt = predictor.cfg['decimation']/predictor.cfg['physics_hz']
        self.history = [[] for _ in range(num_envs)]
        self.last = [None]*num_envs

    def reset(self, ids):
        for i in ids:
            self.history[i] = []
            self.last[i] = None

    def observe(self, position, time, serve, valid, ids=None):
        p = np.asarray(position)
        n = len(self.history)
        if p.shape != (n,3):
            raise ValueError('Expected [env,3] observable positions.')
        output = torch.zeros((n,4), dtype=torch.float32)
        selected=range(n) if ids is None else ids
        ids, x = [], []
        for i in selected:
            if not valid[i] or not np.isfinite(p[i]).all() or not np.isfinite(time[i]):
                self.reset([i]); continue
            stamp = (int(serve[i]), float(time[i]))
            prior = self.last[i]
            if prior is not None and (prior[0]!=stamp[0] or abs(stamp[1]-prior[1]-self.dt)>1e-7):
                self.reset([i])
            self.history[i].append(p[i].copy())
            self.history[i] = self.history[i][-5:]
            self.last[i] = stamp
            if len(self.history[i])==5 and self.predictor.trained:
                ids.append(i); x.append(np.asarray(self.history[i]).reshape(15))
        if ids:
            output[ids,:3] = self.predictor.predict(np.asarray(x))
            output[ids,3] = 1.
        return output


    @torch.no_grad()
    def current(self):
        output=torch.zeros((len(self.history),4),dtype=torch.float32)
        ids=[i for i,h in enumerate(self.history) if len(h)==5]
        if ids and self.predictor.trained:
            output[ids,:3]=self.predictor.predict(np.asarray([self.history[i] for i in ids]).reshape(-1,15))
            output[ids,3]=1.
        return output


def _model_sha256(model):
    """Serialization-independent digest for the exact fixed tensor contents."""
    digest = hashlib.sha256()
    for name, value in sorted(model.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('Frozen predictor requires finite tensor weights.')
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenPredictor:
    """The selected offline model, with auditable inputs and no training state.

    Historical source hashes belong to the immutable preparation snapshot, not
    the subsequently edited inference module. No held-out raw data are opened.
    Histories intentionally restart after checkpoint load/environment reset.
    """
    def __init__(self, num_envs, artifact='configs/pace_frozen.json'):
        artifact = (ROOT / artifact).resolve()
        self.cfg = json.loads(artifact.read_text())
        expected_input = dict(shape=[5, 3], order='oldest_first_xyz',
                              frame='local_table_ground_origin', unit='m',
                              policy_hz=48, normalization='none')
        expected_output = dict(shape=[3], meaning='post_bounce_apex',
                               frame='local_table_ground_origin', unit='m')
        if (self.cfg.get('schema') != 'airbot-e-frozen-v1'
            or self.cfg.get('mode') != 'frozen'
            or self.cfg.get('input') != expected_input
            or self.cfg.get('output') != expected_output):
            raise ValueError('Incompatible frozen artifact schema/input/coordinate contract.')
        required_files = {'checkpoint', 'training', 'validation_report', 'dataset',
                          'train', 'validation', 'source_snapshot'}
        if set(self.cfg['files']) != required_files:
            raise ValueError('Incomplete frozen artifact provenance.')
        paths = {}
        for name, item in self.cfg['files'].items():
            path = ROOT / item['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Missing/changed frozen artifact: {name}: {path}')
            paths[name] = path
        records = {name:json.loads(paths[name].read_text()) for name in
                   ('training', 'validation_report', 'dataset', 'source_snapshot')}
        training, validation, dataset = (records[k] for k in ('training', 'validation_report', 'dataset'))
        cfg = settings()
        if (cfg['schema'] != 'airbot-m6-apex-v1' or cfg['history'] != 5
            or cfg['physics_hz']/cfg['decimation'] != 48
            or cfg['predictor']['hidden'] != [64, 64]
            or any(record['status'] != 'PASS' for record in (training, validation, dataset))
            or training['config'] != cfg or validation['config'] != cfg
            or training['checkpoint_sha256'] != self.cfg['files']['checkpoint']['sha256']
            or validation['checkpoint_sha256'] != training['checkpoint_sha256']
            or training['dataset_sha256'] != self.cfg['files']['dataset']['sha256']
            or validation['collection'] != dataset
            or dataset['distribution'] != self.cfg['distribution']
            or dataset['physics_hz'] != cfg['physics_hz'] or dataset['policy_hz'] != 48):
            raise ValueError('Frozen training/validation/settings/distribution mismatch.')
        for split in ('train', 'validation'):
            row = dataset['splits'][split]
            if (row['sha256'] != self.cfg['files'][split]['sha256']
                or row['collection_config'] != cfg
                or row['seed'] != cfg['splits'][split]['seed']
                or row['all_serves'] != cfg['splits'][split]['serves']):
                raise ValueError('Frozen dataset split/config mismatch.')
        historical = {}
        for record in (training, validation, dataset):
            for path, sha in record['source_sha256'].items():
                if path in historical and historical[path] != sha:
                    raise ValueError('Conflicting historical predictor sources.')
                historical[path] = sha
        if historical != self.cfg['historical_source_sha256']:
            raise ValueError('Frozen historical source manifest mismatch.')
        snapshot = records['source_snapshot']
        if snapshot['schema'] != 'e-frozen-source-snapshot-v1':
            raise ValueError('Incompatible predictor source snapshot.')
        snapshot_sources = {item['path']:item['sha256'] for item in snapshot['files']}
        for path, sha in historical.items():
            # Files inherited unchanged from the base commit need no overlay row.
            actual = snapshot_sources.get(path)
            if actual is None:
                actual = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            if actual != sha:
                raise ValueError(f'Historical predictor source mismatch: {path}')
        for path, sha in self.cfg['runtime_source_sha256'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Frozen runtime settings/source mismatch: {path}')
        data = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
        if (_model_sha256(data['model']) != self.cfg['model_state_sha256']
            or data['updates'] != validation['checkpoint_epoch']):
            raise ValueError('Frozen selected checkpoint weights/epoch mismatch.')
        self.predictor = Predictor(inference_only=True)
        self.predictor.load_state_dict(data)
        self.hook = PredictionHook(self.predictor, num_envs)
        self.provenance = copy.deepcopy(self.cfg)
        self.provenance.update(
            artifact_path=str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())

    def validate_environment(self, strike, physics_hz, decimation):
        fields = ('version', 'serve_position_m', 'serve_velocity_m_s',
                  'position_half_range_m', 'velocity_half_range_m_s', 'physics_hz')
        if (any(strike.get(k) != self.cfg['distribution'][k] for k in fields)
            or physics_hz != self.predictor.cfg['physics_hz']
            or decimation != self.predictor.cfg['decimation']):
            raise ValueError('Frozen predictor requires the validated Z=2.3m serve distribution and 48Hz input.')

    def observe(self, position, time, serve, valid, ids=None):
        return self.hook.observe(position, time, serve, valid, ids)

    def state_dict(self):
        state = self.predictor.state_dict()
        if _model_sha256(state['model']) != self.cfg['model_state_sha256']:
            raise ValueError('Frozen predictor weights changed.')
        return dict(schema='airbot-e-frozen-checkpoint-v1', mode='frozen',
                    provenance=copy.deepcopy(self.provenance), predictor=state)

    def load_state_dict(self, state):
        if (state.get('schema') != 'airbot-e-frozen-checkpoint-v1'
            or state.get('mode') != 'frozen' or state.get('provenance') != self.provenance):
            raise ValueError('Frozen predictor checkpoint mode/provenance mismatch.')
        predictor = state['predictor']
        if (predictor.get('mode') != 'frozen' or 'optimizer' in predictor or 'rng' in predictor
            or _model_sha256(predictor['model']) != self.cfg['model_state_sha256']
            or predictor['updates'] != self.predictor.updates):
            raise ValueError('Frozen predictor checkpoint weights/training state mismatch.')
        self.predictor.load_state_dict(predictor)
        self.hook.reset(range(len(self.hook.history)))


class _MeanModel(torch.nn.Module):
    """A fixed point is model state, never an optimizable parameter."""
    def __init__(self, point):
        super().__init__()
        self.register_buffer('point', torch.tensor(point, dtype=torch.float32))
        self.eval()

    def forward(self, x):
        return self.point.expand(len(x), 3).clone()


class _MeanPredictor:
    def __init__(self, point):
        self.cfg = settings()
        self.model = _MeanModel(point)
        self.optimizer = self.rng = None
        self.trained, self.updates = True, 0
        self._point = list(point)

    def fit_epoch(self, *args, **kwargs):
        raise RuntimeError('Fixed mean predictor cannot train.')

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(torch.as_tensor(x, dtype=torch.float32)).detach()

    def state_dict(self):
        return dict(schema=self.cfg['schema'], mode='mean', config=copy.deepcopy(self.cfg),
                    constant_point_m=list(self._point), trained=True, updates=0,
                    model={k:v.detach().clone() for k,v in self.model.state_dict().items()})

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (set(state) != set(expected)
            or any(state[k] != expected[k] for k in expected if k != 'model')
            or state['trained'] is not True or type(state['updates']) is not int
            or _model_sha256(state['model']) != _model_sha256(expected['model'])):
            raise ValueError('Fixed mean predictor checkpoint constant/config mismatch.')
        self.model.load_state_dict(state['model'], strict=True)
        self.model.eval()


class FixedMeanPredictor:
    """Train-only equal-serve mean, with exactly the learned hint's validity.

    Construction only verifies immutable evidence and creates one constant
    buffer. It never constructs an MLP, optimizer or random generator. Raw
    validation/held-out data are neither read nor used to select the point.
    """
    def __init__(self, num_envs, artifact='configs/pace_mean.json'):
        artifact = (ROOT / artifact).resolve()
        self.cfg = json.loads(artifact.read_text())
        cfg = settings()
        expected_input = dict(shape=[5, 3], order='oldest_first_xyz',
                              frame='local_table_ground_origin', unit='m',
                              policy_hz=48, normalization='none')
        expected_output = dict(shape=[3], meaning='post_bounce_apex',
                               frame='local_table_ground_origin', unit='m')
        expected_recipe = dict(split='train', seed=6101, serves=128, windows=4868,
                               window='windows(state[:,:3],time,serve,ones,target,valid)',
                               reduction='mean_per_serve_then_mean_across_serves',
                               dtype='float32', validation_used=False, heldout_used=False)
        if (self.cfg.get('schema') != 'airbot-e-mean-v1' or self.cfg.get('mode') != 'mean'
            or self.cfg.get('input') != expected_input or self.cfg.get('output') != expected_output
            or self.cfg.get('recipe') != expected_recipe or self.cfg.get('settings') != cfg
            or cfg['history'] != 5 or cfg['physics_hz'] != 960 or cfg['decimation'] != 20):
            raise ValueError('Incompatible mean artifact mode/input/coordinate/settings/recipe contract.')
        required = {'train', 'dataset', 'cross', 'frozen_artifact', 'source_snapshot'}
        if set(self.cfg['files']) != required:
            raise ValueError('Incomplete mean artifact provenance.')
        paths = {}
        for name, item in self.cfg['files'].items():
            path = ROOT / item['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Missing/changed mean artifact: {name}: {path}')
            paths[name] = path
        dataset, cross, frozen, snapshot = (json.loads(paths[name].read_text()) for name in
                                            ('dataset', 'cross', 'frozen_artifact', 'source_snapshot'))
        train = dataset['splits']['train']
        point = self.cfg.get('constant_point_m')
        if (dataset['status'] != 'PASS' or cross['status'] != 'PASS'
            or point != cross['train_mean_apex_m'] or len(point) != 3
            or not np.isfinite(point).all() or np.asarray(point, dtype=np.float32).tolist() != point
            or dataset['distribution'] != self.cfg['distribution']
            or dataset['physics_hz'] != 960 or dataset['policy_hz'] != 48
            or train['sha256'] != self.cfg['files']['train']['sha256']
            or train['collection_config'] != cfg or train['seed'] != 6101 or train['all_serves'] != 128
            or frozen['schema'] != 'airbot-e-frozen-v1' or frozen['mode'] != 'frozen'
            or frozen['input'] != expected_input or frozen['output'] != expected_output
            or frozen['distribution'] != self.cfg['distribution']
            or any(frozen['files'][k] != self.cfg['files'][k] for k in ('train', 'dataset'))):
            raise ValueError('Mean constant/training/distribution/frozen reference mismatch.')
        if snapshot['schema'] != 'airbot-e-mean-source-snapshot-v1':
            raise ValueError('Incompatible mean baseline source snapshot.')
        baseline = {item['path']:item['sha256'] for item in snapshot['files']}
        for name in required - {'source_snapshot'}:
            item = self.cfg['files'][name]
            if baseline.get(item['path']) != item['sha256']:
                raise ValueError(f'Mean evidence differs from immutable baseline: {name}')
        runtime = self.cfg['runtime_source_sha256']
        if (set(runtime) != set(frozen['runtime_source_sha256']) | {'src/airbot_tt/pace_predictor.py'}
            or any(runtime[k] != sha for k, sha in frozen['runtime_source_sha256'].items())):
            raise ValueError('Mean runtime source manifest mismatch.')
        for path, sha in runtime.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Mean runtime settings/source mismatch: {path}')
        self.predictor = _MeanPredictor(point)
        if _model_sha256(self.predictor.model.state_dict()) != self.cfg['model_state_sha256']:
            raise ValueError('Mean constant model state mismatch.')
        self.hook = PredictionHook(self.predictor, num_envs)
        self.provenance = copy.deepcopy(self.cfg)
        self.provenance.update(
            artifact_path=str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())

    def validate_environment(self, strike, physics_hz, decimation):
        fields = ('version', 'serve_position_m', 'serve_velocity_m_s',
                  'position_half_range_m', 'velocity_half_range_m_s', 'physics_hz')
        if (any(strike.get(k) != self.cfg['distribution'][k] for k in fields)
            or physics_hz != self.predictor.cfg['physics_hz']
            or decimation != self.predictor.cfg['decimation']):
            raise ValueError('Mean predictor requires the validated Z=2.3m serve distribution and 48Hz input.')

    def observe(self, position, time, serve, valid, ids=None):
        return self.hook.observe(position, time, serve, valid, ids)

    def state_dict(self):
        state = self.predictor.state_dict()
        if _model_sha256(state['model']) != self.cfg['model_state_sha256']:
            raise ValueError('Fixed mean predictor constant changed.')
        return dict(schema='airbot-e-mean-checkpoint-v1', mode='mean',
                    provenance=copy.deepcopy(self.provenance), predictor=state)

    def load_state_dict(self, state):
        if (set(state) != {'schema', 'mode', 'provenance', 'predictor'}
            or state.get('schema') != 'airbot-e-mean-checkpoint-v1' or state.get('mode') != 'mean'
            or state.get('provenance') != self.provenance
            or _model_sha256(state['predictor']['model']) != self.cfg['model_state_sha256']):
            raise ValueError('Mean predictor checkpoint mode/provenance/constant mismatch.')
        self.predictor.load_state_dict(state['predictor'])
        self.hook.reset(range(len(self.hook.history)))


def actor_frame(q_relative, qd, previous_action, position, valid, age, prediction=None):
    """Build the proposed observable-only single frame (23 or 27 columns).

    M6 tests this bridge; five-frame actor history and the RL env belong to M7.
    Inputs must already be in local table coordinates; no teacher/truth argument.
    """
    n = len(q_relative)
    parts = []
    for value, width in ((q_relative,6),(qd,6),(previous_action,6),(position,3),(valid,1),(age,1)):
        tensor = torch.as_tensor(value, dtype=torch.float32).detach()
        if tensor.shape != (n,width) or not torch.isfinite(tensor).all():
            raise ValueError(f'Expected finite [{n},{width}] actor field.')
        parts.append(tensor)
    if prediction is not None:
        prediction = torch.as_tensor(prediction,dtype=torch.float32).detach()
        if prediction.shape != (n,4) or not torch.isfinite(prediction).all():
            raise ValueError('Expected finite [env,4] prediction and validity.')
        parts.append(prediction)
    return torch.cat(parts,dim=1).clamp(-100,100)


class OnlinePredictor:
    """Independent recent-window MSE updates; PPO never owns these parameters.

    Buffer capacity counts policy instants, each containing up to num_envs windows.
    A restored model starts fresh observation histories when the environment resets.
    """
    def __init__(self,num_envs):
        from collections import deque
        self.cfg=json.loads((ROOT/'configs/pace_online.json').read_text())
        self.predictor=Predictor()
        self.hook=PredictionHook(self.predictor,num_envs)
        self.buffer=deque(maxlen=self.cfg['buffer_policy_steps'])
        self.attempts=0;self.optimizer_steps=0

    def observe(self,position,time,serve,valid,ids=None):
        return self.hook.observe(position,time,serve,valid,ids)

    def collect(self,target,valid):
        x=[];y=[]
        for i,history in enumerate(self.hook.history):
            if valid[i] and len(history)==5 and np.isfinite(target[i]).all():
                x.append(np.asarray(history).reshape(15));y.append(target[i])
        # Empty frames also age the bounded recent buffer.
        self.buffer.append((torch.tensor(np.asarray(x).reshape(-1,15),dtype=torch.float32),
                            torch.tensor(np.asarray(y).reshape(-1,3),dtype=torch.float32)))

    def update(self):
        import time
        started=time.monotonic()
        if self.attempts>=self.cfg['update_iterations']:
            return dict(status='cutoff',updates=self.predictor.updates,optimizer_steps=self.optimizer_steps,seconds=0.)
        self.attempts+=1
        x=torch.cat([a for a,b in self.buffer]) if self.buffer else torch.empty(0,15)
        y=torch.cat([b for a,b in self.buffer]) if self.buffer else torch.empty(0,3)
        if not len(x):return dict(status='no_valid_windows',samples=0,updates=self.predictor.updates,optimizer_steps=self.optimizer_steps,seconds=time.monotonic()-started)
        loss=self.predictor.fit_epoch(x,y,batch_size=self.cfg['batch_size'])
        self.optimizer_steps+=(len(x)+self.cfg['batch_size']-1)//self.cfg['batch_size']
        return dict(status='updated',samples=len(x),mse=loss,updates=self.predictor.updates,
                    optimizer_steps=self.optimizer_steps,seconds=time.monotonic()-started)

    def state_dict(self):
        return dict(config=self.cfg,predictor=self.predictor.state_dict(),attempts=self.attempts,
                    optimizer_steps=self.optimizer_steps,buffer=list(self.buffer))

    def load_state_dict(self,state):
        if state['config']!=self.cfg:raise ValueError('Online predictor configuration mismatch')
        self.predictor.load_state_dict(state['predictor']);self.attempts=state['attempts']
        self.optimizer_steps=state['optimizer_steps'];self.buffer.clear();self.buffer.extend(state['buffer'])
        self.hook.reset(range(len(self.hook.history)))
