"""PACE's original MLP with AIRBOT serve-safe data and observation contracts."""
import hashlib
import importlib
import json
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
    def __init__(self):
        self.cfg = settings()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.cfg['predictor']['seed'])
            self.model = original_mlp()
        self.rng = torch.Generator().manual_seed(self.cfg['predictor']['seed'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['predictor']['lr'])
        self.trained = False
        self.updates = 0

    def fit_epoch(self, x, y):
        if self.updates >= self.cfg['predictor']['epochs']:
            raise RuntimeError('Predictor update budget exhausted.')
        if len(x) == 0:
            raise ValueError('No valid training windows.')
        x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Non-finite training data.')
        self.model.train()
        total = 0.
        for ids in torch.randperm(len(x), generator=self.rng).split(self.cfg['predictor']['batch_size']):
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

    def save(self, path):
        torch.save(dict(schema=self.cfg['schema'], config=self.cfg,
                        source_manifest_sha256=hashlib.sha256((ROOT/'configs/pace_sources.json').read_bytes()).hexdigest(),
                        ball_config_sha256=hashlib.sha256((ROOT/'configs/ball.json').read_bytes()).hexdigest(),
                        model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        trained=self.trained, updates=self.updates, rng=self.rng.get_state()), path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        sha = hashlib.sha256((ROOT/'configs/pace_sources.json').read_bytes()).hexdigest()
        if (data['config'] != self.cfg or data['source_manifest_sha256'] != sha
            or data['ball_config_sha256'] != hashlib.sha256((ROOT/'configs/ball.json').read_bytes()).hexdigest()):
            raise ValueError('Checkpoint source/config mismatch.')
        self.model.load_state_dict(data['model'], strict=True)
        self.optimizer.load_state_dict(data['optimizer'])
        self.trained, self.updates = data['trained'], data['updates']
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

    def observe(self, position, time, serve, valid):
        p = np.asarray(position)
        n = len(self.history)
        if p.shape != (n,3):
            raise ValueError('Expected [env,3] observable positions.')
        output = torch.zeros((n,4), dtype=torch.float32)
        ids, x = [], []
        for i in range(n):
            if not valid[i] or not np.isfinite(p[i]).all():
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
