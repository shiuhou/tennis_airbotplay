"""Fetch and execute unmodified, hash-pinned Orbbec examples; no device logic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'.agents/runs/foundation/camera_calibration/orbbec_examples'
MANIFEST = ROOT/'configs/orbbec_examples.json'


def specification():
    return json.loads(MANIFEST.read_text())


def verify_sources():
    spec = specification()
    for name, expected in spec['files'].items():
        path = SOURCE/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Official source missing/modified: {path}. Run setup; never silently replace edits.')
    return spec


def fetch_sources():
    spec = specification()
    for name, expected in spec['files'].items():
        path = SOURCE/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Existing official file modified: {path}; preserving it.')
            continue
        url = f"https://raw.githubusercontent.com/orbbec/pyorbbecsdk/{spec['commit']}/{name}"
        content = urllib.request.urlopen(url, timeout=30).read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch for {name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    verify_sources()
    print('Official source + LICENSE verified:', spec['commit'])


def run_example(name, extra):
    spec = verify_sources()
    work = ROOT/'.agents/runs/foundation/camera_calibration/official_logs'
    work.mkdir(parents=True, exist_ok=True)
    # The examples' exit codes and messages are preserved, NOT interpreted as hardware PASS.
    return subprocess.call([sys.executable, str(SOURCE/spec['commands'][name]), *extra], cwd=work)


def load_module(relative, name):
    verify_sources()
    sys.path.insert(0, str(SOURCE/'examples'))
    module_spec = importlib.util.spec_from_file_location(name, SOURCE/relative)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    fetch_sources()
