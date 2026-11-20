"""Small, stdlib-only research receipts and descriptive comparisons.

Inspired by the user-supplied RL engineering template's evidence contracts;
independent implementation. See assets/upstream/rl_engineering_v3/provenance.json.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(root, directory, protocol_path, inputs, command, timeout, result_file="result.json"):
    """Run once in a new directory. Child writes result.json via {run_dir}."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    result_path = (directory/result_file).resolve()
    if not result_path.is_relative_to(directory):
        raise ValueError('Result must be inside the run directory')
    if not command or timeout <= 0:
        raise ValueError('A command and positive timeout are required')
    protocol = json.loads(Path(protocol_path).read_text())
    for key in ('question', 'common', 'method', 'kind'):
        if not protocol.get(key):
            raise ValueError(f'Protocol requires {key}')
    if protocol['kind'] not in ('experiment', 'analysis'):
        raise ValueError('kind must be experiment or analysis')
    files = {str(Path(p).resolve()): sha256(p) for p in inputs}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    # Include dirty/untracked source identities without reading ignored datasets or credentials.
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z',
                                     '--cached', '--others', '--exclude-standard']).decode().split('\0')
    source = {name: sha256(root/name) for name in names if name and (root/name).is_file()}
    receipt = dict(schema=1, started_utc=datetime.now(timezone.utc).isoformat(),
                   git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                   source_sha256=source, input_sha256=files, protocol=protocol,
                   launcher=dict(python=sys.version, executable=sys.executable, platform=platform.platform()),
                   command=command, result_file=result_file, timeout_s=timeout, status='RUNNING')
    directory.mkdir(parents=True, exist_ok=False)  # never reuse/overwrite a trial
    write_json(directory/'record.json', receipt)
    argv = [arg.replace('{run_dir}', str(directory)) for arg in command]
    started = time.monotonic()
    code = None
    failure = None
    with (directory/'execution.log').open('wb') as log:
        try:
            proc = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                failure = 'TIMEOUT'
        except OSError as exc:
            failure = str(exc)
    receipt.update(exit_code=code, seconds=time.monotonic()-started, error=failure)
    receipt['status'] = 'FAIL' if failure or code != 0 else 'INCOMPLETE'
    if code == 0 and not failure and result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
            if result.get('status') == 'PASS':
                receipt['status'] = 'PASS'
            elif result.get('status') == 'FAIL':
                receipt['status'] = 'FAIL'
        except (ValueError, OSError):
            receipt['error'] = 'Invalid result.json'
    # Changes during execution make the source identity ambiguous.
    receipt['changed_inputs'] = [p for p, digest in files.items() if not Path(p).is_file() or sha256(p) != digest]
    receipt['changed_sources'] = [p for p, digest in source.items() if not (root/p).is_file() or sha256(root/p) != digest]
    if receipt['changed_inputs'] or receipt['changed_sources']:
        receipt['status'] = 'INCOMPLETE'
    receipt['artifacts_sha256'] = {str(p.relative_to(directory)): sha256(p)
                                   for p in directory.rglob('*') if p.is_file() and p.name != 'record.json'}
    write_json(directory/'record.json', receipt)
    return receipt


def compare(directories, metric):
    """Descriptive training-run summaries, not episode-level statistical inference."""
    groups, common, problems, seen = {}, None, [], set()
    for directory in map(Path, directories):
        record = json.loads((directory/'record.json').read_text())
        protocol = record['protocol']
        if common is None:
            common = protocol['common']
        if protocol['common'] != common:
            raise ValueError('Incompatible common evaluation conditions')
        if protocol['kind'] != 'experiment':
            raise ValueError('Analysis/replots are not independent experiments')
        if record['status'] != 'PASS':
            problems.append(dict(run=str(directory), status=record['status']))
            continue
        for name, digest in record['artifacts_sha256'].items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path) != digest:
                raise ValueError(f'Changed or missing artifact: {directory}/{name}')
        result_name=record.get('result_file', 'result.json')
        if result_name not in record['artifacts_sha256']:
            raise ValueError('Result is not an authenticated artifact')
        result = json.loads((directory/result_name).read_text())
        if not result.get('runtime') or type(result.get('training_seed')) is not int:
            raise ValueError('Comparison requires actual workload runtime and independent training_seed')
        value = result['metrics'][metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Metric must be finite numeric data')
        method = json.dumps(protocol['method'], sort_keys=True)
        identity = (method, result['training_seed'])
        if identity in seen:
            raise ValueError('Duplicate training seed within method; copies are not replications')
        seen.add(identity)
        groups.setdefault(method, []).append(float(value))
    return dict(status='INCOMPLETE' if problems else 'PASS', scientific_conclusion='INCONCLUSIVE',
                meaning='Descriptive only; no superiority test or causal conclusion. Episodes are not training seeds.',
                metric=metric, common=common, failed_or_incomplete=problems,
                groups=[dict(method=json.loads(k), independent_training_runs=len(v), values=v,
                             mean=statistics.mean(v), sd=statistics.stdev(v) if len(v)>1 else None)
                        for k,v in groups.items()])
