#!/usr/bin/env python3
"""Fetch only pinned official text sources, verify hashes, preserve licenses."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'configs/pace_sources.json').read_text())
dest = root/'.agents/runs/foundation/pace/source'
for row in manifest['files']:
    path = dest/row['path']
    if path.exists():
        data = path.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/purdue-tracelab/PACE-ICRA2026/{manifest['commit']}/{row['path']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise RuntimeError(f'Hash mismatch: {path}; existing files will not be overwritten.')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
print(f"Verified {len(manifest['files'])} official source files at {manifest['commit']}")
