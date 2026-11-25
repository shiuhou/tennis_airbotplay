#!/usr/bin/env python3
"""Download original official dataset files at the pinned commit; preserve and verify bytes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'configs/deepmind_sources.json').read_text())
    destination=ROOT/'.agents/runs/foundation/deepmind_audit/source';destination.mkdir(parents=True,exist_ok=True)
    for item in manifest['files']:
        path=destination/item['path']
        if path.exists():payload=path.read_bytes()
        else:
            with urllib.request.urlopen(item['url'],timeout=60) as response:payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=item['sha256'] or len(payload)!=item['bytes']:
            raise ValueError('Pinned source mismatch: '+item['path'])
        if 'git_blob_sha1' in item:
            blob=b'blob '+str(len(payload)).encode()+b'\0'+payload
            assert hashlib.sha1(blob).hexdigest()==item['git_blob_sha1']
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    existing=destination/'manifest.json'
    if existing.exists():assert json.loads(existing.read_text())==manifest
    else:existing.write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS official DeepMind original files, pinned SHA256/Git blobs; data CC-BY-4.0, software Apache-2.0')


if __name__=='__main__':main()
