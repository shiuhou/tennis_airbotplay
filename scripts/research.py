#!/usr/bin/env python3
"""Record an explicit research command, or compare existing run results (no training)."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from airbot_tt.research import run, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run')
    r.add_argument('--directory', type=Path, required=True)
    r.add_argument('--protocol', type=Path, required=True)
    r.add_argument('--input', type=Path, action='append', default=[])
    r.add_argument('--result', default='result.json', help='Relative path to the child result JSON')
    r.add_argument('--timeout', type=float, required=True)
    r.add_argument('command', nargs=argparse.REMAINDER)
    c = sub.add_parser('compare')
    c.add_argument('--metric', required=True)
    c.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    if args.mode == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        result = run(ROOT, args.directory, args.protocol, args.input, command, args.timeout, args.result)
        print(json.dumps({k:result[k] for k in ('status', 'exit_code', 'seconds', 'error')}, indent=2))
    else:
        result = compare(args.directories, args.metric)
        print(json.dumps(result, indent=2))
    return {'PASS': 0, 'FAIL': 1}.get(result['status'], 2)


if __name__ == '__main__':
    sys.exit(main())
