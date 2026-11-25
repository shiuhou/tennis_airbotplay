#!/usr/bin/env python3
"""Run selected unit checks in the CURRENT environment; no simulation/hardware."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
GROUPS = {
    'research': ['test_research_workflow'],
    'core': ['test_ball', 'test_motion', 'test_planning', 'test_prediction',
             'test_racket', 'test_single_shot', 'test_strike'],
    'pace': ['test_pace_m6', 'test_pace_source_audit', 'test_pace_env', 'test_pace_serves', 'test_pace_bc', 'test_recovery_fit'],
    'frozen': ['test_pace_m6.FrozenPredictorTests', 'test_pace_m6.GuidanceTests',
               'test_pace_m6.PredictorTests.test_hook_same_step_reset_and_observable_only',
               'test_pace_env', 'test_pace_runner'],
    'mean': ['test_pace_m6.FixedMeanPredictorTests', 'test_pace_m6.FrozenPredictorTests',
             'test_pace_m6.GuidanceTests',
             'test_pace_m6.PredictorTests.test_hook_same_step_reset_and_observable_only',
             'test_pace_env', 'test_pace_runner'],
    'camera': ['test_camera_calibration'],
}
DEPENDENCIES = {'research': [], 'core': ['numpy', 'scipy'],
                'pace': ['numpy', 'scipy', 'torch'], 'frozen': ['numpy', 'scipy', 'torch'], 'mean': ['numpy', 'scipy', 'torch'], 'camera': ['numpy', 'cv2', 'pyorbbecsdk']}


def status_for(result):
    if not result.wasSuccessful():
        return 'FAIL', 1
    if result.testsRun == 0 or result.skipped:
        return 'INCOMPLETE', 2
    return 'PASS', 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    missing = [name for name in DEPENDENCIES[args.group] if importlib.util.find_spec(name) is None]
    report = dict(group=args.group, python=sys.executable, missing=missing)
    if missing:
        report.update(status='BLOCKED', tests=0, skipped=[])
        code = 2
    else:
        suite = unittest.defaultTestLoader.loadTestsFromNames(GROUPS[args.group])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        status, code = status_for(result)
        report.update(status=status, tests=result.testsRun,
                      skipped=[(str(test), reason) for test, reason in result.skipped],
                      failures=len(result.failures), errors=len(result.errors))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return code


if __name__ == '__main__':
    sys.exit(main())
