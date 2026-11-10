"""Low-impact, resumable regression-only five-model study on audited measured data."""
import argparse
import json
from pathlib import Path
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
from low_impact_runtime import protect_current_process, RunLock, ResourceGuard, StopRequested, status_writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    policy = protect_current_process()  # before numpy/BLAS/torch imports
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'src'))
    args.output.mkdir(parents=True, exist_ok=True)
    with RunLock(args.output):
        status = status_writer(args.output)
        guard = ResourceGuard(args.output, status)
        try:
            status('checking_resources', resources_policy=policy)
            guard.wait_for_start()  # measured start gate plus a verified own-process memory cap
            if os.environ.get('GNN_REQUIRE_TESTS', '1') == '1':
                import unittest
                import io
                status('running_tests')
                suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_real_benchmark.py')
                stream = io.StringIO()
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
                (args.output / 'tests.log').write_text(stream.getvalue(), encoding='utf-8')
                if not result.wasSuccessful() or result.testsRun == 0:
                    raise RuntimeError('Regression checks failed or no tests were discovered; training not started')
                (args.output / 'tests_passed.json').write_text(json.dumps({'tests': result.testsRun, 'passed': True}), encoding='utf-8')
            from gnn_graph_selection.measured_benchmark import run
            result = run(json.loads(args.config.read_text('utf-8-sig')), args.config.resolve().parent.parent,
                         args.output, policy, status, guard, preflight_only=args.preflight_only)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except StopRequested as exc:
            status('stopped', reason=str(exc))
            raise SystemExit(3)
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            (args.output / 'failure.json').write_text(json.dumps({'error': repr(exc), 'pid': policy['pid']}), encoding='utf-8')
            raise


if __name__ == '__main__':
    main()
