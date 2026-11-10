"""Launch all V2 simulation stages with bounded concurrency and partial exports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="algorithm/configs/mechanism_v2_suite.json")
    parser.add_argument("--output", default="algorithm/results/v2_full_suite_20261003")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from",help="Import verified checkpoints into a new output after an I/O-only repair.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from gnn_mechanisms.suite import build_jobs, plan_summary, execute
    print(json.dumps(plan_summary(config, build_jobs(config)), ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        result = execute(config, args.output, resume_from=args.resume_from)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
