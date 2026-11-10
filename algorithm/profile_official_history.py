"""Describe all common available history, without tuning or evaluating models."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.official_history import profile


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True, help='Completed output of analyze_official.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/official_analysis.json')
    a = p.parse_args()
    result = profile(a.raw, a.audit, json.loads(a.config.read_text('utf-8-sig')), a.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
