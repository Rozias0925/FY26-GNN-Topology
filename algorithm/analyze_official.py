"""Run Shihmen QA, hourly alignment, forecast labels, probes and baseline analysis."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.official_analysis import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True, help='Dated directory written by fetch_official.py')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/official_analysis.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.raw, json.loads(a.config.read_text('utf-8-sig')), a.output)
    print(json.dumps({k: result[k] for k in ['raw_turbidity_rows', 'hourly_steps',
                                            'classification_ready', 'regression_ready', 'blockers']},
                     ensure_ascii=False, indent=2))
    print(f'Report: {(a.output / "ANALYSIS_REPORT.md").resolve()}')


if __name__ == '__main__':
    main()
