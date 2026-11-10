"""Fetch official WRA data without credentials; resume cached pages."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.sources import SOURCES, download

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', choices=list(SOURCES), default=['turbidity', 'reservoir'])
    p.add_argument('--output', type=Path, default=ROOT / 'data' / 'raw')
    p.add_argument('--max-pages', type=int, default=2000)
    p.add_argument('--refresh', action='store_true')
    args = p.parse_args()
    for name in args.sources:
        download(name, args.output, args.max_pages, refresh=args.refresh)
