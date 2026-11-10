"""Commands: fetch separately, audit data, simulate, or validate a real case."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.io import write_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['simulate', 'audit', 'real'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'smoke.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--selection', type=Path, help='Frozen simulation selection_rule.json (required for formal transfer)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        from gnn_graph_selection.real_data import audit
        result = audit(args.raw, args.output or ROOT / 'data' / 'audit')
        print(json.dumps({'total_rows':result['total_rows'], 'stations':len(result['stations']), 'download_complete':result['archive_download_complete']}))
        return
    cfg = json.loads(args.config.read_text('utf-8'))
    from gnn_graph_selection.experiment import run_simulation, scenario_specs, initialize_output, run_panel, summarize
    if args.command == 'simulate':
        if args.dry_run:
            scenarios = list(scenario_specs(cfg))
            runs = len(scenarios) * len(cfg['seeds']) * len(cfg['protocol']['horizons']) * 5
            print(json.dumps({'scenarios': len(scenarios), 'model_runs': runs,
                             'candidate_fits': runs * len(cfg['training']['widths']) * len(cfg['training']['learning_rates'])}))
            return
        run_simulation(cfg, args.output or ROOT / 'results' / args.config.stem)
    else:
        from gnn_graph_selection.real_data import prepare_real, save_panel
        if cfg.get('purpose') == 'research' and args.selection is None:
            parser.error('Formal external validation requires --selection from completed simulations')
        rules = None
        if args.selection:
            rules = json.loads(args.selection.read_text('utf-8'))['rules']
        local = {**cfg['protocol'], **cfg['training'], **cfg['data'], 'graph_mode':'estimated'}
        output = args.output or ROOT / 'results' / args.config.stem
        if 'observations_csv' in local:
            from gnn_graph_selection.csv_data import prepare_csv
            panel = prepare_csv(local)
        else:
            panel = prepare_real(args.raw, local)
        initialize_output(output, cfg, {'data': panel.metadata, 'selector_hash': digest(rules)})
        save_panel(panel, Path(output) / 'prepared')
        from inspect_readiness import inspect
        readiness = inspect(panel, local)
        write_json(Path(output) / 'label_readiness.json', readiness)
        if not readiness['ready_for_comparison']:
            manifest_path = Path(output) / 'manifest.json'
            manifest = json.loads(manifest_path.read_text('utf-8'))
            manifest['status'] = 'insufficient_data'
            manifest['issues'] = readiness['issues']
            write_json(manifest_path, manifest)
            from gnn_graph_selection.reporting import write_readiness_report
            write_readiness_report(output, readiness)
            print('Real comparison skipped: insufficient event coverage. See label_readiness.json and report.html.', flush=True)
            return
        rows = []
        for seed in cfg['seeds']:
            result = run_panel(panel, local, Path(output) / f'seed_{seed}', seed, rules)
            rows.extend({**r, 'scenario_id':'shihmen', 'graph_mode':'estimated'} for r in result)
        summarize(rows, output, cfg, selector_rules=rules, real=True)
    print(f'Completed: {args.output or ROOT / "results" / args.config.stem}', flush=True)


if __name__ == '__main__':
    main()
