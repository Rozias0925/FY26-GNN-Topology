"""Resumable experiment runner; all computational research stages share artifacts."""
from __future__ import annotations
import itertools
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import SimulationConfig
from .simulation import simulate
from .protocol import Panel, graph_inputs, prepare
from .training import evaluate_model, baselines
from .models import MODEL_NAMES
from .descriptors import describe, FEATURE_NAMES
from .selection import fit_selection, apply_rule
from .io import digest, source_digest, write_json


def scenario_specs(config):
    grid = config['grid']
    keys = sorted(grid)
    seen = set()
    for mode in config.get('graph_modes', ['estimated']):
        for values in itertools.product(*(grid[k] for k in keys)):
            settings = dict(zip(keys, values))
            # Edge observation flips have no effect in the series-estimation arm.
            if mode == 'estimated':
                settings['edge_noise'] = 0.0
                if settings.get('graph_signal') == 0:
                    settings.update(graph_drift=0., neighbor_heterogeneity=0.)
            identifier = digest({'mode': mode, 'settings': settings})[:12]
            if identifier not in seen:
                seen.add(identifier)
                yield identifier, mode, settings


def initialize_output(output, cfg, extra=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': cfg, 'source_sha256': source_digest(), 'extra': extra}
    fingerprint = digest(identity)
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text('utf-8'))['fingerprint'] != fingerprint:
        raise ValueError('Output has a different config, source version or data checksum. Use a new --output directory.')
    write_json(path, {**identity, 'fingerprint': fingerprint, 'python': platform.python_version(),
                      'numpy': np.__version__, 'torch': torch.__version__,
                      'status': 'running', 'purpose': cfg.get('purpose', 'research')})
    return output


def run_panel(panel, cfg, folder, seed, selector_rules=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    graphs = graph_inputs(panel, cfg)
    graph_seconds = time.perf_counter() - started
    descriptor_file = folder / 'descriptors.json'
    if descriptor_file.exists():
        descriptors = json.loads(descriptor_file.read_text('utf-8'))
    else:
        descriptors = describe(panel, graphs, cfg)
        write_json(descriptor_file, descriptors)
    write_json(folder / 'preparation_runtime.json', {'graph_construction_seconds': graph_seconds,
               'graph_and_descriptor_seconds': time.perf_counter() - started,
               'note': 'Shared preprocessing; add relevant graph construction cost when discussing practical runtime.'})
    rows = []
    for horizon in cfg.get('horizons', [1, 3, 6]):
        data = prepare(panel, cfg, horizon, graphs)
        hfolder = folder / f'h{horizon}'
        write_json(hfolder / 'protocol.json', {**data['metadata'], 'panel': panel.metadata,
                    'target_nodes': panel.target_nodes, 'node_names': panel.node_names,
                    'response_mean': data['response_mean'], 'response_scale': data['response_scale']})
        write_json(hfolder / 'baselines.json', baselines(data, panel))
        if selector_rules is not None:
            rule = next(r for r in selector_rules if r['horizon'] == horizon and r['graph_mode'] == 'estimated')
            # Saved BEFORE real model evaluation; full comparison only evaluates transfer.
            write_json(hfolder / 'frozen_selection.json', apply_rule(rule, descriptors))
        for name in MODEL_NAMES:
            mfolder = hfolder / name.lower().replace(' ', '_')
            completed = all((mfolder / f).exists() for f in ['result.json', 'weights.pt', 'predictions.npz', 'training.json'])
            if completed:
                result = json.loads((mfolder / 'result.json').read_text('utf-8'))
                print(f'Resume {folder.name} h={horizon} {name}', flush=True)
            else:
                print(f'Train {folder.name} h={horizon} {name}', flush=True)
                result = evaluate_model(name, data, panel, cfg, mfolder, seed)
            rows.append({**{k: descriptors[k] for k in FEATURE_NAMES}, 'model': name, 'horizon': horizon,
                         'seed': seed, 'validation_ap': result['validation']['average_precision'],
                         'test_ap': result['test']['average_precision'], 'test_auroc': result['test']['auroc'],
                         'test_brier': result['test']['brier'], 'test_f1': result['test']['f1'],
                         'test_mae': result['test']['mae'], 'test_rmse': result['test']['rmse'],
                         'seconds': result['training_seconds'], 'parameters': result['selected_hyperparameters']['parameter_count']})
    return rows


def summarize(rows, output, cfg, selector_rules=None, real=False):
    output = Path(output)
    df = pd.DataFrame(rows)
    df.to_csv(output / 'metrics.csv', index=False)
    keys = ['graph_mode', 'scenario_id', 'horizon', 'model']
    summary = []
    for group_key, group in df.groupby(keys):
        record = dict(zip(keys, group_key))
        record['replicates'] = len(group)
        for metric in ['test_ap', 'test_auroc', 'test_brier', 'test_f1', 'test_mae', 'test_rmse']:
            values = group[metric].dropna().to_numpy(float)
            record[metric] = float(values.mean()) if len(values) else None
            if len(values) > 1:
                rng = np.random.default_rng(9)
                resamples = rng.choice(values, (1000, len(values)), replace=True).mean(axis=1)
                record[metric + '_ci_low'], record[metric + '_ci_high'] = np.quantile(resamples, [.025, .975]).tolist()
            else:
                record[metric + '_ci_low'] = record[metric + '_ci_high'] = None
        summary.append(record)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    paired = []
    for keys_value, group in df.groupby(['graph_mode', 'scenario_id', 'horizon']):
        for metric in ['test_ap', 'test_brier', 'test_rmse']:
            scores = group.pivot(index='seed', columns='model', values=metric)
            for name in MODEL_NAMES[1:]:
                differences = (scores[name] - scores['MLP']).dropna().to_numpy(float)
                ci = [None, None]
                if len(differences) > 1:
                    means = np.random.default_rng(9).choice(differences, (1000, len(differences)), replace=True).mean(axis=1)
                    ci = np.quantile(means, [.025, .975]).tolist()
                paired.append(dict(zip(['graph_mode','scenario_id','horizon'],keys_value)) | {
                    'model':name,'reference':'MLP','metric':metric,'pairs':len(differences),
                    'mean_difference':float(differences.mean()) if len(differences) else None,'ci_low':ci[0],'ci_high':ci[1]})
    pd.DataFrame(paired).to_csv(output / 'paired_differences.csv', index=False)
    if real:
        selector = selector_rules or []
        write_json(output / 'selection_rule.json', {'schema_version': 1, 'rules': selector,
                   'origin': 'frozen_simulation_rules' if selector else 'smoke_without_transfer_rule'})
        transfer = []
        for (seed, horizon), subset in df.groupby(['seed', 'horizon']):
            path = output / f'seed_{seed}' / f'h{horizon}' / 'frozen_selection.json'
            if not path.exists():
                continue
            selection = json.loads(path.read_text('utf-8'))
            chosen = selection['selected_model']
            actual = subset.loc[subset.model == chosen, 'test_ap']
            selected_ap = float(actual.iloc[0]) if len(actual) and pd.notna(actual.iloc[0]) else None
            best = subset.test_ap.max()
            transfer.append({'seed': int(seed), 'horizon': int(horizon), **selection,
                             'selected_test_ap': selected_ap,
                             'test_oracle_ap_evaluation_only': float(best) if pd.notna(best) else None,
                             'selection_regret': float(best - selected_ap) if selected_ap is not None and pd.notna(best) else None})
        write_json(output / 'transfer_evaluation.json', {'results': transfer,
                   'note': 'Abstention or out-of-domain does not establish successful transfer.'})
    else:
        selector = fit_selection(rows, output / 'selection_rule.json', cfg.get('gray_margin', .01))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode, subset in df.groupby('graph_mode'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for name in MODEL_NAMES:
            item = subset[subset.model == name].groupby('horizon')[['test_ap', 'test_rmse']].mean()
            axes[0].plot(item.index, item.test_ap, marker='o', label=name)
            axes[1].plot(item.index, item.test_rmse, marker='o', label=name)
        axes[0].set(ylabel='Average precision', xlabel='Forecast horizon (steps)')
        axes[1].set(ylabel='RMSE', xlabel='Forecast horizon (steps)')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{cfg.get("purpose", "research")} / {mode} / mean across scenarios (not a selection rule)')
        fig.savefig(output / f'comparison_{mode}.png', dpi=180)
        plt.close(fig)
    write_json(output / 'website_results.json', {'schema_version': 1, 'purpose': cfg.get('purpose', 'research'),
               'source_sha256': source_digest(), 'summary': summary, 'selection': selector,
               'interpretation': 'Smoke runs validate software only; no scientific model ranking is claimed.' if cfg.get('purpose') == 'smoke' else 'See protocol, confidence intervals and held-out scenario validation.'})
    from .reporting import write_report
    write_report(output, cfg, rows)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    manifest['status'] = 'complete'
    write_json(manifest_path, manifest)


def run_simulation(cfg, output):
    output = initialize_output(output, cfg)
    rows = []
    for scenario_id, mode, settings in scenario_specs(cfg):
        for seed in cfg['seeds']:
            simulation = simulate(SimulationConfig(**{**cfg.get('simulation', {}), **settings, 'seed': seed}))
            features = np.concatenate([simulation.node_features, simulation.response[..., None]], axis=-1)
            panel = Panel(features, simulation.response, simulation.response,
                          [f'v{i}' for i in range(simulation.config.n_nodes)], list(range(simulation.config.n_nodes)),
                          np.arange(simulation.config.n_steps), {'kind': 'simulation', 'step_hours': None, 'config': simulation.config.to_dict()},
                          simulation.observed_adjacency)
            folder = output / scenario_id / f'seed_{seed}'
            local_cfg = {**cfg['protocol'], **cfg['training'], 'graph_mode': mode}
            scenario_rows = run_panel(panel, local_cfg, folder, seed)
            write_json(folder / 'scenario.json', {'settings': settings, 'graph_mode': mode,
                         'simulation_config': simulation.config.to_dict(),
                         'mean_realized_edge_replacement': float(simulation.realized_edge_change[1:].mean())})
            regime = {k:v for k,v in settings.items() if k != 'edge_noise'}
            if regime.get('graph_signal') == 0:
                # Graph drift/weight concentration cannot change responses when
                # the graph contribution is zero. Keep these null controls together.
                regime.update(graph_drift=0., neighbor_heterogeneity=0.)
            selection_group = digest(regime)[:12]
            rows.extend({**r, 'scenario_id': scenario_id, 'selection_group':selection_group,
                         'graph_mode': mode, **settings} for r in scenario_rows)
            pd.DataFrame(rows).to_csv(output / 'metrics_partial.csv', index=False)
    summarize(rows, output, cfg)
    return rows
