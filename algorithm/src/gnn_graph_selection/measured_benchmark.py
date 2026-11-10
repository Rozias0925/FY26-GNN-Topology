"""Separate measured regression task; fixed data/splits and no event-loss shortcut.

All candidate fitting/selection finishes before held-out model scoring begins.
The three-node history adapter is exploratory, not a completed synthetic-map
transfer or high-turbidity classification validation.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from gnn_mechanisms.models import SnapshotModel, MODEL_NAMES, pack_graphs, graph_tensors, select_graph
from .io import digest, write_json
from .protocol import Panel, causal_fill, time_splits
from .official_analysis import boundaries, forecast_labels, label_readiness, histories
from .evaluation import regression

_TORCH_CONFIGURED = False


def configure_cpu():
    global _TORCH_CONFIGURED
    torch.set_num_threads(1)
    if not _TORCH_CONFIGURED:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.use_deterministic_algorithms(True)
        _TORCH_CONFIGURED = True


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_hourly_index(timestamps, total):
    # pandas 3 can retain s/us resolution; asi8 is not necessarily nanoseconds.
    dates = pd.to_datetime(timestamps, utc=True).as_unit('ns')
    if len(dates) != total or np.any(np.diff(dates.asi8) != 3_600_000_000_000):
        raise ValueError('Panel must have a uniform, strictly increasing hourly index')
    return dates


def load_audited(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'analysis_manifest.json').read_text('utf-8'))
    if not manifest.get('completed'):
        raise ValueError('Measured analysis is incomplete')
    required = {'analysis_config.json', 'hourly_panel.npz', 'panel_metadata.json', 'causal_graphs.npz'}
    if not required <= set(manifest.get('outputs_sha256', {})):
        raise ValueError('Core analysis artifacts are not all covered by the checksum manifest')
    for name, expected in manifest['outputs_sha256'].items():
        path = (folder / name).resolve()
        if path.parent != folder.resolve() or not path.is_file() or file_sha(path) != expected:
            raise ValueError(f'Analysis artifact identity mismatch: {name}')
    config = json.loads((folder / 'analysis_config.json').read_text('utf-8'))
    cfg = {**config['data'], **config['protocol']}
    if cfg.get('allow_partial_archive') or cfg.get('include_reservoir_covariates'):
        raise ValueError('This run requires the audited complete turbidity-only panel')
    if cfg['target_station'] not in cfg['stations']:
        raise ValueError('Unknown target station')
    metadata = json.loads((folder / 'panel_metadata.json').read_text('utf-8'))
    with np.load(folder / 'hourly_panel.npz', allow_pickle=False) as data:
        panel = Panel(data['features'].copy(), data['response'].copy(), data['graph_series'].copy(),
                      cfg['stations'], [cfg['stations'].index(cfg['target_station'])],
                      data['timestamps'].copy(), metadata, event_response=data['event_response'].copy())
    with np.load(folder / 'causal_graphs.npz', allow_pickle=False) as data:
        static, dynamic = data['static'].copy(), data['dynamic'].copy()
        if data['stations'].tolist() != panel.node_names or not np.array_equal(data['timestamps'], panel.timestamps):
            raise ValueError('Graph node order or timestamps differ from observations')
    total, nodes, _ = panel.features.shape
    dates = validate_hourly_index(panel.timestamps, total)
    if panel.response.shape != (total, nodes) or panel.event_response.shape != (total, nodes):
        raise ValueError('Panel dimensions mismatch')
    for graph, shape in [(static, (nodes, nodes)), (dynamic, (total, nodes, nodes))]:
        if graph.shape != shape or not np.isfinite(graph).all() or np.any((graph != 0) & (graph != 1)):
            raise ValueError('Invalid cached binary graph')
        if not np.array_equal(graph, graph.swapaxes(-1, -2)):
            raise ValueError('Current real adapter requires undirected topology')
    return panel, cfg, static, dynamic, manifest


def prepare_regression(panel, cfg, h, static, dynamic, log_response=True):
    labels = forecast_labels(panel, cfg, h)
    chosen = labels[(labels.split != 'purged') & labels.point_target_ntu.notna()]
    times = chosen.issue_index.to_numpy(int)
    train_end, valid_end = boundaries(len(panel.features), cfg)
    x, mask, scaler = causal_fill(panel.features, train_end, cfg.get('ffill_limit', 3))
    history = histories(np.concatenate([x, mask], -1), times, cfg['history_length']).astype(np.float32)
    target = chosen.point_target_ntu.to_numpy(float)
    split = time_splits(times, h, train_end, valid_end)
    for name, ix in split.items():
        if len(ix) < cfg.get('min_split_samples', 12):
            raise ValueError(f'{name}: insufficient observed regression labels')
    transformed = np.log1p(target) if log_response else target
    mean = float(transformed[split['train']].mean())
    scale = max(float(transformed[split['train']].std()), 1e-6)
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite model inputs or regression labels')
    return dict(x=history, target=target, scaled_target=((transformed - mean) / scale).astype(np.float32),
                times=times, splits=split, target_node=panel.target_nodes[0], horizon=h,
                static=graph_tensors(pack_graphs(static)), dynamic=graph_tensors(pack_graphs(dynamic[times])),
                target_mean=mean, target_scale=scale, log_response=log_response, input_scaler=scaler,
                labels=labels)


def inverse(values, data):
    values = np.asarray(values) * data['target_scale'] + data['target_mean']
    return np.maximum(0., np.expm1(np.clip(values, -20, 30))) if data['log_response'] else values


def predict_scaled(model, data, indices, guard=None, batch_size=128):
    model.eval()
    result = []
    packed = data['dynamic'] if model.name.startswith('Dynamic') else data['static']
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            if guard:
                guard.check()
            ix = indices[start:start + batch_size]
            values = model(torch.from_numpy(data['x'][ix]), select_graph(packed, ix))[:, data['target_node']]
            result.append(values.numpy())
    return np.concatenate(result)


def fit_candidate(name, data, cfg, width, rate, seed, guard=None, status=None):
    configure_cpu()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SnapshotModel(name, input_dim=data['x'].shape[-1], width=width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=cfg.get('weight_decay', 1e-4))
    rng = np.random.default_rng(seed)
    train, valid = data['splits']['train'], data['splits']['validation']
    packed = data['dynamic'] if name.startswith('Dynamic') else data['static']
    best_loss, best_epoch, best, stale = float('inf'), 0, None, 0
    history = []
    for epoch in range(1, cfg['epochs'] + 1):
        if guard:
            guard.check()
        model.train()
        order = rng.permutation(train)
        loss_sum = 0.
        for start in range(0, len(order), cfg['batch_size']):
            if guard:
                guard.check()
            ix = order[start:start + cfg['batch_size']]
            prediction = model(torch.from_numpy(data['x'][ix]), select_graph(packed, ix))[:, data['target_node']]
            # One scalar target per origin (NX08), never broadcast across all nodes.
            target = torch.from_numpy(data['scaled_target'][ix])
            if prediction.shape != target.shape:
                raise ValueError('Target-node output shape mismatch')
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite regression loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ix)
        validation_prediction = predict_scaled(model, data, valid, guard)
        score = float(np.mean((validation_prediction - data['scaled_target'][valid]) ** 2))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation MSE')
        history.append(dict(epoch=epoch, training_mse=loss_sum / len(train), validation_mse=score))
        if score < best_loss - cfg.get('min_delta', 1e-5):
            best_loss, best_epoch, stale = score, epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if status:
            status('training', model=name, horizon=data['horizon'], seed=seed, width=width,
                   learning_rate=rate, epoch=epoch, best_validation_mse=best_loss)
        if stale >= cfg['patience']:
            break
    if best is None:
        raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best)
    return model, dict(width=width, learning_rate=rate, seed=seed, best_epoch=best_epoch,
                       epochs_executed=len(history), validation_mse=best_loss, history=history,
                       selected_epoch_at_cap=best_epoch == cfg['epochs'],
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       loss='MSE on training-standardized log1p(NTU)' if data['log_response'] else 'standardized NTU MSE')


def regression_scores(y, prediction):
    return {**regression(y, prediction),
            'r2_ntu': float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else None,
            'r2_log1p': float(r2_score(np.log1p(y), np.log1p(prediction))) if len(y) > 1 and np.var(np.log1p(y)) > 0 else None}


def validate_config(config):
    if config.get('purpose') != 'exploratory_real_regression' or config.get('runtime') != {
            'device': 'cpu', 'threads': 1, 'workers': 1, 'priority': 'Idle', 'logical_cpus': 1}:
        raise ValueError('Expected the single-worker CPU-only exploratory regression policy')
    for name in ['horizons', 'seeds']:
        values = config[name]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{name}: distinct nonempty values required')
    if any(type(h) is not int or h < 1 for h in config['horizons']):
        raise ValueError('Positive integer horizons required')
    training = config['training']
    for name in ['epochs', 'patience', 'batch_size', 'bootstrap_repetitions', 'bootstrap_block_hours']:
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f'{name}: positive integer required')
    if not training['widths'] or any(type(w) is not int or w < 1 for w in training['widths']):
        raise ValueError('Positive nonempty model widths required')
    if not training['learning_rates'] or any(not np.isfinite(r) or r <= 0 for r in training['learning_rates']):
        raise ValueError('Positive finite learning rates required')


def code_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(list(root.glob('*.py')) + list((root / 'src').rglob('*.py')) +
                   list((root / 'tests').glob('test_real_benchmark.py')))
    return root, paths, digest({p.relative_to(root).as_posix(): file_sha(p) for p in paths})


def candidate_fit(name, data, cfg, seed, width, rate, folder, identity, guard, status):
    folder.mkdir(parents=True, exist_ok=True)
    marker, weights = folder / 'fit.json', folder / 'weights.pt'
    if marker.exists():
        meta = json.loads(marker.read_text('utf-8'))
        if meta.get('plan_identity') != identity or not meta.get('complete') or meta.get('weights_sha256') != file_sha(weights):
            raise ValueError(f'Candidate checkpoint mismatch: {folder}')
        return meta
    print(f'fit h={data["horizon"]} seed={seed} {name} width={width} lr={rate}', flush=True)
    model, meta = fit_candidate(name, data, cfg, width, rate, seed, guard, status)
    temporary = weights.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(weights)
    meta.update(complete=True, plan_identity=identity, weights_sha256=file_sha(weights))
    write_json(marker, meta)
    return meta


def moving_block_indices(times, repetitions, length, seed=47):
    times = np.asarray(times)
    n = len(times)
    boundaries_ = np.r_[0, np.flatnonzero(np.diff(times) != 1) + 1, n]
    ends = np.empty(n, int)
    for first, last in zip(boundaries_[:-1], boundaries_[1:]):
        ends[first:last] = last
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        blocks, count = [], 0
        while count < n:
            start = int(rng.integers(n))
            block = np.arange(start, min(start + length, ends[start]))
            blocks.append(block)
            count += len(block)
        yield np.concatenate(blocks)[:n]


def aggregate(rows, predictions, prepared, seeds, cfg, output, guard):
    pd.DataFrame([{k: v for k, v in row.items() if k != 'selected'} for row in rows]).to_csv(
        output / 'metrics.csv', index=False, encoding='utf-8-sig')
    contrasts = {
        'Static-GAT minus Static-GCN': {'Static-GAT': 1, 'Static-GCN': -1},
        'Dynamic-GAT minus Dynamic-GCN': {'Dynamic-GAT': 1, 'Dynamic-GCN': -1},
        'Dynamic-GCN minus Static-GCN': {'Dynamic-GCN': 1, 'Static-GCN': -1},
        'Dynamic-GAT minus Static-GAT': {'Dynamic-GAT': 1, 'Static-GAT': -1},
        'Architecture-by-dynamicity interaction': {'Dynamic-GAT': 1, 'Static-GAT': -1, 'Dynamic-GCN': -1, 'Static-GCN': 1},
        **{name + ' minus MLP': {name: 1, 'MLP': -1} for name in MODEL_NAMES if name != 'MLP'},
    }
    comparisons = []
    for h, data in prepared.items():
        ix = data['splits']['test']
        y, times = data['target'][ix], data['times'][ix]
        errors = {name: np.stack([np.abs(y - predictions[(h, seed, name)]) for seed in seeds]) for name in MODEL_NAMES}
        deltas = {name: -sum(weight * errors[model] for model, weight in coefficients.items()).mean(0)
                  for name, coefficients in contrasts.items()}
        draws = {name: [] for name in deltas}
        for index in moving_block_indices(times, cfg.get('bootstrap_repetitions', 200),
                                           max(h, cfg.get('bootstrap_block_hours', 24))):
            guard.check()
            for name, values in deltas.items():
                draws[name].append(float(values[index].mean()))
        for name, values in deltas.items():
            comparisons.append(dict(horizon=h, contrast=name, metric='MAE improvement (positive favors first)',
                                    mean_gap=float(values.mean()), ci95=np.quantile(draws[name], [.025, .975]).tolist(),
                                    bootstrap_fraction_positive=float((np.asarray(draws[name]) > 0).mean()),
                                    samples=len(y), block_hours=max(h, cfg.get('bootstrap_block_hours', 24))))
    write_json(output / 'paired_contrasts.json', dict(comparisons=comparisons,
        scope='paired time blocks averaged over fixed training seeds; no independent-dataset claim; not a p-value'))
    table = pd.DataFrame(rows).groupby(['horizon', 'model'])[['mae', 'rmse', 'r2_ntu', 'r2_log1p']].mean().reset_index()
    table.to_csv(output / 'model_summary.csv', index=False, encoding='utf-8-sig')
    lines = ['# 真實濁度回歸比較（探索性）', '',
             'NX08；原 2021 期間與 chronological split 保持不變。五模型採用每站相同 24h 歷史與 missing masks，單一 aggregation layer 接 V2 self branch。此 input adapter 與 synthetic p=8 設計不同。', '',
             '所有候選以 validation regression MSE 選模，完成全部模型選模後才進行 test model scoring。', '',
             '| h | Model | Mean MAE (NTU) | Mean RMSE (NTU) | Mean R² (NTU) |', '|---:|---|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append(f'| {row["horizon"]} | {row["model"]} | {row["mae"]:.4f} | {row["rmse"]:.4f} | {row["r2_ntu"]:.4f} |')
    lines += ['', '各 seed 的結果見 metrics.csv；paired_contrasts.json 使用共同 test origins、24h time blocks，並保持模型間配對。隨機種子代表同一資料集的初始化變異，不是五份獨立水文資料。', '',
              'Persistence 對照僅在 current observation 可用的共同 origins 比較，另記錄排除數。', '',
              '分類 validation 無高濁度正類，未執行事件分類訓練或預警提前量評估。此結果不代表完成高濁度 early warning 或 synthetic phase map 的外部驗證。', '',
              '三站超出 synthetic N=100–1000 支援範圍，GP17 是下游，圖是估計相關圖，歷史同期雨量／流量／水位與發布延遲仍未補齊。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(config, algorithm_root, output, policy, status, guard, preflight_only=False):
    configure_cpu()
    output, algorithm_root = Path(output), Path(algorithm_root)
    validate_config(config)
    status('preflight', resources_policy=policy)
    guard.check(force=True)
    folder = (algorithm_root / config['analysis_directory']).resolve()
    panel, local, static, dynamic, audited = load_audited(folder)
    if len(set(config['seeds'])) != len(config['seeds']) or not config['seeds']:
        raise ValueError('Distinct paired seeds are required')
    if not set(config['horizons']) <= set(local['horizons']):
        raise ValueError('Horizon is outside the audited protocol')
    all_labels = pd.concat([forecast_labels(panel, local, h) for h in config['horizons']])
    readiness = label_readiness(all_labels, local)
    if not readiness.regression_ready.all():
        raise ValueError('Regression sample checks failed')
    if readiness.classification_ready.all():
        class_reason = 'Not in this regression-only execution plan'
    else:
        class_reason = 'Insufficient event-class coverage; threshold and chronological split unchanged'
    root, paths, source = code_identity()
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'torch', 'scikit-learn']}
    candidate_count = len(config['training']['widths']) * len(config['training']['learning_rates'])
    jobs = len(config['horizons']) * len(config['seeds']) * len(MODEL_NAMES)
    plan = dict(config=config, source_sha256=source, python=platform.python_version(), versions=versions,
                analysis_manifest_sha256=file_sha(folder / 'analysis_manifest.json'),
                raw_sources=audited['raw_source_fingerprints'],
                dataset=dict(stations=panel.node_names, target=local['target_station'], start=local['start'], end=local['end']),
                models=MODEL_NAMES, jobs=jobs, candidate_fits=jobs * candidate_count,
                task='regression_only', classification=dict(status='not_run', reason=class_reason),
                selection='training-standardized log1p target validation MSE; no test selection',
                external_validation_completed=False)
    identity = digest(plan)
    write_json(output / 'preflight.json', {**plan, 'plan_identity': identity,
                                          'readiness': readiness.to_dict('records'), 'passed_for_regression': True})
    if preflight_only:
        status('preflight_passed', candidate_fits=plan['candidate_fits'], jobs=jobs)
        return dict(preflight_passed=True, candidate_fits=plan['candidate_fits'], classification=plan['classification'])
    marker = output / 'manifest.json'
    if marker.exists():
        prior = json.loads(marker.read_text('utf-8'))
        if prior.get('plan_identity') != identity:
            raise ValueError('Resume plan/code/data/environment differs; use the frozen code or a new output directory')
    else:
        write_json(marker, {**plan, 'plan_identity': identity, 'created_at': datetime.now(timezone.utc).isoformat()})
        snapshot = output / 'source_snapshot'
        for path in paths:
            destination = snapshot / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        write_json(output / 'config_snapshot.json', config)
    if (output / 'complete.json').exists():
        status('complete', jobs_completed=jobs, candidate_fits_completed=jobs * candidate_count)
        return json.loads((output / 'complete.json').read_text('utf-8'))
    prepared, selected, completed = {}, {}, 0
    train_cfg = config['training']
    for h in config['horizons']:
        guard.check(force=True)
        data = prepare_regression(panel, local, h, static, dynamic, train_cfg.get('log_response', True))
        prepared[h] = data
        write_json(output / f'preprocessing_h{h}.json', {key: data[key] for key in ['input_scaler', 'target_mean', 'target_scale', 'log_response']})
        for seed in config['seeds']:
            for name in MODEL_NAMES:
                job = output / f'h{h}' / f'seed_{seed}' / name
                candidates = []
                for width, rate in itertools.product(train_cfg['widths'], train_cfg['learning_rates']):
                    guard.check()
                    candidate = job / f'w{width}_lr{rate:g}'
                    meta = candidate_fit(name, data, train_cfg, seed, width, rate, candidate, identity, guard, status)
                    candidates.append((meta['validation_mse'], width, rate, candidate, meta))
                    completed += 1
                    status('training', candidate_fits_completed=completed, candidate_fits_total=jobs * candidate_count)
                best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
                selected[(h, seed, name)] = (best[3], best[4])
                write_json(job / 'selected.json', dict(plan_identity=identity, width=best[1], learning_rate=best[2],
                                                      validation_mse=best[0], candidate=best[3].name))
    print('All candidate selection finished; starting held-out regression evaluation.', flush=True)
    status('evaluating', candidate_fits_completed=completed, candidate_fits_total=jobs * candidate_count)
    rows, predictions, baseline = [], {}, []
    for (h, seed, name), (candidate, meta) in selected.items():
        guard.check()
        data = prepared[h]
        model = SnapshotModel(name, input_dim=data['x'].shape[-1], width=meta['width'])
        if file_sha(candidate / 'weights.pt') != meta['weights_sha256']:
            raise ValueError(f'Selected checkpoint changed before evaluation: {candidate}')
        model.load_state_dict(torch.load(candidate / 'weights.pt', map_location='cpu', weights_only=True))
        ix = data['splits']['test']
        prediction = inverse(predict_scaled(model, data, ix, guard), data)
        predictions[(h, seed, name)] = prediction
        current = panel.response[data['times'][ix], panel.target_nodes[0]]
        common = np.isfinite(current)
        row = dict(horizon=h, seed=seed, model=name, samples=len(ix),
                   **regression_scores(data['target'][ix], prediction), selected={k: v for k, v in meta.items() if k != 'history'})
        rows.append(row)
        baseline.append(dict(horizon=h, seed=seed, model=name, samples=int(common.sum()),
                             excluded_missing_current=int((~common).sum()),
                             model_mae=regression(data['target'][ix][common], prediction[common])['mae'],
                             persistence_mae=regression(data['target'][ix][common], current[common])['mae']))
        job = candidate.parent
        temporary = job / 'predictions.tmp.npz'
        np.savez_compressed(temporary, test_issue_indices=data['times'][ix],
                            test_issue_times=panel.timestamps[data['times'][ix]],
                            test_target=data['target'][ix], test_prediction=prediction)
        temporary.replace(job / 'predictions.npz')
        write_json(job / 'result.json', row)
    write_json(output / 'persistence_common_origins.json', baseline)
    aggregate(rows, predictions, prepared, config['seeds'], train_cfg, output, guard)
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(), regression_complete=True,
                    classification=plan['classification'], jobs=jobs, candidate_fits=completed,
                    plan_identity=identity, formal_external_validation_completed=False)
    write_json(output / 'complete.json', complete)
    status('complete', jobs_completed=jobs, candidate_fits_completed=completed)
    return complete
