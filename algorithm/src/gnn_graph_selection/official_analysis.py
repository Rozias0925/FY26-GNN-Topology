"""Measured-data analysis separate from synthetic results and winner selection.

Inputs and graph snapshots at issue t use observations <= t. Labels are never
imputed. Regression and any-exceedance classification have separate validity
criteria. Diagnostics use only the outer training period with purged inner
chronological folds. Test event counts are readiness audits, not tuning criteria.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification as classification_metrics, regression as regression_metrics, extract_events
from .io import write_json, digest, source_digest
from .protocol import causal_fill, graph_inputs, time_splits
from .real_data import audit, load_turbidity, prepare_real, hourly_station, timestamps
from .sources import SOURCES, read_archive, archive_fingerprint

KNOWN_STATIONS = {
    'NW12': ('龍珠灣', 'reservoir_site'),
    'NX08': ('壩前浮台', 'reservoir_site'),
    'GP17': ('後池堰排砂道', 'downstream_site'),
}


def csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def boundaries(total, cfg):
    train = int(total * cfg.get('train_fraction', .6))
    valid = int(total * (cfg.get('train_fraction', .6) + cfg.get('validation_fraction', .2)))
    if not 0 < train < valid < total:
        raise ValueError('Invalid chronological split fractions')
    return train, valid


def forecast_labels(panel, cfg, horizon):
    """Explicit window labels; missing future hours do not become negatives.

    A point target only needs the measurement at t+h. An event target needs
    ALL hourly peaks in (t,t+h]. Boundary-crossing targets are purged.
    """
    if horizon < 1 or len(panel.target_nodes) != 1:
        raise ValueError('Positive horizon and one target station are required')
    total = len(panel.response)
    train_end, val_end = boundaries(total, cfg)
    first = max(cfg['history_length'] - 1, cfg['graph_window'] - 1)
    if first >= train_end or total <= first + horizon:
        raise ValueError('History/calibration window leaves no training forecasts')
    tau = float(cfg['threshold'])
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('Use a finite predefined nonnegative NTU threshold')
    origin = np.arange(first, total - horizon)
    node = panel.target_nodes[0]
    point = panel.response[:, node]
    peaks = (panel.event_response if panel.event_response is not None else panel.response)[:, node]
    future = np.stack([peaks[t + 1:t + horizon + 1] for t in origin])
    complete = np.isfinite(future).all(1)
    events = np.where(complete, (future > tau).any(1).astype(float), np.nan)
    splits = np.full(len(origin), 'purged', dtype=object)
    for name, ix in time_splits(origin, horizon, train_end, val_end).items():
        splits[ix] = name
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    return pd.DataFrame({
        'issue_index': origin, 'issue_time': dates[origin].astype(str),
        'target_time': dates[origin + horizon].astype(str), 'horizon_hours': horizon,
        'split': splits, 'current_ntu': point[origin], 'current_peak_ntu': peaks[origin],
        'point_target_ntu': point[origin + horizon], 'event_target': events,
        'complete_future_event_window': complete, 'threshold_ntu': tau,
    })


def label_readiness(labels, cfg):
    rows = []
    minimum = cfg.get('min_split_samples', 12)
    for horizon, group in labels.groupby('horizon_hours'):
        for split in ['train', 'validation', 'test']:
            subset = group[group.split == split]
            event = subset.event_target.dropna()
            count = int(subset.point_target_ntu.notna().sum())
            positives, negatives = int((event == 1).sum()), int((event == 0).sum())
            rows.append(dict(horizon_hours=int(horizon), split=split,
                             origins=len(subset), regression_windows=count,
                             event_windows=len(event), positives=positives, negatives=negatives,
                             regression_ready=count >= minimum,
                             classification_ready=len(event) >= minimum and positives > 0 and negatives > 0))
    return pd.DataFrame(rows)


def source_inventory(root, period_index, output, turbidity_frame=None):
    """Keep auxiliary snapshots separate; actual time overlap is explicit."""
    rows = []
    for name, spec in SOURCES.items():
        path = root / name / 'manifest.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text('utf-8'))
        row = dict(source=name, title=spec['title'], catalog=spec['catalog'],
                   downloaded_rows=manifest.get('rows', 0),
                   transport_complete=manifest.get('download_complete', False),
                   error=manifest.get('error'),
                   content_sha256=archive_fingerprint(path.parent))
        frame = turbidity_frame if name == 'turbidity' and turbidity_frame is not None else pd.DataFrame(read_archive(path.parent))
        if not frame.empty:
            # Full national station metadata is useful; observations are separately scoped.
            if 'reservoiridentifier' in frame:
                frame = frame[frame.reservoiridentifier.astype(str) == '10201'].copy()
            if name == 'catchment_rain':
                frame = frame[frame.locationid.astype(str) == 'R10201'].copy()
            row['shihmen_rows_if_applicable'] = len(frame)
            field = spec.get('time_field')
            if field in frame:
                dates = timestamps(frame[field])
                row.update(start=str(dates.min()), end=str(dates.max()),
                           observations_in_research_period=int(((dates >= period_index[0]) &
                                                               (dates <= period_index[-1])).sum()))
                if name != 'turbidity':
                    frame['timestamp_utc'] = dates.astype(str)
            if name == 'river_level':
                # Levels below the station datum can legitimately be negative.
                values = pd.to_numeric(frame.waterlevel, errors='coerce')
                verified = frame.checkresult.astype(str).str.lower().eq('true')
                frame['waterlevel_clean_m'] = values.where(np.isfinite(values) & verified & (values > -999))
            if name == 'catchment_rain':
                values = pd.to_numeric(frame.rainfall_raw, errors='coerce')
                frame['rainfall_clean_mm'] = values.where(np.isfinite(values) & (values >= 0))
                frame['accumulation_semantics_verified_for_predictor'] = False
                # Preserve FEWS flags; do not infer QC meanings or cumulative resets.
            csv(frame, output / f'official_{name}.csv') if name != 'turbidity' else None
        rows.append(row)
    inventory = pd.DataFrame(rows)
    csv(inventory, output / 'source_inventory.csv')
    return rows


def station_metadata(frame, root, output):
    catalog = root / 'metadata/turbidity_catalog.html'
    text = catalog.read_text('utf-8-sig') if catalog.exists() else ''
    rows = []
    for station in sorted(frame.station.unique()):
        name, role = KNOWN_STATIONS.get(station, ('未確認', 'unknown'))
        verified = station in KNOWN_STATIONS and station + name in text
        rows.append(dict(station=station, name=name if verified else '未確認',
                         role=role if verified else 'unknown',
                         verified_from_saved_catalog=verified, source=SOURCES['turbidity']['catalog'],
                         coordinates_verified=False, sensor_depth_verified=False))
    csv(pd.DataFrame(rows), output / 'station_metadata.csv')
    return rows


def export_hourly(panel, cfg, frame, output):
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    train, valid = boundaries(len(dates), cfg)
    split = np.where(np.arange(len(dates)) < train, 'train',
                     np.where(np.arange(len(dates)) < valid, 'validation', 'test'))
    tables = []
    monthly = []
    for i, station in enumerate(panel.node_names):
        count = hourly_station(frame[frame.station == station], cfg['minimum_observations_per_hour'],
                               cfg['stuck_hours'])['count']
        tables.append(pd.DataFrame({'timestamp': dates.astype(str), 'station': station, 'split': split,
                                    'last_ntu': panel.response[:, i],
                                    'hourly_peak_ntu': panel.event_response[:, i],
                                    'valid_raw_samples': count.reindex(pd.to_datetime(panel.timestamps)).to_numpy()}))
        for month in sorted(set(dates.strftime('%Y-%m'))):
            values = panel.event_response[dates.strftime('%Y-%m') == month, i]
            monthly.append(dict(station=station, month=month, total_hours=len(values),
                                observed_hours=int(np.isfinite(values).sum()),
                                exceeding_hours=int((values > cfg['threshold']).sum()),
                                missing_hours=int((~np.isfinite(values)).sum())))
    csv(pd.concat(tables, ignore_index=True), output / 'hourly_turbidity.csv')
    csv(pd.DataFrame(monthly), output / 'monthly_coverage_events.csv')
    return monthly


def event_catalog(panel, cfg):
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    train, valid = boundaries(len(dates), cfg)
    rows = []
    for i, station in enumerate(panel.node_names):
        values = panel.event_response[:, i]
        for start, end in extract_events(values, cfg['threshold']):
            left_known = start > 0 and np.isfinite(values[start - 1]) and values[start - 1] <= cfg['threshold']
            right_known = end + 1 < len(values) and np.isfinite(values[end + 1]) and values[end + 1] <= cfg['threshold']
            segment = 'train' if end < train else 'validation' if start >= train and end < valid else 'test' if start >= valid else 'cross_split'
            rows.append(dict(station=station, start=dates[start].isoformat(), end=dates[end].isoformat(),
                             start_index=start, end_index=end, duration_observed_hours=end - start + 1,
                             maximum_ntu=float(values[start:end + 1].max()), split=segment,
                             onset_observed=left_known, end_observed=right_known,
                             censored=not (left_known and right_known)))
    return pd.DataFrame(rows, columns=['station', 'start', 'end', 'start_index', 'end_index',
                                      'duration_observed_hours', 'maximum_ntu', 'split',
                                      'onset_observed', 'end_observed', 'censored'])


def histories(features, times, length):
    # Per-node feature history, not a recurrent temporal model.
    return np.stack([features[t - length + 1:t + 1].transpose(1, 0, 2).reshape(features.shape[1], -1)
                     for t in times])


def inner_folds(origins, horizon, folds=3):
    """Use actual issue indices to purge labels crossing an inner cutoff."""
    origins = np.asarray(origins, int)
    edges = np.unique(np.linspace(len(origins) // 2, len(origins), folds + 1, dtype=int))
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        cutoff = origins[start]
        fit = np.flatnonzero(origins + horizon < cutoff)
        hold = np.arange(start, end)
        if len(fit) >= 12 and len(hold) >= 2:
            yield fit, hold


def mean_neighbors(features, graphs):
    adjacency = np.asarray(graphs, float)
    if adjacency.ndim == 2:
        adjacency = np.broadcast_to(adjacency, (len(features), *adjacency.shape))
    degree = adjacency.sum(-1)
    return np.einsum('tij,tjf->tif', adjacency, features) / np.maximum(degree[:, :, None], 1)


def control_permutations(n, repetitions=5, seed=47):
    """Do not accidentally call the identity permutation a shuffled control.

    Graph automorphisms can still yield identical topology, which is separately
    reported instead of forcing a useful-looking control or selecting by score.
    """
    from math import factorial
    if n < 2 or repetitions < 1:
        raise ValueError('Need at least two nodes and one shuffled control')
    count = min(repetitions, factorial(n) - 1) if n <= 7 else repetitions
    rng = np.random.default_rng(seed)
    permutations = []
    seen = {tuple(range(n))}
    while len(permutations) < count:
        permutation = tuple(rng.permutation(n).tolist())
        if permutation not in seen:
            seen.add(permutation)
            permutations.append(np.asarray(permutation))
    return permutations


def training_diagnostics(panel, cfg, analysis_cfg, labels, static, dynamic):
    """Fixed ridge probes within training. Same inner folds for all contrasts.

    S_X/S_G/D_pred are NX08 point-target, log-NTU R² quantities; H and R_hat
    use complete all-station training snapshots. These are exploratory task
    adapters and are not numerically identical to the synthetic p=8 task.
    """
    from gnn_mechanisms.diagnostics import smoothness, estimate_relevance
    from gnn_mechanisms.graphs import structural_dynamicity
    train_end, _ = boundaries(len(panel.response), cfg)
    result = {'scope': 'outer training period only; purged inner chronological validation',
              'target_scale': 'log1p(NTU)', 'node_count': len(panel.node_names),
              'synthetic_node_range': [100, 1000],
              'within_synthetic_node_support': 100 <= len(panel.node_names) <= 1000,
              'graph_semantics': 'undirected lagged-correlation topology, not verified river connectivity',
              'probe': 'own 24h log-turbidity + missing-mask history; uniform neighbor summary; ridge',
              'R_hat_scope': 'source-feature gating on complete training snapshots; exploratory, no oracle R',
              'horizons': []}
    first = cfg['graph_window'] - 1
    result['D_struct'] = structural_dynamicity(dynamic[first:train_end])
    # No preprocessing statistics from validation/test; fitting each inner fold below is separate.
    node = panel.target_nodes[0]
    # Training-median-filled values are used only for H/gate descriptives, never inner probe scores.
    filled, mask, _ = causal_fill(panel.features[:train_end], train_end, cfg.get('ffill_limit', 3))
    for h, group in labels.groupby('horizon_hours'):
        selected = group[(group.split == 'train') & group.point_target_ntu.notna()]
        origins = selected.issue_index.to_numpy(int)
        target = np.log1p(selected.point_target_ntu.to_numpy(float))
        truth = []
        permutations = control_permutations(len(panel.node_names),
                                            analysis_cfg.get('shuffled_graph_repetitions', 5),
                                            analysis_cfg.get('seed', 47))
        control_names = [f'shuffled_{i}' for i in range(len(permutations))]
        predictions = {name: [] for name in ['feature', 'dynamic', 'static', *control_names]}
        fold_records = []
        shuffled = {name: dynamic[origins][:, permutation][:, :, permutation]
                    for name, permutation in zip(control_names, permutations)}
        for fit, hold in inner_folds(origins, int(h), analysis_cfg.get('probe_folds', 3)):
            # Train medians/scales strictly before the inner validation cutoff.
            cutoff = int(origins[hold[0]])
            x, observed, _ = causal_fill(panel.features[:train_end], cutoff, cfg.get('ffill_limit', 3))
            history = histories(np.concatenate([x, observed], -1), origins, cfg['history_length'])
            own = history[:, node]
            summaries = {'dynamic': mean_neighbors(history, dynamic[origins])[:, node],
                         'static': mean_neighbors(history, static)[:, node],
                         **{name: mean_neighbors(history, graphs)[:, node] for name, graphs in shuffled.items()}}
            designs = {'feature': own, **{k: np.concatenate([own, z], -1) for k, z in summaries.items()}}
            for name, design in designs.items():
                probe = make_pipeline(StandardScaler(), Ridge(alpha=analysis_cfg.get('probe_ridge_alpha', 10)))
                probe.fit(design[fit], target[fit])
                predictions[name].append(probe.predict(design[hold]))
            truth.append(target[hold])
            fold_records.append({'fit_last_issue': int(origins[fit[-1]]),
                                 'fit_last_target': int(origins[fit[-1]] + h),
                                 'validation_first_issue': cutoff, 'validation_origins': len(hold)})
        scores = {name: float(r2_score(np.concatenate(truth), np.concatenate(pred)))
                  for name, pred in predictions.items()} if truth else {}
        if scores:
            scores['shuffled_mean'] = float(np.mean([scores[name] for name in control_names]))
        record = dict(horizon_hours=int(h), training_regression_windows=len(origins), folds=fold_records,
                      probe_scores=scores, S_X=scores.get('feature'),
                      S_G=scores['dynamic'] - scores['feature'] if scores else None,
                      D_pred=scores['dynamic'] - scores['static'] if scores else None,
                      graph_vs_shuffled=scores['dynamic'] - scores['shuffled_mean'] if scores else None,
                      shuffled_control_permutations=[p.tolist() for p in permutations],
                      shuffled_topology_change_fraction={name: float(np.any(graphs != dynamic[origins], axis=(1, 2)).mean())
                                                         for name, graphs in shuffled.items()})
        y = panel.response[origins + h]
        all_known = np.isfinite(y).all(1)
        use = origins[all_known]
        record['H'] = smoothness(np.log1p(y[all_known]), dynamic[use]) if len(use) else None
        record['H_complete_snapshots'] = len(use)
        maximum = analysis_cfg.get('gate_max_snapshots', 300)
        sampled = np.unique(np.linspace(0, len(use) - 1, min(maximum, len(use)), dtype=int)) if len(use) else []
        if len(sampled) >= 12:
            # Gate is a training-only descriptive estimator, not a held-out predictor.
            ix = use[sampled]
            record['R_hat'] = estimate_relevance(filled[ix], np.log1p(panel.response[ix + h]), dynamic[ix],
                                                epochs=analysis_cfg.get('gate_epochs', 40),
                                                seed=analysis_cfg.get('seed', 47))
            record['R_hat_snapshots'] = len(ix)
        else:
            record.update(R_hat=None, R_hat_snapshots=len(sampled))
        result['horizons'].append(record)
    return result


def block_paired_intervals(labels, prediction_a, prediction_b, replicates=200, block_hours=24, seed=47):
    """Time blocks respect missing-origin gaps; positive MAE gap favors A."""
    origins = labels.issue_index.to_numpy(int)
    y = labels.point_target_ntu.to_numpy(float)
    if len(y) < 2:
        return None
    difference = np.abs(y - prediction_b) - np.abs(y - prediction_a)
    runs = np.split(np.arange(len(y)), np.flatnonzero(np.diff(origins) != 1) + 1)
    blocks = [run[start:start + block_hours] for run in runs for start in range(len(run))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        chosen = []
        count = 0
        while count < len(y):
            block = blocks[int(rng.integers(len(blocks)))]
            chosen.append(block)
            count += len(block)
        index = np.concatenate(chosen)[:len(y)]
        draws.append(float(difference[index].mean()))
    return dict(metric='MAE(B)-MAE(A)', mean=float(difference.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist(), bootstrap_p_delta_positive=float(np.mean(np.asarray(draws) > 0)),
                replicates=replicates, block_hours=block_hours, method='paired moving time blocks, no gap crossing')


def baselines(labels, analysis_cfg):
    rows = []
    paired = []
    predictions = []
    for h, group in labels.groupby('horizon_hours'):
        train = group[group.split == 'train']
        median = float(train.point_target_ntu.median())
        prevalence = float(train.event_target.mean())
        for split in ['validation', 'test']:
            frame = group[group.split == split].copy()
            reg = frame[frame.point_target_ntu.notna() & frame.current_ntu.notna()]
            for name, estimate in [('persistence', reg.current_ntu.to_numpy()),
                                   ('training_median', np.full(len(reg), median))]:
                if len(reg):
                    rows.append(dict(horizon_hours=int(h), split=split, task='regression', baseline=name,
                                     samples=len(reg), **regression_metrics(reg.point_target_ntu, estimate)))
            if len(reg) >= 2:
                paired.append(dict(horizon_hours=int(h), split=split, A='persistence', B='training_median',
                                   **block_paired_intervals(reg, reg.current_ntu.to_numpy(), np.full(len(reg), median),
                                                           replicates=analysis_cfg.get('bootstrap_replicates', 200),
                                                           block_hours=analysis_cfg.get('bootstrap_block_hours', 24),
                                                           seed=analysis_cfg.get('seed', 47))))
                part = reg[['issue_time', 'target_time', 'horizon_hours', 'split', 'point_target_ntu', 'event_target']].copy()
                part['persistence_ntu'] = reg.current_ntu.to_numpy()
                part['training_median_ntu'] = median
                predictions.append(part)
            event = frame[frame.event_target.notna() & frame.current_peak_ntu.notna()]
            for name, score in [('current_state_indicator', (event.current_peak_ntu > event.threshold_ntu).to_numpy(float)),
                                ('training_prevalence', np.full(len(event), prevalence))]:
                if len(event) and np.isfinite(prevalence):
                    values = event.event_target.to_numpy(int)
                    rows.append(dict(horizon_hours=int(h), split=split, task='classification', baseline=name,
                                     samples=len(event), positives=int(values.sum()), negatives=int(len(values) - values.sum()),
                                     **classification_metrics(values, score, .5)))
    return pd.DataFrame(rows), paired, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def plots(panel, cfg, monthly, static, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    train, valid = boundaries(len(dates), cfg)
    fig, axes = plt.subplots(len(panel.node_names), 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(dates, panel.event_response[:, i], lw=.6, color='#244c49')
        ax.axhline(cfg['threshold'], color='#806341', ls='--', lw=.9, label=f"{cfg['threshold']:g} NTU")
        for cutoff in [train, valid]:
            ax.axvline(dates[cutoff], color='#777', ls=':', lw=.9)
        ax.set_yscale('symlog', linthresh=1)
        finite = panel.event_response[:, i][np.isfinite(panel.event_response[:, i])]
        ax.set_ylim(0, 1.15 * max(cfg['threshold'], float(finite.max()) if len(finite) else 0))
        ax.set_ylabel(panel.node_names[i] + '\nNTU')
        ax.grid(alpha=.2)
    axes[0].set_title('Hourly turbidity peaks; gaps remain missing; dotted lines = fixed chronological splits')
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output / 'hourly_turbidity.png', dpi=170)
    plt.close(fig)
    target = pd.DataFrame(monthly)
    target = target[target.station == cfg['target_station']]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(target.month, target.exceeding_hours, color='#244c49')
    axes[0].set_ylabel('Hours above threshold')
    axes[0].set_title(cfg['target_station'] + ': fixed-threshold event prevalence and coverage')
    axes[1].bar(target.month, 100 * target.missing_hours / target.total_hours, color='#897961')
    axes[1].set_ylabel('Missing hours (%)')
    axes[1].tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(output / 'monthly_events_coverage.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(static, vmin=0, vmax=1, cmap='Greys')
    ax.set_xticks(range(len(static)), panel.node_names)
    ax.set_yticks(range(len(static)), panel.node_names)
    ax.set_title('Static graph: first 168h calibration only\nEstimated relevance, not physical flow direction')
    fig.tight_layout()
    fig.savefig(output / 'static_graph.png', dpi=170)
    plt.close(fig)


def report(result, inventory, readiness, diagnostics, event_rows, baseline_rows, output):
    lines = ['# 石門水庫官方資料分析', '',
             f"產生時間：{result['generated_at']}；資料與設定 SHA-256 見 `analysis_manifest.json`。", '',
             '## 研究範圍與資料', '',
             '本流程只分析與預測濁度，不求解取水、供水、水庫操作或淨水配置。', '',
             f"官方濁度下載共 {result['raw_turbidity_rows']:,} 筆。固定分析期間 {result['period']['start']} 至 {result['period']['end_exclusive']}（右端不含），共 {result['hourly_steps']:,} 小時。",
             f"目標站 {result['target_station']}；閾值 {result['threshold_ntu']:g} NTU；預測 1／3／6 小時。", '',
             '| 官方資料 | 筆數 | 此研究期間內的觀測 | 下載狀態 |',
             '|---|---:|---:|---|']
    for row in inventory:
        lines.append(f"| {row['title']} | {row['downloaded_rows']:,} | {row.get('observations_in_research_period', '測站 metadata')} | {'完整傳輸' if row['transport_complete'] else '未完整'} |")
    lines += ['', '「完整傳輸」表示已讀完 API 當次回傳，並不證明所有歷史年份均公開。原始 JSON 壓縮頁、URL、擷取時間及內容雜湊留存在官方下載資料夾；CSV 為整理後副本。', '',
              'NW12＝龍珠灣；NX08＝壩前浮台；GP17＝後池堰排砂道。GP17 是下游位置，不可直接稱為上游測站。NW22 未在當次官方目錄取得位置定義，因此只列入原始資料盤點。', '',
              '## 品質與標籤定義', '',
              '每小時以 (t−1h,t] 聚合：最後有效觀測是 point-target 的近似，最大值用於高濁度事件標籤；至少 6 筆有效子小時觀測。負濁度、無法解析值及衝突重複值不作有效觀測；連續常數讀值超過 24 小時後因果性地標為缺失，不回頭刪除較早時段。零讀值仍保留並在品質盤點中標記。', '',
              'Classification：未來 (t,t+h] 任何一小時的 peak > 1000 NTU。只有所有未來小時均有觀測才給 0／1 標籤，缺失不當成未發生。Regression：t+h 小時最後有效觀測；只要求該目標小時已觀測。兩種 task 分開判斷可用性，不再因 classification 缺類別就擋住 regression 分析。', '',
              '60%／20%／20% 按時間切割，跨切割界線的預測目標 purge。閾值與切割沿用原設定，不依 test 事件數調整。圖用最初 168 小時作 static calibration，dynamic 每 6 小時用截至當下的 168 小時資料更新；缺失輸入只用最多 3 小時的過去值與 training median，另附 missing mask。這是估計的無向相關圖，並非已驗證的河道連接圖。', '',
              '## 目前能否比較預測模型', '',
              '| h | 切割 | Regression windows | Event windows | 正類 | 負類 | Classification 可驗證 |',
              '|---:|---|---:|---:|---:|---:|---|']
    for row in readiness.to_dict('records'):
        lines.append(f"| {row['horizon_hours']} | {row['split']} | {row['regression_windows']} | {row['event_windows']} | {row['positives']} | {row['negatives']} | {'是' if row['classification_ready'] else '不足'} |")
    lines += ['', f"Regression 數量檢查：{'通過' if result['regression_ready'] else '未通過'}；Classification 類別與數量檢查：{'通過' if result['classification_ready'] else '未通過'}。通過數量檢查不等於通過資料、物理圖、發布延遲與外部有效性檢查。", '',
              '事件清單見 `events.csv`：記錄 onset/end、峰值、持續觀測時數、所在切割及 censoring。這是事件描述，不是「提前預警成功」的證據；真正預警需後續模型在事件起點前的 alarm、一對一事件配對、提前時間、漏報與誤報。', '',
              '## 僅以訓練期間估計的 diagnostics', '',
              '| h | S_X | S_G | H | R_hat | D_pred |',
              '|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for row in diagnostics['horizons']:
        lines.append('| ' + ' | '.join([str(row['horizon_hours'])] + [fmt(row.get(key)) for key in ['S_X', 'S_G', 'H', 'R_hat', 'D_pred']]) + ' |')
    lines += ['', f"D_struct＝{diagnostics['D_struct']:.4f}，是相鄰 hourly graph 的平均 1−Jaccard；本設定每 6 小時更新，因此中間不變的 snapshot 也納入計算。", '',
              'S_X 是固定 ridge feature-only probe 在 training 內部的時間驗證 R²（log-NTU）；S_G 是加入 dynamic neighbor history summary 的 R² 增量；D_pred 是相同 probe 的 dynamic−static R²。每個內部 fold 均重新用先前資料估計 imputation/scaling，並 purge 越過其界線的標籤。負增量保留，不截成 0。另提供多個不含 identity permutation 的 shuffled graph 對照，報其平均 probe score。只有三個節點時，部分 graph automorphism 仍可能不改變拓撲，JSON 逐一列出改變比例；不依 target 或模型分數挑選置換。', '',
              'H＝1−observed Dirichlet energy／random node permutation 的精確期望，只用訓練期間 all-station 完整目標；R_hat 是簡單 source-feature gate 的 normalized entropy concentration，使用最多 300 個固定等距 training snapshots。真實 R 無法取得；R_hat 不是真正鄰居重要性的保證。', '',
              '此處只有 3 個測站；synthetic 的 N 範圍是 100–1000，feature 與 target 也不同。這些 diagnostics 是 task adapter 的探索性分析，不可直接將此數值代入 simulation phase diagram 宣稱確定 winner；完整 real-world validation 仍需事先記錄預期並檢驗五模型 held-out 表現。', '',
              '## 可重現的 baseline 分析', '',
              '已計算 persistence（未來值＝目前值）與 training-median regression baseline；以原始 NTU 報 MAE／RMSE，並用同一批觀測做 paired 24h time-block bootstrap。Classification 提供 current-state indicator 與 training-prevalence 對照，q 固定 0.5；這些不是訓練完成的五個模型，也不是 early-warning 系統。', '',
              '詳見 `baseline_metrics.csv`、`baseline_paired_comparisons.json`、`baseline_predictions.csv`。單一類別的 AUROC、無正類的 AP／recall 等保留 NA，不填入虛假成功率。', '',
              '## 仍需補足的資料與限制', '']
    lines += [f'- {issue}' for issue in result['blockers']]
    lines += ['', '原始水庫水情的累積雨量不等於每小時雨量；FEWS 雨量保留 value_type、time_step、quality_flag，未核對其 accumulation semantics 與 QC flag 意義前不作 predictors。河川水位可低於 datum，不把所有負水位一律判成無效。', '',
              '正式網站可以先引用資料盤點、定義、圖表與研究限制；不應把 baseline 或 readiness audit 寫成完成高濁度外部驗證。']
    (output / 'ANALYSIS_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze(raw_root, config, output):
    root, output = Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Avoid presenting stale outputs as belonging to a different analysis setting.
    marker = output / 'analysis_manifest.json'
    if marker.exists():
        prior = json.loads(marker.read_text('utf-8'))
        if prior['config_sha256'] != digest(config):
            raise ValueError('Output already belongs to another config; choose a new output directory')
    write_json(marker, {'completed': False, 'config_sha256': digest(config),
                        'started_at': datetime.now(timezone.utc).isoformat()})
    cfg = {**config['data'], **config['protocol']}
    analysis_cfg = config.get('analysis', {})
    print('analysis: verify archive checksums and build hourly panel', flush=True)
    frame = load_turbidity(root / 'turbidity')
    basic_audit = audit(root, output, frame=frame)
    panel = prepare_real(root, cfg, output, frame=frame)
    inventory = source_inventory(root, pd.to_datetime(panel.timestamps, utc=True), output, turbidity_frame=frame)
    names = station_metadata(frame, root, output)
    monthly = export_hourly(panel, cfg, frame, output)
    event_rows = event_catalog(panel, cfg)
    csv(event_rows, output / 'events.csv')
    labels = pd.concat([forecast_labels(panel, cfg, h) for h in cfg['horizons']], ignore_index=True)
    csv(labels, output / 'forecast_labels.csv')
    readiness = label_readiness(labels, cfg)
    csv(readiness, output / 'label_readiness.csv')
    print('analysis: causal graphs and training-only diagnostic probes', flush=True)
    static, dynamic, _ = graph_inputs(panel, cfg)
    np.savez_compressed(output / 'causal_graphs.npz', static=static, dynamic=dynamic,
                        timestamps=np.asarray(panel.timestamps, dtype=str), stations=np.asarray(panel.node_names))
    diagnostics = training_diagnostics(panel, cfg, analysis_cfg, labels, static, dynamic)
    write_json(output / 'training_diagnostics.json', diagnostics)
    print('analysis: baselines, paired time-block intervals and figures', flush=True)
    baseline_rows, paired, predictions = baselines(labels, analysis_cfg)
    csv(baseline_rows, output / 'baseline_metrics.csv')
    csv(predictions, output / 'baseline_predictions.csv')
    write_json(output / 'baseline_paired_comparisons.json', paired)
    plots(panel, cfg, monthly, static, output)
    blockers = []
    for row in readiness.to_dict('records'):
        if not row['classification_ready']:
            blockers.append(f"h={row['horizon_hours']}，{row['split']}：正類 {row['positives']}、負類 {row['negatives']}，不足以可靠比較事件辨識與漏報。")
    covariates = [r for r in inventory if r['source'] in ['reservoir', 'river_level', 'catchment_rain']]
    for row in covariates:
        if not row.get('observations_in_research_period', 0):
            blockers.append(f"{row['title']}：當次官方快照與研究期間無重疊，尚不能當成同期 predictors。")
    if not cfg.get('include_reservoir_covariates'):
        blockers.append('目前模型輸入只有測站濁度；同期上游雨量／流量／水位尚未納入。雨量站基本資料是 metadata，不是雨量時間序列。')
    blockers += ['官方觀測時間未證明資料在當時即可公開取得；發布延遲尚未驗證，不能宣稱部署的實際提前量。',
                 '物理流向、測站座標與感測深度尚未完整驗證；GP17 下游資料不應誤寫成上游因果訊號。',
                 '3 個測站位於 synthetic N=100–1000 的支援範圍外；目前未完成五模型外部驗證或 preregistered model expectation。']
    train_end, valid_end = boundaries(len(panel.features), cfg)
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(), raw_turbidity_rows=basic_audit['total_rows'],
                  hourly_steps=len(panel.features), target_station=cfg['target_station'], threshold_ntu=cfg['threshold'],
                  period=dict(start=cfg['start'], end_exclusive=cfg['end']),
                  splits=dict(train_end_exclusive=dates[train_end].isoformat(),
                              validation_end_exclusive=dates[valid_end].isoformat()),
                  regression_ready=bool(readiness.regression_ready.all()),
                  classification_ready=bool(readiness.classification_ready.all()), blockers=blockers,
                  input_stations=panel.node_names, station_metadata=names, diagnostics=diagnostics,
                  official_source_inventory=inventory,
                  future_label_definition='point NTU at t+h; any hourly peak > threshold in (t,t+h]',
                  formal_external_validation_completed=False)
    write_json(output / 'analysis_summary.json', result)
    write_json(output / 'analysis_config.json', config)
    report(result, inventory, readiness, diagnostics, event_rows, baseline_rows, output)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
             if p.is_file() and p.name != marker.name}
    write_json(marker, {'completed': True, 'generated_at': result['generated_at'], 'config_sha256': digest(config),
                        'config': config, 'source_code_sha256': source_digest(),
                        'raw_source_fingerprints': {r['source']: r['content_sha256'] for r in inventory},
                        'outputs_sha256': files})
    return result
