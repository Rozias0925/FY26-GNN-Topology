"""One forecast protocol shared by synthetic and measured hourly panels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .estimators import estimate_static_graph, estimate_dynamic_graphs


@dataclass
class Panel:
    features: np.ndarray  # [T,N,F]; response history must be included by caller
    response: np.ndarray  # [T,N]; NEVER imputed for labels
    graph_series: np.ndarray  # [T,N], one physically documented variable per node
    node_names: list[str]
    target_nodes: list[int]
    timestamps: np.ndarray
    metadata: dict
    observed_graphs: np.ndarray | None = None
    event_response: np.ndarray | None = None  # hourly peak when measured subhourly


def time_splits(issue, horizon, train_end, validation_end):
    """Exclusive boundaries: all targets of a window belong to the same split."""
    issue = np.asarray(issue)
    if not 0 < train_end < validation_end or horizon < 1:
        raise ValueError('Invalid temporal split')
    return {
        'train': np.flatnonzero(issue + horizon < train_end),
        'validation': np.flatnonzero((issue >= train_end) & (issue + horizon < validation_end)),
        'test': np.flatnonzero(issue >= validation_end),
    }


def causal_fill(values, train_end, limit=3):
    """Past-only forward fill, then training medians; retain original missing mask."""
    values = np.asarray(values, float)
    shape = values.shape
    frame = pd.DataFrame(np.where(np.isfinite(values), values, np.nan).reshape(shape[0], -1))
    original_mask = np.isfinite(values).astype(np.float32)
    filled = frame.ffill(limit=limit) if limit else frame.copy()
    medians = filled.iloc[:train_end].median().fillna(0)
    filled = filled.fillna(medians).to_numpy().reshape(shape)
    mean = filled[:train_end].mean(axis=0)
    scale = filled[:train_end].std(axis=0)
    scale[scale < 1e-6] = 1
    return ((filled - mean) / scale).astype(np.float32), original_mask, {'mean': mean, 'scale': scale, 'median': medians.to_numpy().reshape(shape[1:])}


def graph_inputs(panel, cfg):
    """Static graph fitted to calibration prefix; dynamic graph uses data <= t.

    Calibration prefix is excluded from learning targets and is earlier than
    all forecast origins. Graph estimation therefore never uses future targets.
    """
    n = panel.features.shape[1]
    window = cfg['graph_window']
    k = min(cfg.get('top_k', 4), n - 1)
    if n < 2:
        raise ValueError('At least two aligned stations/nodes are needed')
    series, _, _ = causal_fill(panel.graph_series, window, cfg.get('ffill_limit', 3))
    mode = cfg.get('graph_mode', 'estimated')
    if mode == 'observed':
        if panel.observed_graphs is None:
            raise ValueError('Observed graph mode is only available for simulations')
        static = (panel.observed_graphs[:window].mean(axis=0) >= .5).astype(float)
        dynamic = panel.observed_graphs.copy()
    elif mode == 'estimated':
        args = dict(top_k=k, max_lag=cfg.get('max_lag', 3), symmetrize='union', minimum_relevance=cfg.get('minimum_relevance', .05))
        static = estimate_static_graph(series[:window], **args).adjacency
        times = np.arange(window - 1, len(series), cfg.get('graph_stride', 6))
        estimates = estimate_dynamic_graphs(series, times, window_length=window, **args)
        dynamic = np.zeros((len(series), n, n), dtype=np.float32)
        for i, (t, estimate) in enumerate(zip(times, estimates)):
            end = times[i + 1] if i + 1 < len(times) else len(series)
            dynamic[t:end] = estimate.adjacency
    else:
        raise ValueError('graph_mode must be observed or estimated')
    return static.astype(np.float32), dynamic.astype(np.float32), series


def prepare(panel, cfg, horizon, graphs):
    total, n, _ = panel.features.shape
    train_end = int(total * cfg.get('train_fraction', .6))
    valid_end = int(total * (cfg.get('train_fraction', .6) + cfg.get('validation_fraction', .2)))
    length = cfg['history_length']
    if not 0 < train_end < valid_end < total or cfg['graph_window'] >= train_end or length < 1 or horizon < 1:
        raise ValueError('Invalid split, calibration window, history length or horizon')
    first = max(length - 1, cfg['graph_window'] - 1)
    target_nodes = panel.target_nodes
    y = panel.response[:, target_nodes].astype(float)
    event_series = panel.event_response if panel.event_response is not None else panel.response
    ye = event_series[:, target_nodes].astype(float)
    if cfg.get('threshold') is None:
        tau = float(np.nanquantile(ye[:train_end], cfg.get('threshold_quantile', .9)))
        threshold_kind = 'training_quantile'
    else:
        tau = float(cfg['threshold'])
        threshold_kind = 'predefined_absolute'
    if not np.isfinite(tau):
        raise ValueError('No observed training targets for threshold')
    x, mask, scaler = causal_fill(panel.features, train_end, cfg.get('ffill_limit', 3))
    x = np.concatenate([x, mask], axis=-1)
    times = np.arange(first, total - horizon)
    if not len(times):
        raise ValueError('Insufficient history after calibration and horizon')
    future = np.stack([y[t + 1:t + horizon + 1] for t in times])
    # Missing future values cannot be silently labeled negative.
    event_future = np.stack([ye[t + 1:t + horizon + 1] for t in times])
    good = np.isfinite(future).all(axis=(1, 2)) & np.isfinite(event_future).all(axis=(1, 2))
    times, future, event_future = times[good], future[good], event_future[good]
    regression = future[:, -1]
    event = (event_future > tau).any(axis=1).astype(np.float32)
    splits = time_splits(times, horizon, train_end, valid_end)
    minimum = cfg.get('min_split_samples', 12)
    for name, ix in splits.items():
        if len(ix) < minimum:
            raise ValueError(f'{name}: only {len(ix)} complete forecast windows; need {minimum}')
        if name != 'test' and len(np.unique(event[ix])) < 2:
            raise ValueError(f'{name}: only one event class; cannot support five-model comparison')
    history = np.stack([x[t - length + 1:t + 1] for t in times])
    transformed = np.log1p(regression) if cfg.get('log_response', False) else regression
    mean = transformed[splits['train']].mean(axis=0)
    scale = transformed[splits['train']].std(axis=0)
    scale[scale < 1e-6] = 1
    return dict(x=history, event=event, regression=regression, scaled_regression=((transformed - mean) / scale).astype(np.float32),
                times=times, splits=splits, static=graphs[0], dynamic=graphs[1][times],
                target_nodes=target_nodes, threshold=tau, response_mean=mean, response_scale=scale,
                log_response=cfg.get('log_response', False), horizon=horizon,
                metadata={'train_end': train_end, 'validation_end': valid_end, 'threshold': tau,
                          'threshold_kind': threshold_kind, 'label': 'any strictly greater in (t,t+h]',
                          'removed_missing_label_windows': int((~good).sum()), 'input_scaler': scaler,
                          'split_counts': {k:len(v) for k,v in splits.items()},
                          'split_prevalence': {k:float(event[v].mean()) for k,v in splits.items()},
                          'single_class_test': len(np.unique(event[splits['test']])) < 2})


def inverse_response(scaled, data):
    values = scaled * data['response_scale'] + data['response_mean']
    return np.maximum(0., np.expm1(np.clip(values, -20, 30))) if data['log_response'] else values
