"""Observable selection features computed entirely before validation/test."""
import numpy as np
from .estimators import estimate_static_graph, bootstrap_dynamic_adjacencies
from .indicators import graph_utility_cv, structural_drift_to_noise_ratio, neighbor_relevance_concentration, graph_dirichlet_energy

FEATURE_NAMES = ['graph_utility', 'sdnr', 'nrc', 'dirichlet_energy', 'density']


def describe(panel, graphs, cfg):
    end = int(len(panel.features) * cfg.get('train_fraction', .6))
    window = cfg['graph_window']
    series = graphs[2][:end]
    n = series.shape[1]
    k = min(cfg.get('top_k', 4), n - 1)
    times = np.unique(np.linspace(window - 1, end - 1, cfg.get('descriptor_windows', 5), dtype=int))
    if len(times) < 2:
        raise ValueError('Insufficient training history for SDNR')
    args = dict(top_k=k, max_lag=cfg.get('max_lag', 3), symmetrize='union', minimum_relevance=cfg.get('minimum_relevance', .05))
    bootstrap = bootstrap_dynamic_adjacencies(series, times, window_length=window,
                  n_bootstrap=cfg.get('descriptor_bootstraps', 8), block_length=cfg.get('descriptor_block', 12), seed=47, **args)
    estimate = estimate_static_graph(series, **args)
    sdnr = structural_drift_to_noise_ratio(bootstrap)
    nrc = neighbor_relevance_concentration(estimate.relevance, estimate.adjacency)
    # Use causal graph snapshots, NOT an estimate fitted to the full training
    # period, in the expanding validation used to measure graph utility.
    issue = np.arange(window - 1, end - 1)
    a = graphs[1][issue]
    weights = a / np.maximum(a.sum(-1, keepdims=True), 1)
    neighbors = np.einsum('tij,tj->ti', weights, series[issue])
    utility = []
    for node in panel.target_nodes:
        utility.append(graph_utility_cv(series[issue, node], neighbors[:, node], series[issue + 1, node],
                                        n_splits=3, purge_gap=1, ridge=.01))
    return {'graph_utility': float(np.mean(utility)), 'sdnr': sdnr.value,
            'nrc': nrc.value, 'nrc_coverage': nrc.coverage,
            'dirichlet_energy': float(np.nanmean(graph_dirichlet_energy(series, estimate.adjacency, center=True))),
            'density': float(estimate.adjacency.sum() / (n * (n - 1))),
            'training_end': end, 'sdnr_between': sdnr.between_time_change, 'sdnr_within': sdnr.within_time_uncertainty,
            'definition': 'Observable training-only statistics; estimated-series SDNR/NRC; never latent weights.'}
