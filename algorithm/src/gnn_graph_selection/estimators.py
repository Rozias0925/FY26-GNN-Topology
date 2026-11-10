"""Future-safe static and trailing-window graph estimators.

The first estimator uses lagged partial correlation as a transparent baseline.
It is not assumed to recover causal hydrological structure. More specialized
estimators can implement the same ``GraphEstimate`` interface later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GraphEstimate:
    adjacency: FloatArray
    relevance: FloatArray
    n_observations: int
    start_time: int
    end_time: int


def _ridge_residual(target: FloatArray, controls: FloatArray, ridge: float) -> FloatArray:
    if controls.size == 0:
        return target - target.mean(axis=0)
    design = np.column_stack((np.ones(controls.shape[0]), controls))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return target - design @ coefficients


def lagged_partial_relevance(
    series: ArrayLike,
    *,
    max_lag: int = 3,
    ridge: float = 1e-6,
    eps: float = 1e-12,
) -> FloatArray:
    """Estimate source-to-target relevance after controlling target history.

    ``relevance[target, source]`` is the signed partial correlation with the
    largest absolute magnitude across source lags 1..``max_lag``. The target's
    own lags 1..``max_lag`` are used as controls.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, node]")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    if values.shape[0] <= 2 * max_lag + 3:
        raise ValueError("series is too short for the requested max_lag")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains missing or non-finite values")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    n_times, n_nodes = values.shape
    relevance = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    sources = np.stack([values[max_lag-lag:n_times-lag] for lag in range(1,max_lag+1)],axis=-1).reshape(n_times-max_lag,-1)
    for target_node in range(n_nodes):
        target = values[max_lag:, target_node]
        own_history = np.column_stack(
            [values[max_lag - lag : n_times - lag, target_node] for lag in range(1, max_lag + 1)]
        )
        target_residual = _ridge_residual(target, own_history, ridge)
        target_norm = np.linalg.norm(target_residual)
        if target_norm <= eps:
            continue
        # The same control design applies to every source/lag. Solve all RHS
        # together, preserving the scalar estimator while avoiding O(N*lags)
        # repeated matrix factorizations per target.
        residuals = _ridge_residual(sources, own_history, ridge)
        denominator = target_norm * np.linalg.norm(residuals, axis=0)
        correlations = np.divide(target_residual @ residuals, denominator,
                                 out=np.zeros_like(denominator), where=denominator > eps).reshape(n_nodes,max_lag)
        best = np.abs(correlations).argmax(axis=1)
        relevance[target_node] = correlations[np.arange(n_nodes),best]
        relevance[target_node,target_node] = 0.0
    return relevance


def top_k_adjacency(
    relevance: ArrayLike,
    top_k: int,
    *,
    minimum_relevance: float = 0.0,
    symmetrize: str = "none",
) -> FloatArray:
    """Build ``adjacency[target, source]`` from absolute relevance rankings.

    ``symmetrize`` can be ``none`` (directed), ``union``, or ``mutual``.
    """
    scores = np.asarray(relevance, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("relevance must be a square matrix")
    n_nodes = scores.shape[0]
    if top_k < 1 or top_k >= n_nodes:
        raise ValueError("top_k must satisfy 1 <= top_k < n_nodes")
    if minimum_relevance < 0:
        raise ValueError("minimum_relevance must be nonnegative")
    if symmetrize not in {"none", "union", "mutual"}:
        raise ValueError("symmetrize must be one of: none, union, mutual")
    adjacency = np.zeros_like(scores, dtype=np.float64)
    for target in range(n_nodes):
        magnitudes = np.abs(scores[target]).copy()
        magnitudes[target] = -np.inf
        order = np.argsort(magnitudes)[::-1]
        selected = [source for source in order[:top_k] if magnitudes[source] >= minimum_relevance]
        adjacency[target, selected] = 1.0
    if symmetrize == "union":
        adjacency = np.maximum(adjacency, adjacency.T)
    elif symmetrize == "mutual":
        adjacency = np.minimum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def estimate_static_graph(
    series: ArrayLike,
    *,
    top_k: int,
    max_lag: int = 3,
    minimum_relevance: float = 0.0,
    symmetrize: str = "none",
    ridge: float = 1e-6,
    start_time: int = 0,
) -> GraphEstimate:
    """Estimate one graph from a fixed training interval."""
    values = np.asarray(series, dtype=np.float64)
    relevance = lagged_partial_relevance(values, max_lag=max_lag, ridge=ridge)
    adjacency = top_k_adjacency(
        relevance,
        top_k,
        minimum_relevance=minimum_relevance,
        symmetrize=symmetrize,
    )
    return GraphEstimate(
        adjacency=adjacency,
        relevance=relevance,
        n_observations=values.shape[0],
        start_time=start_time,
        end_time=start_time + values.shape[0] - 1,
    )


def estimate_dynamic_graphs(
    series: ArrayLike,
    issue_times: ArrayLike,
    *,
    window_length: int,
    top_k: int,
    max_lag: int = 3,
    minimum_relevance: float = 0.0,
    symmetrize: str = "none",
    ridge: float = 1e-6,
) -> list[GraphEstimate]:
    """Estimate one graph per issue time using only a trailing window."""
    values = np.asarray(series, dtype=np.float64)
    times = np.asarray(issue_times, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, node]")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("issue_times must be a non-empty vector")
    minimum_window = 2 * max_lag + 4
    if window_length < minimum_window:
        raise ValueError(f"window_length must be at least {minimum_window}")
    estimates: list[GraphEstimate] = []
    for issue_time in times:
        issue_time = int(issue_time)
        if issue_time < window_length - 1 or issue_time >= values.shape[0]:
            raise ValueError("every issue time must have a complete trailing window")
        start = issue_time - window_length + 1
        estimate = estimate_static_graph(
            values[start : issue_time + 1],
            top_k=top_k,
            max_lag=max_lag,
            minimum_relevance=minimum_relevance,
            symmetrize=symmetrize,
            ridge=ridge,
            start_time=start,
        )
        estimates.append(estimate)
    return estimates


def moving_block_bootstrap(
    series: ArrayLike,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Resample circular consecutive blocks while preserving local ordering."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, node]")
    if block_length < 1 or block_length > values.shape[0]:
        raise ValueError("block_length must lie between 1 and the series length")
    starts = rng.integers(0, values.shape[0], size=int(np.ceil(values.shape[0] / block_length)))
    indices = np.concatenate(
        [(start + np.arange(block_length)) % values.shape[0] for start in starts]
    )[: values.shape[0]]
    return values[indices]


def bootstrap_dynamic_adjacencies(
    series: ArrayLike,
    issue_times: ArrayLike,
    *,
    window_length: int,
    top_k: int,
    n_bootstrap: int,
    block_length: int,
    max_lag: int = 3,
    minimum_relevance: float = 0.0,
    symmetrize: str = "none",
    ridge: float = 1e-6,
    seed: int = 0,
) -> FloatArray:
    """Return graphs shaped [issue_time, bootstrap, node, node] for SDNR."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    values = np.asarray(series, dtype=np.float64)
    times = np.asarray(issue_times, dtype=np.int64)
    rng = np.random.default_rng(seed)
    output = np.zeros(
        (times.size, n_bootstrap, values.shape[1], values.shape[1]), dtype=np.float64
    )
    for time_index, issue_time in enumerate(times):
        issue_time = int(issue_time)
        start = issue_time - window_length + 1
        if start < 0 or issue_time >= values.shape[0]:
            raise ValueError("every issue time must have a complete trailing window")
        window = values[start : issue_time + 1]
        for bootstrap in range(n_bootstrap):
            sampled = moving_block_bootstrap(window, block_length=block_length, rng=rng)
            estimate = estimate_static_graph(
                sampled,
                top_k=top_k,
                max_lag=max_lag,
                minimum_relevance=minimum_relevance,
                symmetrize=symmetrize,
                ridge=ridge,
            )
            output[time_index, bootstrap] = estimate.adjacency
    return output
