"""Candidate observable indicators from the research specification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_batches(signal: ArrayLike, adjacency: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(signal, dtype=np.float64)
    a = np.asarray(adjacency, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    if a.ndim == 2:
        a = np.broadcast_to(a, (y.shape[0], *a.shape))
    if y.ndim != 2 or a.ndim != 3 or a.shape[0] != y.shape[0]:
        raise ValueError("expected signal [batch, node] and adjacency [batch, node, node]")
    if a.shape[1:] != (y.shape[1], y.shape[1]):
        raise ValueError("signal and adjacency node dimensions disagree")
    return y, a


def graph_dirichlet_energy(
    signal: ArrayLike,
    adjacency: ArrayLike,
    *,
    center: bool = False,
    eps: float = 1e-12,
) -> FloatArray | float:
    """Compute normalized Dirichlet energy on symmetrized nonnegative graphs.

    Returns ``nan`` for samples with zero signal norm after optional centering.
    """
    original = np.asarray(signal)
    y, a = _as_batches(signal, adjacency)
    if np.any(a < 0):
        raise ValueError("adjacency must be nonnegative")
    if center:
        y = y - y.mean(axis=1, keepdims=True)
    symmetric = 0.5 * (a + np.swapaxes(a, -1, -2))
    np.einsum("bii->bi", symmetric)[:] = 0.0
    differences = y[:, :, None] - y[:, None, :]
    numerator = 0.5 * np.sum(symmetric * differences**2, axis=(1, 2))
    denominator = np.sum(y**2, axis=1)
    energy = np.full_like(denominator, np.nan)
    valid = denominator > eps
    energy[valid] = numerator[valid] / denominator[valid]
    return float(energy[0]) if original.ndim == 1 else energy


def _ridge_fit_predict(
    x_train: FloatArray,
    y_train: FloatArray,
    x_valid: FloatArray,
    ridge: float,
) -> FloatArray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    valid = (x_valid - mean) / scale
    train = np.column_stack((np.ones(train.shape[0]), train))
    valid = np.column_stack((np.ones(valid.shape[0]), valid))
    penalty = np.eye(train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    return valid @ coefficients


def graph_utility_cv(
    self_features: ArrayLike,
    neighbor_features: ArrayLike,
    target: ArrayLike,
    *,
    n_splits: int = 4,
    min_train_fraction: float = 0.5,
    purge_gap: int = 0,
    ridge: float = 1e-6,
    eps: float = 1e-12,
) -> float:
    """Blocked out-of-sample relative MSE improvement from neighbor summaries."""
    x_self = np.asarray(self_features, dtype=np.float64)
    x_neighbor = np.asarray(neighbor_features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x_self.ndim == 1:
        x_self = x_self[:, None]
    if x_neighbor.ndim == 1:
        x_neighbor = x_neighbor[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if not (x_self.shape[0] == x_neighbor.shape[0] == y.shape[0]):
        raise ValueError("all inputs must have the same number of ordered samples")
    if n_splits < 1 or not 0 < min_train_fraction < 1 or ridge < 0 or purge_gap < 0:
        raise ValueError("invalid cross-validation settings")
    n_samples = y.shape[0]
    first_valid = max(2, int(np.ceil(n_samples * min_train_fraction)))
    remaining = n_samples - first_valid
    if remaining < n_splits:
        raise ValueError("not enough samples for requested blocked splits")
    validation_blocks = np.array_split(np.arange(first_valid, n_samples), n_splits)
    self_errors: list[FloatArray] = []
    augmented_errors: list[FloatArray] = []
    augmented = np.column_stack((x_self, x_neighbor))
    for valid_indices in validation_blocks:
        train_end = int(valid_indices[0]) - purge_gap
        if train_end < 2:
            raise ValueError("purge_gap leaves too few training samples")
        prediction_self = _ridge_fit_predict(
            x_self[:train_end], y[:train_end], x_self[valid_indices], ridge
        )
        prediction_augmented = _ridge_fit_predict(
            augmented[:train_end], y[:train_end], augmented[valid_indices], ridge
        )
        self_errors.append((y[valid_indices] - prediction_self) ** 2)
        augmented_errors.append((y[valid_indices] - prediction_augmented) ** 2)
    loss_self = float(np.mean(np.concatenate(self_errors, axis=0)))
    loss_augmented = float(np.mean(np.concatenate(augmented_errors, axis=0)))
    return (loss_self - loss_augmented) / max(loss_self, eps)


@dataclass(frozen=True, slots=True)
class SDNRResult:
    value: float
    between_time_change: float
    within_time_uncertainty: float


def structural_drift_to_noise_ratio(
    bootstrap_adjacencies: ArrayLike, *, eps: float = 1e-12
) -> SDNRResult:
    """Compute SDNR from arrays shaped [time_window, bootstrap, node, node]."""
    estimates = np.asarray(bootstrap_adjacencies, dtype=np.float64)
    if estimates.ndim != 4 or estimates.shape[0] < 2 or estimates.shape[1] < 2:
        raise ValueError("expected at least [2 time windows, 2 bootstraps, node, node]")
    if estimates.shape[2] != estimates.shape[3]:
        raise ValueError("adjacency matrices must be square")
    means = estimates.mean(axis=1)
    between = float(np.mean(np.sum(np.diff(means, axis=0) ** 2, axis=(1, 2))))
    deviations = estimates - means[:, None, :, :]
    within = float(np.mean(np.sum(deviations**2, axis=(2, 3))))
    return SDNRResult(between / (within + eps), between, within)


@dataclass(frozen=True, slots=True)
class NRCResult:
    value: float
    node_values: FloatArray
    valid_nodes: NDArray[np.bool_]
    coverage: float


def neighbor_relevance_concentration(
    relevance: ArrayLike,
    adjacency: ArrayLike | None = None,
    *,
    eps: float = 1e-12,
) -> NRCResult:
    """Compute entropy-normalized concentration for nodes with >=2 neighbors."""
    scores = np.asarray(relevance, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("relevance must be a square matrix")
    if adjacency is None:
        mask = np.isfinite(scores) & (scores != 0)
    else:
        graph = np.asarray(adjacency, dtype=np.float64)
        if graph.shape != scores.shape:
            raise ValueError("adjacency and relevance shapes differ")
        mask = graph > 0
    np.fill_diagonal(mask, False)
    node_values = np.full(scores.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(scores.shape[0], dtype=np.bool_)
    for node in range(scores.shape[0]):
        neighbors = np.flatnonzero(mask[node] & np.isfinite(scores[node]))
        if neighbors.size < 2:
            continue
        magnitudes = np.abs(scores[node, neighbors])
        total = magnitudes.sum()
        if total <= eps:
            continue
        probabilities = magnitudes / total
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        node_values[node] = 1.0 - entropy / np.log(neighbors.size)
        valid[node] = True
    value = float(np.nanmean(node_values)) if valid.any() else float("nan")
    return NRCResult(value, node_values, valid, float(valid.mean()))
