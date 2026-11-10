"""Leakage-aware forecast-window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .simulation import SimulationResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class ForecastDataset:
    horizon: int
    threshold: float
    issue_times: IntArray
    history: FloatArray  # [sample, history_time, node, feature + response]
    current_observed_adjacency: FloatArray  # [sample, node, node]
    regression_target: FloatArray  # [sample, node], exactly t + h
    event_target: NDArray[np.bool_]  # [sample, node], any exceedance in (t, t+h]


def build_forecast_dataset(
    simulation: SimulationResult,
    horizon: int,
    history_length: int,
    threshold: float,
) -> ForecastDataset:
    """Create targets without using information after each issue time."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    n_steps = simulation.config.n_steps
    first_issue = history_length - 1
    last_issue = n_steps - horizon - 1
    if first_issue > last_issue:
        raise ValueError("history_length and horizon leave no forecast samples")
    issue_times = np.arange(first_issue, last_issue + 1, dtype=np.int64)
    combined = np.concatenate(
        (simulation.node_features, simulation.response[..., None]), axis=-1
    )
    history = np.stack(
        [combined[t - history_length + 1 : t + 1] for t in issue_times]
    )
    current_graph = simulation.observed_adjacency[issue_times].copy()
    regression = np.stack(
        [simulation.response[t + horizon] for t in issue_times]
    )
    events = np.stack(
        [
            np.any(simulation.response[t + 1 : t + horizon + 1] > threshold, axis=0)
            for t in issue_times
        ]
    )
    return ForecastDataset(
        horizon=horizon,
        threshold=float(threshold),
        issue_times=issue_times,
        history=history,
        current_observed_adjacency=current_graph,
        regression_target=regression,
        event_target=events,
    )

