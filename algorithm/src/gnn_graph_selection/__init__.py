"""Controlled simulation primitives for graph-model selection research."""

from .config import SimulationConfig
from .dataset import ForecastDataset, build_forecast_dataset
from .indicators import (
    NRCResult,
    SDNRResult,
    graph_dirichlet_energy,
    graph_utility_cv,
    neighbor_relevance_concentration,
    structural_drift_to_noise_ratio,
)
from .estimators import (
    GraphEstimate,
    bootstrap_dynamic_adjacencies,
    estimate_dynamic_graphs,
    estimate_static_graph,
    lagged_partial_relevance,
    top_k_adjacency,
)
from .simulation import SimulationResult, simulate

__all__ = [
    "ForecastDataset",
    "GraphEstimate",
    "NRCResult",
    "SDNRResult",
    "SimulationConfig",
    "SimulationResult",
    "build_forecast_dataset",
    "bootstrap_dynamic_adjacencies",
    "estimate_dynamic_graphs",
    "estimate_static_graph",
    "graph_dirichlet_energy",
    "graph_utility_cv",
    "lagged_partial_relevance",
    "neighbor_relevance_concentration",
    "simulate",
    "structural_drift_to_noise_ratio",
    "top_k_adjacency",
]
