"""Controlled graph/time-series data generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .graphs import (
    corrupt_adjacency,
    drift_graph,
    neighbor_weights,
    ring_lattice,
    update_edge_scores,
)

FloatArray = NDArray[np.float64]
SCHEMA_VERSION = 2


@dataclass(slots=True)
class SimulationResult:
    config: SimulationConfig
    node_features: FloatArray  # [time, node, feature]
    response: FloatArray  # [time, node]
    true_adjacency: FloatArray  # [time, node, node]
    observed_adjacency: FloatArray  # [time, node, node]
    true_neighbor_weights: FloatArray  # [time, node, node]
    edge_scores: FloatArray  # [time, node, node], nan for non-edges
    beta: FloatArray  # [feature]
    projection: FloatArray  # [feature]

    @property
    def realized_edge_change(self) -> FloatArray:
        changes = np.zeros(self.config.n_steps, dtype=np.float64)
        edge_count = self.true_adjacency[0].sum() / 2
        if edge_count == 0:
            return changes
        for t in range(1, self.config.n_steps):
            changed = np.abs(self.true_adjacency[t] - self.true_adjacency[t - 1]).sum() / 2
            changes[t] = changed / (2 * edge_count)
        return changes

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {"schema_version": SCHEMA_VERSION, "config": self.config.to_dict()},
            sort_keys=True,
        )
        np.savez_compressed(
            destination,
            metadata=np.array(metadata),
            node_features=self.node_features,
            response=self.response,
            true_adjacency=self.true_adjacency,
            observed_adjacency=self.observed_adjacency,
            true_neighbor_weights=self.true_neighbor_weights,
            edge_scores=self.edge_scores,
            beta=self.beta,
            projection=self.projection,
        )
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> "SimulationResult":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported simulation schema version")
            return cls(
                config=SimulationConfig(**metadata["config"]),
                node_features=archive["node_features"].copy(),
                response=archive["response"].copy(),
                true_adjacency=archive["true_adjacency"].copy(),
                observed_adjacency=archive["observed_adjacency"].copy(),
                true_neighbor_weights=archive["true_neighbor_weights"].copy(),
                edge_scores=archive["edge_scores"].copy(),
                beta=archive["beta"].copy(),
                projection=archive["projection"].copy(),
            )


def simulate(config: SimulationConfig) -> SimulationResult:
    """Generate one fully reproducible controlled scenario."""
    # Independent streams keep exogenous data fixed when graph parameters change.
    streams = np.random.SeedSequence(config.seed).spawn(5)
    rng, score_rng, observation_rng, feature_rng, response_rng = [
        np.random.default_rng(s) for s in streams
    ]
    total_steps = config.n_steps + config.burn_in
    shape_graph = (total_steps, config.n_nodes, config.n_nodes)
    true_adjacency = np.zeros(shape_graph, dtype=np.float64)
    observed_adjacency = np.zeros(shape_graph, dtype=np.float64)
    true_neighbor_weights = np.zeros(shape_graph, dtype=np.float64)
    edge_scores = np.full(shape_graph, np.nan, dtype=np.float64)

    true_adjacency[0] = ring_lattice(config.n_nodes, config.degree, rng)
    edge_scores[0] = update_edge_scores(true_adjacency[0], None, None, score_rng)
    true_neighbor_weights[0] = neighbor_weights(
        true_adjacency[0], edge_scores[0], config.neighbor_heterogeneity
    )
    observed_adjacency[0] = corrupt_adjacency(
        true_adjacency[0], config.edge_noise, observation_rng
    )
    for t in range(1, total_steps):
        true_adjacency[t] = drift_graph(
            true_adjacency[t - 1], config.graph_drift, rng
        )
        edge_scores[t] = update_edge_scores(
            true_adjacency[t], true_adjacency[t - 1], edge_scores[t - 1], score_rng
        )
        true_neighbor_weights[t] = neighbor_weights(
            true_adjacency[t], edge_scores[t], config.neighbor_heterogeneity
        )
        observed_adjacency[t] = corrupt_adjacency(
            true_adjacency[t], config.edge_noise, observation_rng
        )

    beta = np.linspace(0.5, 1.5, config.feature_dim, dtype=np.float64)
    beta /= np.linalg.norm(beta)
    projection = np.linspace(1.5, 0.5, config.feature_dim, dtype=np.float64)
    projection /= np.linalg.norm(projection)

    node_features = np.zeros(
        (total_steps, config.n_nodes, config.feature_dim), dtype=np.float64
    )
    response = np.zeros((total_steps, config.n_nodes), dtype=np.float64)
    node_features[0] = feature_rng.normal(size=(config.n_nodes, config.feature_dim))
    response[0] = response_rng.normal(scale=config.response_noise, size=config.n_nodes)
    for t in range(total_steps - 1):
        innovations = feature_rng.normal(size=(config.n_nodes, config.feature_dim))
        node_features[t + 1] = (
            config.feature_ar * node_features[t]
            + config.feature_noise * innovations
        )
        local_signal = node_features[t] @ beta
        scalar_neighbor_signal = np.tanh(node_features[t] @ projection)
        neighbor_signal = true_neighbor_weights[t] @ scalar_neighbor_signal
        response[t + 1] = (
            config.response_ar * response[t]
            + local_signal
            + config.graph_signal * neighbor_signal
            + config.response_noise * response_rng.normal(size=config.n_nodes)
        )

    return SimulationResult(
        config=config,
        node_features=node_features[config.burn_in:],
        response=response[config.burn_in:],
        true_adjacency=true_adjacency[config.burn_in:],
        observed_adjacency=observed_adjacency[config.burn_in:],
        true_neighbor_weights=true_neighbor_weights[config.burn_in:],
        edge_scores=edge_scores[config.burn_in:],
        beta=beta,
        projection=projection,
    )
