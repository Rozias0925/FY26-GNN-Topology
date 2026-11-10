"""Run one small simulation and print invariant/indicator summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gnn_graph_selection import (  # noqa: E402
    SimulationConfig,
    build_forecast_dataset,
    graph_dirichlet_energy,
    neighbor_relevance_concentration,
    simulate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional .npz output path")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = SimulationConfig(seed=args.seed)
    result = simulate(config)
    training_end = int(config.n_steps * 0.6)
    threshold = float(np.quantile(result.response[:training_end], 0.9))
    datasets = {
        horizon: build_forecast_dataset(result, horizon, 12, threshold)
        for horizon in (1, 3, 6)
    }
    energy = graph_dirichlet_energy(
        result.response, result.observed_adjacency, center=True
    )
    nrc = neighbor_relevance_concentration(
        result.true_neighbor_weights[-1], result.true_adjacency[-1]
    )
    initial_degree = result.true_adjacency[0].sum(axis=1)
    final_degree = result.true_adjacency[-1].sum(axis=1)

    print("Controlled simulation smoke run")
    print(f"  seed: {config.seed}")
    print(f"  arrays: X={result.node_features.shape}, r={result.response.shape}")
    print(f"  degree preserved: {bool(np.array_equal(initial_degree, final_degree))}")
    print(f"  mean realized edge change: {result.realized_edge_change[1:].mean():.4f}")
    print(f"  training-only event threshold (90th percentile): {threshold:.4f}")
    for horizon, dataset in datasets.items():
        print(
            f"  h={horizon}: samples={len(dataset.issue_times)}, "
            f"event prevalence={dataset.event_target.mean():.4f}"
        )
    print(f"  mean centered Dirichlet energy: {np.nanmean(energy):.4f}")
    print(f"  latent-weight NRC: {nrc.value:.4f} (coverage={nrc.coverage:.2%})")
    if args.output:
        destination = result.save_npz(args.output)
        print(f"  saved: {destination}")


if __name__ == "__main__":
    main()

