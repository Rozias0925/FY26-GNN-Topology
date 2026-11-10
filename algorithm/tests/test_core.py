from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_graph_selection import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    build_forecast_dataset,
    bootstrap_dynamic_adjacencies,
    estimate_dynamic_graphs,
    estimate_static_graph,
    graph_dirichlet_energy,
    graph_utility_cv,
    neighbor_relevance_concentration,
    simulate,
    structural_drift_to_noise_ratio,
)
from gnn_graph_selection.graphs import (  # noqa: E402
    drift_graph,
    is_connected,
    neighbor_weights,
    ring_lattice,
    update_edge_scores,
)


class GraphTests(unittest.TestCase):
    def test_regular_connected_graph_and_degree_preserving_drift(self) -> None:
        rng = np.random.default_rng(4)
        adjacency = ring_lattice(20, 4, rng)
        drifted = drift_graph(adjacency, 0.8, rng)
        np.testing.assert_array_equal(adjacency.sum(axis=1), np.full(20, 4.0))
        np.testing.assert_array_equal(drifted.sum(axis=1), np.full(20, 4.0))
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(drifted, drifted.T)
        self.assertTrue(is_connected(adjacency))
        self.assertTrue(is_connected(drifted))
        self.assertGreater(np.abs(adjacency - drifted).sum(), 0)

    def test_zero_drift_is_exactly_static(self) -> None:
        rng = np.random.default_rng(5)
        adjacency = ring_lattice(12, 4, rng)
        np.testing.assert_array_equal(drift_graph(adjacency, 0.0, rng), adjacency)

    def test_zero_kappa_gives_uniform_neighbor_weights(self) -> None:
        rng = np.random.default_rng(6)
        adjacency = ring_lattice(10, 4, rng)
        scores = update_edge_scores(adjacency, None, None, rng)
        weights = neighbor_weights(adjacency, scores, 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_allclose(weights[adjacency > 0], 0.25)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            n_nodes=10, feature_dim=2, n_steps=40, degree=4, seed=11
        )

    def test_reproducible_shapes_and_graph_invariants(self) -> None:
        first = simulate(self.config)
        second = simulate(self.config)
        np.testing.assert_array_equal(first.node_features, second.node_features)
        np.testing.assert_array_equal(first.response, second.response)
        np.testing.assert_array_equal(first.true_adjacency, second.true_adjacency)
        self.assertEqual(first.node_features.shape, (40, 10, 2))
        self.assertEqual(first.response.shape, (40, 10))
        np.testing.assert_array_equal(first.true_adjacency.sum(axis=2), 4.0)
        np.testing.assert_allclose(first.true_neighbor_weights.sum(axis=2), 1.0)

    def test_round_trip_archive(self) -> None:
        result = simulate(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            result.save_npz(path)
            loaded = SimulationResult.load_npz(path)
        self.assertEqual(loaded.config, result.config)
        np.testing.assert_array_equal(loaded.response, result.response)
        np.testing.assert_array_equal(loaded.edge_scores, result.edge_scores)

    def test_forecast_windows_do_not_include_future_inputs(self) -> None:
        result = simulate(self.config)
        threshold = float(np.quantile(result.response[:24], 0.9))
        dataset = build_forecast_dataset(result, horizon=3, history_length=5, threshold=threshold)
        self.assertEqual(dataset.history.shape, (33, 5, 10, 3))
        first_issue = int(dataset.issue_times[0])
        np.testing.assert_array_equal(
            dataset.history[0, :, :, -1], result.response[first_issue - 4 : first_issue + 1]
        )
        np.testing.assert_array_equal(
            dataset.regression_target[0], result.response[first_issue + 3]
        )
        expected = np.any(
            result.response[first_issue + 1 : first_issue + 4] > threshold, axis=0
        )
        np.testing.assert_array_equal(dataset.event_target[0], expected)


class IndicatorTests(unittest.TestCase):
    def test_dirichlet_energy_boundaries(self) -> None:
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(graph_dirichlet_energy([2.0, 2.0], adjacency), 0.0)
        self.assertAlmostEqual(graph_dirichlet_energy([1.0, -1.0], adjacency), 2.0)
        self.assertTrue(np.isnan(graph_dirichlet_energy([2.0, 2.0], adjacency, center=True)))

    def test_graph_utility_detects_neighbor_information(self) -> None:
        rng = np.random.default_rng(12)
        n = 240
        self_feature = rng.normal(size=(n, 1))
        neighbor_feature = rng.normal(size=(n, 1))
        target = 0.1 * self_feature[:, 0] + 2.0 * neighbor_feature[:, 0] + rng.normal(scale=0.1, size=n)
        utility = graph_utility_cv(self_feature, neighbor_feature, target)
        self.assertGreater(utility, 0.9)

    def test_sdnr_and_nrc(self) -> None:
        estimates = np.zeros((3, 4, 3, 3), dtype=float)
        estimates[1, :, 0, 1] = estimates[1, :, 1, 0] = 1.0
        estimates[2, :, 0, 2] = estimates[2, :, 2, 0] = 1.0
        sdnr = structural_drift_to_noise_ratio(estimates)
        self.assertGreater(sdnr.value, 1e10)

        adjacency = np.ones((3, 3)) - np.eye(3)
        uniform = neighbor_relevance_concentration(adjacency, adjacency)
        self.assertAlmostEqual(uniform.value, 0.0)
        concentrated_scores = np.array([[0, 100, 1], [100, 0, 1], [100, 1, 0]], dtype=float)
        concentrated = neighbor_relevance_concentration(concentrated_scores, adjacency)
        self.assertGreater(concentrated.value, 0.9)
        self.assertEqual(concentrated.coverage, 1.0)


class EstimatorTests(unittest.TestCase):
    def test_lagged_estimator_recovers_known_source(self) -> None:
        rng = np.random.default_rng(18)
        n_times = 400
        series = rng.normal(size=(n_times, 4))
        for t in range(1, n_times):
            series[t, 1] = 0.15 * series[t - 1, 1] + 2.5 * series[t - 1, 0] + rng.normal(scale=0.1)
        estimate = estimate_static_graph(series[:300], top_k=1, max_lag=2)
        self.assertEqual(estimate.adjacency[1, 0], 1.0)
        self.assertEqual(estimate.adjacency[1].sum(), 1.0)
        self.assertGreater(abs(estimate.relevance[1, 0]), 0.9)

    def test_dynamic_estimator_is_future_safe(self) -> None:
        rng = np.random.default_rng(19)
        series = rng.normal(size=(100, 5))
        altered = series.copy()
        altered[61:] = 1000 * rng.normal(size=altered[61:].shape)
        first = estimate_dynamic_graphs(
            series, [40, 60], window_length=30, top_k=2, max_lag=2
        )
        second = estimate_dynamic_graphs(
            altered, [40, 60], window_length=30, top_k=2, max_lag=2
        )
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a.adjacency, b.adjacency)
            np.testing.assert_array_equal(a.relevance, b.relevance)
            self.assertLessEqual(a.end_time, 60)

    def test_bootstrap_graphs_feed_sdnr(self) -> None:
        rng = np.random.default_rng(20)
        series = rng.normal(size=(100, 5))
        estimates = bootstrap_dynamic_adjacencies(
            series,
            [50, 70, 90],
            window_length=40,
            top_k=2,
            n_bootstrap=4,
            block_length=5,
            max_lag=2,
            seed=21,
        )
        self.assertEqual(estimates.shape, (3, 4, 5, 5))
        result = structural_drift_to_noise_ratio(estimates)
        self.assertTrue(np.isfinite(result.value))


if __name__ == "__main__":
    unittest.main()
