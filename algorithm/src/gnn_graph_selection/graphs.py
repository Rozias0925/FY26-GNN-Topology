"""Graph construction, drift, latent weights, and observation noise."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ring_lattice(n_nodes: int, degree: int, rng: np.random.Generator) -> FloatArray:
    """Return a randomly relabeled, connected, undirected regular graph."""
    if degree < 2 or degree >= n_nodes or degree % 2:
        raise ValueError("degree must be even and satisfy 2 <= degree < n_nodes")
    base = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    half = degree // 2
    for node in range(n_nodes):
        for offset in range(1, half + 1):
            neighbor = (node + offset) % n_nodes
            base[node, neighbor] = base[neighbor, node] = 1.0
    permutation = rng.permutation(n_nodes)
    return base[np.ix_(permutation, permutation)]


def is_connected(adjacency: FloatArray) -> bool:
    n_nodes = adjacency.shape[0]
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == n_nodes


def _edge_list(adjacency: FloatArray) -> list[tuple[int, int]]:
    rows, cols = np.where(np.triu(adjacency, k=1) > 0)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def drift_graph(
    adjacency: FloatArray,
    drift_rate: float,
    rng: np.random.Generator,
    max_attempt_factor: int = 50,
) -> FloatArray:
    """Apply connected, degree-preserving double-edge swaps.

    ``drift_rate`` controls the fraction of existing edges targeted for
    replacement. A successful swap replaces two edges. Failed proposals are
    skipped, which is possible in small or dense graphs.
    """
    if not 0.0 <= drift_rate <= 1.0:
        raise ValueError("drift_rate must lie in [0, 1]")
    result = np.asarray(adjacency, dtype=np.float64).copy()
    if drift_rate == 0:
        return result
    edges = _edge_list(result)
    target_swaps = int(np.ceil(rng.binomial(len(edges), drift_rate) / 2))
    completed = 0
    attempts = 0
    max_attempts = max(max_attempt_factor * max(target_swaps, 1), 1)
    while completed < target_swaps and attempts < max_attempts:
        attempts += 1
        edges = _edge_list(result)
        chosen = rng.choice(len(edges), size=2, replace=False)
        (a, b), (c, d) = edges[int(chosen[0])], edges[int(chosen[1])]
        if len({a, b, c, d}) < 4:
            continue
        proposals = [((a, d), (c, b)), ((a, c), (b, d))]
        rng.shuffle(proposals)
        accepted = False
        for (u, v), (x, y) in proposals:
            u, v = sorted((u, v))
            x, y = sorted((x, y))
            if u == v or x == y or result[u, v] or result[x, y]:
                continue
            candidate = result.copy()
            candidate[a, b] = candidate[b, a] = 0.0
            candidate[c, d] = candidate[d, c] = 0.0
            candidate[u, v] = candidate[v, u] = 1.0
            candidate[x, y] = candidate[y, x] = 1.0
            if is_connected(candidate):
                result = candidate
                completed += 1
                accepted = True
                break
        if not accepted:
            continue
    return result


def update_edge_scores(
    adjacency: FloatArray,
    previous_adjacency: FloatArray | None,
    previous_scores: FloatArray | None,
    rng: np.random.Generator,
) -> FloatArray:
    """Keep scores for persistent edges and sample scores for new edges."""
    scores = np.full_like(adjacency, np.nan, dtype=np.float64)
    for i, j in _edge_list(adjacency):
        persists = (
            previous_adjacency is not None
            and previous_scores is not None
            and previous_adjacency[i, j] > 0
            and np.isfinite(previous_scores[i, j])
        )
        score = previous_scores[i, j] if persists else rng.normal()
        scores[i, j] = scores[j, i] = float(score)
    return scores


def neighbor_weights(
    adjacency: FloatArray, edge_scores: FloatArray, concentration: float
) -> FloatArray:
    """Return row-normalized weights over each node's neighbors."""
    if concentration < 0:
        raise ValueError("concentration must be nonnegative")
    weights = np.zeros_like(adjacency, dtype=np.float64)
    for node in range(adjacency.shape[0]):
        neighbors = np.flatnonzero(adjacency[node] > 0)
        if neighbors.size == 0:
            continue
        logits = concentration * edge_scores[node, neighbors]
        logits = logits - np.max(logits)
        exponentials = np.exp(logits)
        weights[node, neighbors] = exponentials / exponentials.sum()
    return weights


def corrupt_adjacency(
    adjacency: FloatArray, edge_noise: float, rng: np.random.Generator
) -> FloatArray:
    """Independently flip undirected, off-diagonal edge indicators."""
    if not 0.0 <= edge_noise <= 1.0:
        raise ValueError("edge_noise must lie in [0, 1]")
    n_nodes = adjacency.shape[0]
    result = np.asarray(adjacency, dtype=np.float64).copy()
    upper = np.triu_indices(n_nodes, k=1)
    flips = rng.random(upper[0].size) < edge_noise
    values = result[upper]
    values[flips] = 1.0 - values[flips]
    result[upper] = values
    result[(upper[1], upper[0])] = values
    np.fill_diagonal(result, 0.0)
    return result

