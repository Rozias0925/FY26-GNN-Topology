"""Validated configuration for the controlled simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_nodes: int = 20
    feature_dim: int = 3
    n_steps: int = 240
    degree: int = 4
    graph_signal: float = 0.6
    graph_drift: float = 0.05
    edge_noise: float = 0.05
    neighbor_heterogeneity: float = 1.0
    feature_ar: float = 0.7
    response_ar: float = 0.5
    feature_noise: float = 1.0
    response_noise: float = 0.5
    seed: int = 7
    burn_in: int = 100

    def __post_init__(self) -> None:
        import math
        for name, value in self.to_dict().items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if self.n_nodes < 6:
            raise ValueError("n_nodes must be at least 6")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.n_steps < 8:
            raise ValueError("n_steps must be at least 8")
        if self.degree < 2 or self.degree >= self.n_nodes or self.degree % 2:
            raise ValueError("degree must be even and satisfy 2 <= degree < n_nodes")
        for name in ("graph_drift", "edge_noise"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.graph_signal < 0:
            raise ValueError("graph_signal must be nonnegative")
        if self.neighbor_heterogeneity < 0:
            raise ValueError("neighbor_heterogeneity must be nonnegative")
        if abs(self.feature_ar) >= 1 or abs(self.response_ar) >= 1:
            raise ValueError("AR coefficients must have absolute value below 1")
        if self.feature_noise <= 0 or self.response_noise <= 0:
            raise ValueError("noise scales must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
