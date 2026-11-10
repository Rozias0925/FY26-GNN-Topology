"""Explicit, validated generator and experiment design."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math


@dataclass(frozen=True)
class Condition:
    n_nodes: int = 500
    n_steps: int = 40
    degree: int = 10
    phi: float = .6
    graph_signal: float = .75
    kappa: float = 0.
    delta: float = 0.
    eta: float = 0.
    tau: float = 0.
    noise: float = .3
    observation_noise: float = .5
    graph_family: str = "regular"
    regime: str = "mechanism"
    task: str = "regression"
    attention: str = "source"
    smooth_normalize: bool = True
    train_fraction: float = .6
    validation_fraction: float = .2
    classification_quantile: float = .8

    def __post_init__(self):
        if self.n_nodes < 12 or self.n_steps < 15:
            raise ValueError("Need at least 12 nodes and 15 snapshots.")
        if not 2 <= self.degree < self.n_nodes or self.degree % 2:
            raise ValueError("degree must be positive, even, and smaller than N.")
        for key in ("phi", "graph_signal", "delta", "eta", "train_fraction",
                    "validation_fraction", "classification_quantile"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must lie in [0,1].")
        if self.phi >= 1 or not 0 < self.classification_quantile < 1:
            raise ValueError("Nonstationary features or invalid quantile.")
        for key in ("kappa", "tau", "noise", "observation_noise"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Need a nonempty test period.")
        if self.graph_family not in {"regular", "heterogeneous", "sbm", "knn"}:
            raise ValueError("Unknown graph family.")
        if self.regime not in {"mechanism", "smoothness"}:
            raise ValueError("Unknown regime.")
        if self.task not in {"regression", "classification"}:
            raise ValueError("Unknown task.")
        if self.attention not in {"source", "query"}:
            raise ValueError("Unknown relevance mechanism.")
        if self.regime == "smoothness" and (self.delta or self.eta):
            raise ValueError("The isolated smoothness experiment has fixed clean topology.")
        if self.graph_family=="regular":
            max_turnover=min(1.,(self.n_nodes-1-self.degree)/self.degree)
            if max(self.delta,self.eta)>max_turnover:
                raise ValueError("Requested edge replacement exceeds available non-edges.")

    def to_dict(self):
        return asdict(self)

    @property
    def identifier(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def design(config, stage):
    """One-axis interventions share all other defaults and random streams."""
    base = Condition(**config.get("generator", {}))
    if stage == "A":
        specs = [
            ("feature_only", dict(graph_signal=0)),
            ("uniform_neighbors", dict(graph_signal=.9, kappa=0)),
            ("selective_neighbors", dict(graph_signal=.9, kappa=4)),
            ("smooth_signal", dict(regime="smoothness", tau=2)),
            ("predictive_dynamics", dict(graph_signal=.9, delta=.4)),
            ("noisy_dynamics", dict(graph_signal=.9, eta=.4)),
        ]
        common = dict(graph_signal=.75,kappa=0,delta=0,eta=0,tau=0,regime="mechanism")
        return [dict(stage=stage, axis=name, value=None,
                     condition=replace(base, **{**common,**changes}))
                for name, changes in specs]
    if stage == "B":
        levels = {
            "graph_signal": [0, .1, .25, .5, .75, .9, 1],
            "kappa": [0, .5, 1, 2, 4],
            "tau": [0, .1, .5, 1, 2, 5],
            "delta": [0, .05, .10, .20, .40],
            "eta": [0, .05, .10, .20, .40],
        }
        result = []
        for axis, values in levels.items():
            for value in config.get("axes", {}).get(axis, values):
                changes = dict(graph_signal=.75, kappa=0, delta=0, eta=0,
                               tau=0, regime="smoothness" if axis == "tau" else "mechanism")
                changes[axis] = value
                result.append(dict(stage=stage, axis=axis, value=value,
                                   condition=replace(base, **changes)))
        return result
    if stage == "C":
        result = []
        grids = [
            ("signal_relevance", "graph_signal", [0,.25,.5,.75,1], "kappa", [0,1,2,4]),
            ("drift_noise", "delta", [0,.1,.2,.4], "eta", [0,.1,.2,.4]),
            ("smooth_signal", "tau", [0,.5,2,5], "graph_signal", [0,.25,.5,.75,1]),
        ]
        for name, x, xs, y, ys in grids:
            for xv in xs:
                for yv in ys:
                    changes = dict(graph_signal=.75, kappa=0, delta=0, eta=0,
                                   tau=0, regime="mechanism")
                    changes.update({x:xv, y:yv})
                    result.append(dict(stage=stage, axis=name, value={x:xv,y:yv},
                                       condition=replace(base, **changes)))
        return result
    if stage == "D":
        import numpy as np
        from scipy.stats import qmc
        count = config.get("multi_axis_conditions", 200)
        if not 1 <= count <= 10000:
            raise ValueError("Invalid Latin-hypercube sample size.")
        sampler = qmc.LatinHypercube(5, seed=config.get("design_seed", 20261002))
        samples = sampler.random(count)
        ranges = np.array([[0,1],[0,4],[0,.4],[0,.4],[0,2.]])
        return [dict(stage=stage, axis="latin_hypercube", value=i,
                     condition=replace(base, regime="mechanism",
                         **dict(zip(("graph_signal","kappa","delta","eta","tau"),
                                    ranges[:,0]+sample*(ranges[:,1]-ranges[:,0])))))
                for i, sample in enumerate(samples)]
    if stage == "sample_size":
        return [dict(stage=stage, axis="n_nodes", value=n,
                     condition=replace(base,n_nodes=n,kappa=0,delta=0,eta=0,tau=0))
                for n in config.get("sample_sizes",[100,300,500,1000])]
    if stage == "robustness":
        result = []
        base=replace(base,regime="mechanism",kappa=0,delta=0,eta=0,tau=0)
        for kappa in (0,4):
            result.append(dict(stage=stage,axis="architecture_control",value=kappa,
                               condition=replace(base,kappa=kappa,attention="source")))
        for family in ("heterogeneous","sbm","knn"):
            for kappa in (0,4):
                result.append(dict(stage=stage,axis="graph_family",value=f"{family}_k{kappa}",
                    condition=replace(base,graph_family=family,kappa=kappa)))
        for noise in (.1,.5,1):
            result.append(dict(stage=stage,axis="noise",value=noise,
                              condition=replace(base,noise=noise)))
        for kappa in (0,4):
            result.append(dict(stage=stage,axis="classification",value=kappa,
                condition=replace(base,task="classification",kappa=kappa)))
        result.append(dict(stage=stage,axis="query_dependent",value=4,
                           condition=replace(base,kappa=4,attention="query")))
        return result
    raise ValueError(f"Unknown stage: {stage}")


def plans(config, stages):
    result = []
    for stage in stages:
        result.extend(design(config,stage))
    return result
