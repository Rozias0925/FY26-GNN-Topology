"""Independent feature blocks and training-calibrated synthetic mechanisms."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from .config import Condition
from .graphs import initial_graph, rewire, consensus, source_weights, relevance_entropy, edge_turnover


@dataclass
class Dataset:
    condition: Condition
    seed: int
    features: np.ndarray
    target: np.ndarray
    latent: np.ndarray
    true_graphs: np.ndarray
    observed_graphs: np.ndarray
    static_graph: np.ndarray
    origin_times: np.ndarray
    target_times: np.ndarray
    splits: dict
    oracle: dict


def time_split(n_steps, train_fraction=.6, validation_fraction=.2, horizon=1):
    origins = np.arange(1,n_steps-horizon+1)
    targets = origins+horizon
    train_end = int(n_steps*train_fraction)
    valid_end = int(n_steps*(train_fraction+validation_fraction))
    splits = dict(train=np.flatnonzero(targets<=train_end),
                  validation=np.flatnonzero((targets>train_end)&(targets<=valid_end)),
                  test=np.flatnonzero(targets>valid_end))
    if any(len(indices)<2 for indices in splits.values()):
        raise ValueError("Each chronological split needs at least two target snapshots.")
    return origins,targets,splits


def standardized(values, training):
    mean = float(values[training].mean())
    scale = float(values[training].std())
    if scale < 1e-10:
        raise ValueError("Degenerate generator signal.")
    return (values-mean)/scale,dict(mean=mean,scale=scale)


def smooth(values, adjacency, tau):
    if not tau:
        return values.copy()
    a = csr_matrix(adjacency.astype(float))
    laplacian = diags(np.asarray(a.sum(1)).ravel())-a
    return spsolve(eye(len(adjacency),format="csr")+tau*laplacian,values)


def generate(condition: Condition, seed: int):
    streams = np.random.SeedSequence(seed).spawn(8)
    graph_rng,drift_rng,noise_rng,feature_rng,response_rng,smooth_rng,tie_rng,extra_rng = [
        np.random.default_rng(stream) for stream in streams]
    c = condition
    true = np.empty((c.n_steps,c.n_nodes,c.n_nodes),dtype=bool)
    observed = np.empty_like(true)
    true[0] = initial_graph(c.n_nodes,c.degree,c.graph_family,graph_rng)
    for t in range(1,c.n_steps):
        true[t] = rewire(true[t-1],c.delta,drift_rng)
    for t in range(c.n_steps):
        # Corruption is independent at each snapshot, relative to the TRUE graph.
        observed[t] = rewire(true[t],c.eta,noise_rng)
    train_end = int(c.n_steps*c.train_fraction)
    static = consensus(observed[:train_end],tie_rng)
    raw = np.empty((c.n_steps,c.n_nodes,8))
    raw[0] = feature_rng.normal(size=(c.n_nodes,8))
    for t in range(1,c.n_steps):
        raw[t] = c.phi*raw[t-1]+np.sqrt(1-c.phi**2)*feature_rng.normal(size=(c.n_nodes,8))
    horizon = 0 if c.regime == "smoothness" else 1
    origins,targets,splits = time_split(c.n_steps,c.train_fraction,c.validation_fraction,horizon)
    train = splits["train"]
    oracle = dict(horizon=horizon, feature_blocks_independent=True,
                  true_turnover=[edge_turnover(a,b) for a,b in zip(true[:-1],true[1:])],
                  corruption_turnover=[edge_turnover(a,b) for a,b in zip(true,observed)],
                  true_edge_count=int(true[0].sum()/2),
                  static_edge_count=int(static.sum()/2))
    if c.regime == "smoothness":
        z0 = smooth_rng.normal(size=(c.n_steps,c.n_nodes))
        z = np.stack([smooth(row,true[0],c.tau) for row in z0])
        if c.smooth_normalize:
            z,oracle["smooth_calibration"] = standardized(z,train)
        x = raw.copy()
        x[:,:,0] = z+c.observation_noise*extra_rng.normal(size=z.shape)
        latent = z
        oracle["R_true"] = None
        oracle["latent_snr"] = float(z[train].var()/max(c.noise**2,1e-15))
    else:
        x = raw.copy()
        if c.tau:
            # Multi-axis extension: smooth only the graph feature block.
            # Its driving innovations remain independent of the self block.
            x[:,:,4:] = np.stack([smooth(row[:,4:],a,c.tau) for row,a in zip(raw,true)])
            mean=x[train,:,4:].mean(axis=(0,1))
            scale=x[train,:,4:].std(axis=(0,1))
            x[:,:,4:] = (x[:,:,4:]-mean)/np.maximum(scale,1e-8)
            oracle["graph_feature_calibration"] = dict(mean=mean.tolist(),scale=scale.tolist())
        own,neighbor_features = x[:,:,:4],x[:,:,4:]
        u = own[:,:,0]+.5*own[:,:,1]**2+np.sin(own[:,:,2])+.5*own[:,:,2]*own[:,:,3]
        content = (neighbor_features[:,:,0]+.5*neighbor_features[:,:,1]**2+
                   .5*np.sin(neighbor_features[:,:,2])+.3*neighbor_features[:,:,3])
        b = np.array([1.,-.5,.25,.5])
        b /= np.linalg.norm(b)
        scores = neighbor_features@b
        v = np.empty((c.n_steps,c.n_nodes))
        relevance = []
        for t in range(c.n_steps):
            if c.attention == "source":
                weights = source_weights(true[t],scores[t],c.kappa)
            else:
                # Explicit robustness mechanism requiring query-dependent ranks.
                logits = c.kappa*(neighbor_features[t]@neighbor_features[t].T)/2
                mask=true[t]|np.eye(c.n_nodes,dtype=bool)
                logits[~mask]=-np.inf
                logits-=logits.max(1,keepdims=True)
                weights=np.exp(logits)
                weights/=weights.sum(1,keepdims=True)
            v[t]=weights@content[t]
            relevance.append(float(relevance_entropy(weights).mean()))
        u,um = standardized(u[:-1],train)
        v,vm = standardized(v[:-1],train)
        latent=np.sqrt(1-c.graph_signal)*u+np.sqrt(c.graph_signal)*v
        oracle.update(u_moments=um,v_moments=vm,
                      standardized_u_variance=float(u[train].var()),
                      standardized_v_variance=float(v[train].var()),
                      u_v_covariance=float(np.mean(u[train]*v[train])),
                      latent_variance=float(latent[train].var()),
                      R_true=float(np.mean(np.array(relevance)[train])),
                      b=b.tolist(),self_loop_in_generator=True,
                      lambda_interpretation="relative calibrated component variances, excluding noise")
        x = x[:-1]
    target=latent+c.noise*response_rng.normal(size=latent.shape)
    if c.task=="classification":
        oracle["classification_threshold"]=float(np.quantile(latent[train],c.classification_quantile))
        oracle["classification_label"]="latent_without_response_noise > training_quantile"
    return Dataset(c,seed,x.astype(np.float32),target.astype(np.float32),
                   latent.astype(np.float32),true[:len(x)],observed[:len(x)],
                   static,origins,targets,splits,oracle)
