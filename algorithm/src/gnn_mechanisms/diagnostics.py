"""Training-only, chronologically cross-fitted observable diagnostics.

Oracle quantities remain in a separate field; phase maps use R_hat by default.
Ridge probes use a fixed nonlinear basis and no architecture/hyperparameter search.
"""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from .graphs import consensus, structural_dynamicity, relevance_entropy
from .models import pack_graphs, graph_tensors, select_graph, expand_graph, gather


def lift(x):
    shape=x.shape
    flat=np.asarray(x,float).reshape(-1,shape[-1])
    phi=PolynomialFeatures(2,include_bias=False).fit_transform(flat)
    return np.concatenate((phi,np.sin(flat)),axis=1).reshape(*shape[:-1],-1)


def neighbor_summary(phi,graphs):
    packed=pack_graphs(graphs)
    result=[]
    for t,row in enumerate(phi):
        graph_index=t if len(packed["indices"])>1 else 0
        ix=packed["indices"][graph_index]
        mask=packed["mask"][graph_index]
        neighbors=row[ix]
        result.append((neighbors*mask[:,:,None]).sum(1)/mask.sum(1)[:,None])
    return np.asarray(result)


def chronological_folds(count,folds=3):
    first=max(4,count//2)
    boundaries=np.unique(np.linspace(first,count,folds+1,dtype=int))
    return [(np.arange(stop),np.arange(stop,end))
            for stop,end in zip(boundaries[:-1],boundaries[1:]) if end>stop]


def smoothness(y,graphs):
    """Exact expectation under uniform within-snapshot node permutation."""
    numerator=denominator=0.
    n=y.shape[1]
    for values,graph in zip(y,graphs):
        i,j=np.where(np.triu(graph,1))
        numerator+=float(np.square(values[i]-values[j]).sum())
        denominator+=len(i)*2*n/(n-1)*float(values.var())
    return float(1-numerator/denominator) if denominator>1e-12 else None


def diagnostic_target(dataset):
    if dataset.condition.task=="classification":
        train=dataset.splits["train"]
        threshold=float(np.quantile(dataset.latent[train],
                                    dataset.condition.classification_quantile))
        return (dataset.latent>threshold).astype(float)
    return dataset.target.astype(float)


def probe_diagnostics(features,target,graphs,seed=47,folds=3):
    """Public adapter: pass ONLY the training period from synthetic/real data."""
    x=np.asarray(features,float)
    y=np.asarray(target,float)
    graphs=np.asarray(graphs,dtype=bool)
    if x.shape[:2]!=y.shape or len(graphs)!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned, complete training-only node panel is required.")
    if len(x)<8 or x.shape[1]<2 or graphs.shape!=(len(x),x.shape[1],x.shape[1]):
        raise ValueError("Need at least eight aligned training snapshots and two nodes.")
    if not np.array_equal(graphs,graphs.transpose(0,2,1)):
        raise ValueError("The current diagnostic adapter requires undirected topology.")
    phi=lift(x)
    dynamic=neighbor_summary(phi,graphs)
    permutation=np.random.default_rng(seed+1).permutation(x.shape[1])
    shuffled=graphs[:,permutation][:,:,permutation]
    random_summary=neighbor_summary(phi,shuffled)
    true_values=[]
    predictions={name:[] for name in ("feature","dynamic","static","shuffled")}
    records=[]
    for fit,holdout in chronological_folds(len(x),folds):
        static=consensus(graphs[fit],np.random.default_rng(seed))
        fixed=neighbor_summary(phi,static)
        designs=dict(feature=phi,dynamic=np.concatenate((phi,dynamic),-1),
                     static=np.concatenate((phi,fixed),-1),
                     shuffled=np.concatenate((phi,random_summary),-1))
        for name,design in designs.items():
            probe=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            probe.fit(design[fit].reshape(-1,design.shape[-1]),y[fit].ravel())
            predictions[name].append(probe.predict(design[holdout].reshape(-1,design.shape[-1])))
        true_values.append(y[holdout].ravel())
        records.append(dict(fit_origin_indices=fit.tolist(),holdout_origin_indices=holdout.tolist()))
    truth=np.concatenate(true_values)
    scores={name:float(r2_score(truth,np.concatenate(values))) for name,values in predictions.items()}
    return dict(S_X=scores["feature"],S_G=scores["dynamic"]-scores["feature"],
                S_G_shuffled=scores["shuffled"]-scores["feature"],
                graph_vs_shuffled=scores["dynamic"]-scores["shuffled"],
                H=smoothness(y,graphs),D_struct=structural_dynamicity(graphs),
                D_pred=scores["dynamic"]-scores["static"],probe_scores=scores,
                folds=records,probe="fixed quadratic+sin basis; standardized ridge alpha=10",
                S_G_scope="incremental information accessible to a uniform-summary probe",
                n_snapshots=len(x))


class EdgeGatingProbe(nn.Module):
    """Small source-feature gate and linear value/readout, not a trained GNN."""
    def __init__(self,raw_dim,basis_dim):
        super().__init__()
        self.gate=nn.Linear(raw_dim,1,bias=False)
        nn.init.zeros_(self.gate.weight)
        self.readout=nn.Linear(2*basis_dim,1)

    def forward(self,x,phi,graph):
        graph=expand_graph(graph,len(x))
        logits=gather(self.gate(x),graph["indices"]).squeeze(-1)
        weights=torch.softmax(logits.masked_fill(~graph["mask"],-torch.inf),-1)
        aggregate=(weights[:,:,:,None]*gather(phi,graph["indices"])).sum(-2)
        predicted=self.readout(torch.cat((phi,aggregate),-1)).squeeze(-1)
        return predicted,weights,graph["mask"]


def estimate_relevance(features,target,graphs,epochs=80,seed=47):
    x=np.asarray(features,float)
    phi=lift(x)
    x=(x-x.mean((0,1)))/np.maximum(x.std((0,1)),1e-6)
    phi=(phi-phi.mean((0,1)))/np.maximum(phi.std((0,1)),1e-6)
    y=np.asarray(target,float)
    y=(y-y.mean())/max(y.std(),1e-6)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model=EdgeGatingProbe(x.shape[-1],phi.shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01)
    tx=torch.as_tensor(x,dtype=torch.float32)
    tp=torch.as_tensor(phi,dtype=torch.float32)
    ty=torch.as_tensor(y,dtype=torch.float32)
    graph=graph_tensors(pack_graphs(graphs))
    rng=np.random.default_rng(seed)
    for _ in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(x),4):
            indices=order[start:start+4]
            prediction,_,_=model(tx[indices],tp[indices],select_graph(graph,indices))
            loss=(prediction-ty[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite gating probe.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
    with torch.no_grad():
        values=[]
        for start in range(0,len(x),4):
            indices=np.arange(start,min(start+4,len(x)))
            _,weights,mask=model(tx[indices],tp[indices],select_graph(graph,indices))
            weights=weights.numpy()
            entropy=-(weights*np.log(np.maximum(weights,1e-30))).sum(-1)
            size=mask.numpy().sum(-1)
            values.append(np.where(size>1,1-entropy/np.log(np.maximum(size,2)),0).ravel())
    return float(np.clip(np.concatenate(values),0,1).mean())


def measure(dataset,config=None):
    config=config or {}
    train=dataset.splits["train"]
    x=dataset.features[train]
    y=diagnostic_target(dataset)[train]
    graphs=dataset.observed_graphs[train]
    result=probe_diagnostics(x,y,graphs,seed=config.get("probe_seed",47),
                             folds=config.get("folds",3))
    result["R_hat"]=estimate_relevance(x,y,graphs,epochs=config.get("gate_epochs",80),
                                      seed=config.get("probe_seed",47))
    result["training_target_times"]=dataset.target_times[train].tolist()
    result["R_true"]=dataset.oracle["R_true"]
    result["R_estimation_error"]=(result["R_hat"]-result["R_true"]
                                  if result["R_true"] is not None else None)
    result["scope"]="training-only; R_true is oracle and excluded from observable phase-map axes"
    return result
