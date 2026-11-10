"""Snapshot-only, permutation-equivariant models with sparse neighborhoods.

No response history, node ID embedding, true edge weights, or recurrent memory.
One aggregation layer follows a nonlinear node encoder in every graph model.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ["MLP","Static-GCN","Static-GAT","Dynamic-GCN","Dynamic-GAT"]
EXTRA_MODELS = ["Static-GATv2","Dynamic-GATv2"]


def pack_graphs(adjacency):
    a = np.asarray(adjacency,dtype=bool)
    if a.ndim == 2:
        a=a[None]
    a=a.copy()
    n=a.shape[-1]
    a[:,np.arange(n),np.arange(n)]=True
    degree=a.sum(-1)
    width=int(degree.max())
    indices=np.zeros((*degree.shape,width),dtype=np.int64)
    mask=np.zeros_like(indices,dtype=bool)
    weights=np.zeros_like(indices,dtype=np.float32)
    for t,row in enumerate(a):
        for i in range(n):
            neighbors=np.flatnonzero(row[i])
            size=len(neighbors)
            indices[t,i,:size]=neighbors
            mask[t,i,:size]=True
            weights[t,i,:size]=1/np.sqrt(degree[t,i]*degree[t,neighbors])
    return dict(indices=indices,mask=mask,gcn_weights=weights)


def graph_tensors(packed,device="cpu"):
    return {key:torch.as_tensor(value,device=device) for key,value in packed.items()}


def select_graph(packed,indices):
    return {key:value[indices] if value.shape[0]>1 else value for key,value in packed.items()}


def expand_graph(packed,batch):
    return {key:value.expand(batch,-1,-1) if value.shape[0]==1 else value
            for key,value in packed.items()}


def gather(values,indices):
    batch=torch.arange(len(values),device=values.device)[:,None,None]
    return values[batch,indices]


class SnapshotModel(nn.Module):
    def __init__(self,name,input_dim=8,width=32):
        super().__init__()
        if name not in MODEL_NAMES+EXTRA_MODELS:
            raise ValueError(name)
        self.name=name
        self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),
                                   nn.Linear(width,width),nn.ReLU())
        self.value=nn.Linear(width,width)
        if "GAT" in name:
            self.source=nn.Parameter(torch.empty(width))
            nn.init.normal_(self.source,std=.05)
            if "GATv2" in name:
                self.query=nn.Linear(width,width,bias=False)
            else:
                self.target=nn.Parameter(torch.empty(width))
                nn.init.normal_(self.target,std=.05)
        self.head=nn.Sequential(nn.Linear(2*width,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,x,packed=None,return_attention=False):
        h=self.encoder(x)
        value=self.value(h)
        attention=None
        if self.name == "MLP":
            message=value
        else:
            graph=expand_graph(packed,len(x))
            neighbor=gather(value,graph["indices"])
            if "GAT" in self.name:
                if "GATv2" in self.name:
                    logits=(F.leaky_relu(self.query(h)[:,:,None,:]+neighbor,.2)*self.source).sum(-1)
                else:
                    # Original additive GAT: source ranking is query independent.
                    source_scores=(h@self.source).unsqueeze(-1)
                    logits=F.leaky_relu((h@self.target)[:,:,None]+
                                       gather(source_scores,graph["indices"]).squeeze(-1),.2)
                attention=torch.softmax(logits.masked_fill(~graph["mask"],-torch.inf),-1)
                message=(attention[:,:,:,None]*neighbor).sum(-2)
            else:
                message=(graph["gcn_weights"][:,:,:,None]*neighbor).sum(-2)
        prediction=self.head(torch.cat((h,message),-1)).squeeze(-1)
        return (prediction,attention) if return_attention else prediction
