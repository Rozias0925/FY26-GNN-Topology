"""Five trained baselines with one shared node-wise temporal MLP encoder.

GCN: Kipf & Welling (2017), symmetric normalized self-loop adjacency.
GAT: Velickovic et al. (2018), one-head additive masked attention.
Dynamic refers to A at forecast origin, not merely changing attention weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ['MLP', 'Static GCN', 'Static GAT', 'Dynamic GCN', 'Dynamic GAT']


def normalize_graph(adjacency):
    a = adjacency + torch.eye(adjacency.shape[-1], device=adjacency.device)
    degree = a.sum(-1).clamp_min(1e-8).rsqrt()
    return degree[..., :, None] * a * degree[..., None, :]


class GCN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x, a):
        return normalize_graph(a) @ self.linear(x) + self.bias


class GAT(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.target = nn.Parameter(torch.empty(width))
        self.source = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.target, std=width ** -.5)
        nn.init.normal_(self.source, std=width ** -.5)
        self.bias = nn.Parameter(torch.zeros(width))

    def attention(self, x, a):
        h = self.linear(x)
        logits = F.leaky_relu((h @ self.target)[..., :, None] + (h @ self.source)[..., None, :], negative_slope=.2)
        allowed = (a > 0) | torch.eye(a.shape[-1], dtype=torch.bool, device=a.device)
        return h, torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)

    def forward(self, x, a):
        h, weights = self.attention(x, a)
        return weights @ h + self.bias


class ForecastModel(nn.Module):
    def __init__(self, name, history_length, input_dim, width=32):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(name)
        self.name = name
        self.encoder = nn.Sequential(nn.Linear(history_length * input_dim, width), nn.ReLU())
        self.spatial = GCN(width) if name.endswith('GCN') else GAT(width) if name.endswith('GAT') else nn.Linear(width, width)
        self.head = nn.Linear(2 * width, 2)

    def forward(self, history, adjacency, target_nodes=None):
        # Every model sees identical histories. MLP never mixes node dimensions.
        b, length, n, f = history.shape
        h = self.encoder(history.permute(0, 2, 1, 3).reshape(b, n, length * f))
        message = self.spatial(h) if self.name == 'MLP' else self.spatial(h, adjacency)
        # Equal self branch lets graph models retain local predictive information.
        outputs = self.head(torch.cat([h, F.relu(message)], dim=-1))
        if target_nodes is not None:
            outputs = outputs[:, target_nodes]
        return outputs[..., 0], outputs[..., 1]
