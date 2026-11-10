"""Binary graph topology, exact-density consensus, and degree-preserving swaps."""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def initial_graph(n, degree, family, rng):
    seed = int(rng.integers(2**31))
    if family == "regular":
        for attempt in range(20):
            graph = nx.random_regular_graph(degree,n,seed=seed+attempt)
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError("Could not generate a connected regular graph.")
    elif family == "heterogeneous":
        graph = nx.barabasi_albert_graph(n,max(1,degree//2),seed=seed)
    elif family == "sbm":
        sizes = [n//2,n-n//2]
        graph = nx.stochastic_block_model(sizes,
            [[min(.9,1.6*degree/n),min(.9,.4*degree/n)],
             [min(.9,.4*degree/n),min(.9,1.6*degree/n)]],seed=seed)
        components = list(nx.connected_components(graph))
        for left,right in zip(components,components[1:]):
            graph.add_edge(min(left),min(right))
    else:
        from sklearn.neighbors import NearestNeighbors
        points = rng.normal(size=(n,3))
        nearest = NearestNeighbors(n_neighbors=degree+1).fit(points)
        neighbors = nearest.kneighbors(points,return_distance=False)[:,1:]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from((i,int(j)) for i in range(n) for j in neighbors[i])
        components = list(nx.connected_components(graph))
        for left,right in zip(components,components[1:]):
            graph.add_edge(min(left),min(right))
    return nx.to_numpy_array(graph,nodelist=range(n),dtype=bool)


def edge_turnover(before, after):
    mask = np.triu_indices(len(before),1)
    count = int(before[mask].sum())
    return float((before[mask] & ~after[mask]).sum()/count) if count else 0.


def rewire(adjacency, rate, rng):
    """Replace an even rounded count of ORIGINAL edges, without undoing swaps.

    Realized turnover is |E_old - E_new| / |E_old|. A swap replaces two
    original edges. Degrees and the total edge count are exactly conserved.
    """
    if not 0 <= rate <= 1:
        raise ValueError("Swap rate must lie in [0,1].")
    original = np.asarray(adjacency,dtype=bool)
    if rate == 0:
        return original.copy()
    edges = np.column_stack(np.where(np.triu(original,1)))
    requested = 2*int(round(len(edges)*rate/2))
    if requested == 0:
        return original.copy()
    for _ in range(8):
        a = original.copy()
        available = list(range(len(edges)))
        removed = 0
        attempts = 0
        while removed < requested and len(available) >= 2 and attempts < 50*requested:
            attempts += 1
            positions = rng.choice(len(available),2,replace=False)
            (i,j),(k,l) = edges[[available[int(p)] for p in positions]]
            if len({int(i),int(j),int(k),int(l)}) < 4:
                continue
            proposals = [((i,k),(j,l)),((i,l),(j,k))]
            rng.shuffle(proposals)
            for (u,v),(w,z) in proposals:
                if a[u,v] or a[w,z] or original[u,v] or original[w,z]:
                    continue
                a[i,j]=a[j,i]=a[k,l]=a[l,k]=False
                a[u,v]=a[v,u]=a[w,z]=a[z,w]=True
                for position in sorted(map(int,positions),reverse=True):
                    available[position] = available[-1]
                    available.pop()
                removed += 2
                break
        if removed == requested and connected_components(csr_matrix(a),directed=False)[0] == 1:
            return a
    raise RuntimeError("Could not realize requested turnover while retaining connectivity.")


def consensus(snapshots, rng):
    """Highest-frequency M undirected edges; random seeded tie-breaking."""
    a = np.asarray(snapshots,dtype=bool)
    if a.ndim != 3 or len(a) == 0:
        raise ValueError("Need at least one training snapshot.")
    n = a.shape[1]
    upper = np.triu_indices(n,1)
    frequencies = a[:,upper[0],upper[1]].mean(0)
    edge_count = int(round(a.sum(axis=(1,2)).mean()/2))
    order = np.lexsort((rng.random(len(frequencies)),-frequencies))
    result = np.zeros((n,n),dtype=bool)
    chosen = order[:edge_count]
    result[upper[0][chosen],upper[1][chosen]] = True
    return result | result.T


def structural_dynamicity(snapshots):
    a = np.asarray(snapshots,dtype=bool)
    scores = []
    for left,right in zip(a[:-1],a[1:]):
        union = np.logical_or(left,right).sum()
        scores.append(1-float(np.logical_and(left,right).sum()/union) if union else 0.)
    return float(np.mean(scores)) if scores else 0.


def source_weights(adjacency, scores, kappa):
    mask = np.asarray(adjacency,dtype=bool) | np.eye(len(adjacency),dtype=bool)
    logits = np.broadcast_to(kappa*scores,(len(scores),len(scores))).copy()
    logits[~mask] = -np.inf
    logits -= logits.max(axis=1,keepdims=True)
    weights = np.exp(logits)
    return weights/weights.sum(axis=1,keepdims=True)


def relevance_entropy(weights):
    """Normalized NEGATIVE entropy: uniform -> R=0, concentrated -> R=1."""
    weights = np.asarray(weights,float)
    sizes = (weights>0).sum(axis=-1)
    entropy = -(weights*np.log(np.maximum(weights,1e-300))).sum(axis=-1)
    normalized = np.divide(entropy,np.log(np.maximum(sizes,2)),
                           out=np.ones_like(entropy),where=sizes>1)
    return np.clip(1-normalized,0,1)
