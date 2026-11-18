# GNN Primer

A two-page reference for L34 and Lab 8.

---

## The setup

A graph G = (V, E):
- V = node set, |V| = n
- E ⊆ V × V = edge set
- A ∈ {0, 1}^{n × n} = adjacency matrix
- D = diag(deg(v_1), ..., deg(v_n)) = degree matrix
- L = D - A = graph Laplacian
- L_norm = I - D^(-1/2) A D^(-1/2) = symmetric normalized Laplacian

Each node has feature vector x_v ∈ ℝ^d. Each edge optionally has feature e_{uv} ∈ ℝ^d_edge.

A GNN learns a function f(G, x, e) that produces either:
- Per-node outputs (node classification, segmentation).
- A graph-level output (graph classification, regression). Lab 8 is this case: one E_app per graph.
- Per-edge outputs (link prediction).

---

## The message-passing framework

A GNN layer:

```
For each node v:
  1. Messages from neighbors:
     m_{u -> v} = message(h_u, h_v, e_{uv})       for u in N(v)
  2. Aggregate (permutation-invariant):
     m_v = AGGREGATE({m_{u -> v} : u in N(v)})
  3. Update:
     h_v_new = update(h_v, m_v)
```

Every GNN architecture is a choice of `message`, `aggregate`, `update`.

---

## GCN (Kipf & Welling, ICLR 2017)

The simplest workhorse.

```
h^{(l+1)} = sigma(D̃^(-1/2) Ã D̃^(-1/2) h^{(l)} W^{(l)} + b^{(l)})

where Ã = A + I (self-loops), D̃ = degree matrix of Ã.
```

In message-passing form, with d̃_v = deg(v) + 1 the degree counting the self-loop:

```
m_{u -> v} = h_u / sqrt(d̃_u * d̃_v)
m_v = h_v / d̃_v + sum_{u in N(v)} m_{u -> v}        # the self-loop term, then the neighbors
h_v_new = sigma(W * m_v + b)
```

**Pros.** Simple. Computationally efficient. Degree-normalized neighbor mixing tends to smooth features (the
"low-pass" intuition); repeated mixing can make node features less distinguishable, so deep GCNs risk over-smoothing.

**Cons.** Its normalized averaging cannot distinguish some non-isomorphic graphs that the 1-WL (Weisfeiler-Lehman)
test separates (Xu et al. 2019). `GCNConv` takes no edge feature vectors (at most a scalar weight per edge).

---

## GIN (Xu, Hu, Leskovec & Jegelka, ICLR 2019)

More expressive than GCN: as powerful as the 1-WL graph-isomorphism test, which Xu et al. prove is the upper bound for
message-passing GNNs of this kind.

```
m_{u -> v} = h_u
m_v = sum_{u in N(v)} m_{u -> v}                # SUM, not mean
h_v_new = MLP((1 + eps) * h_v + m_v)            # eps learnable
```

**With edge features (what Lab 8 uses).** PyG's `GINEConv` adds each edge's features to its message:
m_v = sum_{u in N(v)} ReLU(h_u + W_e e_{uv}). With `edge_dim=4`, Lab 8's four edge features are mapped linearly to
the hidden width.

**Pros.** Maximally discriminative among these message-passing GNNs. A common default for graph-level prediction.

**Cons.** The sum grows with the number of neighbors, so on graphs with very high-degree nodes the features can get
large; consider normalization or GraphSAGE-style sampling if your graphs are large.

---

## GAT (Veličković et al., ICLR 2018) and GATv2 (Brody, Alon & Yahav, ICLR 2022)

Both learn *attention weights* per edge. GAT:

```
alpha_{uv} = softmax_u(LeakyReLU(a^T [W h_u | W h_v]))
h_v_new = sigma( sum_{u in N(v)} alpha_{uv} * (W h_u) )
```

GATv2, the version Lab 8 uses, moves the nonlinearity inside the attention score:

```
alpha_{uv} = softmax_u( a^T LeakyReLU(W_t h_u + W_s h_v + W_e e_{uv}) )      # W_e e_{uv} only with edge_dim
h_v_new = sum_{u in N(v)} alpha_{uv} * (W_t h_u)
```

In GAT, the ranking of neighbors by attention score is the same for every receiving node ("static" attention), because
`a` is applied before the nonlinearity. GATv2 applies `a` after it, so the ranking can depend on the receiving node
("dynamic" attention). With `edge_dim`, the attention also sees the edge features.

**Multi-head:** run K attention computations in parallel; concatenate or average. Lab 8 uses
`GATv2Conv(64, 16, heads=4, concat=True, edge_dim=4)`: four heads of 16, concatenated back to width 64. `GATv2Conv`
also adds a self-loop to each node by default, so a node attends to itself as well as to its neighbors.

**Pros.** Learnable per-edge importance. Especially useful when edges have different meanings.

**Cons.** The attention weights sum to one over each neighborhood, so the aggregate is a weighted average: like mean
aggregation, it cannot tell one neighbor from several identical ones (the GIN paper's argument for sum). How GATv2 and
GIN compare on Lab 8's graphs, in error, parameter count and time, is what Task 3 measures.

---

## GraphSAGE (Hamilton, Ying & Leskovec, NeurIPS 2017)

For very large graphs.

```
m_v = AGG({h_u : u in SAMPLED_NEIGHBORS(v)})    # sample fixed-size set
h_v_new = sigma(W * concat(h_v, m_v) + b)
```

Sampling makes it scalable to graphs with millions of nodes. Lab 8's graphs (2 to 5 nodes) need no sampling.

---

## PyTorch Geometric idioms

**Installation.**

On ARCC, PyTorch Geometric 2.7.0 (with torch 2.5.1) is already in the course environment. Check it once:

```bash
source /project/me5475/setup5475.sh
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

Never `pip install` anything into the course environment. The optional compiled extensions torch_scatter,
torch_sparse and torch_cluster are not installed there, and the Lab 8 scripts do not need them. On your own machine,
`pip install torch_geometric` is enough.

**One runnable sequence.** The snippets below run in order in the course environment; the shapes match Lab 8
(7 node features, 4 edge features, hidden width 64).

*Data representation.*

```python
import torch
import torch.nn as nn
from torch_geometric.data import Data

# A toy cell with 3 inclusions, joined 0-1 and 1-2; each undirected edge appears in both directions
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 7)              # node features: (n_nodes, 7), as in Lab 8
edge_attr = torch.randn(4, 4)      # edge features: (n_edges, 4), one row per directed edge
y = torch.tensor([0.7])            # graph-level target (E_app in Lab 8)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

Lab 8's graphs (`microstructure_to_graph.py`) have one node per inclusion, 2 to 5 per cell. `x` is (n, 7): cx, cy,
major, minor, area = πab, e·cos 2φ, e·sin 2φ, with e = (major − minor)/(major + minor) and φ the direction of the major
axis. `edge_index` joins neighboring inclusions in both directions (the Delaunay triangulation of the centers for
three or more, the single pair for two). `edge_attr` is (n_edges, 4): dx, dy, dist, dist². `y` is the cell's E_app
from Lab 6, and `row` is its Lab 6 row.

*Batching.* PyG's `DataLoader` joins several graphs into one "disjoint union" graph (a `Batch` object). Its field
`batch.batch` says which graph each node belongs to.

```python
from torch_geometric.loader import DataLoader

graphs = [data, data.clone()]                  # two toy graphs
loader = DataLoader(graphs, batch_size=2, shuffle=False)
batch = next(iter(loader))                     # a Batch object
print(batch.x.shape, batch.edge_index.shape, batch.edge_attr.shape, batch.batch)
# torch.Size([6, 7]) torch.Size([2, 8]) torch.Size([8, 4]) tensor([0, 0, 0, 1, 1, 1])
```

*Layers.* Lift the 7 input features to the hidden width first, then apply the message-passing layers.

```python
from torch_geometric.nn import GCNConv, GINEConv, GATv2Conv

lift = nn.Linear(7, 64)
h = lift(batch.x)                                            # (6, 64)

gcn = GCNConv(64, 64)
h_gcn = gcn(h, batch.edge_index)                             # (6, 64); GCNConv takes no edge features

# The two layers Lab 8 uses; both read the 4 edge features through edge_dim=4
gin = GINEConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)), train_eps=True, edge_dim=4)
gat = GATv2Conv(64, 64 // 4, heads=4, concat=True, edge_dim=4)   # 4 heads of 16, concatenated: 64
h_gin = gin(h, batch.edge_index, batch.edge_attr)            # (6, 64)
h_gat = gat(h, batch.edge_index, batch.edge_attr)            # (6, 64)
```

*Pooling for a graph-level prediction.* Pool with the assignment vector `batch.batch` (not the `Batch` object).

```python
from torch_geometric.nn import global_add_pool, global_mean_pool

g_sum = global_add_pool(h_gin, batch.batch)     # (2, 64): one row per graph; Lab 8's default (--pooling sum)
g_mean = global_mean_pool(h_gin, batch.batch)   # (2, 64); --pooling mean
```

Sum pooling keeps the number of nodes (here, inclusions) in the graph embedding; mean pooling does not. Xu et al. use a
sum readout for the same reason they use sum aggregation.

---

## When to use which architecture

| Scenario | Common starting point |
|----------|-------------|
| Small dense graphs (< 100 nodes) | GIN or GAT/GATv2 |
| Large sparse graphs (>> 10k nodes) | GraphSAGE |
| FE mesh-based time-evolving simulation | MeshGraphNet |
| Atomic/molecular structure with rotation invariance | NequIP, MACE (E(3)-equivariant) |
| Simple regression baseline | GCN |

Lab 8 trains both GIN (`GINEConv`) and GATv2 (`GATv2Conv`), with edge features, hidden width 64, 4 layers, and sum
pooling by default. Which one does better on these graphs is what Task 3 measures.

---

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Loss decreases but predictions are constant | Possibly over-smoothing (too many layers), among other causes | Try fewer layers or residual connections, chosen on validation error |
| Training is slow | Batch size too small | Increase batch_size; PyG handles variable-size graphs |
| A node's own features counted twice | Explicit duplicate self-loop edges, or a custom layer that adds self-loops on top of ones already in `edge_index` (PyG's `GCNConv` adds only *missing* self-loops; `GATv2Conv` removes and re-adds them) | Keep self-loops in one place; check `edge_index` for duplicates |
| Test error >> train | Overfitting on graph structure | More data; or simpler architecture |
| Edge attributes ignored | Used `GCNConv` or plain `GINConv`, which take no edge feature vectors | Use a layer with `edge_dim`: `GINEConv` or `GATv2Conv` (both Lab 8 models), or `NNConv` |
| Optimistic test error | Layers, pooling or seed chosen by test error | Choose on validation error and evaluate the test set once (the Lab 8 script keeps the best-validation weights) |

---

## A note on degree normalization

For graphs where node degrees vary widely (e.g., one node with 100 neighbors, others with 2), the GCN's `1/sqrt(deg)`
normalization sometimes hurts, and GIN's unnormalized sum can grow large. Lab 8's graphs have 2 to 5 nodes, so each node
has 1 to 4 neighbors and neither effect can be large; whether sum or mean pooling suits E_app better is an experiment
(`--pooling`). Make any such choice on validation error, never on test error.
