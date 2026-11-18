"""
microstructure_to_graph.py
===========================

Module 8 / Lab 8 -- convert the Lab 6 microstructure dataset to graphs, one graph per cell.

Lab 6 stores, for every cell, the exact inclusion list it drew (`ellipses`: a list of (cx, cy, a, b, theta) per
cell, a ragged object array) and the apparent modulus MOOSE computed (`moduli`, E_app). Each cell becomes a PyTorch
Geometric `Data` object:

  - x:          (n_nodes, 7) node features, one node per inclusion:
                cx, cy                      center (cell coordinates in [0, 1])
                major, minor                semi-axes, major >= minor
                area                        pi * a * b
                e*cos(2 phi), e*sin(2 phi)  orientation of the major axis, phi in [0, pi), weighted by the
                                            elongation e = (major - minor) / (major + minor): zero for a circle,
                                            and the same value for (a, b, theta) and (b, a, theta + pi/2), which
                                            describe the same ellipse
  - edge_index: (2, n_edges) edges between neighboring inclusions: the Delaunay triangulation of the centers for
                three or more inclusions (both directions of each edge), the single pair for two
  - edge_attr:  (n_edges, 4) dx, dy, distance, distance^2 from the source center to the target center
  - y:          E_app of that cell (MOOSE, Lab 6)
  - row:        the cell's row in the Lab 6 dataset, so the GNN can use Lab 6's exact train/val/test split

The graph keeps each inclusion's exact center, size, shape and orientation, but not the pixel image: it is a
different representation of the same cells, not a lossy copy of the image. Features are standardized later, in
gnn_microstructure.py, with training-set statistics.

Usage
-----
    python microstructure_to_graph.py --in rve_dataset.npz --out microstructure_graphs.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError
from torch_geometric.data import Data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="input", type=Path, required=True,
                   help="rve_dataset.npz from Lab 6 (with the ellipses field)")
    p.add_argument("--out", type=Path, default=Path("microstructure_graphs.pt"))
    return p.parse_args()


def node_features(ellipses):
    """(cx, cy, major, minor, area, e cos 2phi, e sin 2phi) for each inclusion."""
    feats = []
    for (cx, cy, a, b, theta) in ellipses:
        major, minor = (a, b) if a >= b else (b, a)
        phi = theta if a >= b else theta + np.pi / 2       # direction of the major axis
        e = (major - minor) / (major + minor)
        feats.append([cx, cy, major, minor, np.pi * a * b, e * np.cos(2 * phi), e * np.sin(2 * phi)])
    return np.array(feats, dtype=np.float32)


def neighbor_edges(centers):
    """Directed edge list (both directions) between neighboring inclusion centers."""
    n = len(centers)
    pairs = set()
    if n == 2:
        pairs = {(0, 1)}
    elif n >= 3:
        try:
            for simplex in Delaunay(centers).simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u, v = sorted((int(simplex[i]), int(simplex[j])))
                        pairs.add((u, v))
        except QhullError:
            # all centers on one line (practically never happens with Lab 6's sampling): connect every pair
            pairs = {(u, v) for u in range(n) for v in range(u + 1, n)}
    edges = sorted(pairs)
    edges = edges + [(v, u) for (u, v) in edges]            # undirected: both directions
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)              # a single inclusion: an isolated node
    return np.array(edges, dtype=np.int64).T


def cell_to_graph(ellipses, e_app, row):
    if len(ellipses) == 0:
        raise ValueError(f"row {row} has no inclusions; Lab 6 cells always have at least two")
    x = node_features(ellipses)
    edge_index = neighbor_edges(x[:, :2])
    src, dst = edge_index
    d = x[dst, :2] - x[src, :2]
    dist = np.linalg.norm(d, axis=1)
    edge_attr = np.stack([d[:, 0], d[:, 1], dist, dist ** 2], axis=1).astype(np.float32)
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr).reshape(-1, 4),
        y=torch.tensor([e_app], dtype=torch.float32),
        row=torch.tensor([row], dtype=torch.long),
    )


def main():
    args = parse_args()
    print(f"Loading {args.input} ...")
    data_npz = np.load(args.input, allow_pickle=True)        # ellipses is a ragged object array
    ellipses_all = data_npz["ellipses"]
    moduli_all = data_npz["moduli"]
    print(f"  {len(ellipses_all)} cells, E_app range {moduli_all.min():.4f} -- {moduli_all.max():.4f}")

    graphs = [cell_to_graph(ell, float(m), i) for i, (ell, m) in enumerate(zip(ellipses_all, moduli_all))]
    assert len(graphs) == len(moduli_all)                     # one graph per row, in row order

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, args.out)
    print(f"Converted {len(graphs)} cells; saved {args.out}")

    n_nodes = np.array([g.num_nodes for g in graphs])
    n_edges = np.array([g.edge_index.shape[1] // 2 for g in graphs])
    counts = {k: int((n_nodes == k).sum()) for k in sorted(set(n_nodes.tolist()))}
    print(f"Nodes per graph (= inclusions): {counts}")
    print(f"Undirected edges per graph: min={n_edges.min()}, max={n_edges.max()}, mean={n_edges.mean():.2f}")


if __name__ == "__main__":
    main()
