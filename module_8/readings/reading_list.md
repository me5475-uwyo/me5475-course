# Module 8 — Reading List

M8's lectures: L34 Fri Nov 20, L35 Mon Nov 23, L36 Mon Nov 30 (no class Wed–Fri Nov 25–27). L34 condenses lectures 8–9
of Steve Sun's *Geometric Learning for Solid Mechanics* (2023); those slides are the instructor's source material and
are not distributed.

---

## Before starting Lab 8 (released Wed Nov 18; due Sun Dec 6, 11:59 PM MT)

### [Primary] PyTorch Geometric documentation, *Introduction by Example* tutorial.

pytorch-geometric.readthedocs.io (Get Started). `Data`, `edge_index` and mini-batching with `DataLoader`: what the
Lab 8 scripts use. PyG 2.7.0 is already in the course environment on ARCC: check that `import torch_geometric` works,
and do not `pip install` it there.

### [Primary] `module_8/readings/gnn_primer.md`.

---

## Before Lecture 35 (Mon Nov 23; assigned in Lecture 34, Fri Nov 20)

### [Primary] Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.

GCN paper. Read Sections 1–3.

### [Primary] Gilmer, Schoenholz, Riley, Vinyals & Dahl, *Neural Message Passing for Quantum Chemistry*, ICML 2017.

MPNN framework paper. Read Section 2 (the message-passing framework).

### [Optional] Vlassis, Ma & Sun, *Geometric deep learning for computational mechanics Part I: anisotropic hyperelasticity*, CMAME 371, 113299, 2020.

Polycrystals as graphs, in mechanics.

### [Optional] Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*, arXiv 2104.13478, 2021.

The unified view, tutorial style.

---

## Before Lecture 36 (Mon Nov 30; assigned in Lecture 35, Mon Nov 23)

### [Primary] Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia, *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021.

MeshGraphNet. Read Sections 1–3.

### [Primary] `module_8/readings/meshgraphnet_primer.md`.

### [Optional] Xu, Hu, Leskovec & Jegelka, *How Powerful are Graph Neural Networks?*, ICLR 2019. GIN paper.

### [Optional] Veličković, Cucurull, Casanova, Romero, Liò & Bengio, *Graph Attention Networks*, ICLR 2018. GAT paper.

### [Optional] Brody, Alon & Yahav, *How Attentive are Graph Attention Networks?*, ICLR 2022. GATv2, the attention layer Lab 8 uses.

---

## Reference

### Mechanics applications of GNNs

- Vlassis, Ma & Sun (above).
- Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec & Battaglia, *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020. Particle-based simulators (GNS): fluids, rigid solids, deformable materials.
- Allen, Rubanova, Lopez-Guevara, Whitney, Sanchez-Gonzalez, Battaglia & Pfaff, *Learning Rigid Dynamics with Face Interaction Graph Networks*, ICLR 2023.

### Advanced GNN architectures (for the final project)

- Battaglia et al., *Relational inductive biases, deep learning, and graph networks*, arXiv 1806.01261, 2018. The unifying perspective.
- E(3)-equivariant GNNs for atomistic simulation: Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*, Nature Communications 13, 2453, 2022 (NequIP); Batatia, Kovács, Simm, Ortner & Csányi, *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022.
- Fuchs, Worrall, Fischer & Welling, *SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks*, NeurIPS 2020. Equivariant attention.

### Software

- `torch_geometric` (PyG) — the standard. Used in Lab 8 (2.7.0 on ARCC).
- `dgl` (Deep Graph Library) — alternative; comparable functionality.
- `jraph` (JAX) — for JAX users; archived (read-only) by its owner in May 2025.
