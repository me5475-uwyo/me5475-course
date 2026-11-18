# Module 8 — Graph Neural Networks for Mesh-Based Mechanics

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 3 lectures, Fri Nov 20 → Mon Nov 30 (no class Nov 25–27)
**Status:** Audited Nov 15, 2026 `[fallback]` (pending Reviewer)

---

## What this module does

The three-lecture GNN overview the syllabus locked in. By the end of M8 students can:

- State why GNNs are the right tool for *irregular* mesh-based mechanics.
- Build a small message-passing GNN (GIN or GATv2) in PyTorch Geometric.
- Apply a GNN to the M6 microstructure dataset, treating each microstructure as a graph rather than an image.
- Compare GNN vs CNN on exactly the same test cells.
- Identify when MeshGraphNet-style surrogates are the right choice for FE-mesh-based mechanics (vs CNN/U-Net for image-based, or FNO for regular-grid).

This module is intentionally an overview. L34 condenses lectures 8–9 of Steve Sun's *Geometric Learning for Solid Mechanics* (2023), the instructor's source material, which is not distributed to students; the course adapts the high-value subset (per the early planning decision).

**Module deliverable (Lab 8)** (released Wed Nov 18, due Sun Dec 6, 11:59 PM MT). Represent each Lab 6 cell as a graph (one node per inclusion, carrying its exact center, semi-axes, area and orientation; edges between neighboring inclusions), train GIN and GATv2 regressors for the apparent modulus E_app, and compare them with your Lab 6 CNN on exactly the same test cells. Optional stretch: a MeshGraphNet-style rollout on a toy mass-spring membrane.

---

## Where this module fits

**Back to M6:** Lab 8 reuses the M6 microstructure dataset. The image-vs-graph representation comparison is the pedagogical anchor.

**Back to M7:** GNN is the natural complement to FNO for irregular meshes. Together M7+M8 cover "surrogates for PDE solutions": FNO for regular grids, GNN for arbitrary meshes.

**Connection to Steve Sun's course:** L34 adapts lectures 8 and 9 of his *Geometric Learning for Solid Mechanics* (graph learning overview; GCN for solid systems). His later lectures on GCNs (Part II) and graph pooling are out of scope for our three-lecture treatment.

**Forward to final project:** GNN surrogates on FE meshes are a natural final-project direction. Lab 8 is the prerequisite.

---

## File index

```
module_8/
├── README.md
├── lectures/
│   ├── L34_graphs_in_mechanics.md                  (Fri Nov 20) why graphs, GCN essentials
│   ├── L35_message_passing_meshgraphnets.md        (Mon Nov 23) MPNN, GIN, GAT, MeshGraphNets
│   └── L36_gnn_surrogates_lab.md                   (Mon Nov 30) Lab 8 walkthrough
├── examples/
│   ├── microstructure_to_graph.py                   Lab 6 dataset -> one graph per cell (7 node, 4 edge features)
│   ├── gnn_microstructure.py                         GIN / GATv2 for E_app, on Lab 6's split
│   ├── meshgraphnet_demo.py                          MeshGraphNet-style rollout on a toy mass-spring membrane
│   └── gnn_train.sbatch                              ARCC SLURM: convert | gin | gat | meshgraph (CPU, partition mb)
├── homework/
│   ├── lab_8.md                                      microstructure-as-graph: GNN vs CNN
│   └── starter_prompts.md
└── readings/
    ├── reading_list.md
    ├── gnn_primer.md                                 message passing, GCN, GIN, GAT/GATv2, PyG idioms
    └── meshgraphnet_primer.md                        Pfaff et al. (ICLR 2021) architecture
```

---

## Acknowledgments

L34 is adapted from Steve Sun's *Geometric Learning for Solid Mechanics* (2023):

- L34 (graphs in mechanics) condenses his lecture 8 motivation slides and definitions.
- L34's GCN section condenses his lecture 9 spectral derivation.
- The MeshGraphNet treatment uses Pfaff et al. (ICLR 2021) as the primary citation.

What we *don't* cover (relative to his full coverage): spectral GCN derivation in depth, graph pooling, advanced GNN architectures (E(3)-equivariant GNNs, MACE/NequIP). These are forward references for the final project.

---

## Open items

- PyTorch Geometric on ARCC: resolved. torch 2.5.1 and torch_geometric 2.7.0 are in the course environment (torch_scatter, torch_sparse and torch_cluster are not installed and not needed); nobody `pip install`s into it. Locally, `pip install torch_geometric` is enough. Convert, GIN, GATv2 and the MeshGraphNet demo were smoke-tested on ARCC (Nov 15, 40-cell demo set; the full 500-cell pipeline with the rewritten scripts the same day, `readings/measured_results.md`).
- Lab 8 depends on each student's Lab 6 dataset and CNN checkpoint. Fallback (no shared copy needed): rerun Lab 6
  Task 1a–1b and Task 2 with the default seeds (Lab 8 handout, Hints).
