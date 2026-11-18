# Module 8 — Measured results

Every number quoted in M8 lectures, readings and Canvas pages comes from a run recorded here. Format follows
`module_0/readings/measured_results.md`. Accuracy, GIN-vs-GAT and GNN-vs-CNN comparisons, and training-time
comparisons are Lab 8 answers: they are held in `team/reviews/2026-11-15_M8_lab8_HELD_until_Dec6/` until Lab 8
closes (Sun Dec 6, 11:59 PM MT) and are not quoted anywhere before then.

## §1 · Environment (ARCC, Nov 15, 2026)

Course environment via `source /project/me5475/setup5475.sh`: torch 2.5.1, torch_geometric 2.7.0, scipy 1.17.1,
networkx 3.6.1; torch_scatter, torch_sparse and torch_cluster are not installed, and the Lab 8 scripts do not need
them (GINEConv, GATv2Conv, global pooling and the PyG DataLoader run without them).

## §2 · Lab 8 pipeline, run as students run it (ARCC, Nov 15, 2026)

- **Data:** Lab 6's `data_generation.sbatch` with its default seeds (job array 25244994, ten workers, 48–56 s each)
  and `aggregate_parts.py` (job 25245006): 500 cells, the standard Lab 6 dataset.
- **Conversion:** `gnn_train.sbatch convert` (job 25245007, 19 s elapsed): 500 graphs, one per row; cells with
  2 / 3 / 4 / 5 inclusions: 136 / 134 / 126 / 104; undirected edges per graph 1–9, mean 3.99.
- **GIN** (`gnn_train.sbatch gin`, job 25245008) and **GATv2** (`gnn_train.sbatch gat`, job 25245009): both loaded
  500 graphs, split 350 / 75 / 75 with split seed 42; first five test rows [473, 132, 472, 46, 40]; GIN 39,301
  parameters, GATv2 39,553 parameters; both stopped early on validation. Their errors and times are held.
- **Lab 6 CNN on the same split** (`cnn_train.sbatch modulus`, job 25245010, node mba30-005, `Device: cuda (NVIDIA
  A30)`): 97,761 parameters; the same first five test rows. Its error is a Lab 6 result.
- **Split check:** the CNN, GIN and GATv2 checkpoints store identical `split.test_idx` (75 rows); the GNN checkpoints
  load with plain `torch.load` and hold `test_pred` (75 values each), as Lab 8 Task 4 needs.
- **MeshGraphNet stretch** (`gnn_train.sbatch meshgraph`, defaults, job 25245011, node mbcpu-008): membrane of 100
  nodes and 342 springs; 100 trajectories × 50 steps, split by trajectory 80 / 20; 305,410 parameters; 30 epochs;
  exited 0 after 15 min 07 s (inside the job's 1 h limit), writing `meshgraphnet_demo.pt` and `.png`. Its one-step
  and rollout errors are held.
- **Earlier smoke test** (job 25244750, the L30 demo's 40-cell set, original scripts): conversion, GIN, GAT and the
  old MeshGraphNet demo all exited 0 (before the Nov 15 rewrite).
