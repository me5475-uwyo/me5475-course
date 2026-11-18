# Starter Prompts for Lab 8

Lead and review in **separate sessions, no shared context**. Logged prompts mandatory.

Every Lab 8 job is a CPU job on ARCC partition `mb`, submitted with `gnn_train.sbatch` (which sources the course
setup itself). PyTorch Geometric is already in the course environment: if an agent tells you to `pip install
torch_geometric` there, or to move the tensors to a GPU, do not follow it.

---

## Task 1 — Graph conversion verification

**Lead prompt (after running microstructure_to_graph.py):**

```
I converted my Lab 6 dataset (<N> cells) to graphs with microstructure_to_graph.py: one
graph per cell, in row order. Verify the conversion is correct:

1. Each graph's node count = the number of inclusions in that cell's Lab 6 `ellipses`
   entry (between 2 and 5).
2. Edges: two inclusions -> the single pair; three or more -> the Delaunay triangulation
   of the centers, which for n points with h on the convex hull has 3n - 3 - h
   undirected edges (3 for n = 3, 5 or 6 for n = 4, 7 to 9 for n = 5). Each undirected
   edge appears twice in edge_index, once in each direction.
3. No graph has self-loops (edge_index[0] != edge_index[1] everywhere).
4. Node features x are (n, 7): cx, cy, major, minor, area = pi*major*minor,
   e*cos(2 phi), e*sin(2 phi), with major >= minor, e = (major - minor)/(major + minor)
   in [0, 1), and phi the direction of the major axis.
5. Edge features edge_attr are (n_edges, 4): dx, dy, dist, dist^2, with (dx, dy) = target
   center - source center, dist = sqrt(dx^2 + dy^2), and the reverse edge carrying
   (-dx, -dy).
6. y is the cell's E_app (Lab 6's `moduli` at that row) and row is its Lab 6 row.

Spot check 3 random graphs by:
  - Counting nodes vs the cell's inclusion count.
  - Counting undirected edges against 3n - 3 - h.
  - Checking edge_attr[k] consistency and the node features against the ellipse.
```

**Review prompt:**

```
Read the lead's verification. Independently confirm:

1. Delaunay needs at least three non-collinear points: the converter uses the single
   pair for two inclusions and connects every pair if the centers are collinear
   (QhullError), and no graph has self-loops.
2. The conversion preserves the labels: graph i has row == i, y == moduli[i] (E_app),
   and as many nodes as ellipses[i] has inclusions (no shuffling between the cell list
   and the E_app list).
3. The graph file stores raw features (centers in [0, 1], E_app in Lab 6's units);
   standardization happens later, in gnn_microstructure.py, with training-set
   statistics only. Nothing in the converter uses statistics of the whole dataset.

PASS / FAIL each.

Verification:
<paste>
```

---

## Task 4 — CNN vs GNN comparison

The critical methodology check.

**Lead prompt:**

```
Write comparison.py for Lab 8 Task 4. Everything is in cnn_vs_gnn/:
  - my Lab 6 CNN checkpoint cnn_modulus.pt, cnn_microstructure_to_modulus.py, and
    rve_dataset.npz
  - my Lab 8 checkpoints gin_modulus.pt and gat_modulus.pt

For the comparison to be fair, all models must be scored on the SAME test rows, in the
same order. Do not rebuild the split by hand: read it from the checkpoints.

The script:
1. Loads the CNN checkpoint with torch.load(path, map_location="cpu", weights_only=False)
   and each GNN checkpoint with torch.load(path, map_location="cpu"), reads
   ckpt["split"]["test_idx"] from each, and stops with an error if they differ.
2. Rebuilds MicrostructureCNN from cnn_microstructure_to_modulus.py with the
   checkpoint's "args", loads "model_state", predicts the images at those test rows
   (as Lab 6 fed them: float32, shape (N, 1, 64, 64)), and converts the output back to
   E_app with the checkpoint's scalers mu_y and sd_y.
3. Takes each GNN's test predictions from ckpt["test_pred"] (already in E_app units, in
   test_idx order).
4. Writes, for each model, MAE, MAE / mean(E_app) over the test rows, and the largest
   relative error to comparison.txt.
5. Plots comparison_plot.png: CNN parity | GNN parity | the MAEs side by side.
```

**Review prompt:**

```
Read the script. Verify:

1. It reads split.test_idx from every checkpoint and stops if they differ; it does not
   recompute the split.
2. The CNN sees the images exactly as Lab 6's script fed them (no image
   standardization), and its output is converted back with the CNN checkpoint's own
   mu_y and sd_y.
3. The GNN predictions come from ckpt["test_pred"], already in E_app units: they are
   not un-standardized a second time, and they are paired with the true E_app of the
   same rows in the same order.
4. The "MAE / mean(E_app)" formula divides by the mean of the test set's true E_app
   values, not the mean of train+test.
5. The MAE it reports for each model matches that checkpoint's stored test_mae (to
   rounding).

PASS / FAIL each.

Script:
<paste>
```

---

## Meta-tip — before you explain the gap

Whichever model has the lower test error, check these before you write the explanation (Task 4d) or the reflection
(Task 6):

1. **Did both converge?** Read each training-history plot: has the validation loss leveled off, and did training stop
   on the patience rule rather than the epoch limit? A best epoch at the very end is a prompt to look closer at the
   trend, not proof that training failed.
2. **Are the test rows identical?** Compare `split.test_idx` across the CNN and GNN checkpoints; `comparison.py` should
   refuse to run otherwise.
3. **Is the gap larger than the seed spread?** Compare it with your seed reruns: Task 3's seed-42 and seed-7 runs of both GNNs,
   Lab 6 Task 3 for the CNN.
4. **What does each representation carry?** The graph has each inclusion's exact center, semi-axes, area and
   orientation, plus the offsets between neighbors, but no pixels. The CNN sees a 64 × 64 rasterized image of the same
   cell, with no explicit list of inclusions.

Document either outcome honestly in your reflection.
