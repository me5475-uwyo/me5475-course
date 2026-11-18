# Lab 8 — Microstructure as a Graph: GNN vs CNN

**Module:** 8 — Graph Neural Networks
**Released:** Wed Nov 18, 2026 (at L33)
**Due:** Sun Dec 6, 2026, 11:59 PM Mountain Time
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_8` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

> **Spans Thanksgiving and the final presentations.** Lab 7 is due Mon Nov 23; there is no class Wed–Fri Nov 25–27;
> final-project presentations are Wed Dec 2 and Fri Dec 4. The compute is small (CPU jobs, no MOOSE: you reuse your
> Lab 6 data), so plan to finish the main tasks before the break. M8's lectures run alongside: graphs and the GCN
> rule L34 (Fri Nov 20), message passing and MeshGraphNet L35 (Mon Nov 23), the GNN pipeline L36 (Mon Nov 30).

---

## What this lab does

Lab 6 represented each cell as a 64 × 64 image and trained a CNN to predict its apparent modulus E_app (the
mixed-boundary plane-strain response MOOSE computed; Lab 6 explains exactly what it is). Lab 8 represents **the same
cells** as graphs: one node per inclusion, with that inclusion's exact center, size, shape and orientation as node
features, and edges between neighboring inclusions. You train two graph neural networks (GIN and GATv2) on the graphs
and compare them with your Lab 6 CNN **on exactly the same test cells**.

The two representations carry different information: the graph has the exact ellipse parameters the generator drew
but no pixels; the CNN sees a rasterized picture. Which predicts E_app better is your measurement; no answer is given
here, and an honest comparison, whichever way it comes out, is the result.

---

## Before you start

- **Get the files.** `git checkout main && git pull`. You need, in `module_8/examples/`: `microstructure_to_graph.py`,
  `gnn_microstructure.py`, `gnn_train.sbatch`, and for the stretch task `meshgraphnet_demo.py`. Readings are in
  `module_8/readings/`; agent prompts in `module_8/homework/starter_prompts.md`.
- **You need two Lab 6 files:** your dataset `labs/lab_6/<you>/data_generation/rve_dataset.npz` and your Task 2 CNN
  checkpoint `labs/lab_6/<you>/cnn_training/checkpoints/cnn_modulus.pt` (trained with the default `--split-seed 42`).
- **Work on ARCC.** Everything runs as CPU jobs on partition `mb`. PyTorch Geometric is already in the course
  environment; do not `pip install` it there.
- **Set your username once per terminal session:**

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first: your GitHub username, no angle brackets
```

---

## Deliverables

```
labs/lab_8/<your-github-username>/
├── data_conversion/
│   ├── microstructure_to_graph.py, gnn_train.sbatch
│   ├── microstructure_graphs.pt                one graph per Lab 6 cell
│   ├── graph_visualization.png                 four cells drawn as graphs (Task 1b)
│   └── prompts_data.md
├── gin_training/
│   ├── gnn_microstructure.py, gnn_train.sbatch
│   ├── checkpoints/gin_modulus.pt, checkpoints/gin_modulus.png
│   ├── checkpoints/gin_seed7.pt, checkpoints/gin_seed7.png    (Task 3 seed check)
│   ├── m8_gnn-<jobid>.out                      one per run
│   └── prompts_gin.md
├── gat_comparison/
│   ├── checkpoints/gat_modulus.pt, checkpoints/gat_modulus.png
│   ├── checkpoints/gat_seed7.pt, checkpoints/gat_seed7.png
│   ├── m8_gnn-<jobid>.out                      one per run
│   └── prompts_gat.md
├── cnn_vs_gnn/
│   ├── comparison.py                            your script (Task 4b)
│   ├── comparison.txt
│   ├── comparison_plot.png
│   └── prompts_comparison.md
├── stretch_meshgraphnet/                        (optional, +10)
│   ├── meshgraphnet_demo.png, m8_gnn-<jobid>.out
│   └── prompts_meshgraphnet.md
└── reflection.md                                ≤ 400 words
```

---

## Tasks

### Task 1 — Convert the cells to graphs (20 points)

**1a.** Copy the scripts and your Lab 6 dataset, then convert:

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first
cd ~/me5475-"$U" && git checkout main && git pull
git checkout -b lab_8
mkdir -p labs/lab_8/"$U"/{data_conversion,gin_training,gat_comparison,cnn_vs_gnn,stretch_meshgraphnet}
cd labs/lab_8/"$U"/data_conversion
cp ../../../../module_8/examples/{microstructure_to_graph.py,gnn_train.sbatch} .
cp ../../../lab_6/"$U"/data_generation/rve_dataset.npz .
sbatch gnn_train.sbatch convert
```

The printout reports the number of cells (it must equal your Lab 6 count: one graph per cell, in row order), how many
cells have 2, 3, 4 and 5 inclusions, and the edges per graph.

**1b.** Draw four cells as graphs: each inclusion as its ellipse (center, semi-axes, orientation from your Lab 6
`ellipses`), the edges as line segments between centers. Commit `graph_visualization.png`. In two sentences: what does
the graph keep that the 64 × 64 image does not, and what does the image keep that the graph does not?

### Task 2 — Train the GIN (30 points)

```bash
cd ../gin_training
cp ../../../../module_8/examples/{gnn_microstructure.py,gnn_train.sbatch} .
cp ../data_conversion/microstructure_graphs.pt .
sbatch gnn_train.sbatch gin
```

The script uses **Lab 6's split** (the same 70 / 15 / 15 rows as your CNN with `--split-seed 42`), keeps the weights of
the best validation epoch, and evaluates the test set once. Report the parameter count, the test MAE and MAE / mean(E),
and read the parity plot: where along the E_app range is it worst?

### Task 3 — Train the GATv2 for comparison (15 points)

```bash
cd ../gat_comparison
cp ../gin_training/{gnn_microstructure.py,gnn_train.sbatch,microstructure_graphs.pt} .
sbatch gnn_train.sbatch gat
```

**Seed check, decided in advance for both models.** One run each is one sample, so rerun **both** architectures with
`--seed 7` (initialization and minibatch order only; `--split-seed 42` and the split stay), each to its own file:

```bash
cd ../gin_training && sbatch gnn_train.sbatch gin --seed 7 --out checkpoints/gin_seed7.pt
cd ../gat_comparison && sbatch gnn_train.sbatch gat --seed 7 --out checkpoints/gat_seed7.pt
```

Report all four runs (GIN and GATv2, seeds 42 and 7) in one table: test MAE / mean(E), parameter count, and training
time (the `elapsed:` line; say which node ran each, from `sacct -j <jobid> -o JobID,Elapsed,NodeList`). Do not pick a
"best" seed by its test error. Is the GIN–GATv2 gap larger than the seed-to-seed difference within each architecture?
Commit the two seed-7 checkpoints, plots and printouts beside the main ones.

### Task 4 — CNN vs GNN on the same test cells (25 points)

**4a.** Copy your Lab 6 CNN checkpoint, its script and your dataset into `cnn_vs_gnn/`, with both GNN checkpoints.

**4b.** Write `comparison.py` (agents welcome; log the prompts) that:

1. reads `ckpt["split"]["test_idx"]` from the CNN checkpoint and from each GNN checkpoint, and **stops with an error
   if they differ** (same rows, same order);
2. rebuilds your CNN from `cnn_microstructure_to_modulus.py` (`MicrostructureCNN`, with the checkpoint's `args`),
   predicts the test images, and converts back to E_app with the checkpoint's `scalers`;
3. takes the GNN test predictions from the GNN checkpoints (`ckpt["test_pred"]`, in the same row order);
4. reports, for each model, MAE, MAE / mean(E_app) and the largest relative error, in `comparison.txt`.

Run it as a short job: copy `gnn_train.sbatch` to `comparison.sbatch` and change its `srun` line to
`srun python comparison.py`.

**4c.** Make `comparison_plot.png`: CNN parity | GNN parity | the MAEs side by side.

**4d.** In `comparison.txt`: which model has the lower test error, by how much, and is the gap larger than the
seed-to-seed variation you measured (Task 3 here, Lab 6 Task 3 for the CNN)? Then explain the result from what each
representation carries and how each network uses it. Compare with your Lab 6 area-fraction baseline if it helps.

### Task 5 (optional, +10 bonus) — A MeshGraphNet on a mass-spring membrane

```bash
cd ../stretch_meshgraphnet
cp ../../../../module_8/examples/{meshgraphnet_demo.py,gnn_train.sbatch} .
sbatch gnn_train.sbatch meshgraph
```

The demo simulates a toy mass-spring membrane (not a finite-element model), trains a MeshGraphNet-style network to
predict each free node's acceleration, and **rolls out** unseen test trajectories from their initial states, feeding
its own predictions back. With the defaults it takes about 15 minutes on a CPU node (`module_8/readings/measured_results.md`
§2). Commit `meshgraphnet_demo.png` and the printout. Compare the one-step test error with the
rollout error: why do they differ, and how does the rollout error grow with the step? Then rerun with
`--noise-std 0.003 --out meshgraphnet_noise.pt` and report what changes. This option is a **simplified augmentation
ablation**: it adds noise to the training input velocities but leaves the acceleration targets unchanged. It is *not*
the noise-correcting training of Pfaff et al. (2021, Appendix A.2.2), which also adjusts the targets so the model learns
to undo the noise. Does it improve the rollout error, worsen it, or leave it about the same, and why might that be?

### Task 6 — Reflection (5 points, ≤ 400 words)

1. **GIN vs GATv2.** Over the four runs, how large is the gap between architectures, and is it larger than the
   seed-to-seed difference within each?
2. **GNN vs CNN.** Report the numbers from Task 4. Explain the result from the two representations.
3. **When would a GNN clearly win?** Sketch a mechanics problem where you would expect a graph model to beat an image
   model, and say why (irregular meshes, varying resolution, many small features, …).
4. **One thing your prompts caught.** As in every lab.

---

## Grading rubric (out of 100, +10 bonus)

| Component | Points |
|---|---|
| Task 1: graphs converted, four drawn, keep/lose answered | 20 |
| Task 2: GIN trained, metrics and parity read | 30 |
| Task 3: GATv2 trained; both architectures rerun with seed 7; all four runs reported | 15 |
| Task 4: CNN vs GNN on identical test cells, explained | 25 |
| Task 6: reflection, all four questions answered concretely | 5 |
| Prompts logged in each sub-folder | 5 |
| Task 5 (optional): MeshGraphNet rollout | +10 bonus |

---

## Hints

- **If a job ends within seconds**, read both `.err` and `.out`. The job script sources the course setup itself, so it
  does not matter whether you sourced it before `sbatch`.
- **Arguments after the variant go to the script:** `sbatch gnn_train.sbatch gin --hidden 32 --out
  checkpoints/gin_h32.pt`. Use a new `--out` for each variant so you do not overwrite your main result.
- **The split.** Keep `--split-seed 42` (the default) in both labs; `--seed` changes only initialization and order.
  The graph file must hold one graph per Lab 6 row; the training script checks this.
- **Graph sizes.** Lab 6 cells have 2 to 5 inclusions, so each graph has 2 to 5 nodes; three or more centers are joined
  by their Delaunay triangulation, two by a single edge.
- **No Lab 6 dataset or CNN checkpoint?** Regenerate them with Lab 6's own commands: Task 1a and 1b in a
  `data_generation/` folder (the default seeds give the standard dataset, in a few minutes of ARCC jobs), then Task 2's
  `sbatch cnn_train.sbatch modulus` in `cnn_training/`. Use the default `--split-seed 42`.
- **Loading checkpoints.** The GNN checkpoints load with `torch.load(path, map_location="cpu")`. The Lab 6 CNN
  checkpoint loads with `torch.load(path, map_location="cpu", weights_only=False)`.

---

## What this lab leads into

- **L35–L36** connect it to message passing and MeshGraphNet on real meshes.
- **Final project.** GNN surrogates on FE meshes are a natural direction; Lab 8 is the prerequisite.

---

## Academic integrity

Agents are encouraged. Logged prompts in each sub-folder are mandatory. Your data, networks and numbers must be your
own runs; copying a classmate's files is plagiarism. Discussing strategy with classmates is fine and encouraged.

---

## How to submit

A pull request from branch `lab_8`, with your files under `labs/lab_8/<your-github-username>/`
(`module_0/readings/git_submission_guide.md` walks through it).

```
U=YOUR_GITHUB_USERNAME            # EDIT first, as above
cd ~/me5475-"$U"                  # the repository root: the paths below start here
git checkout lab_8                # the branch you made in Task 1a
printf 'm8_gnn-*.err\n__pycache__/\n*/rve_dataset.npz\ngin_training/microstructure_graphs.pt\ngat_comparison/microstructure_graphs.pt\ncnn_vs_gnn/*.pt\nstretch_meshgraphnet/*.pt\n' > labs/lab_8/"$U"/.gitignore
git add labs/lab_8/"$U"
git commit -m "Lab 8: what you did"
git push -u origin lab_8
# then open the pull request on github.com -- that PR is the submission
```
