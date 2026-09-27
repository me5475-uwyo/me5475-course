# Lab 3 — PINN for Plate-with-Hole, End-to-End

**Module:** 3 — PINNs for Solid Mechanics
**Released:** Mon Sep 28, 2026 (the day Lecture 12 is taught)
**Due:** Mon Oct 19, 2026, 11:59 PM Mountain Time (the day Lecture 20 is taught)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_3` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

---

## What this lab does

Lab 3 is the spine of Module 3. You solve the **same** plate-with-hole problem five ways:

1. A forward PINN with soft boundary conditions — the course's PyTorch port of Min Lin's notebook.
2. A quantitative comparison against a converged **MOOSE reference of exactly the same problem**.
3. The same PINN with **hard** (exactly imposed) displacement conditions — a controlled comparison.
4. An Optuna-tuned PINN configuration (the Lab 2 pattern, on GPUs).
5. *(stretch)* An inverse problem — recovering E and ν from sparse displacement measurements **plus one measured force**.

**The problem, everywhere in this lab:** the square [0, 1] × [0, 1] minus a quarter hole of radius 0.1 at the origin; plane-strain linear elasticity, E = 1, ν = 0.3. Symmetry on the left edge (u₁ = 0, σ₁₂ = 0) and the bottom edge (u₂ = 0, σ₁₂ = 0); the right edge is pulled to u₁ = 1 with σ₁₂ = 0; the top edge and the hole are traction-free. **Ten boundary conditions — two per edge, two on the hole.**

---

## Before you start

**Every training step is a submitted job.** Anything that trains a network runs as a batch job on a GPU node, never on the login node ([ARCC HPC policy](https://www.uwyo.edu/arcc/policies/hpcpolicy.html)). The job scripts are in `module_3/examples/` and in `/project/me5475/examples/`:

- **Tasks 1 and 3** — `pinn_train.sbatch` (one PINN, one GPU; about 10 minutes on an A30);
- **Task 4** — `pinn_optuna_sweep.sbatch` (six workers, **at most two GPUs at a time**);
- **Task 5** — `pinn_inverse_ensemble.sbatch` (ten seeds, **at most two GPUs at a time**).

**Course rule: never run more than two of your GPU jobs at once.** The array scripts enforce it with `%2`; do not remove it, and do not submit a second array while one is running. The GPU partitions are shared with the whole university.

On the login node you only copy files, extract reference data, submit jobs (`sbatch`), watch them (`squeue -u $USER`) and run the short analyses. Log in, update your repository, and activate the course environment:

```bash
cd ~/me5475/me5475-<your-github-username>
git pull
source /project/me5475/setup5475.sh
```

**Every command block below starts at the root of your course repository.** From `labs/lab_3/<your-github-handle>/<task-folder>/`, the repository root is `../../../../`.

**Shared reference files** (read-only; copy from, never write into): `/project/me5475/examples/lab3_reference/` holds the MOOSE reference solution (`plate_square_hole_reference_out.e`), its input file, and its measured quantities (`plate_square_hole_reference_out.csv`). How it was built and checked: `module_3/readings/measured_results.md` §1.

---

## Deliverables

```
labs/lab_3/<your-github-handle>/
├── task1_port/
│   ├── plate_with_hole_fixed.py        your copy (possibly modified)
│   ├── pinn_train.sbatch
│   ├── pinn-<jobid>.out                the job's log
│   ├── plate_with_hole_fixed.png       u_1, u_2, sigma fields and von Mises
│   ├── plate_with_hole_fixed.loss.npz  loss history
│   └── prompts_task1.md
├── task2_moose_comparison/
│   ├── reference_grid.csv              from extract_moose_reference.py
│   ├── comparison_script.py
│   ├── comparison_plot.png             PINN vs MOOSE, side by side
│   ├── comparison_metrics.txt
│   └── prompts_task2.md
├── task3_hard_bc/
│   ├── plate_with_hole_hard_bc.py      your copy
│   ├── ansatz_check.txt                the three displacement identities, checked numerically
│   ├── soft_vs_hard.png                comparable loss components, both runs
│   ├── soft_vs_hard_metrics.txt
│   └── prompts_task3.md
├── task4_optuna_sweep/
│   ├── pinn_optuna_sweep.py, pinn_optuna_sweep.sbatch
│   ├── pinn_opt.db                     the Optuna study
│   ├── pinn_sweep-*.out                a few worker logs (all show the same split fingerprint)
│   ├── study_summary.txt, parallel_coords.html, optimization_history.html
│   ├── retrain_best.py, retrain_log.txt
│   └── prompts_task4.md
├── task5_inverse/                      (stretch — recommended, +20 bonus)
│   ├── measurements.csv
│   ├── inverse_results_seed_*.txt      one per seed
│   ├── ensemble_summary.txt, ensemble_plot.png
│   └── prompts_task5.md
└── reflection.md                       <= 600 words
```

---

## Tasks

### Task 1 — Port Min's notebook and run it on ARCC (20 points)

**1a.** Read Min Lin's original notebook, `module_3/examples/min_lin_2D_hole_example.ipynb` (TensorFlow backend, outputs cleared — read it, you do not need to run it). Find its ten boundary conditions. One deliberate difference in its physics: its constants C₁₁ = E/(1 − ν²), C₁₂ = Eν/(1 − ν²) are **plane stress**; this lab's problem, the port and the MOOSE reference are **plane strain**. Keep the port plane strain.

**1b.** Read the course's PyTorch port, `module_3/examples/plate_with_hole_fixed.py`. Check that it imposes the **same ten** conditions, and understand every line before you submit it. (An earlier version of this port dropped the three σ₁₂ = 0 conditions on the left, bottom and right edges. With them missing, the PINN reached the same tiny training loss but a displacement error of about 60 % and almost no stress concentration at the hole — `measured_results.md` §2. A small PINN loss does not certify that the problem was posed right.)

**1c.** Train it as a job:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_3/<your-github-handle>/task1_port
cd labs/lab_3/<your-github-handle>/task1_port
cp ../../../../module_3/examples/plate_with_hole_fixed.py ../../../../module_3/examples/pinn_train.sbatch .
sbatch pinn_train.sbatch            # Adam 50 000 + L-BFGS; about 10 minutes on an A30
squeue -u $USER
```

**1d.** Submit the figure the job writes (`plate_with_hole_fixed.png`: u₁, u₂, σ₁₁, σ₂₂, σ₁₂ and the plane-strain von Mises stress).

### Task 2 — Compare with the MOOSE reference (15 points)

**2a.** Extract the reference on a 200 × 200 grid (only points inside the plate are kept; this takes seconds and is not training):

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_3/<your-github-handle>/task2_moose_comparison
cd labs/lab_3/<your-github-handle>/task2_moose_comparison
python /project/me5475/examples/extract_moose_reference.py --mode grid --nx 200 --ny 200 --output reference_grid.csv
```

`reference_grid.csv` has columns `x_1, x_2, u_1, u_2, sigma_11, sigma_22, sigma_12, sigma_vm`.

**2b.** Write `comparison_script.py`. It rebuilds the problem, loads your trained network, and evaluates it on the grid points. The checkpoint holds the L-BFGS optimizer state, so load only the network weights:

```python
import sys, glob
from pathlib import Path
sys.path.insert(0, "../task1_port")
import numpy as np, pandas as pd, torch, deepxde as dde
from plate_with_hole_fixed import build_problem

data, net = build_problem()
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)                      # any compile; we only predict
ckpt = sorted(glob.glob("../task1_port/plate_with_hole_fixed.pt-*.pt"))[-1]
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
ref = pd.read_csv("reference_grid.csv")
pred = model.predict(ref[["x_1", "x_2"]].values)    # columns: u_1, u_2, sigma_11, sigma_22, sigma_12
```

**2c.** Report in `comparison_metrics.txt`:

- the **relative L2 error** ‖PINN − MOOSE‖ / ‖MOOSE‖ over the grid, for (u₁, u₂) together and for each stress component — field norms, not pointwise relative errors (those blow up where the reference stress is near zero);
- the **stress at the top of the hole**: evaluate your PINN exactly at (x₁, x₂) = (0, 0.1) and compare with the reference's value there, σ₁₁ = **3.2418** (MOOSE, `measured_results.md` §1). Say where you evaluated.

Plot PINN and MOOSE side by side for each field (`comparison_plot.png`).

**2d.** Discuss: how close is the PINN? Which quantity is hardest for it, and where on the plate are its errors largest?

### Task 3 — Hard boundary conditions: a controlled comparison (15 points)

**3a. What hard BCs guarantee.** `module_3/examples/plate_with_hole_hard_bc.py` builds the displacement conditions into the network's output (L15):

    u₁ = x₁ + x₁ (1 − x₁) N₁(x),      u₂ = x₂ N₂(x)

Show on paper that u₁ = 0 on x₁ = 0, u₁ = 1 on x₁ = 1 and u₂ = 0 on x₂ = 0 for **any** network outputs N₁, N₂. Then check it numerically: feed random raw outputs through `output_transform` at points on those three edges and print the largest violation (`ansatz_check.txt`). Finally, list the **seven** conditions that are still in the loss — the transform fixes only u₁ and u₂, so all the traction conditions (including σ₁₂ = 0 on the left, bottom and right) remain soft. Ten soft conditions become seven.

**3b. Train both, controlled.** Run the soft and the hard PINN with **the same seed**. The two scripts then use the same 4 000 + 600 collocation points, the same starting weights, the same network and the same optimizer schedule (Adam 50 000 + L-BFGS), so the *only* difference is how the three displacement conditions are enforced:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_3/<your-github-handle>/task3_hard_bc
cd labs/lab_3/<your-github-handle>/task3_hard_bc
cp ../../../../module_3/examples/plate_with_hole_fixed.py ../../../../module_3/examples/plate_with_hole_hard_bc.py \
   ../../../../module_3/examples/pinn_train.sbatch .
SEED=43 sbatch pinn_train.sbatch            # soft
SEED=43 sbatch pinn_train.sbatch hard_bc    # hard
```

**3c. Measure** (`soft_vs_hard_metrics.txt`, `soft_vs_hard.png`), for both runs:

- relative L2 error of displacement and of each stress component against `reference_grid.csv` (Task 2's script works for both; import `build_problem_hard_bc` for the hard run);
- the largest displacement-condition error on **freshly sampled** points of the three edges (not the training points) — the hard run should give zero; what does the soft run give?
- σ₁₁ at (0, 0.1), as in Task 2;
- training cost: wall time of each phase (the job log prints `'train' took … s` twice — Adam, then L-BFGS) and the number of L-BFGS steps (`steps` in the `.loss.npz`);
- loss curves of the **comparable components only**. Both `.loss.npz` files store one column per term. The first five are the PDE residuals in both. The seven traction terms are columns 8–14 in the soft file and 5–11 in the hard file; the soft file's columns 5–7 are the three displacement conditions the hard run does not have. Do not compare the two *total* losses: they sum different terms.

**3d. Conclude, from your numbers:** What did hard enforcement *guarantee*? Which measured quantities improved and which got worse in your run? What extra work did the ansatz cost? For which uses would you choose it? Either method can be the right answer if your evidence supports it. (Three measured pairs, for orientation — yours will differ: `measured_results.md` §6.)

### Task 4 — Optuna sweep on the GPUs (20 points)

**4a.** Set up the sweep folder and create the study (login node):

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_3/<your-github-handle>/task4_optuna_sweep
cd labs/lab_3/<your-github-handle>/task4_optuna_sweep
cp ../../../../module_3/examples/pinn_optuna_sweep.py ../../../../module_3/examples/pinn_optuna_sweep.sbatch \
   ../../../../module_3/examples/plate_with_hole_hard_bc.py ../../../../module_3/examples/plate_with_hole_fixed.py .
cp ../task2_moose_comparison/reference_grid.csv .
optuna create-study --study-name pinn_lab3_sweep --storage "sqlite:///$(pwd)/pinn_opt.db" --direction minimize
```

**4b.** Submit: six workers × four trials = 24 trials, at most two running at a time. In our full rehearsal this took about an hour and a half of elapsed time, and about 2.6 GPU-hours in total (`measured_results.md` §5) — start it early; queue wait is extra.

```bash
sbatch pinn_optuna_sweep.sbatch
squeue -u $USER
grep -h "split fingerprint" pinn_sweep-*.out | sort -u     # must print exactly one line
```

**How many trials finish is up to the pruner, not you.** In our rehearsal 13 of the 24 completed and 11 were pruned early (`measured_results.md` §5) — that is the sweep working. What counts is that all 24 were *attempted*. **If a worker dies for a cluster reason** (timeout, node failure — its log ends without "Worker done"), resubmit just that worker once; the study keeps every finished trial and the new worker adds its four:

```bash
sbatch --array=<task-id> pinn_optuna_sweep.sbatch       # e.g. --array=3 for pinn_sweep-<jobid>_3.out
```

Then count the attempted trials — completed, pruned and failed: `python -c "import optuna; s=optuna.load_study(study_name='pinn_lab3_sweep', storage='sqlite:///pinn_opt.db'); print(sum(t.state.is_finished() for t in s.trials))"` should print 24 or more. (The trial a killed worker was running stays "running" in the study and is not counted; the resubmitted worker's four trials make up for it.) If a resubmitted worker fails again, stop and say so in `prompts_task4.md` — do not keep resubmitting.

**Validation and test.** The sweep splits the reference grid once (`--split-seed 42`, the same for every worker) into a **validation half**, which every trial is scored on, and a **test half**, which the Optuna objective never uses. Tasks 2 and 3 already looked at the whole grid, so the test half is "unused by the Optuna objective", not an untouched end-to-end test set.

**4c.** Analyse with the Module 2 script (login node, seconds):

```bash
python /project/me5475/examples/analyze_study.py --storage sqlite:///pinn_opt.db --study-name pinn_lab3_sweep --out-dir . | tee study_summary.txt
```

**4d.** Which hyperparameter mattered most here, and does that match what you expected from L9–L11 and the L17 reading? Treat the importance ranking as exploratory, as in Lab 2.

**4e.** Retrain at the best configuration and report its relative L2 error **on the test half** (`split_reference(ref, 42)` in `pinn_optuna_sweep.py` returns the halves), next to your Task 3 hard-BC run's error on the same test half. `retrain_best.py` trains, so submit it as a job: copy `pinn_train.sbatch`, and point `SCRIPT`/`ARGS` at your script.

### Task 5 — Inverse problem (stretch, +20 bonus)

*(L18, Wed Oct 14, covers inverse PINNs.)*

**Why a force is needed.** The plate is loaded by a **prescribed displacement**, so the displacement field does not depend on E at all — only on ν. Stress, and so force, scales with E. Displacement data alone can identify ν but never E. `plate_with_hole_inverse.py` therefore also uses **one reaction force**: the total force on the loaded edge, F = ∫₀¹ σ₁₁(1, x₂) dx₂, which it reads from the reference's measured quantities (1.0734 for E = 1; `measured_results.md` §1) and perturbs with the same 1 % noise as the displacements.

**5a.** Build the measurement file (200 random points; the script picks 20 and adds noise):

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_3/<your-github-handle>/task5_inverse
cd labs/lab_3/<your-github-handle>/task5_inverse
cp ../../../../module_3/examples/plate_with_hole_inverse.py ../../../../module_3/examples/pinn_inverse_ensemble.sbatch .
python /project/me5475/examples/extract_moose_reference.py --mode random --n-points 200 --output measurements.csv
```

**5b.** Run seeds 0–9 (two at a time; about 10 minutes each on an A30):

```bash
sbatch pinn_inverse_ensemble.sbatch
```

**5c.** Collect the ten `inverse_results_seed_*.txt` rows into `ensemble_summary.txt`: mean, standard deviation and range of E and ν, and every run's values. **Keep every run.** Report any run whose values are inadmissible (E ≤ 0, or ν outside (0, 0.5)) or whose training failed — do not drop a run because it lands far from the true values; that is a result.

**5d.** Plot the ten (E, ν) estimates (`ensemble_plot.png`), with the truth (1.0, 0.3) and the ensemble mean marked.

**5e.** Interpret the spread honestly. Each seed changes *several* things at once — which 20 points are measured, the noise on them and on the force, the collocation points and the initial weights — so the spread measures their **combined** effect, not the uncertainty of one experiment. Ten runs are not enough to claim a calibrated 95 % interval. What does the spread tell you, and what would you change to separate measurement uncertainty from optimisation randomness?

### Task 6 — Reflection (≤ 600 words)

Answer all five:

1. **PINN vs MOOSE.** Quantitatively, how did your PINN compare with MOOSE? Was your finding consistent with L12's claim that "FE beats PINN on linear, well-posed forward problems on simple geometries" — and in what sense does "beats" apply here (accuracy, cost, effort)?
2. **Hard vs soft BCs.** What did the hard BCs guarantee, what did your controlled comparison show, and was the extra code worth it for this problem?
3. **HP tuning at the PINN scale.** Did the sweep produce a meaningfully better PINN on the test half? Did the most important hyperparameter match your intuition?
4. **PINN failure modes.** Did you meet any of L17's pathologies across the tasks? If you tried a trick from L17, describe what happened.
5. **Inverse problem (if completed).** How well were E and ν recovered, and what does your ensemble's spread mean?

---

## Grading rubric (out of 100, plus +20 stretch)

| Component | Points |
|-----------|--------|
| Task 1: port + run on ARCC | 20 |
| Task 2: MOOSE comparison with quantitative metrics | 15 |
| Task 3: controlled hard-vs-soft comparison | 15 |
| Task 4: Optuna sweep + analysis | 20 |
| Task 5 (stretch): inverse problem with ensemble | +20 bonus |
| Reflection: all five questions answered concretely | 20 |
| Prompts logged in all sub-folders | 10 |

**Penalty conditions.**

- Failure to compare to MOOSE quantitatively (no L2 error reported): -10.
- The Optuna study holds fewer than 24 attempted trials (completed, pruned and failed all count — pruning is expected and is not penalised): -5.
- Reflection generic or boilerplate: -10.

---

## Hints

- **Read Min's notebook at least twice** before reading the port. Each pass reveals a layer.
- **A low PINN loss is not a correct solution.** Task 1b's missing-BC story is the reason Task 2 exists: always compare against an independent reference.
- **Exact BCs do not remove the need for enough collocation points.** With 1 500 points the hard-BC PINN fitted its training points and oscillated between them (test loss 7.7 against a training loss of 8e-7); with 4 000 it did not (`measured_results.md` §3). If your test loss is far above your training loss, suspect this first.
- **For Task 4, narrow the learning-rate range with care** — L9 and L10's lessons apply. Don't spend trials on rates that cannot work.
- **Task 5 runs can land far from the truth.** Report what fraction of your runs did, and check whether their force and displacement fits were still good — that tells you whether it is an optimisation failure or a genuinely ambiguous fit.

---

## What this lab leads into

- **Midterm project (due Sun Nov 1).** The full spec is released at the start of M4. The "PINN inverse problem" option builds directly on Task 5; tuning is recommended for every option.
- **Module 5.** You will deploy a trained constitutive network inside MOOSE and run the same plate problem — a three-way comparison: MOOSE (M0), PINN (M3), NN-augmented MOOSE (M5).
- **Module 7.** Operator learning generalises the parametric PINN to whole input functions; you will see DeepONet on the same geometry.

---

## Academic integrity reminder

Agents encouraged. Logged prompts mandatory. Numerical results must be yours; copying a classmate's checkpoints, databases or result files is plagiarism. Discussing strategy with classmates is fine and encouraged.
