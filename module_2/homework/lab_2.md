# Lab 2 — Hyperparameter Sweep with Optuna on ARCC

**Module:** 2 — DL Practice + ARCC at Scale
**Released:** Thu Sep 24, 2026 (scheduled for Fri Sep 18 but released late — the deadline moved with it)
**Due:** Mon Oct 5, 2026, 11:59 PM Mountain Time (the day Lecture 15 is taught; extended from Wed Sep 30 by the instructor)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_2` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

> **Update 2026-09-25:** `optuna_sweep.py` now uses one data split for all workers (before, each worker scored its trials on different validation rows). Pull the update and re-copy `optuna_sweep.py` and `optuna_sweep.sbatch` into your `sweep/` folder before running your sweep. If you already ran it, re-run with a fresh `optuna.db` (move the old one aside, then repeat steps 2b–2d). Task 4 now reuses the same split, so its test rows are ones no trial ever saw.

---

## What this lab does

Lab 2 takes the constitutive MLP you trained in Lab 1 and *parallelizes* its hyperparameter search across ARCC. You will run 50 trials in a few minutes of wall time once the workers start (2 min 38 s in our test run, `module_2/readings/measured_results.md` §6; queue wait is extra and varies), identify the best configuration, and analyze which hyperparameters actually mattered. The pattern you build here is a reusable workflow you can bring to later labs and your midterm project.

The lab also rehearses three new practitioner-level skills introduced in M2: choosing an LR via the LR-finder, reading Optuna parallel-coordinates plots, and writing a SLURM array that does something more interesting than running the same job 5 times.

---

## Deliverables

```
labs/lab_2/<your-github-handle>/
├── preliminaries/
│   ├── lr_finder_output.png             figure from running module_2/examples/lr_finder.py
│   ├── loss_curve_gallery.png           figure from module_2/examples/loss_curve_gallery.py
│   ├── reproducibility_check.txt        output of reproducibility_checklist.py
│   ├── lab2_preliminaries.sbatch        the Task 1 job script, as you submitted it
│   ├── prelim-*.out                     the Task 1 job's log
│   └── prompts_l9_l10.md
├── sweep/
│   ├── optuna_sweep.py                  your copy (possibly modified search space)
│   ├── optuna_sweep.sbatch              ARCC SLURM script (your account, partition)
│   ├── optuna.db                        SQLite database from the completed sweep
│   ├── lab2-*.out                       SLURM worker output logs (a few; all show the same split fingerprint)
│   └── prompts_sweep.md
├── analysis/
│   ├── analyze_study.py                 your copy (may be unchanged)
│   ├── study_summary.txt                stdout of analyze_study.py
│   ├── parallel_coords.html             from analyze_study.py
│   ├── optimization_history.html
│   ├── param_importances.html
│   ├── slice.html
│   └── prompts_analysis.md
├── verification/
│   ├── retrain_best.py                  retrain at best params, get final test MSE
│   ├── retrain_best.sbatch              the Task 4 job script, as you submitted it
│   ├── retrain_log.txt                  output of retrain (includes the split fingerprint)
│   ├── (optional) ood_test_results.md   stretch task results
│   └── prompts_verification.md
└── reflection.md                        <= 500 words
```

---

## Tasks

**Before you start.** Log in to ARCC and activate the course environment (it already has PyTorch, Optuna and Plotly — do not `pip install` anything):

```bash
source /project/me5475/setup5475.sh
```

**Where each step runs — every training step is a submitted job.** Anything that *trains* a network runs as a batch job on a compute node, never on the login node: ARCC's login nodes are for setup, submitting jobs and looking at results ([ARCC HPC policy](https://www.uwyo.edu/arcc/policies/hpcpolicy.html)). Lab 2 gives you one job script per training step, all in `module_2/examples/`:

- **Task 1** — `lab2_preliminaries.sbatch` runs the LR finder, the loss-curve gallery and the reproducibility check;
- **Task 2** — `optuna_sweep.sbatch` runs the ten sweep workers;
- **Task 4** — `retrain_best.sbatch` runs your `retrain_best.py`.

On the login node you only copy files, create the study, submit jobs (`sbatch`), watch them (`squeue -u $USER`) and run the analysis in Task 3. Each job writes its output to a `*-<jobid>.out` file in the folder you submitted from.

**Every command block below starts at the root of your course repository** (the folder that contains `labs/` and `module_2/`). Each block begins with `cd "$(git rev-parse --show-toplevel)"`, which takes you back there from anywhere inside the repository, so you can run the blocks one after another. All paths inside a block are relative to the folder it `cd`s into; from `labs/lab_2/<your-github-handle>/<sub-folder>/`, the repository root is `../../../../` and your Lab 1 folder is `../../../lab_1/<your-github-handle>/`.

### Task 1 — Preliminaries (from L9, L10)

Three short scripts to run before launching the sweep, **as one batch job**. Copy them and the job script into your `preliminaries/` directory and submit from there:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_2/<your-github-handle>/preliminaries
cd labs/lab_2/<your-github-handle>/preliminaries
cp ../../../../module_2/examples/lr_finder.py \
   ../../../../module_2/examples/loss_curve_gallery.py \
   ../../../../module_2/examples/reproducibility_checklist.py \
   ../../../../module_2/examples/lab2_preliminaries.sbatch .
sbatch lab2_preliminaries.sbatch
squeue -u $USER               # wait until the job has gone from the list
cat prelim-*.out              # the job's log
```

The job reads your Lab 1 dataset from `../../../lab_1/<your-github-handle>/data_generation/data/single_element.csv` (the handle comes from the folder you submit from) and stops at once with `ERROR: Lab 1 dataset not found` if it is not there. It writes `lr_finder_output.png`, `loss_curve_gallery.png` and `reproducibility_check.txt` into `preliminaries/`.

**1a. LR finder.** Open `lr_finder_output.png`. Read off the recommended LR. Use this to inform your Optuna search range — if the LR-finder says ~3e-3 is good, you can narrow Optuna's search to `[1e-4, 1e-2]` instead of the default `[1e-5, 1e-1]`, which doubles your effective trial budget for the range that actually matters.

**1b. Loss-curve gallery** and **1c. Reproducibility checklist.** For 1b, look at all six panels of `loss_curve_gallery.png` carefully — when you read your Optuna trials' loss curves in Task 3 you will need to recognize each pattern. For 1c, the checklist should print PASS on items 1 and 2; commit `reproducibility_check.txt`.

### Task 2 — Run the Optuna sweep on ARCC

**2a.** Copy `module_2/examples/optuna_sweep.py` and `optuna_sweep.sbatch` into your `sweep/` directory, **and copy your Lab 1 dataset next to them** — the sweep reads `data/single_element.csv` from the directory you submit from:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_2/<your-github-handle>/sweep/data
cd labs/lab_2/<your-github-handle>/sweep
cp ../../../../module_2/examples/optuna_sweep.py \
   ../../../../module_2/examples/optuna_sweep.sbatch .
cp ../../../lab_1/<your-github-handle>/data_generation/data/single_element.csv data/
```

If the file is missing, every worker stops at once with `ERROR: dataset not found` rather than failing ten times. Then check the SLURM header of `optuna_sweep.sbatch`. For this course it is:

```bash
#SBATCH --account=me5475          # the course allocation
#SBATCH --partition=mb            # MedicineBow CPU partition
#SBATCH --mail-user=<your-email>  # your address, not the instructor's
```

There is no `pytorch` module on MedicineBow — PyTorch comes from the shared conda environment, which
`source /project/me5475/setup5475.sh` activates (verified 2026-09-18: Python 3.11.15, torch 2.5.1).
GPU partitions `mb-l40s` / `mb-a30` exist but this sweep is CPU-only and does not need them.

**2b.** Create the empty study from a login node, inside `sweep/` (the `$(pwd)` makes the database path absolute, so the workers and this command agree on one file):
```bash
cd "$(git rev-parse --show-toplevel)"
cd labs/lab_2/<your-github-handle>/sweep
optuna create-study --study-name lab2_sweep \
    --storage "sqlite:///$(pwd)/optuna.db" \
    --direction minimize
```

**2c.** Submit, from the same folder:
```bash
cd "$(git rev-parse --show-toplevel)"
cd labs/lab_2/<your-github-handle>/sweep
sbatch optuna_sweep.sbatch
squeue -u $USER          # 10 array tasks should appear
```

**2d.** Wait for completion (expect roughly ten minutes after the first worker starts; queue wait is on top of that). Then check that every worker trained and scored on the same rows — each worker log prints its split fingerprint, and this must print exactly **one** line:

```bash
cd "$(git rev-parse --show-toplevel)"
cd labs/lab_2/<your-github-handle>/sweep
grep -h "split fingerprint" lab2-*.out | sort -u
```

Two or more different lines mean the workers used different `--split-seed` values: fix that, move `optuna.db` aside, and re-run 2b–2c. Commit `optuna.db` and a few worker logs.

**Possible modifications.** If your LR-finder (Task 1a) suggested a tighter LR range, modify `optuna_sweep.py` to use it. Do not change `--split-seed` or `load_and_split` — Task 4 depends on them. Document your choice and reasoning in `prompts_sweep.md`.

### Task 3 — Analyze the study

Copy `analyze_study.py` into your `analysis/` directory, point it at the database in `sweep/`, and write the plots there:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_2/<your-github-handle>/analysis
cd labs/lab_2/<your-github-handle>/analysis
cp ../../../../module_2/examples/analyze_study.py .
python analyze_study.py --storage sqlite:///../sweep/optuna.db --study-name lab2_sweep --out-dir . | tee study_summary.txt
```

This writes the four HTML files and saves the printed summary to `study_summary.txt`.

Examine the plots:

- **Parallel coordinates.** Each line is one trial. First find the axis labelled **Objective Value** — in Optuna's default plot it is the *first* (leftmost) axis, on a linear scale, and the line colour also encodes it. (The L11 slide draws a custom version with the axes in a different order and log MSE, so do not expect your plot to look the same.) Which hyperparameters' good values cluster tightly and which spread widely?
- **Optimization history.** Did the objective improve over time? Did pruning kick in (gaps in the trial sequence)?
- **Parameter importances.** Optuna estimates importance via fANOVA. Which hyperparameter ranks highest in *your* study, and does that match what the parallel-coordinates plot shows? Treat the result as exploratory: it is computed from your completed trials only, within your search space, and a re-run can reorder the smaller entries. Say what you think explains your ranking.

### Task 4 — Verification: retrain at the best params, evaluate on test set

The comparison is only fair if both models are scored on the **same test rows, in the same units**, and those rows were never used by any trial. `optuna_sweep.py` already defines that split, so reuse it rather than writing your own: import `load_and_split` (and `MLP`) from your `sweep/optuna_sweep.py` and call it with `split_seed=42`, the value every worker used.

1. Write `retrain_best.py` in `verification/`. It loads `study.best_params` from the study, calls `load_and_split(path, split_seed=42)`, and trains **two** models on the train+val rows combined (`X_train`+`X_val`, `Y_train`+`Y_val`): one with the best params and one with Lab 1's default config (lr=1e-3, hidden=32, layers=4, no weight decay). There is no validation set left for early stopping, so train both for the same fixed number of epochs (the sweep's 2000 is a reasonable choice) and record that number.
2. Evaluate both on `data["X_test"]`, `data["Y_test"]` and report both test MSEs in the same standardised units as the sweep's objective (the train-only scaling `load_and_split` applies). Print `data["split"]["fingerprint"]` into `retrain_log.txt`; it must match the fingerprint in your worker logs. By how much (in percentage MSE) did the sweep improve over the default?
3. If the improvement is small (< 10%), discuss whether HP tuning was worth the compute. If large (> 50%), discuss what specifically the best config does that the default did not.

A starting point (run from `verification/`; the training loop is yours to write):

```python
import sys
from pathlib import Path

import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sweep"))
from optuna_sweep import MLP, load_and_split          # the sweep's own split and model

data = load_and_split(Path("../sweep/data/single_element.csv"), split_seed=42)
print("split fingerprint:", data["split"]["fingerprint"])   # must match the worker logs
X_fit = torch.cat([data["X_train"], data["X_val"]])
Y_fit = torch.cat([data["Y_train"], data["Y_val"]])
best = optuna.load_study(study_name="lab2_sweep",
                         storage="sqlite:///../sweep/optuna.db").best_params
# best["batch_size"] may be the string "full": use len(X_fit) then.
# ... train one MLP with `best` and one with Lab 1's default, same epoch budget,
# ... then MSE on data["X_test"], data["Y_test"] for each.
```

It trains two networks, so **run it as a batch job** with `retrain_best.sbatch`, submitted from `verification/`. The job saves everything the script prints to `retrain_log.txt`:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p labs/lab_2/<your-github-handle>/verification
cd labs/lab_2/<your-github-handle>/verification
cp ../../../../module_2/examples/retrain_best.sbatch .
sbatch retrain_best.sbatch     # after writing retrain_best.py in this folder
squeue -u $USER
cat retrain_log.txt
```

### Task 5 (stretch — recommended, not required) — Out-of-distribution generalization

Train data was sampled with strain magnitudes in [-0.005, +0.005]. Test the best Optuna config on strains in [-0.01, +0.01] — twice the training range. Generate the OOD test data by re-running `module_1/examples/generate_data.py` with `--max-strain 0.01` and `--seed 99` (different from training seed).

Two specific questions:

- Does the OOD test MSE blow up dramatically, or does linear extrapolation still work? (For linear elasticity, it *should* work — the truth is linear, so a well-trained network should extrapolate.)
- Does the Optuna-best config OOD-generalize better than Lab 1's default? Or does HP tuning sometimes hurt OOD generalization (overfitting to the in-distribution range)?

### Task 6 — Reflection (≤ 500 words)

Answer all four:

1. **Best config.** What hyperparameters did the sweep pick? Were any surprising? How does the best config differ from Lab 1's default?
2. **Importance.** Which hyperparameter mattered most by fANOVA? Was this consistent with what the parallel-coordinates plot showed visually?
3. **Compute cost.** Total CPU-hours spent on the sweep (look at `sacct`). Was the improvement in val MSE worth the cost? Hypothetically, if you could afford 500 trials instead of 50, would you expect significant further improvement, or would you hit diminishing returns?
4. **Stretch (if done): OOD.** Did the best config OOD-generalize better or worse than the default? What does this tell you about HP tuning for scientific ML where the test distribution may differ from the training distribution?

---

## Grading rubric (out of 100)

| Component | Points |
|-----------|--------|
| Preliminaries (Task 1: LR finder, gallery, reproducibility) | 15 |
| Sweep runs to completion on ARCC (Task 2) | 20 |
| Analysis: all four HTML files exist and are inspected (Task 3) | 20 |
| Verification: retrain at best, compare to default (Task 4) | 20 |
| Stretch OOD task (Task 5) | +10 bonus |
| Reflection: all four questions answered concretely (Task 6) | 20 |
| Prompts logged in all sub-folders | 5 |

**Penalty conditions.**

- The sweep ran only one trial / locally (not on ARCC): -10.
- `optuna.db` is empty or missing: -10.
- Reflection generic or boilerplate: -10.

---

## Hints

- **Pre-stage the study before submitting workers.** Running `optuna.create_study(load_if_exists=True)` inside each worker is also fine, but a single explicit creation prevents race conditions on first submit.
- **SQLite on ARCC is the course's observed configuration, not a guarantee.** Your home directory on ARCC is a network file system (NFS), every `trial.report()` is a database write, and Optuna's own FAQ advises against parallel SQLite, particularly on NFS ([Optuna 4.8 FAQ](https://optuna.readthedocs.io/en/v4.8.0/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3)). What we observed (`module_2/readings/measured_results.md` §6): reporting every epoch crashed 2 of 10 workers with `database is locked`; with the shipped settings — a report every 10 epochs and a 60 s lock timeout — the ten workers finished all 50 trials with no failures. That is what we saw, not a promise; the timeout only makes a worker wait longer for the lock. **If you do hit `database is locked`:** note which worker logs show it, keep the trials that finished (they are safe in the database), and resubmit the sweep — a rerun joins the same study and adds another 5 trials per worker. If it keeps happening, tell the instructor; the fallback is Optuna's journal storage, which needs coordinated changes to the worker, analysis and retrain scripts (see `module_2/readings/slurm_sweep_patterns.md`). Either way, record what happened in `prompts_sweep.md`.
- **Pruning is your friend.** With Median pruning, expect some trials to be stopped early — in the corrected ARCC run (Sep 25), 23 of the 50 were, 21 of them at epoch 100, the first step the pruner may act (`measured_results.md` §6). How many get pruned depends on your search range and data. Zero pruned trials is not automatically a problem: the pruner only acts after `n_warmup_steps`, only once enough trials have completed (5 here), and only when a trial's best reported loss so far is worse than the median of the completed trials' losses at the same step. If you see none, check those three things before changing anything.
- **For Task 5,** generate a *new* OOD dataset with a different random seed (so it's not a subset of training data). Run `module_1/examples/generate_data.py` with `--max-strain 0.01 --seed 99`.
- **Reading parallel-coordinates plots.** Each vertical axis is one hyperparameter, plus one labelled **Objective Value** — in Optuna's default plot that is the leftmost axis (linear scale), and the line colour repeats it. Follow the lines that start low on the Objective axis (good val MSE) across the other axes — they tell you what configurations work. If multiple lines bunch tightly at one value on some axis, that hyperparameter has a clear optimum; if they spread widely, it may matter less within your search range — check that against the importance plot rather than assuming it.

---

## What this lab leads into

- **Module 3 (Wk 5).** PINN training is much more sensitive to hyperparameters than constitutive surrogate training. Lab 3's Task 4 is an Optuna sweep of a PINN — the workflow you build today carries straight over.
- **Midterm project.** A hyperparameter sweep is recommended, not required, for the midterm project — whichever option you pick, this workflow (shared study, many workers, analysis afterwards, a test set no trial touched) is ready to reuse.
- **Module 9 (Wk 14).** HPC at scale revisits the Optuna pattern with distributed (multi-GPU) training and a larger search space.

---

## Academic integrity reminder

Agents encouraged. Logged prompts mandatory. Numerical results must be yours — copying a classmate's `optuna.db` is plagiarism. Discussing strategy with classmates is fine and encouraged.
