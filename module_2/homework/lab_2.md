# Lab 2 — Hyperparameter Sweep with Optuna on ARCC

**Module:** 2 — DL Practice + ARCC at Scale
**Released:** Thu Sep 24, 2026 (scheduled for Fri Sep 18 but released late — the deadline moved with it)
**Due:** Mon Oct 5, 2026, 11:59 PM Mountain Time (the day Lecture 15 is taught; extended from Wed Sep 30 by the instructor)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_2` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

---

## What this lab does

Lab 2 takes the constitutive MLP you trained in Lab 1 and *parallelizes* its hyperparameter search across ARCC. You will run 50 trials in roughly ten minutes of wall time (indicative, not a guarantee — it depends on queue wait and node load), identify the best configuration, and analyze which hyperparameters actually mattered. The pattern you build here is what every subsequent module's lab reuses.

The lab also rehearses three new practitioner-level skills introduced in M2: choosing an LR via the LR-finder, reading Optuna parallel-coordinates plots, and writing a SLURM array that does something more interesting than running the same job 5 times.

---

## Deliverables

```
labs/lab_2/<your-github-handle>/
├── preliminaries/
│   ├── lr_finder_output.png             figure from running module_2/examples/lr_finder.py
│   ├── loss_curve_gallery.png           figure from module_2/examples/loss_curve_gallery.py
│   ├── reproducibility_check.txt        output of reproducibility_checklist.py
│   └── prompts_l9_l10.md
├── sweep/
│   ├── optuna_sweep.py                  your copy (possibly modified search space)
│   ├── optuna_sweep.sbatch              ARCC SLURM script (your account, partition)
│   ├── optuna.db                        SQLite database from the completed sweep
│   ├── lab2-*.out                       SLURM worker output logs (a few)
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
│   ├── retrain_log.txt                  output of retrain
│   ├── (optional) ood_test_results.md   stretch task results
│   └── prompts_verification.md
└── reflection.md                        <= 500 words
```

---

## Tasks

### Task 1 — Preliminaries (from L9, L10)

Three short scripts students should run before launching the sweep:

**1a. LR finder.** Copy `module_2/examples/lr_finder.py` into your `preliminaries/` directory and run it there on your Lab 1 dataset:

```bash
cd labs/lab_2/<your-github-handle>/preliminaries
python lr_finder.py --data ../../../lab_1/<your-github-handle>/data_generation/data/single_element.csv --out lr_finder_output.png
```

Save the figure. Read off the recommended LR. Use this to inform your Optuna search range — if the LR-finder says ~3e-3 is good, you can narrow Optuna's search to `[1e-4, 1e-2]` instead of the default `[1e-5, 1e-1]`, which doubles your effective trial budget for the range that actually matters.

**1b. Loss-curve gallery.** Run `python loss_curve_gallery.py`. Save the figure. Look at all six panels carefully — when you read your Optuna trials' loss curves in Task 3 you will need to recognize each pattern.

**1c. Reproducibility checklist.** Run `python reproducibility_checklist.py`. It should print PASS on items 1 and 2. Commit the output.

### Task 2 — Run the Optuna sweep on ARCC

**2a.** Copy `module_2/examples/optuna_sweep.py` and `optuna_sweep.sbatch` into your `sweep/` directory, **and copy your Lab 1 dataset next to them** — the sweep reads `data/single_element.csv` from the directory you submit from:

```bash
cd labs/lab_2/<your-github-handle>/sweep
mkdir -p data
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

**2b.** Create the empty study from a login node:
```bash
optuna create-study --study-name lab2_sweep \
    --storage "sqlite:///$(pwd)/optuna.db" \
    --direction minimize
```

**2c.** Submit:
```bash
sbatch optuna_sweep.sbatch
squeue -u $USER          # 10 array tasks should appear
```

**2d.** Wait for completion (expect roughly ten minutes after the first worker starts; queue wait is on top of that). Commit `optuna.db` and a few worker logs.

**Possible modifications.** If your LR-finder (Task 1a) suggested a tighter LR range, modify `optuna_sweep.py` to use it. Document your choice and reasoning in `prompts_sweep.md`.

### Task 3 — Analyze the study

From your `analysis/` directory, point the script at the database in `sweep/` and write the plots here:

```bash
cd labs/lab_2/<your-github-handle>/analysis
python analyze_study.py --storage sqlite:///../sweep/optuna.db --study-name lab2_sweep --out-dir . | tee study_summary.txt
```

This writes the four HTML files and saves the printed summary to `study_summary.txt`.

Examine the plots:

- **Parallel coordinates.** Each line is one trial. The objective is the rightmost column. Which hyperparameters' good values cluster tightly (high importance) and which spread widely (low importance)?
- **Optimization history.** Did the objective improve over time? Did pruning kick in (gaps in the trial sequence)?
- **Parameter importances.** Optuna estimates importance via fANOVA. For this problem, learning rate should dominate; weight decay should be mostly irrelevant given the noise-free labels.

### Task 4 — Verification: retrain at the best params, evaluate on test set

After identifying `study.best_params`:

1. Write `retrain_best.py` that loads the best params from the study, retrains the model with them on the train+val combined dataset, and reports MSE on the *test set*.
2. Compare to Lab 1's default config (lr=1e-3, hidden=32, layers=4, no weight decay). By how much (in percentage MSE) did the sweep improve over the default?
3. If the improvement is small (< 10%), discuss whether HP tuning was worth the compute. If large (> 50%), discuss what specifically the best config does that the default did not.

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
- **SQLite + 10 concurrent workers works if the workers write rarely.** Your home directory on ARCC is a network file system, and every `trial.report()` is a database write. The script therefore reports every 10 epochs (not every epoch) and waits up to 60 s for the lock; reporting every epoch crashed workers with "database is locked" when we tested it. If you change those settings and see that error in your worker logs, change them back — or switch to Optuna's journal storage (read `module_2/readings/slurm_sweep_patterns.md`).
- **Pruning is your friend.** With Median pruning, expect many trials to be killed early — in our test run on ARCC, most of the 50 were (it depends on your search range and data). This is healthy. If `study.trials_dataframe()` shows zero pruned trials, your `n_warmup_steps` is too high or your training is too fast for pruning to kick in.
- **For Task 5,** generate a *new* OOD dataset with a different random seed (so it's not a subset of training data). Run `module_1/examples/generate_data.py` with `--max-strain 0.01 --seed 99`.
- **Reading parallel-coordinates plots.** Each vertical axis is one hyperparameter. Trace each line from left to right — lines that end at low objective (good val MSE) tell you what configurations work. If multiple lines bunch tightly at one value on some axis, that hyperparameter has a clear optimum; if they spread widely, it doesn't matter much.

---

## What this lab leads into

- **Module 3 (Wk 5).** PINN training is much more sensitive to hyperparameters than constitutive surrogate training. The Optuna pattern you build today is what you reuse next week for the PINN module's Lab 3.
- **Midterm project.** Whichever flavor of midterm you pick, you will run an Optuna sweep as part of it. The framework you build today is the framework you reuse.
- **Module 9 (Wk 14).** HPC at scale revisits the Optuna pattern with distributed (multi-GPU) training and a larger search space.

---

## Academic integrity reminder

Agents encouraged. Logged prompts mandatory. Numerical results must be yours — copying a classmate's `optuna.db` is plagiarism. Discussing strategy with classmates is fine and encouraged.
