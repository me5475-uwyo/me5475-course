# Module 2 — measured results

Every computed number in a Module 2 reading must appear here first, with what was run, where, and
when. A reading cites this file; it never carries a number that was not measured. If a value is
wanted but unmeasured, the reading says **(not measured)**.

---

## Environments

| id | where | versions |
|---|---|---|
| **LAPTOP** | macOS, `~/ME5475-demo-env` | python 3.9, torch 2.8.0, numpy 2.0.2, pandas 2.3.3 |
| **ARCC** | MedicineBow login node `mblog1`, `/project/me5475/envs/ml4sm` | python 3.11.15, torch 2.5.1+cu121, numpy 2.4.4, pandas 3.0.3, matplotlib 3.10.9 |
| **MOOSE** | `/project/me5475/software/rom_opt_arcc/rom_opt-opt`, built 2026-06-09, app version `437fbe5082` | not on `PATH`; call by full path |

---

## 1 · Plane-strain stiffness, E = 1.0, ν = 0.3

Run: `python module_1/examples/generate_data_analytic.py`
Date: 2026-09-21 · Environments: **LAPTOP** and **ARCC**, identical output

```
[ 1.346154   0.576923   0.000000 ]
[ 0.576923   1.346154   0.000000 ]
[ 0.000000   0.000000   0.384615 ]
```

The (3,3) entry is **G = E/(2(1+ν)) = 1/2.6**, and it multiplies the **engineering** shear
γ_xy = 2ε_xy. Closed-form and seeded, so both machines agree to every printed digit, as do all 200
data rows.

## 2 · MOOSE agrees with that stiffness

Run: `rom_opt-opt -i single_element_loadsweep.i` · Date: 2026-09-21 · Environment: **MOOSE**

MOOSE reported ε_xx = 5.0e-4, ε_xy = −2.5e-4 (so γ_xy = −5.0e-4), ε_yy = 0, and:

| component | MOOSE | C·ε from §1 |
|---|---|---|
| σ_xx | 6.7307692e-04 | 6.730769e-04 |
| σ_yy | 2.8846154e-04 | 2.884615e-04 |
| σ_xy | −1.9230769e-04 | −1.923077e-04 |

**This is the measurement that licenses `generate_data_analytic.py`** as a stand-in for MOOSE in the
L9–L11 demos. It does **not** license it as a substitute for the Lab 1 deliverable.

## 3 · LR-range test on the analytic dataset — seed-dependent

Run: `python module_2/examples/lr_finder.py --data data/single_element_analytic.csv --seed <s>`
Date: 2026-09-21 · Environment: **LAPTOP**

| seed | suggested LR |
|---|---|
| 0 | 1.07e-03 |
| 1 | 1.00e-02 |
| **42** (script default) | **1.00e-02** |
| 7 | *no suggestion printed* |
| 123 | *no suggestion printed* |

On **ARCC**, seed 42 also gives **1.00e-02**, on a different torch.

**Read this carefully before citing it.** `1.00e-02` is one tenth of `--lr-max`, the top of the
sweep. The script reports a "divergence" at the last point of its own range, so the agreement
across two machines and two torch versions is the *artifact* being stable, not the *measurement*
being reproducible. The test points at a decade, not a value. See `team/UNRESOLVED.md` U9-7.

## 4 · Plate with a hole — mesh convergence

Run: `rom_opt-opt -i plate_with_hole.i Mesh/uniform_refine=<r>` · Date: 2026-09-21 · **MOOSE**
Geometry: quarter annulus, rmin 0.1, rmax 1.0, nr 8, nt 24; E = 1, ν = 0.3; σ_∞ = 1 traction on `rmax`.

| `uniform_refine` | elements | max σ_xx | max von Mises | wall |
|---|---|---|---|---|
| 0 | 192 | 1.6221 | 1.2458 | 1 s |
| 1 | 768 | 2.0048 | 1.5610 | 1 s |
| 2 | 3 072 | 2.3841 | 1.9328 | 1 s |
| 3 | 12 288 | 2.6790 | 2.2542 | 2 s |
| 4 | 49 152 | 2.8762 | 2.4784 | 5 s |

Analytical **K_t = 3** for a circular hole in an *infinite* plate. The sequence approaches it from
below; the remaining gap is the finite outer radius (rmax/rmin = 10) plus residual discretisation.

## 5 · Lab 1 dataset generation on ARCC

Run: `python generate_data.py --n 20 --moose-exe /project/me5475/software/rom_opt_arcc/rom_opt-opt`
Date: 2026-09-21 · Environments: **ARCC** + **MOOSE**, with `module load gcc/14.2.0`

**20 runs in 19 s → 200 runs ≈ 3 minutes.**

Requires `module load gcc/14.2.0` after `source setup5475.sh`: conda's `libstdc++` is older than the
one MOOSE was built against, and without it the driver fails with `GLIBCXX_3.4.31 not found`.
MOOSE alone works and python alone works; only the combination fails.

## 6 · Lab 2 sweep on ARCC — 50 trials, ten workers, one shared split

Run: `DATA=…/single_element_analytic.csv sbatch optuna_sweep.sbatch` (10 workers × 5 trials,
`--max-epochs 2000`, TPE + `MedianPruner(n_startup_trials=5, n_warmup_steps=100)`, reports every 10
epochs, SQLite with a 60 s lock timeout, **`--split-seed 42` for every worker**)
Date: 2026-09-25 · Environment: **ARCC** (`ml4sm`, optuna 4.8.0), partition `mb`, job 18801657
Data: the analytic Lab 1 dataset (`generate_data_analytic.py`, 200 points, noise-free).
Evidence: `team/reviews/2026-09-25_L11_evidence/` (all 50 trials as CSV, `study_summary.txt`,
`sacct_18801657.txt`).

**All ten worker logs show the same split fingerprint** (`d45e25d089`), so every trial was trained
and scored on the same rows. **All 50 trials ran within 2 min 38 s** (first trial start 09:17:13, last
finish 09:19:51) once the workers started. **27 completed, 23 pruned, 0 failed**; 21 of the 23 were
pruned at epoch 100, the first check-in the warm-up allows.

| rank | trial | val MSE (standardised) | lr | width | n_layers | weight_decay | batch |
|---|---|---|---|---|---|---|---|
| 1 | 33 | 5.09e-5 | 1.02e-2 | 32 | 2 | 2.7e-3 | 16 |
| 2 | 34 | 5.82e-5 | 1.01e-2 | 32 | 2 | 2.0e-3 | 16 |
| 3 | 16 | 7.01e-5 | 2.20e-3 | 128 | 3 | 8.1e-3 | 16 |
| worst completed | 1 | 4.66e-2 | 2.89e-5 | 16 | 6 | 2.1e-4 | 64 |

The five best completed trials span 0.73 decades of learning rate (1.9e-3 to 1.0e-2). The lowest
completed rate (2.9e-5) and the two highest (7.6e-2, 9.0e-2) are all among the four worst.

fANOVA importance (`analyze_study.py`): **lr 0.746**, n_layers 0.098, width 0.077, batch_size 0.055,
weight_decay 0.023. **Exploratory:** 27 adaptively chosen completed trials in this search space —
a rerun can reorder the small values. This run does **not** measure how much time pruning saves
(no pruning-off comparison was run).

**History — two earlier runs, kept for the record, not for conclusions.**
- *Job 18490083 (2026-09-24):* reporting to SQLite **every epoch** on the NFS home directory, two of
  ten workers crashed with `database is locked`. Reporting every 10 epochs with a 60 s timeout then
  ran cleanly — one observed run; Optuna itself advises against parallel SQLite on NFS.
- *Job 18493245 (2026-09-24):* 50/50 trials, but **each worker used its own data split** (the
  per-worker `--seed` also seeded the split), so trials were scored on different validation rows.
  Found by the Reviewer 2026-09-25; its ranking is withdrawn. Evidence kept in
  `team/reviews/2026-09-25_L11_evidence/history_job18493245_mixed_split/`.
