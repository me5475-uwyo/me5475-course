# Module 4 — measured results

Every computed number in a Module 4 reading or in Lab 4 must appear here first, with what was run, where,
and when. A reading cites this file; it never carries a number that was not measured. If a value is
wanted but unmeasured, the reading says **(not measured)**.

`[fallback]` — measured by Co-Worker subagents (Opus 5.5) on ARCC, 2026-10-03, in two passes:

- **Current (§1–§3):** the Lab 4 data and models with **fully reversed loading cycles**
  (0 → +a → −a → 0, three times), adopted by the instructor on 2026-10-03 (`team/DECISIONS.md`, D3).
  - Evidence: `team/reviews/2026-10-03_M4_reversed_evidence/`.
  - Report: `team/reviews/agent_reports/2026-10-03_coworker_subagent_M4_reversed_cycles.md`.
  - ARCC work folder: `~/me5475-demo/M4-reversed/`.
- **Superseded (§5):** the same morning's validation with radial, never-reversed cycles
  (0 → a → 0, three times). Kept as history.
  - Evidence: `team/reviews/2026-10-03_M4_lab4_validation_evidence/`.
  - Report: `team/reviews/agent_reports/2026-10-03_coworker_subagent_M4_lab4_validation.md`.
  - ARCC work folder: `~/me5475-demo/M4-validate/`.

§1.1 (the boundary-condition defect), §2.3, §3 and §4 do not depend on the loading shape.

---

## Environments

| id | where | versions |
|---|---|---|
| **ARCC-CPU** | MedicineBow partition `mb` (AMD nodes `mbcpu-001/002/003/005/006/008/011/012/013/015`), `/project/me5475/envs/ml4sm`; training on 4 cores via `train_constitutive.sbatch`, MOOSE on 1 core via `generate_data.sbatch` | python 3.11.15, torch 2.5.1+cu121 (run on CPU), numpy 2.4.4, pandas 3.0.3, matplotlib 3.10.9 |
| **MOOSE** | as run for §1–§3: `/project/me5475/software/rom_opt_arcc/rom_opt-opt` (not on `PATH`), module stack of `module_0/examples/run_cantilever.sbatch`; the course build from 2026-10: `/project/me5475/software/moose_torch/me5475/me5475-opt` (§1.2: identical data) | MOOSE git 437fbe5082 (2026-03-09), libMesh f8a1758, PETSc 3.24.4 (both builds) |

Final files (md5), as run for §1–§3: `j2_cyclic_loading.i` 2ef550512e08677a2c5defe8fa1bb07b (released as c44c6973f5cbd716077e4022cdb6f064 after comments-only edits on 2026-10-03; jobs 22750009 and 22941265 regenerated `trajectories.npz`, identical SHA-256 `bc6bb355…f725`),
`generate_plasticity_data.py` c78dc1309973fb4982417d33d3b9ff16, `generate_data.sbatch`
f8d9009fcf0ba62c6c282bb0715eed0e (updated to 9808e64d598a51281413c65193acfc9a for the new course executable and its library path and run in §1.2; released as 55b88ae029caa5278ef608deec3c312a, which differs from that run copy only in two comment lines), `stateless_mlp_baseline.py` 7ddb7f9e8291a650b6a87616e700cb22,
`lstm_plasticity.py` 7aa92728cfdc782e37f9374ed47d90fc (released as f1684e2dd05ea691d59bc11e08749200 after a module-docstring-only edit on 2026-10-03; the code's AST is unchanged), `levelset_yield_surface.py`
b5661881cb093c804851ef07a5896f1d (released as c8fcdf7da71b5ec561d7f1b5e1883a14 after docstring/comment-only edits on 2026-10-03; the code's AST is unchanged), `icnn_hyperelastic.py` 87e9860eca02e8d1334c7f8d4e4edd3a (unchanged),
`train_constitutive.sbatch` fcdcb7066e37aa8a0b9a11cedfbbede0. Only the first two changed with the reversed
cycles: the `cyclic_factor` expression and comments, and the driver's docstring and help text. The
driver's code is unchanged.

**Every printed number that was run more than once came out identical, digit for digit.** That covers
the data file on `mbcpu-001` and `mbcpu-002`, and the training numbers on `mbcpu-001`, `-002` and `-005`
(the scripts are seeded and run on CPU). Times vary with node load.

---

## 1 · Task 1 — the data (`j2_cyclic_loading.i`, `generate_plasticity_data.py`)

### 1.1 · The input file as shipped was wrong — and what the fix measures

One run of the shipped input with its default amplitudes (ε_xx = 0.01, ε_yy = γ_xy = 0), job 22579830.
These runs used the radial cyclic factor of the time. The defect is in the boundary conditions, not in the
time function.

| | requested | as shipped | after the fix (job 22579831) |
|---|---|---|---|
| element-average ε_xx at a peak | 0.01 | **0.005** | 0.01 |
| element-average γ_xy at a peak | 0 | **−0.005** | 0 |
| σ_xx at a peak | — | 0.673 | 1.34615 (= (λ + 2μ) ε_xx, elastic) |

- **Cause:** separate Dirichlet BCs on the left/right/top/bottom edges disagree at the shared corner
  nodes, and the last one applied wins. The single element is sheared and carries about half the
  requested strain.
- **The fix** applies u = ε(t)·x on all four edges (one BC per component). A mixed run (0.015, −0.01,
  0.02), job 22579832, then carries exactly those strains.
- **Deprecation:** the shipped input also used the deprecated `Modules/TensorMechanics/Master` block,
  which printed two deprecation warnings with stack traces per run.
- **Cost:** one MOOSE run takes 1.0–1.6 s and 0.26–0.27 GB (`/usr/bin/time`).

A 200-trajectory dataset from the **shipped** input (job 22579833, SHA-256 `f571b4ec…`; radial factor)
gave 120/200 yielding, and a largest |ε_xx| at a peak of 0.0138 against a requested maximum of 0.02. It is
not used for anything below.

**Re-check with the reversed factor**, the same mixed amplitudes (0.015, −0.01, 0.02): job 22587297,
2026-10-03 12:59, `mbcpu-001`, 1.15 s, 0.26 GB, no warnings in the log.

- **Strains:** the element-average strain equals +a at t = 0.25, 1.25, 2.25 and −a at t = 0.75, 1.75,
  2.75. The difference is 0 to the printed digits. |ε| ≤ 2.1e-16 at t = 0.5, 1, …, 3.
- **Plastic flow:** in both directions in every cycle.
- **Peak σ_xx:** its magnitude grows each cycle (isotropic hardening): 0.918 → 1.002 → 1.072 at +a, and −0.962 →
  −1.038 → −1.103 at −a.

### 1.2 · Running the lab's command

- **As written, on ARCC:** `python generate_plasticity_data.py --n 200 --out trajectories.npz` (login
  node, course env) exits at once: `MOOSE executable not found: moose-opt`. MOOSE is not on `PATH` on ARCC.
  (Measured in the morning pass; the driver's code has not changed since.)
- **With `generate_data.sbatch`** (the driver plus the course MOOSE build, 1 core of `mb`), submitted from a
  shell that had sourced `/project/me5475/setup5475.sh`:

| job | date | node | elapsed | MaxRSS | result |
|---|---|---|---|---|---|
| 22587295 (student route) | 2026-10-03 12:59 | mbcpu-001 | **2 min 38 s** | 0.22 GB | 200/200 runs, `trajectories.npz` |
| 22587296 (independent folder) | 12:59 | mbcpu-001 | 2 min 39 s | 0.22 GB | identical file |
| 22587607 (`--exclude=mbcpu-001`) | 13:02 | mbcpu-002 | 2 min 57 s | 0.28 GB | identical file |
| 22580620 (morning, the then-option input; same non-comment lines) | 09:24 | mbcpu-001 | 2 min 43 s | 0.22 GB | identical file |
| 22941265 (**new course build** `me5475-opt`, released files) | 22:37 | mbcpu-001 | **6 min 26 s** | — | identical file |

`trajectories.npz` is **440 480 bytes (0.44 MB)**, with SHA-256
`bc6bb355bc12e1639b9e2d54410f44b5eaff18210eb0912549e3f60db6f7f725` in all five runs. Run time with a student's
own local MOOSE: **(not measured)**.

**The new course build (2026-10-03).** Job 22941265 ran the updated files (`generate_data.sbatch` md5
9808e64d598a51281413c65193acfc9a, an executable and library-path update; `j2_cyclic_loading.i` md5
c44c6973f5cbd716077e4022cdb6f064, a comments-only update; the released job script, 55b88ae0…, changes only two
comment lines from the run copy) with `/project/me5475/software/moose_torch/me5475/me5475-opt`:
the same file, in 6 min 26 s. The extra time is start-up. Job 22941859 (`mbcpu-001`, one core) timed 5 launches of
each executable, twice: `--version` alone took 3.11 s and 3.25 s on the old executable and 8.06 s and 9.22 s on the
new one; 5 single-element runs took 5.68 s and 5.89 s against 10.70 s and 11.38 s. So each of the generator's 200
launches costs about 1 s more, which is consistent with loading the added libtorch dependencies (this small sample
does not isolate the cause).

**Course reference copy:** this file, with a README and `SHA256SUMS`, is in `/project/me5475/examples/lab4_reference/`
(staged 2026-10-03 after the Reviewer gate).

### 1.3 · What is in it

The checks below come from validation-only scripts `jobs/check_reversed.py` and
`jobs/check_reversed_extra.py` (jobs 22587608 and 22587751).

**Arrays.** `epsilon`, `sigma`, `eps_p` are **(200, 61, 3)** float32: 60 steps of dt = 0.05 plus t = 0,
with no NaN. There is no `alpha` array (the input writes no accumulated plastic strain).
`n_proportional` = 100.

**The load path.** Every path is fully reversed and radial: ε(t) = f(t)·ε_amp, with one triangle wave f that
goes 0 → +1 → −1 → 0 three times (period 1).
- f = +1 at t = 0.25, 1.25, 2.25; f = −1 at t = 0.75, 1.75, 2.75; f = 0 at every half-integer and
  integer t.
- Max |ε(t) − f(t) ε_amp| over the whole file is 1.1e-9. At the first −peak, ε = −ε_amp exactly.
- So the strain stays on one line through the origin. Its sign flips, but its direction never rotates.
- Trajectories 0–99 scale a random unit direction by 0.5–1 × 0.02 (|ε_amp| 0.0101–0.0200).
  Trajectories 100–199 draw each component in ±0.02 (|ε_amp| 0.0035–0.0285).
- The driver calls the second set "non-proportional", but its strain direction does not change either.

**Amplitude against the elastic limit.** The ratio is the elastic trial von Mises stress at the first peak
(plane strain) divided by the yield stress 1.0:

| set | range | median |
|---|---|---|
| 0–99 | 0.57–1.78 | 1.05 |
| 100–199 | 0.28–2.42 | 1.40 |
| all 200 | 0.28–2.42 | 1.17 |

A trajectory yields exactly when this ratio exceeds 1. The smallest ratio that yields is 1.001; the
largest that does not is 0.990.

**Plasticity.**
- **134/200 yield** (55 of 0–99, 79 of 100–199), and **all 134 yield again in every cycle, in both
  directions**. In each of the three cycles there is plastic flow on the way out to +a and on the way out
  to −a in all 134.
- **66 never yield.** None yields in the first cycle only.
- **The first step after every peak is elastic** in all 134, in all six peaks.
- **Where reverse flow starts in cycle 1:**
  - 113 start once the strain has crossed zero, at f ≤ −0.2;
  - 14 start on the step from ε = 0 (f = 0 → −0.2);
  - 7 start while ε is still positive (f = 0.2 → 0).

  In cycles 2 and 3, flow happens only on the way out to ±a.
- **Plastic steps:** 6–29 of 60 per yielding trajectory (median 15).
- **Hardening:** in all 134, |σ| at the same peak strain grows every cycle, at both +a and −a. From
  cycle 1 to cycle 3 it rises by 0–32 % at +a (median 2 %) and 0–28 % at −a (median 2 %). The 66 elastic
  trajectories repeat exactly (relative change 0).

**Loops with area.** The net work Σσ·dε over each closed strain cycle is > 1e-6 in exactly the 134
yielding trajectories, in every cycle:

| cycle | median over the 134 | median over all 200 | max |
|---|---|---|---|
| 1 | 9.5e-3 | 4.3e-3 | 0.0435 |
| 2 | 1.08e-2 | 4.9e-3 | 0.0482 |
| 3 | 9.6e-3 | 4.3e-3 | 0.0468 |

**The stress–strain curve does not close.** These are closed *strain* cycles: the strain is back at 0 at t = 1, 2
and 3. The stress is not. In the single mixed-amplitude run (job 22587297, a = (0.015, −0.01, 0.02);
`single_run/check_single_22587297.txt` in the reversed-cycle evidence):
- σ_xx at ε = 0 is 0, 0.480, 0.404 and 0.340 at t = 0, 1, 2 and 3;
- σ_xx at the +a peak is 0.918, 1.002 and 1.072 in cycles 1–3.

So each cycle traces drifting hysteretic branches, not a closed stress–strain loop. The net work above is therefore
not, by itself, the dissipated energy: the elastic energy held at ε = 0 by the residual stress also changes from
cycle to cycle.

**In the lab's σ_xx–ε_xx plot:**
- 186 of 200 curves visit both the first quadrant (ε_xx > 0, σ_xx > 0) and the third (ε_xx < 0,
  σ_xx < 0). That includes 127 of the 134 yielding ones.
- All 134 yielding trajectories have plastic flow at some step where ε_xx < 0.

**The history the data carry.**
- **Residual stress |σ| at ε = 0:**
  - at t = 0.5, after the first excursion to +a: median 0.107, max 0.856, above 0.01 in 132;
  - at t = 1, after the full first cycle: median 0.098, max 0.939, above 0.01 in 132.
- **The same strain in a later cycle:** |σ(t = 1.25) − σ(t = 0.25)| has median 0.037 over the yielding
  trajectories (max 0.173).
- **Ranges:** σ_xx −3.58…3.59, σ_yy −3.46…3.47, σ_xy −0.61…0.605; the largest |ε^p| component is
  0.0100.

**The handout's one-liners** (in job 22587608):
- `(np.abs(np.diff(eps_p[:, 20:], axis=1)).max(axis=(1, 2)) > 1e-10).sum()` (plastic flow after the
  first cycle) prints **134**;
- `(np.abs(eps_p).max(axis=(1, 2)) > 1e-10).sum()` (yields at all) also prints 134, and the two select
  the same trajectories.

Plot: `task1_data/reversed_task1c_loops.png` (the lab's own σ_xx–ε_xx plot) in the evidence folder.

## 2 · Tasks 2 and 3 — stateless MLP vs LSTM

**Setup.** Defaults throughout: seed 42, 2 000 epochs, Adam lr 1e-3. The data are split by trajectory:
160 train / 40 test, i.e. 9 760 / 2 440 (ε, σ) pairs; 111 of the training and 23 of the test trajectories
yield.

**"Test MSE"** is the scripts' own number: the mean squared error on the test trajectories, with each stress
component standardized by its training-set mean and std.

### 2.1–2.2 · Results and analysis — posted after Lab 4 closes

The Task 2–3 test errors and the analysis of how well each model reproduces the history are added to this
section after Lab 4 closes (Fri Nov 6). Until then, your own runs are the result.

### 2.3 · Job-script check

The shipped `train_constitutive.sbatch` (GPU partition, bare `python`) was submitted from a shell that had
sourced `setup5475.sh`: job 22580012, `mba30-004`.
- **Result:** it **failed in 1 s** with `ModuleNotFoundError: No module named 'matplotlib'` (the base
  interpreter ran).
- None of the five scripts uses a GPU.
- The fixed script (CPU partition, `$CONDA_PREFIX/bin/python`) ran every variant from the same kind of
  shell, in both passes.

## 3 · Task 4 — level-set yield surface (`levelset_yield_surface.py --kappa 1.0 --epochs 5000`)

This task does not use `trajectories.npz`. Jobs 22580008, 22580382, 22580640 and, after the reversed
change, 22587606 (`mbcpu-003`, 21 s, 0.33 GB) printed the same digits (18–28 s, 0.33–0.57 GB):
- losses at epoch 4 500: total 1.85e-4, data 1.67e-4, grad 1.80e-5;
- **`Max SDF error: 1.9526e-01`**.

The analysis (`jobs/analyze_levelset.py` in the morning evidence; jobs 22580389 and 22581363, same
training, same digits):

- **The learned yield surface:**
  - the zero level sits at radius 0.99534–0.99841 on 720 rays (max |r − κ| = 4.7e-3, i.e. within 0.5 %);
  - the outward normal is within 0.27° of the analytical one (mean 0.10°);
  - |∇φ| on the surface is 1.025–1.058.
- **Where the printed maximum comes from:** σ ≈ (−0.01, 0.01), the cone tip of the exact SDF |σ| − κ.
  - φ_NN(0, 0) = −0.791 (exact −1).
  - Max |error| on the script's 200 × 200 grid over [−2, 2]²: 0.119 where |σ| ≥ 0.1κ; 0.033 where
    |σ| ≥ 0.25κ; 0.026 where |σ| ≥ 0.5κ.
  - Mean |error| on the grid: 8.2e-3.
- **Outside the sampled box** [−3, 3]²: φ_NN(5, 0) = 3.209 (exact 4). The max error on the circle of
  radius r is 0.014 at r = 2, 0.019 at r = 3, 0.28 at r = 4 and 0.88 at r = 5.
- **|∇φ|** (not trained; the loss matches only the gradient's direction):
  - median 0.44 for |σ| < 0.25;
  - 0.96–1.09 for 0.75 ≤ |σ| < 2;
  - median 0.91 (min 0.60) for 3 ≤ |σ| < 4.3.

The shipped script printed a literal `\n` in the first panel's title. That is fixed; no numbers changed.

## 4 · L21 demo — `icnn_hyperelastic.py` (not a Lab 4 task)

The script is unchanged and does not use the Lab 4 data, so it was not re-run after the reversed change.
Jobs 22580009, 22580384 and 22580641 (25–33 s, 0.32–0.78 GB) gave the same digits each time:

```
Kept 331 of 500 samples after rejecting extreme deformations
epoch     0  total=7.4261e+04  data=7.1870e+04  stress=2.3910e+03
Test psi MSE:    2.6141e+00
Test sigma MSE:  7.9859e+00
Test sigma rel-err: 657.1029%
```

The default 5 000 epochs are far from converged: the loss is 17.3 at epoch 4 500 and still falling.
Variants (135 held-out F from the training distribution):

| run | job | test ψ MSE | test σ relative error | elapsed |
|---|---|---|---|---|
| H¹ (α = 1), 5 000 epochs — the default | 22580641 | 2.61 | **657 %** | 25 s |
| MSE only (α = 0), 5 000 epochs | 22581064 | 13.4 | 1 614 % | 26 s |
| H¹, 20 000 epochs | 22580423 | 2.47e-04 | **10.2 %** | 1 min 13 s |
| MSE only, 20 000 epochs | 22580424 | 3.32e-03 | **42.6 %** | 1 min 25 s |
| H¹, lr 1e-2, 5 000 epochs | 22580425 | 2.13e-02 | 93.7 % | 32 s |
| MSE only, lr 1e-2, 5 000 epochs | 22580426 | 0.178 | 261 % | 34 s |

H² is not implemented in the script; its cost and accuracy are **(not measured)**.

**Is the trained ICNN polyconvex?** Analysis: `jobs/analyze_icnn.py`, jobs 22581062 and 22581063; it
reproduces the printed test numbers. Polyconvexity implies rank-one convexity: the curvature d²ψ/dt² of
ψ(F + t a⊗b) must be ≥ 0. Over 2 700 random (F, a, b) at the held-out F:

| training | rank-one curvature < 0 | min curvature | ∂ψ/∂(I₁−3) < 0 | ∂ψ/∂(I₂−3) < 0 | ∂ψ/∂(−ln J) < 0 |
|---|---|---|---|---|---|
| 5 000 epochs (default) | **1 785 / 2 700 (66 %)** | −36.6 | 98 % of F | 90 % | 100 % |
| 20 000 epochs | 0 / 2 700 | +0.20 | 0 % | 96 % | 43 % |

The neo-Hookean truth has 0 / 2 700 negative.

- **By construction:** the network is convex in its four inputs.
- **Why that is not enough:** convexity in (I₁, I₂, J, −ln J) gives polyconvexity in F only if ψ is also
  non-decreasing in I₁, I₂ and −ln J. The unconstrained skip weights let those slopes go negative.
- **So** polyconvexity is **not** a property of this architecture. The 20 000-epoch model is rank-one
  convex at these samples only because it fits a convex target.

## 5 · Superseded 2026-10-03 — radial loading (0 → a → 0, three times, never reversed)

**Status.** These were the Lab 4 numbers until the instructor adopted fully reversed cycles the same day
(`team/DECISIONS.md`, D3). Nothing in Lab 4 uses them now; they are kept as the record of that design.
- **Input:** the boundary-condition-fixed `j2_cyclic_loading.i` (md5 2ea7522b9baeb0da4b7e40a95e346d0b)
  with the radial factor
  `if(t < 0.5, 2*t, if(t < 1, 2*(1-t), if(t < 1.5, 2*(t-1), if(t < 2, 2*(2-t), if(t < 2.5, 2*(t-2), 2*(3-t))))))`.
- **Driver:** `generate_plasticity_data.py` md5 5df6108638ed329d6712fe7ae7373f3b (the code is the same as
  now).
- **Evidence:** `team/reviews/2026-10-03_M4_lab4_validation_evidence/`.

### 5.1 · The data (was §1.2–§1.3)

| job | date | node | elapsed | MaxRSS | result |
|---|---|---|---|---|---|
| 22579834 | 2026-10-03 09:12 | mbcpu-001 | 2 min 47 s | 0.22 GB | 200/200 runs, `trajectories.npz` |
| 22580636 | 09:24 | mbcpu-001 | 2 min 45 s | 0.22 GB | identical file |
| 22581414 (final driver) | 09:38 | mbcpu-001 | 2 min 50 s | 0.22 GB | identical file |

**The file:** 440 480 bytes, SHA-256 `9c9dbf81ca169b6ad1ad326e9523f53a56d6f30b7231500182462c8d2ec3a13f`.
It is the same shape, (200, 61, 3), with the same amplitudes and the same trial ratios as §1.3 (0.28–2.42,
median 1.17).

**The path:** radial and never reversed, with one triangle wave going 0 → 1 → 0 three times. Max
|ε(t) − f(t) ε_amp| = 1.1e-9.

**Plasticity:**
- 134/200 yield on the first ramp (55 of 0–99, 79 of 100–199).
- **127 yield only on the first ramp**, then unload and reload elastically.
- **7 yield again in every cycle** and draw loops with area: trajectories 100, 105, 114, 116, 118, 135,
  176, all with ratio ≥ 2.12.
- 66 never yield.

**Other numbers:**
- Net work over cycle 2: 0 in 193 trajectories, up to 4.9e-3 in the 7.
- Residual |σ| at ε = 0 at t = 1: median 0.107, max 0.856, above 0.01 in 132.
- Ranges: σ_xx −3.52…3.09, σ_yy −3.43…3.25, σ_xy −0.563…0.539; largest |ε^p| 0.0100.
- The then-handout's one-liner `np.where(np.abs(np.diff(eps_p[:, 20:], axis=1)).max(axis=(1, 2)) > 1e-10)[0]`
  printed `[100 105 114 116 118 135 176]`.

### 5.2 · Tasks 2 and 3 (was §2.1–§2.2)

Posted after Lab 4 closes (Fri Nov 6), together with §2.1–§2.2.

## 6 · Numbers stated before this validation, checked

| where | stated | measured (reversed data, §1–§3, unless noted) |
|---|---|---|
| `lab_4.md` 1b (old) | "~5–10 min locally" | ARCC job 2 min 38 s – 2 min 57 s (radial: 2 min 45 s); local **(not measured)** |
| `lab_4.md` 1b (old) | ".npz typically < 5 MB" | 0.44 MB |
| `lab_4.md` 1c (old) | "Confirm you see hysteresis loops" | 134/200 trajectories draw loops in every cycle (radial: 7/200) |
| `starter_prompts.md` Task 2 (old) | "Reverse-loading branch into the third quadrant" | all 134 yielding trajectories yield in reverse; the σ_xx–ε_xx curve visits quadrants I and III in 127 of them (radial: never reversed) |
| `lab_4.md` 1c, L19 source/deck | T ≈ 60; "≈50 steps", "~10,000 triples" | T = 61 rows (60 steps); 200 × 61 = 12 200 |
| L19 source/deck | `alpha` array in the file | not in the file |
| L19 source/deck | amplitudes "to ≈5× the elastic limit" | at most 2.42× (median 1.17×) |
| L19 deck, L22 deck | non-proportional paths, "direction rotates" | none: every path radial to 1.1e-9 (it reverses sign; it does not rotate) |
| L19 source, L22 source/deck | monotonic paths in the data | none: every path cycles three times |
| L22 source/deck | "~5 min per 1 000 epochs on a CPU" (batch 32) | 2 000 epochs at batch 32: 41 s (radial: 1 min 42 s); batch 16: 43 s – 1 min 52 s (4 cores) |
| L22 deck demo | `python lstm_plasticity.py --epochs 2000 --hidden 32 --layers 2` | error: `--data` is required (morning pass; the script is unchanged) |
| `sobolev_training_primer.md` (old) | MSE-only σ "off by 10–20 %", H¹ "<1 %" | 42.6 % / 10.2 % after 20 000 epochs; 1 614 % / 657 % at the default 5 000 |
| `sobolev_training_primer.md` | H¹ "~2×", H² "~(2+N)×" cost | (not measured) — pass-count heuristics |
| L20 source/deck | Way 3 "~10×" better on rotated test set | (not measured) |


## 7 · After-L19 constitutive-model examples — ARCC validation

**ARCC-Agent validation, 2026-10-16.** These results apply only to the four optional after-L19
companions and do not change or reveal the held Lab 4 Task 2–3 results. At the time of the ARCC report, the companions were
unpublished pending Co-Worker audit and independent review; see the audit update in Section 7.4.

### 7.1 · Build, inputs and execution

- Executable: `/project/me5475/software/moose_torch/me5475/me5475-opt`, SHA-256
  `083a3c61596230246d31abcd010fdcdfff399b703ed0ad00417354530a47ec14`.
- MOOSE revision: `437fbe50823cac0ab39dcf77b2347a0c98bb575a`; course-app revision:
  `d0305f4610037ed93a0ab144f492e7fecd718038`; reported application version: `437fbe5082`.
- Runtime stack: `arcc/1.0`, `gcc/14.2.0`, `openmpi/5.0.5`,
  `hdf5/1.14.3__hl_True__fortran_False-ompi`, `miniconda3/24.3.0`, Python 3.11.15.
- All five validation-file hashes matched the handoff exactly. The eight baseline jobs ran the
  unmodified candidates; each input first passed `--check-input` and then completed with exit code
  `0:0`.

| model | dt = 0.01 job · elapsed · MaxRSS | dt = 0.005 job · elapsed · MaxRSS | states |
|---|---|---|---|
| Drucker-Prager | 24624299 · 10 s · 275960 K | 24624300 · 12 s · 277412 K | 301 / 601 |
| kinematic | 24624301 · 10 s · 263568 K | 24624302 · 13 s · 277528 K | 301 / 601 |
| Voce | 24624303 · 9 s · 275228 K | 24624304 · 11 s · 271952 K | 301 / 601 |
| Hill | 24624305 · 9 s · 263952 K | 24624306 · 11 s · 275496 K | 301 / 601 |

Every CSV was finite through t = 3 with strictly increasing times. The observed step ranges were
0.01 ± 2.3e-16 and 0.005 ± 3.4e-16: no adaptive cutbacks occurred. The largest imposed-strain
error was 1.60e-15 at dt = 0.01 and 3.34e-15 at dt = 0.005. Thus ε_xx = 0.02 f(t) and
ε_yy = ε_zz = ε_xy = 0 were reproduced to the stated 1e-10 absolute tolerance. All four models
developed nonzero plastic strain; maximum plastic-tensor norms at dt = 0.01 were 0.00629148,
0.00527514, 0.00527727 and 0.00202455 in table order.

### 7.2 · Halved-timestep comparison

The 301 coarse states were compared with the dt = 0.005 results at the same times. For each output
column, the scale-normalized difference is
`max(|coarse - fine|) / max(max(|fine|), 1e-12)`. Pointwise relative differences exclude fine-grid
values below `max(1e-10, 1e-6 × column scale)`; they remain ill-conditioned near sign crossings, so
the scale-normalized values below are the primary comparison.

| model | max stress abs. diff. | max stress scale-normalized diff. | max plastic-strain abs. diff. | max plastic scale-normalized diff. |
|---|---:|---:|---:|---:|
| Drucker-Prager | 3.38e-9 | 2.83e-9 | 1.97e-11 | 1.03e-8 |
| kinematic | 6.80e-5 | 2.88e-5 (0.00288 %) | 8.85e-7 | 2.05e-4 (0.0205 %) |
| Voce | 6.41e-9 | 2.56e-9 | 8.33e-11 | 1.93e-8 |
| Hill | 1.25e-3 | 8.05e-4 (0.0805 %) | 1.85e-5 | 1.31e-2 (1.31 %) |

This single halving quantifies timestep sensitivity; it is not a proof of an asymptotic convergence
rate. Hill is the most timestep-sensitive of these four runs, especially in plastic-strain
components that cross zero.

### 7.3 · Model-specific checks

- **Drucker-Prager — validated correction applied locally.** The original input parsed and ran, but
  `TensorMechanicsHardeningConstant` and `TensorMechanicsPlasticDruckerPrager` are expired aliases
  on the pinned build. Replacing them with `SolidMechanicsHardeningConstant` and
  `SolidMechanicsPlasticDruckerPrager` removed the warnings; corrected job 24624328 completed in
  8 s with a CSV numerically identical to the original (maximum difference 0). The pinned source
  gives outer-tip A = 0.14858308, K = 1.22468600 and flow B = 0.07094363 for the chosen angles and
  cohesion. The largest cone residual on plastic increments was 9.07e-7 in the baseline, within
  the 0.001 smoothing scale. A controlled equibiaxial in-plane variant (job 24624329) has the same
  deviatoric strain-amplitude norm but a different pressure path: it first yielded at t = 0.13,
  ε_xx = ε_yy = 0.0104 and P = -1.72957, versus t = 0.19, ε_xx = 0.0152 and P = -1.26093 for the
  x-only case. This demonstrates pressure sensitivity. Across both tests the largest normal-stress
  magnitude was 4.94702, over 2e5 times smaller than the 1e6 cap strengths; the caps are therefore
  inferred inactive, although no direct cap-activity flag was output.
- **Kinematic — pass.** `isotropic_hardening_constant`, `q` and `b` are all zero. The pinned source
  updates backstress as `(2/3) C_k Δε^p - γ X Δp`; thus the input's `kinematic_hardening_modulus = 10`
  is C_k before the 2/3 factor, and γ = 20 supplies nonlinear recovery. At dt = 0.01, first forward
  plastic flow appeared at t = 0.17 and σ_eq = 1.00368; reverse plastic flow appeared at t = 0.58
  and σ_eq = 0.96134. The reduced reverse magnitude and continuing plastic-strain evolution confirm
  active kinematic translation/Bauschinger behavior.
- **Voce — pass.** Pinned source evaluates `hardening_function` at old accumulated effective plastic
  strain plus the current increment, then subtracts the separately supplied initial yield stress.
  The function must therefore return total yield stress, as written. Reconstructing path-accumulated
  p from the dt = 0.005 plastic-strain increments gave p = 0.0362260 at the last plastic point and
  σ_Y = 1.25772147; measured equivalent stress agreed within 2.34e-9. The theoretical tangent fell
  from Qb = 10 to 4.84557, consistent with saturation toward σ0 + Q = 1.5 rather than dependence on
  simulation time.
- **Hill — pass with measured timestep sensitivity above.** Pinned source constructs the stated
  Hill matrix and returns ω = sqrt(Q_H/2). During plastic increments, maximum |ω - 1| was 1.64e-10
  for x loading and 9.33e-14 for y loading. With material axes fixed, the x-directed run first
  yielded at t = 0.22, ε_xx = 0.0176 and σ_xx = 2.36569; the y-directed controlled variant
  (job 24624330) first yielded at t = 0.21, ε_yy = 0.0168 and σ_yy = 2.22696. This directional
  difference is consistent with the ideal-uniaxial normalized thresholds 1.34840 (x) and 1.24035
  (y), while the measured runs remain constrained plane-strain rather than uniaxial-stress tests.
  `hardening_constant = 0` was retained.

### 7.4 · Evidence and release status

Compact evidence is in `team/reviews/2026-10-16_M4_constitutive_arcc_evidence/`; the validation
report is `team/reviews/2026-10-16_arcc_constitutive_examples_validation.md`. Full run logs, syntax
dump and CSVs remain under
the instructor's validation folder in the course project space on MedicineBow.

Verdict: kinematic, Voce and Hill pass this execution gate. Drucker-Prager fails **as submitted**
because it uses expired object aliases; the exact two-name correction was validated and is
numerically identical. The Co-Worker audit below applies that patch and checks the evidence.
The independent Reviewer cleared the package on Oct 19 (`2026-10-16_reviewer_constitutive_reading_regate.md`), and it was published the same day.


**Co-Worker audit, October 16 [fallback].** Applied the validated `SolidMechanics...` object names to the local companion. All four input bodies match the ARCC-tested versions after comments are removed; status comments are updated. Recomputed the diagnostics from all 11 CSVs and reproduced the reported values. Sections 1–6 and held Lab 4 placeholders are unchanged. The reading and release candidates are updated; publication still awaits independent Reviewer clearance. Audit evidence: `team/reviews/2026-10-16_M4_constitutive_coworker_audit/`.

---

## 8 · Frame-indifference primer — worked arithmetic (`frame_indifference_checks.py`)

**Worked arithmetic and transformation checks, not fitted-model performance.** Nothing here is trained or
measured. These are exact tensor operations, rounded for printing, and they back every number in
`frame_indifference_primer.md` and on L20 slides 3–4.

- **Run:** 2026-10-19, `[fallback]` Co-Worker (Opus 5.5).
- **Where:** the instructor's laptop, in the course demo environment (`~/ME5475-demo-env`): Python 3.9.6, NumPy
  2.0.2, macOS arm64. Cluster job: not applicable.
- **Command:** `python module_4/examples/frame_indifference_checks.py`. It is deterministic, with no random numbers.
- **Script:** SHA-256 `b8bd7febba8624a49cb6fd071f44a8fc4d7a6c0d295717f883f05436993d562d`.
- **Independent rerun:** the Reviewer reran it on 2026-10-19 and got byte-identical output.
- **Inputs, all chosen for illustration:**
  - **Passive rotation:** θ = 30°, with Q_ij = e'_i · e_j, so Q = [[cos θ, sin θ], [−sin θ, cos θ]].
  - **Stress:** σ = [[2, 1], [1, 0]].
  - **Strain:** ε = [[1, 0.5], [0.5, 0]] × 10⁻³.
  - **Isotropic stiffness:** plane strain, E = 1, ν = 0.3.
  - **Orthotropic stiffness:** C₁₁₁₁ = 10, C₂₂₂₂ = 1, C₁₁₂₂ = 0.3, C₁₂₁₂ = 0.5.
  - **Deformation:** F = [[1.1, 0.2], [0, 0.95]], with a superposed rotation R(45°).
  - **Neo-Hookean energy:** ψ = ½μ (tr C − 2) − μ ln J + ½λ (ln J)², with E = 1, ν = 0.3.
  - **Non-objective energy:** ψ_bad = ½μ |F − I|².

| item | quantity | value |
|---|---|---|
| 1 | v = (1, 0) → Q v | (0.866, −0.5), length 1 |
| 2 | σ' = Q σ Qᵀ | [[2.366, −0.366], [−0.366, −0.366]]; trace 2, det −1, principal values 2.4142, −0.4142 (unchanged) |
| 3 | ε' (× 10⁻³); γ' = 2ε'₁₂ | [[1.183, −0.183], [−0.183, −0.183]]; −0.366 × 10⁻³ |
| 4 | isotropic C₁₁₁₁; C' = C | 1.3462; true |
| 5 | orthotropic C'₁₁₁₁, C'₂₂₂₂, C'₁₁₂₂, C'₁₂₁₂, C'₁₁₁₂ | 6.1750, 1.6750, 1.8750, 2.0750, −2.8579; σ' = C' : ε' equals Q (C : ε) Qᵀ: true |
| 6 | C = FᵀF before and after R | [[1.21, 0.22], [0.22, 0.9425]], both; b* = R b Rᵀ: true |
| 7 | neo-Hookean ψ(F), ψ(RF) | 0.012956, 0.012956; σ* = R σ Rᵀ, P* = R P, S* = S: true |
| 8 | ψ_bad(F), ψ_bad(RF) | 0.010096, 0.186638 |
| 9 | small strain under a pure 45° rotation; E; 1° rotation | ε₁₁ = ε₂₂ = −0.2929; E = 0; ε₁₁ = −1.523 × 10⁻⁴ |

The full output is in `team/reviews/2026-10-19_L20_fix_evidence/frame_indifference_checks.out`.

