# Module 3 — measured results

Every computed number in a Module 3 reading or handout must appear here first, with what was run,
where, and when. A reading cites this file; it never carries a number that was not measured. If a
value is wanted but unmeasured, the reading says **(not measured)**.

`[fallback]` — measured by the Co-Worker (Opus 5.5) and a Co-Worker subagent, 2026-09-24/25.

---

## Environments

| id | where | versions |
|---|---|---|
| **ARCC-GPU** | MedicineBow `mb-a30` (A30), `/project/me5475/envs/ml4sm`, via `pinn_train.sbatch` | python 3.11.15, torch 2.5.1+cu121, DeepXDE 1.15.0 |
| **MOOSE** | `/project/me5475/software/rom_opt_arcc/rom_opt-opt` | not on `PATH`; call by full path |

---

## 1 · The shared MOOSE reference — square plate, quarter hole

Input `module_3/examples/plate_square_hole_reference.i` (the PINN's exact problem: [0,1]² minus a
quarter disk r = 0.1; u₁ = 0 left, u₂ = 0 bottom, u₁ = 1 right; top and hole traction-free; plane
strain, E = 1, ν = 0.3). Staged: `/project/me5475/examples/lab3_reference/` (`…_out.e`, `…_out.csv`).
Date 2026-09-24, ARCC jobs 18630270_[0-3], 18630271.

| mesh (nt/nr) | elements | job wall time | σ_xx at (0, 0.1) | rel. L2(u) vs finest |
|---|---|---|---|---|
| 24/48 | 2 304 | 7 s | 3.23799 | 9.4e-6 |
| 48/96 | 9 216 | 9 s | 3.24086 | 2.3e-6 |
| 96/192 | 36 864 | 20 s | 3.24160 | 4.8e-7 |
| **192/384 (shared)** | 147 456 | 1 min 33 s | **3.24179** | — |

Job wall time is SLURM's `Elapsed` for the whole job (start-up included) on 8 CPU cores of partition
`mb` (`sacct`, read 2026-09-27); the staged run, job 18630271, which also writes the full output, took
2 min 27 s. For comparison, each of the six PINN jobs in §6 took 11–13 minutes on an A30 GPU (whole job;
9.9–11.8 minutes of training).

Finest level: **reaction on the loaded edge `reaction_right_x` = 1.0734459** (left edge −1.0734458;
imbalance 1.8e-7); average σ_xx on the right edge 1.07345; area 1 − π/400 to 1e-14.

> **Note (2026-09-27) on §2–§6 — the sampler.** These results use the released DeepXDE sampler, which includes some
> points inside the removed hole. At the forward script's default seed 42, 52 of 4 600 training points are
> affected, including left- and bottom-edge BC samples. The field errors remain measured against the physical
> MOOSE reference. The soft/hard pairs share the same sampler, so they compare the released implementations;
> they do not isolate enforcement on a corrected sampling of the intended domain. A one-seed filtered run
> changed errors in both directions.

## 2 · Soft-BC PINN vs the reference — the three σ₁₂ = 0 conditions matter

`plate_with_hole_fixed.py`, defaults (Adam 50 000 + L-BFGS), `sbatch pinn_train.sbatch`, ARCC-GPU.
Error = relative L2 on the 200×200 reference grid inside the domain (39 671 points).

| | as shipped (σ₁₂ BCs on left/bottom/right missing) | with the three BCs restored |
|---|---|---|
| job, date | 18630631, 2026-09-24 | 18923098, 2026-09-25 |
| final training loss | 7.3e-6 | 6.6e-6 |
| rel. L2 error, u = (u₁, u₂) | **0.597** | **1.20e-3** |
| rel. L2 error, σ₁₁ / σ₂₂ / σ₁₂ | 0.706 / 9.67 / 9.12 | 2.5e-3 / 7.7e-2 / 3.4e-2 |
| σ₁₁ at the nearest grid point to the hole top, (0, 0.1005) (MOOSE 3.204 there; 3.2418 exactly at (0, 0.1), §1) | **0.063** | 3.100 |
| wall time | 9 min 09 s | 8 min 20 s |

Same training budget, near-identical training loss, but the shipped script solved an
under-constrained problem. **A small PINN loss does not certify the problem was posed right.**

## 3 · Collocation budget — 1 500 vs 4 000 domain points (all with the three BCs)

ARCC-GPU, 2026-09-25, `sbatch pinn_train.sbatch [hard_bc]`. "4 000" = `num_domain=4000,
num_boundary=600` (1 500 = 1500/300, Min Lin's notebook value). Validation variants in
`~/me5475-demo/L3-validate/`. rel. L2 on the 39 671-point reference grid. **Schedules differ:** soft runs
used Adam 50 000 + L-BFGS, hard runs the then-default Adam 25 000 + L-BFGS — so wall times here are not a
soft-vs-hard speed comparison (§6 is the controlled one). The σ₁₁ column is the value at the nearest grid
point to the hole top, (0, 0.1005), where MOOSE gives 3.204 (3.2418 exactly at (0, 0.1), §1).

| run | job | wall | final train / **test** loss | rel. L2 u | σ₁₁ at (0, 0.1005) (MOOSE 3.204) |
|---|---|---|---|---|---|
| soft BC, 1 500 (Min's budget) | 18923098 | 8 min 20 s | 6.6e-6 / 1.4e-3 | 1.20e-3 | 3.100 |
| **soft BC, 4 000** | 18957698 | 10 min 25 s | 8.2e-6 / **1.8e-5** | **6.0e-4** | 3.174 |
| hard BC, 1 500 | 18923099 | 5 min 48 s | 8.3e-7 / **7.7** | 4.8e-2 | 0.239 |
| hard BC, 1 500 + resample every 1 000 Adam steps | 18935477 | 7 min 31 s | 9.4e-6 / 0.93 | 1.8e-2 | 2.084 |
| **hard BC, 4 000** | 18935475 | 8 min 07 s | 9.1e-6 / **1.9e-5** | **1.04e-3** | **3.2015** |

With 1 500 points the hard-BC PINN fits the fixed collocation points and oscillates between them:
its x-momentum residual on the test points grows from ~1e-3 to 7.5 while the training residual
falls. The one resampling schedule tried (every 1 000 Adam steps, then L-BFGS on one fixed set) did not
cure it; the measured 4 000/600 run did (one run — not a minimum adequate count). The shipped scripts now default to 4 000/600 for all four PINN scripts.

## 4 · Inverse PINN — E and ν from displacements plus one force

`plate_with_hole_inverse.py`, seed 0, 20 displacement points + the reaction force, 1 % noise on
both (force measurement 1.06876 vs reference 1.07345), Adam 30 000 + L-BFGS, ARCC-GPU.

| collocation | job | wall | final test loss | E (truth 1.0) | ν (truth 0.3) |
|---|---|---|---|---|---|
| 1 500 | 18923101 | 12 min 05 s | 3.97 | 1.0001 | 0.2930 |
| **4 000 (shipped)** | 18957700 | 9 min 05 s | **1.3e-4** | **0.9992** | **0.2949** |

Two more seeds, 2026-09-26, through `pinn_inverse_ensemble.sbatch` (`--array=1-2`, job 19273831),
each with its own 20 measurement points, noise draws, collocation points and initial weights:

| seed | wall | final test loss | E | ν |
|---|---|---|---|---|
| 1 | 8 min 40 s | 2.1e-4 | 1.0005 | 0.3005 |
| 2 | 9 min 18 s | 2.1e-4 | 1.0084 | 0.2968 |

Three seeds (0–2): E within 0.9 % and ν within 1.8 % of the truth; all admissible. The full 10-seed
spread (Task 5) is **(not measured)**; each seed varies several sources at once (Task 5e).

## 5 · Sweep — full 6 × 4 rehearsal of Task 4

`pinn_optuna_sweep.sbatch` exactly as shipped (`--array=0-5%2`, 4 trials per worker), job 19273830,
2026-09-26, ARCC-GPU, with the final scripts (hashes: `optionA/code_md5.txt`). One reference-split
fingerprint (`3edc926b52`, 19 835 validation points) in every worker log; 0 errors.

| | |
|---|---|
| trials | **24: 13 completed, 11 pruned, 0 failed** |
| worker wall times | 5 min 36 s – 43 min 13 s (limit 2 h) |
| whole sweep, two GPUs at a time | 11:55:46 → 13:22:12 = **1 h 26 min** |
| GPU time, all six workers | **2.6 GPU-hours** |
| best validation rel. L2 (u) | 2.66e-4 (trial 4: lr 2.6e-3, 5 × 50, Adam 50 000, bc weight 11.2) |
| fANOVA importance | lr 0.36, adam_epochs 0.28, bc_loss_weight 0.28, width 0.08, layers 0.003 (exploratory) |

**Note for grading:** this correct run completed 13 trials; pruning decides how many complete. (Earlier
2-worker × 1-trial smoke test, job 18957872, 2026-09-25: both COMPLETED, 0 errors.)

## 6 · Soft vs hard BCs — controlled, three seeds (Task 3's comparison)

*See the sampler note before §2: both runs of each pair use the released sampler.*

2026-09-26, ARCC-GPU, jobs 19273824–19273829, `SEED=<s> sbatch pinn_train.sbatch [hard_bc]` with the
final scripts (hashes in `~/me5475-demo/L3-validate/optionA/code_md5.txt`). Within each pair: the same
4 000 + 600 sampled points and the same initial weights (checked: identical arrays at seeds 42 and 43),
the same network (6 × 50) and the same schedule (Adam 50 000, lr 1e-3, then L-BFGS). Errors are
relative L2 on the 39 671-point reference grid; σ₁₁ is evaluated exactly at (0, 0.1), where MOOSE gives
**3.2418** (§1); the Dirichlet error is the largest violation of u₁ = 0 (left), u₂ = 0 (bottom), u₁ = 1
(right) on 500 freshly sampled points per edge. Scoring: `optionA/score_pair.py` (validation only).

| seed | BCs | u | σ₁₁ | σ₂₂ | σ₁₂ | σ₁₁(0, 0.1) | Dirichlet err | Adam / L-BFGS (s) | L-BFGS steps |
|---|---|---|---|---|---|---|---|---|---|
| 42 | soft | 6.01e-4 | 1.77e-3 | 3.29e-2 | 1.96e-2 | 3.1999 | 1.8e-3 | 461 / 176 | 9 824 |
| 42 | hard | 9.81e-4 | 1.71e-3 | 2.15e-2 | 1.64e-2 | 3.2350 | 0 | 508 / 167 | 9 211 |
| 43 | soft | 6.99e-4 | 2.01e-3 | 3.76e-2 | 2.16e-2 | 3.1327 | 2.0e-3 | 463 / 155 | 8 607 |
| 43 | hard | 8.28e-4 | 1.42e-3 | 2.12e-2 | 1.33e-2 | 3.2158 | 0 | 501 / 204 | 11 034 |
| 44 | soft | 5.55e-4 | 1.88e-3 | 3.84e-2 | 2.06e-2 | 3.1267 | 2.4e-3 | 437 / 158 | 9 290 |
| 44 | hard | 2.62e-3 | 3.32e-3 | 3.13e-2 | 2.97e-2 | 3.2167 | 0 | 489 / 170 | 9 425 |

**What the three pairs show (a release check, not proof):** soft had the lower displacement error in
3/3 pairs; hard had the lower σ₂₂ error in 3/3, lower σ₁₁ and σ₁₂ errors in 2/3, and σ₁₁ at the hole
top closer to MOOSE in 3/3; hard met the displacement conditions exactly (soft: ~2e-3). **Under the same
schedule the hard runs were not faster** — 6–14 % more training time, Adam + L-BFGS from the table (5–13 % for
the whole job, SLURM `Elapsed`); Adam alone, with the same 50 000 steps, took 8–12 % longer: the transform
adds work per step;
the "2× faster" of earlier drafts compared unequal schedules (hard had half the Adam steps) and is
superseded by this controlled comparison. Seed 42 soft
reproduces §3's run exactly (6.014e-4): the runs are deterministic on this hardware.

## 7 · L13 — 1-D and 2-D diffusion PINNs (`pinn_diffusion_1d.py`, `pinn_diffusion_2d.py`)

Measured 2026-09-26 by a Co-Worker subagent (Opus 5.5) `[fallback]`; merged here by the Co-Worker. Full
evidence (job logs, plots, validation-only code): `team/reviews/2026-09-26_L13_evidence/`; report:
`team/reviews/agent_reports/2026-09-26_coworker_subagent_L13_prep.md`.

### 7.0 · Environments for §7

| id | where | versions |
|---|---|---|
| **ARCC-CPU** | MedicineBow partition `mb`, node `mbcpu-001` (AMD EPYC 9454), `/project/me5475/envs/ml4sm`, 4 cores unless stated, via `module_3/examples/pinn_diffusion.sbatch` | python 3.11.15, torch 2.5.1+cu121 (CPU), DeepXDE 1.15.0, numpy 2.4.4 |
| **ARCC-GPU** | partition `mb-a30,mb-l40s`, node `mba30-007` (A30) | same env |
| **Laptop** | instructor's M2 Max, `~/ME5475-demo-env` interpreter (python 3.9.6, torch 2.8.0); DeepXDE 1.15.0 + its 7 missing dependencies **side-loaded from a scratch directory via `PYTHONPATH`** — the env itself was not modified (`pip list` identical before/after) | torch 2.8.0, DeepXDE 1.15.0 |

Script hashes (md5). **Rehearsed** (jobs 19296464/65, `~/me5475-demo/L13-validate/rehearsal/`):
`pinn_diffusion_1d.py` 0519114c93ce1b129d54f00eb8499f56, `pinn_diffusion_2d.py` c45e90b450efa40eba8dbfbcb3ef6149,
`pinn_diffusion.sbatch` a079287e4ce1bf322d57d4dbfc407f40. **Final** (repository and
`~/me5475-demo/L13-validate/live/`, the class folder): 477b7e7902cb88d69d48a770192ec25d,
26f82c161276a1caacccdfa337213d52, a0338ade38951dbec1602b23c5342cae — they differ from the rehearsed files
**only in comment and docstring lines** (run times worded from this ledger; `tail -F`), checked with `diff`
on ARCC. Jobs 19291289–19291303 ran an earlier revision that differs only in plot styling (legend position,
colour-bar ticks); their training numbers are identical to the final revision's, digit for digit.

### 7.1 · The two teaching scripts (seed 42, defaults) — ARCC-CPU

| | 1-D (`pinn_diffusion_1d.py`) | 2-D (`pinn_diffusion_2d.py`) |
|---|---|---|
| collocation points | 52 = 50 inside + 2 ends (residual at all 52; BC at the 2 ends) | 2 200 = 2 000 inside + 200 on the edges |
| test points | 102 = 100 evenly spaced (k/101) + the 2 training ends | 1 224 = 1 024 (32 × 32 grid) + the 200 training edge points |
| end of Adam, 5 000 its: ℒ_PDE / ℒ_BC | 5.78e-5 / 8.40e-7 | 4.74e-4 / 5.07e-3 |
| end of Adam: test rel. L2 error | **1.09e-3** | **9.61e-2** |
| L-BFGS iterations (stops itself) | **60** | **1 501** |
| end of L-BFGS: ℒ_PDE / ℒ_BC | 1.21e-5 / 6.82e-10 | 1.94e-6 / 9.03e-7 |
| end of L-BFGS: test rel. L2 error | **2.77e-5** | **1.05e-3** |
| max \|u_NN − u\| | 3.54e-5 (1 001 points) | 2.54e-3 (101 × 101 grid) |
| Adam / L-BFGS time (s) | 8.5–11.9 / 0.15–0.21 | 64–124 / 23–37 |
| job elapsed (start → end) | 26–29 s | 1 min 50 s – 2 min 33 s (the shipped script, inside a longer job: 2 min 56 s) |

Jobs: baseline as shipped 19290697; final 19291289/90 (run_a), 19291291/92 (run_b), 19291293/94 (1 core),
19291295 (2 cores), 19291296/97 (8 cores); rehearsal 19296464/65. **Every CPU run printed identical
numbers** — 1-D: 6 runs plus the `l13_experiments.py` mirror; 2-D: 7 runs; 1 to 8 cores, alone or with 14 other
jobs on the node (the shipped scripts' baseline 19290697 printed no L-BFGS test error, but every value it did
print matches). (All ran on
`mbcpu-001`; another CPU model could differ in the last digits.) Times vary with node load: 2-D Adam
took 64 s in one run and 124 s in another with the same settings.

Loss table, 1-D (the "they need not move together" evidence): step 3000 [9.27e-5, 4.73e-7] test error
7.26e-4; 4000 [6.80e-5, 5.37e-8] 3.57e-4; **5000 [5.78e-5, 8.40e-7] 1.09e-3** — the total loss fell
(6.81e-5 → 5.86e-5) while the test error rose ×3.05; ℒ_BC rose ×15.6. √(8.40e-7) = 9.2e-4 is the size of the
boundary miss, consistent with the 1.1e-3 error.

Loss table, 2-D test error: 0.865, 0.145, 0.124, 0.116, 0.105, 0.0961 at steps 0–5000; 1.49e-3 at 6000;
1.05e-3 at 6501. A spike in ℒ_PDE at step 4000 (4.35e-2) under Adam.

### 7.2 · The classroom sequence, rehearsed — ARCC-CPU, 2026-09-26 13:50

Exact demo commands from `~/me5475-demo/L13-validate/rehearsal/` (both jobs submitted back to back):

| | 1-D (19296464) | 2-D (19296465) |
|---|---|---|
| queue wait | 18 s | 18 s |
| job elapsed | 26 s | 1 min 51 s |
| submit → job end | **44 s** | **2 min 9 s** |
| Adam / L-BFGS | 10.3 s / 0.18 s | 72.0 s / 22.9 s |

Every earlier job that day started within 1–20 s (Saturday; a Wednesday 13:00 queue is **not measured**).
`scp` of both PNGs and both logs to the laptop: 5 s. **The loss table streams live:** with
`PYTHONUNBUFFERED=1` under `srun`, rows appeared in the `.out` file as printed (2-D poll: 1 row at +32 s,
3 at +81 s, 5 at +112 s, 9 at +160 s; `arcc_logs/live_stream_poll_2d.log`). `import deepxde` on a cold
node took 22 s once (job 19290697); warm, the whole 1-D job is 26 s.

### 7.3 · Claims tested

| claim (old deck / source) | test | result |
|---|---|---|
| "50 random points" | `introspect` in 19290697; `keystone_and_source_checks.py` (19291299) | DeepXDE 1.15 default `train_distribution="Hammersley"`; points identical under seeds 42 and 7; first 8 in order 0, 1, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375; **all 52 are multiples of 1/64** |
| keystone: "both curves give the same ℒ_PDE" | analytic, at DeepXDE's own points (19291299) | the old drawing was wrong (see report); for **u_bad = sin(πx) + 0.1 sin(64πx)**: ℒ_PDE = 9.6e-22 at the 52 training points, ℒ_BC = 2.2e-31; ℒ_PDE = 8.25e6 at the 100 interior test points; test rel. L2 error **0.100**; max \|u_bad − u\| 0.100 |
| "≈1e-3 after Adam, ≈1e-6 after L-BFGS" | §7.1 | 1-D ℒ_PDE 5.8e-5 → 1.2e-5; 2-D 4.7e-4 → 1.9e-6 |
| "L-BFGS first, from a random start, is the fastest way to NaN" | `--mode lbfgs` (19291299, 19291300) | **false here.** 1-D: 391 its, 1.93 s, test error **9.2e-6**, max 1.29e-5, finite. 2-D: 1 657 its, 35.7 s, test error **7.4e-4**, max 2.57e-3, finite |
| "L-BFGS runs to its own convergence" | §7.1 + seeds + float64 | float32 1-D: 60 its (seed 42), 89 (seed 1), 2 (seed 2), 5 (seed 3); 2-D: 1 501. float64 1-D (19291299): 15 765 its (DeepXDE cap 15 000 per call, checked between 1 000-its steps), 58 s, test error 2.7e-6 — different init under float64, so not a controlled comparison |
| seeds matter (1-D) | seeds 1, 2, 3 (19291299) | end-of-Adam test error 4.1e-2 / 9.6e-5 / 5.6e-5; final 1.7e-5 / 9.3e-5 / 5.4e-5 (seed 42: 1.09e-3 → 2.77e-5) |
| "needs 2 000 points in 2-D" | `--num-domain 50 / 500` (19291302, 19296314, 19296320) | final test error, seeds 42/1/2 — **50 inside: 1.17e-3 / 1.61e-3 / 1.74e-3**; 500 inside (seed 42): 8.3e-4; **2 000 inside: 1.05e-3 / 8.72e-4 / 1.64e-3** (19291290, 19296312, 19296318) |
| "a 2-D field needs 6 hidden layers" | `--hidden 4` (19291301, 19296316, 19296321) | **4 hidden: 6.40e-4 / 1.19e-3 / 9.00e-4** (seeds 42/1/2) vs 6 hidden above |
| 2-D after Adam | the six 2-D runs above | test error 8.5–10.4 % after 5 000 Adam steps in all three seeds (6 layers, 2 000 points) |
| "if slow, --iterations 2000" | 19291303 | Adam 62 s + L-BFGS 52 s (1 632 its): barely shorter; final 1.72e-3 |
| "a 1-D PINN doesn't need a GPU"; GPU for 2-D | 19291298 (A30) | 2-D on GPU: Adam 27.4 s, L-BFGS 21.9 s (1 670 its), job 66 s; different numbers from CPU (final test error 6.51e-4); prints a harmless cuBLAS context warning. GPU queue started at once on this Saturday; weekday not measured. 1-D on GPU **not measured** |
| lecture source's code block (`np.sin`, `dde.maps.FNN`, `epochs=5000`) | 19291299 | **fails** at the first step: `RuntimeError: Can't call numpy() on Tensor that requires grad`. With `torch.sin` it runs; `epochs=` still accepted with *"epochs is deprecated … Use iterations instead"* |
| `dde.maps` | 19290697 | exists in 1.15; `dde.maps is dde.nn` → True |
| "put DDE_BACKEND after the import: silently ignored" | DeepXDE 1.15 source, `backend/__init__.py` | the variable is read only at import; DeepXDE prints `Using backend: …` either way; with no variable and no `~/.deepxde/config.json` it prints "No backend selected. Finding available backend…" and writes the config file |

### 7.4 · Finite elements for the same 1-D problem — laptop, numpy

`team/reviews/2026-09-26_L13_evidence/fe_comparison_1d.py`: N equal linear elements, load by 5-point Gauss per element, dense solve; errors against
sin(πx); "rel L2 (test)" on DeepXDE's own 102 test points; median of 2 000 assemble+solve repeats (naive Python
loop assembly).

| N | max nodal error | max error (100 001 points) | rel. L2 (test points) | assemble + solve |
|---|---|---|---|---|
| 4 | 3.3e-13 | **7.0e-2** | 5.6e-2 | 131 µs |
| 16 | 3.3e-16 | 4.8e-3 | 3.5e-3 | 274 µs |
| 50 | 4.0e-15 | 4.9e-4 | 3.7e-4 | 700 µs |
| 100 | 2.7e-15 | 1.2e-4 | 1.1e-4 | 1.3 ms |
| **200** | 6.1e-15 | **3.1e-5** | 2.3e-5 | **2.7 ms** |

Nodal values are exact to rounding for every N (1-D Galerkin superconvergence); "four elements to machine
precision" is true only at the nodes. 200 elements match the PINN's max error (3.5e-5) in 2.7 ms.

### 7.5 · Laptop (for the venue decision only)

Final scripts, `~/ME5475-demo-env/bin/python` + side-loaded DeepXDE (`team/reviews/2026-09-26_L13_evidence/laptop_logs/laptop_final.log`):

| | 1-D | 2-D |
|---|---|---|
| Adam | 40.4 s (same with 1 or 4 OMP threads) | 66.4 s |
| L-BFGS | **2 its**, 0.14 s | 742 its, 39.4 s |
| test error, end of Adam → end | 1.23e-4 → 1.13e-4 | 7.82e-2 → 1.24e-3 |
| max error | 1.33e-4 | 3.16e-3 |
| wall | 44 s (73 s on the first, cold run) | 109 s |

**Different numbers from ARCC** (torch 2.8 vs 2.5.1: different initial weights from the same seed —
initial ℒ_PDE 45.8 vs 50.2), and on the laptop the 1-D L-BFGS "drop" does not happen. What
`pip install deepxde` would add to the demo env (`pip install --dry-run`, report in `team/reviews/2026-09-26_L13_evidence/laptop_logs/`):
DeepXDE 1.15.0, scikit-learn 1.6.1, scikit-optimize 0.10.2, scipy 1.13.1, joblib 1.5.3, threadpoolctl 3.7.0,
pyaml 26.7.0, PyYAML 6.0.3 — 8 packages, 131 MB installed; numpy, matplotlib and torch already satisfy it.

## 8 · Plane stress vs plane strain — the same plate, both models (for the plane-stress/strain primer)

**Date and job:** 2026-09-27, ARCC job 19576019. Node `mbcpu-001`, partition `mb`, 8 MPI ranks; 1 min 52 s
for the whole job, including two grid extractions. Evidence: `team/reviews/2026-09-27_planestress_evidence/`,
containing the job script, the comparison script, the job log and the postprocessor CSV.

**What was run.**
- The shared plane-strain input, `module_3/examples/plate_square_hole_reference.i`, on the shared mesh
  (nt = 192, nr = 384), unchanged except for two command-line overrides:
  - `youngs_modulus = E(1 + 2ν)/(1 + ν)² = 0.9467455621301775`;
  - `poissons_ratio = ν/(1 + ν) = 0.23076923076923078`.
- Plane strain with these constants is exactly plane stress with E = 1 and ν = 0.3, the constants of Min
  Lin's notebook. The mapping is exact for u₁, u₂, σ₁₁, σ₂₂ and σ₁₂; this run's σ₃₃ and von Mises values are
  not plane-stress values and are not used.
- Both solutions were sampled on the same 200 × 200 grid with `extract_moose_reference.py`: 39 671 points,
  0 NaN.

| quantity | plane stress | plane strain (§1) | ratio |
|---|---|---|---|
| σ₁₁ at (0, 0.1) | 2.9500246 | 3.2417853 | 0.9100000 |
| mean σ₁₁ on the loaded edge | 0.9768370 | 1.0734472 | 0.9100000 |
| reaction on the loaded edge | 0.9768360 | 1.0734459 | 0.9100002 |
| σ₁₁(0, 0.1) / mean σ₁₁ on the loaded edge | 3.01998 | 3.01998 | 1 |

Plane-stress equilibrium check: the left and right reactions sum to 3.3e-8.

**Relative L2 difference on the grid, plane stress vs plane strain:**
- u₁: 1.02e-3;
- u₂: **2.92e-1**;
- σ₁₁, σ₂₂ and σ₁₂: **9.0000e-2 each**.

**Out-of-plane stress in plane strain.** σ₃₃ = ν(σ₁₁ + σ₂₂) reaches a maximum of 0.966 on the grid; its
relative L2 size is ‖σ₃₃‖/‖σ₁₁‖ = 0.301.

**Reading.**
- **The exact result.** For this plate, each in-plane stress component (σ₁₁, σ₂₂, σ₁₂) is exactly (1 − ν²) = 0.91 times its plane-strain value.
  - It holds for these boundary conditions only: frictionless constant-u₁ grips left and right, bottom symmetry, and a traction-free top and hole, with homogeneous isotropic small-strain elasticity and no body force.
  - It is not a general conversion factor. The argument is in `plane_stress_and_plane_strain_primer.md` §5.
  - σ₃₃ is not scaled: it is zero in plane stress and generally nonzero in plane strain.
- **The FE numbers agree within numerical error.** On the grid, ‖σ_ps − 0.91 σ_pe‖ / ‖σ_pe‖ is:
  - σ₁₁: 2.0e-10;
  - σ₂₂: 1.0e-8;
  - σ₁₂: 5.1e-9.

  The largest pointwise |σ_ps − 0.91 σ_pe| is 8.5e-8, against field maxima of 0.90–3.20. This came from `scaled_residual.py` on the saved grids (arithmetic on a login node, no new solve); its output is in the evidence folder. The reaction ratio, 0.9100002, is likewise within numerical error of 0.91.
- **The displacements do change.** The relative L2 difference in u₂ is 29 %, because the effective
  Poisson ratio is 0.3 in plane stress and 0.43 in plane strain.
