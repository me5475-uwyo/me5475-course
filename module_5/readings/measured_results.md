# Module 5 — measured results

Every computed number in a Module 5 reading or in Lab 5 must appear here first, with what was run, where, and when.
A reading cites this file; it never carries a number that was not measured. If a value is wanted but unmeasured, the
reading says **(not measured)**.

`[fallback]` — measured by a Co-Worker subagent (Opus 5.5) on ARCC, 2026-10-03 (evening) and 2026-10-04 (early
morning).
- Evidence: `team/reviews/2026-10-03_M5_lab5_evidence/` (job scripts, every `.out`/`.err`, `sacct`, figures).
- Report: `team/reviews/agent_reports/2026-10-03_coworker_subagent_M5_lab5.md`.
- ARCC work folder: `/gscratch/xzhang16/me5475-demo/M5-validate/`.

**Held until Lab 5 closes (Wed Nov 11, 11:59 PM MT):** values computed from the network that a student reports in
Lab 5: the network run's strain states and coverage (§3.1), the network-vs-elastic error values (§3.2), the extrapolation error values and the Task 4a probe (§4.2), and
the Task 5 network values (§6.2). With the course data and Lab 1's default settings, our two training runs on ARCC
gave byte-identical networks (§1.3), so a student with the same setup can expect these values to be the answers.
They are kept in `team/reviews/2026-10-03_M5_lab5_evidence/measured_results_M5_HELD_until_Nov11.md` and are restored
here after the lab closes.
- **Published:** run times, Newton iterations, failure messages, and the elastic and analytical reference values.
  Lab 0's handout already gives the elastic peaks, and Lab 3's ledger the reference values.
- **Status:** approved by the instructor on 2026-10-26 (`team/DECISIONS.md`, H1).

---

## Environments

| id | where | versions |
|---|---|---|
| **ARCC-CPU** | MedicineBow partition `mb`, nodes `mbcpu-001/002/003/010/016`; Python on the login node `mblog2` for the seconds-long steps (export, verify, comparisons, grid extraction) | `/project/me5475/envs/ml4sm`: python 3.11.15, torch 2.5.1+cu121 (CPU), numpy 2.4.4, pandas 3.0.3, matplotlib 3.10.9 |
| **MOOSE** | `/project/me5475/software/moose_torch/me5475/me5475-opt`, runtime block of `team/arcc_build/README.md`, 4 MPI ranks | MOOSE 437fbe5082, libtorch 2.5.1 CPU. `me5475-opt` SHA-256 `083a3c61…ec14`; `lib/libme5475-opt.so.0` SHA-256 `89f7afea…c9b5` (the hardened `ComputeTorchScriptStress`, relinked 2026-10-03 23:03 MDT, job 22952328). Every MOOSE result below was run with these two files, except the data job 22947144 (§1.1). The validation jobs printed both hashes at start and end. For the `run_nn_simulation.sbatch` jobs (§7) the hashes were read at submission, and every job before and after them (03:06 to 03:51) printed the same two. |
| **laptop** | macOS, `~/ME5475-demo-env` (export test only, §5) | torch 2.8.0 |

Files revised after the October 4 review and validated October 5 (md5): `export_torchscript.py` 2cdb6f6ddff2a435011027d955f0aa33,
`verify_torchscript.py` 0749647627e8b33f8439541982813e24, `plate_with_hole_nn_material.i`
4057a0edbc4dcc78637946ab5d692272, `run_nn_simulation.sbatch` a73cdf83dc6c29294cb08e1fbadd3248 (unchanged),
`comparison_three_methods.py` 6840b654e475175c55228b5a5f9b6379 (the validated computational version; the released
file, md5 6dd9a5818905f8e6485b1c2b9f9f646c, differs only in its module docstring, revised 2026-10-26). A release review on 2026-10-26
cleared these files with wording fixes; it did not rerun the computation.
- **Before the gate:** export 407aba6e…, verify d6092560…, input f49c1045…, comparison cf932dc4…. The MOOSE runs
  below used the earlier files. The input changed only in comments; the Python checks of §2 and §7 were re-run with
  the final scripts (job 23569018).
- The MOOSE runs of §3–§4 used input version 654cf64a… (failure tests, §5), e5ab0958… (mesh and load sweeps) or
  the final version's non-comment content (the student route, §7; checked by `diff` with comments stripped). The
  earlier versions differ from the final one in comments, the default `refine` (4; every sweep run set it on the
  command line) and the default `nl_max_its` (50; it only matters for runs that do not converge, §4.1).
  The final file passed `--check-input` with either material choice (job 22979863).
- The verify outputs in §2 come from the final script. The comparisons in §3 and §4 were re-run with the final script
  on the same MOOSE outputs (§7).

---

## 1 · The Lab 1 course data and the course reference network

### 1.1 · The data (`module_1/examples/generate_data.py`, `single_element_loadsweep.i`, unchanged)

`generate_data.py --n 200 --out data/single_element.csv --moose-exe <the course executable>` (defaults: seed 42,
`--max-strain 0.005`), as a 1-core job on `mb`:

| job | date, node | time (`/usr/bin/time`) | SHA-256 of `single_element.csv` |
|---|---|---|---|
| 22947144 | 2026-10-03 22:55, mbcpu-001 | 347.3 s (5 min 51 s job) | `8f060b178a8662a31eb34689fe599685c523c0d19fe5013d4acb7ea7f61b65ed` |
| 22977143 | 2026-10-04 03:06, mbcpu-001 | 340.8 s (5 min 44 s job) | identical |

Job 22947144 ran before the 23:03 relink, with the previous `libme5475-opt.so.0` (SHA-256 `f3b51b7c…5a9d`); the data
use only `ComputeLinearElasticStress`, and job 22977143, after the relink, gave the same file.

- **Cost per run:** about 1.7 s per single-element MOOSE run with this executable. Lab 1's handout quotes 20 runs in
  19 s with the older executable on a login node; the new build adds about 1 s per launch (Phase 1 switch record,
  `team/fallback-log.md`).
- **Consistency:** every row satisfies σ = Cε (plane strain, E = 1, ν = 0.3) to 2.1e-16.

### 1.2 · What the single element carries

The same seed in `generate_data_analytic.py` gives the strains the driver *requests* (SHA-256 `178e005f…3864`). The
MOOSE file's strains are not those. A least-squares fit (exact to 2.1e-16) shows that the element carries a fixed
linear mix of the requested strains r:

- ε_xx = 0.5 r_xx + 0.25 r_γ;
- ε_yy = 0.5 r_yy;
- γ_xy = −0.5 r_xx + 0.5 r_yy + 0.25 r_γ.

The per-edge boundary conditions of `single_element_loadsweep.i` conflict at the element's corner nodes; this is the
same defect measured in Lab 4's input before its fix (`module_4/readings/measured_results.md` §1.1).

**Consequences:**
- **The targets are still exact:** σ = Cε holds for every row. But the strain samples are not distributed as the
  driver intended, and a network learns from the samples it gets. So this does not show that the learned stiffness
  is what the intended data would have given (not measured).
- **The strains lie in a parallelepiped, not the cube:** the image of the requested cube [−0.005, 0.005]³ under the
  map above. That parallelepiped reaches |ε_xx| ≤ 3.75e-3, |ε_yy| ≤ 2.5e-3 and |γ_xy| ≤ 6.25e-3, narrower than the
  cube in ε_xx and ε_yy, wider in γ_xy. The CSV's 200 points cover |ε_xx| ≤ 3.35e-3, |ε_yy| ≤ 2.50e-3 and
  |γ_xy| ≤ 5.02e-3, already beyond the cube in γ_xy.
- **Lab 5's coverage checks use the convex hull of all 200 CSV rows,** the "full dataset hull." Lab 1 trains on 140
  of them (`split_data` in `mlp_constitutive.py`: 70/15/15, seed 42). The full hull is therefore a proxy for the
  training data, not the training hull (§2).

### 1.3 · The network (`module_1/examples/mlp_constitutive.py` via `train_mlp.sbatch`, defaults)

`train_mlp.sbatch` with its two mail lines removed (5 000 epochs, width 32, 4 Linear layers, lr 1e-3, seed 42; 4 CPU
cores):

| job | node | time | checkpoint SHA-256 |
|---|---|---|---|
| 22952483 | mbcpu-001 | 21 s | `87e4b206532c43454f8b1a087270a33968079365e64e2c11ca4f02be51b2a343` |
| 22977144 | mbcpu-001 | 20 s | identical (byte for byte) |

- 2 339 parameters.
- Test-set MAE: σ_xx 5.9874e-6, σ_yy 8.1932e-6, σ_xy 2.1193e-6.
- Implied stiffness at zero strain (as printed):

  ```
  [[ 1.3351e+00  5.6934e-01  3.9705e-03]
   [ 5.7346e-01  1.3313e+00 -5.6421e-03]
   [-2.1501e-03  4.6215e-04  3.8295e-01]]
  ```

  C11 is 0.82 % low, C22 1.10 % low, C12 1.31 % low, C33 0.43 % low.

**Reproducibility:** the data and the checkpoint came out byte-identical in two independent runs on ARCC `mb` nodes
with the course environment. A student who ran Lab 1 with the defaults and the MOOSE route in that environment can
expect this network. Byte identity on other hardware, software versions or settings is not promised.

---

## 2 · Task 1 — export and verify (student route, §7; login node mblog2, 2026-10-04 03:13–03:45)

**`export_torchscript.py --checkpoint mlp_constitutive.pt`:** 3.7 s.
- **File:** `mlp_constitutive_ts.pt`, 26 343 bytes (25.7 KB); another export of the same checkpoint in another folder
  was 25.5 KB.
- **Reload checks on 1 000 random strains in [−0.005, 0.005]³:**
  - the float32 file against the eager model: max |diff| = 0;
  - the float64 copy (what MOOSE runs) against the eager float32 model: max |diff| = 1.099e-9, which is 1.5e-7 of the
    largest stress (float32 round-off). The revised script draws the strains in float64 and gives 1.154e-9, again
    1.5e-7 (job 23569018);
  - (revised script) the float64 copy against the network evaluated in float64: max |diff| = 0, gate 1e-12 of the
    largest stress.
- **Not byte-reproducible:** three exports of the same checkpoint gave three different SHA-256 values (the archive also
  stores the file name). The numbers are identical. Compare outputs, not hashes.

**`verify_torchscript.py --model … --data single_element.csv --checkpoint …`:** 5.6 s.

| # | strain (ε_xx, ε_yy, γ_xy) | rel. error vs Hooke | inside the full dataset hull (200 rows; decides judging) | inside the training-split hull (140 rows; information) |
|---|---|---|---|---|
| 0 | (0.001, 0, 0) | 0.73 % | yes | yes |
| 1 | (0, 0.001, 0) | 0.36 % | yes | yes |
| 2 | (0, 0, 0.001) | 0.63 % | yes | yes |
| 3 | (0.002, 0, 0) | 0.34 % | yes | yes |
| 4 | (0.001, −0.001, 0) | 1.08 % | yes | yes |
| 5 | (0.001, 0.001, 0) | 0.24 % | yes | yes |
| 6 | (0.0015, −0.0008, 0.0005) | 1.49 % | **no** (not judged) | no |
| 7 | (−0.001, 0, 0) | 0.25 % | yes | yes |
| 8 | (0, 0, −0.002) | 1.21 % | yes | yes |
| 9 | (0.002, 0.002, 0.002) | 0.49 % | yes | **no** |

- **Verdict:** `RESULT: PASS`, 9/9 judged (at least 5 judged are required). Without `--data` all ten are judged
  and it also passes.
- **Training-split column:** computed in job 23569018 from `split_data`'s default split of this CSV. The verify
  script does not use it, because it cannot know a student's split.
- **The test removed from the battery:** (0.003, 0.003, 0.003), outside the full dataset hull, gave 5.97 % and failed
  without `--data`; it was replaced by test 9.
- **σ(0)** = (−2.521e-6, 1.179e-6, 7.10e-8).
- **Tangent by autograd:** ‖J − C‖/‖C‖ = 1.03 % at zero strain (J equals Lab 1's implied stiffness to the printed
  digits) and 2.72 % at test 6. σ(0) and the tangent are printed as diagnostics; they do not decide the result.
- **File against the eager model (check 4; 1 010 strains drawn in float64):** max |diff| = 0 in float32 and 0 in
  float64. The gates are 1e-5 and 1e-12 of the largest stress (7.533e-3).

**Re-run after the October 4 review** (job 23569018, mbcpu-016, 2026-10-05; the verify and export scripts revised so
that check 4, coverage and non-finite values decide the result). Every value above is unchanged. Negative controls,
each detected as required:
- **Verify FAILs:**
  - an export of 1.01 C against a checkpoint of C (Reviewer's synthetic pair): check 4, 9.55e-5;
  - the reference network with a tiny strain cube as `--data`: 0 of 10 judged, NOT VALIDATED;
  - the `eps.float()` wrapper of §5: check 4 in float64, 3.853e-10 against a gate of 7.5e-15;
  - the same, with the verify script beside an export script carrying the cast;
  - the unstandardized network of §5 (check 4, 4.07e-3, and the Hooke tests);
  - a non-finite network, and a 2C network.
- **Verify stops with an error:** the 2-output network and the bare float cast of §5.
- **Verify PASSes:** the reference export with and without `--checkpoint` (without it, the result says check 4 was
  not run), the laptop torch 2.8.0 export, and a fresh export by the revised script.
- **The revised export script:**
  - on the reference checkpoint: float32 and float64 differences both 0;
  - with an `eps_raw.float()` added to the wrapper's `forward`: it stops (3.853e-10, 5.1e-8 of the largest stress).
- **Full record:** `team/reviews/2026-10-05_M5_lab5_gatefix_evidence/`.

---

## 3 · Tasks 2–3 — Lab 0's plate, network vs elastic (`plate_with_hole_nn_material.i`, σ∞ = 1e-3, refine 3)

### 3.1 · The load and the final element-average strains

- **The plate:** 12 288 elements and 12 545 nodes. The polygonal area is 0.776989; the exact quarter annulus is
  π/4 (1 − 0.1²) = 0.777544.
- **The elastic strains (a published reference value; σ∞ = 1e-3):** the elastic run's largest element-average ε_xx
  is 2.3752e-3 (postprocessor `max_strain_xx`, elastic column, job 22977474). The elastic problem is linear, so at
  Lab 0's load (σ∞ = 1) the strains are 1 000 times larger: up to about 2.4.
- **The network run's strain states and coverage: values held until Wed Nov 11** (see the header). The per-element
  strain ranges, how many elements have a final element-average strain inside the convex hull of all 200 strains in
  the CSV (the full dataset hull), the largest outside state and its location, and the refine-4 maximum are in the
  held file (§3.1 there).
- **What the coverage check means:** it is a **coverage proxy**, not a guarantee. The material evaluates the network
  at quadrature points, whose strains differ from the element average, and at every Newton iterate, which is not
  checked. The hull also includes Lab 1's validation and test rows (§1.2).

### 3.2 · Network vs elastic — **values held until Wed Nov 11** (see the header)

The relative L2 differences (area-weighted) of u, σ_xx, σ_yy, σ_xy and both von Mises measures, the peak values and
every postprocessor, from `comparison_three_methods.py` on the job 22977474 output, are in the held file.
- **One published fact (for the input header and Task 2b):** across refine 0–4 (job 22954802), the relative L2
  difference of u changes by less than 0.1 % of itself and that of σ_xx by less than 7 %. These L2 differences are
  stable over the tested meshes. The peak values are more mesh-sensitive: their network-vs-elastic difference changes
  several-fold over the same meshes (held, §3.2).

### 3.3 · Cost: Newton iterations and run times (`run_nn_simulation.sbatch`, 4 MPI ranks)

| job | node | refine | network run | elastic run | whole job |
|---|---|---|---|---|---|
| 22977474 (student route) | mbcpu-001 | 3 | 43.27 s, 3 Newton iterations, 4 residual evaluations | 3.46 s, 1 iteration, 2 evaluations | 48 s |
| 22978020 | mbcpu-002 | 3 | 42.85 s | 3.13 s | 49 s |
| 22978021 | mbcpu-002 | 3 | 43.08 s | 3.18 s | 49 s |
| 22978022 | mbcpu-002 | 4 (49 152 elements) | 183.18 s, 4 iterations | 5.84 s, 1 iteration | 3 min 11 s |

- **Ratios:** the network run takes about 13× the elastic run at refine 3, and refine 4 costs 4.2× refine 3.
- **Newton history of the network run:** |R| = 8.02e-5 → 3.54e-7 → 3.77e-10 → 1.98e-15. The elastic run goes
  8.00e-5 → 5.2e-17.
- **The two initial residuals differ** (8.02e-5 vs 8.00e-5) because the network's σ(0) ≠ 0 (§2) acts at every
  quadrature point.
- **Mesh levels** (job 22954802; mbcpu-001, which was shared with build jobs at the time, so these times are less
  reliable): the network run took 5.9, 18.1, 13.8, 39.6 and 179.7 s at refine 0–4, and the elastic run 2.9–5.5 s.
  There were 3 Newton iterations at refine 0–3 and 4 at refine 4.
- **Reaction on x = 0 (`reaction_x_left`):** −1.0000010e-3 in the elastic run, equilibrium with the applied traction
  (about −σ∞). The network run's value is held (§3.2; Task 2d).
- **Exodus files:** 2.77 MB each at refine 3, 10.75 MB at refine 4.
- **Memory** (`sacct` MaxRSS of the network step, largest rank): 279 MB at refine 3, 369 MB at refine 4, 648–660 MB
  on Lab 3's plate (§6). The job script asks for 8 GB.

### 3.4 · von Mises and σ_zz

- MOOSE's `vonmises_stress` uses the material's σ_zz: ν(σ_xx + σ_yy) for the elastic material, 0 for the network.
- `vonmises_pe` rebuilds σ_zz = ν(σ_xx + σ_yy) from the in-plane element stresses for both runs.
- **Sizes of the differences:** held (§3.2; Task 3d). Published: the effect is an *increase* of MOOSE's peak von
  Mises in the network run. At the top of the hole the state is close to uniaxial, and for uniaxial σ_xx the von
  Mises stress is σ_xx with σ_zz = 0 but σ_xx √(1 − ν + ν²) = 0.889 σ_xx with σ_zz = ν σ_xx (analytic).
- **The linear-network check** (job 22979862, 2026-10-04, mbcpu-002, input 468faa67… = the final file before a comment fix, refine 0 and 2, the build's
  test network σ = Cv from `team/arcc_build/tests/make_models.py`) reproduced every postprocessor of the elastic run
  except `max_vonmises_stress`: exactly at refine 0 and within 2.4e-13 relative at refine 2.
- **Element-average effect:** in the elastic run, `max_vonmises_pe` (2.25368e-3) and MOOSE's `max_vonmises_stress`
  (2.25419e-3) differ by 2e-4 relative. One is the von Mises of element-average stresses, the other the element
  average of quadrature-point values.

---

## 4 · Task 4 — outside the training data

### 4.1 · What the solver did (published)

**Load sweep with the reference network** (refine 3):
- The rows up to 2e-3 come from job 22968858 (mbcpu-001, input e5ab0958…, no iteration limit) and agree with the
  student-route jobs 22977475–77 (final input).
- "Converged" means the input's tolerance (1e-10 relative).

| σ∞ | outcome | Newton iterations | network run time |
|---|---|---|---|
| 1e-4 | converged | 2 | 32 s (student route: 35 s) |
| 2e-4 | converged | 3 | 41 s |
| 5e-4 | converged | 3 | 40 s |
| 1e-3 | converged | 3 | 43 s |
| 2e-3 | converged | 5 | 60 s (student route: 59 s) |
| 5e-3 | **not converged:** |R| stalled near 2.2e-5 (start 4.0e-4); `DIVERGED_MAX_IT` | 50 | 1 492 s |
| 1e-2 | **not converged:** `DIVERGED_LINE_SEARCH` (|R| 8.0e-4 → 3.7e-4) | 14 | 522 s |
| 3e-2 | **not converged:** `DIVERGED_LINE_SEARCH` (2.4e-3 → 2.0e-3) | 23 | 1 011 s |
| 1e-1 | **not converged:** `DIVERGED_LINE_SEARCH` (8.0e-3 → 7.6e-3) | 13 | 666 s |
| 1 | **crash:** one Newton step (|R| 8.00e-2 → 7.96e-2); the next LU factorization fails (`FACTOR_NUMERIC_ZEROPIVOT`); the update is NaN, and the material stops the run: "returned a non-finite stress at element …" | 1 | 19.6 s (student route, job 22977475: 24.4 s, job 26 s) |

- **A failed Steady run aborts** ("Aborting as solve did not converge"). Its CSV and Exodus hold only the t = 0 state,
  so a failed run cannot be compared, and `run_nn_simulation.sbatch` stops before the elastic run.
- **The final input stops Newton after 15 iterations,** so the 5e-3 to 1e-1 runs would now stop sooner. Their times
  with that limit are (not measured).
- **The 2e-3 run** converged, but only part of the plate's final element-average strains lay inside the full
  dataset hull (value held, §4.2).
- **What the failures show:** Newton did not converge from 5e-3 to 1e-1 with this solver, load step and zero initial
  guess. They do not show that the network material has no solution at those loads.

### 4.2 · Error values — **held until Wed Nov 11**

The relative L2 differences at every converged load, the share of the plate inside the full dataset hull, and the
network's stress at growing strains (Task 4a's probe) are in the held file.

---

## 5 · What mistakes look like (job 22977145, 2026-10-04 03:06, mbcpu-001; refine 2, σ∞ = 1e-3; Python checks on mblog2)

| file given to MOOSE | what happened |
|---|---|
| the correct export (torch 2.5.1, ARCC) | converged in 3 Newton iterations, 15.1 s |
| the same checkpoint exported **on a laptop with torch 2.8.0** | loaded and ran; every postprocessor identical to the ARCC export (max relative difference 0.0), 13.9 s |
| the raw network traced **without the standardization wrapper** | **converged without complaint** in 2 iterations, to a wrong plate: `max_strain_xx` 5.27e-3 (correct 2.10e-3), `min_strain_yy` −3.71e-3 (correct −7.18e-4). `verify_torchscript.py` failed it on all 10 tests. At ε = (1e-3, 0, 0) it returns σ = (−2.2e-4, 1.96e-3, 2.5e-4); Hooke gives (1.35e-3, 5.8e-4, 0). |
| the wrapper with **`eps.float()` in `forward`** | Newton stalled: |R| from 1.13e-4 down to 1.29e-11 (about 1e-7 of the start) and no further; stopped at the 15-iteration limit, 58.1 s, exit 1. In Python the float64 copy runs (the float64 buffers promote the input back), and it differs from the correct file by up to 1.9e-10, 3.4e-8 of the largest stress. |
| a bare layer with `x.float()` in `forward` (Python only) | the float64 call fails: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double` (torch 2.5.1 on ARCC, and 2.8.0 on the laptop) |
| a network returning 2 values | "…returned 2 values; it must return 3 (sigma_xx, sigma_yy, sigma_xy) for one strain." 4.4 s; `verify_torchscript.py` stops: "returned shape (10, 2); MOOSE needs (N, 3)" |
| a file name that does not exist | "Error while loading torchscript file …nothere.pt! open file failed because of errno 2 on fopen: No such file or directory", 2.1 s |
| **the Lab 1 checkpoint** instead of the export | "Error while loading torchscript file …mlp_constitutive.pt! PytorchStreamReader failed locating file constants.pkl: file not found", 2.1 s |
| the correct file with the **old executable** `rom_opt-opt` | "A 'ComputeTorchScriptStress' is not a registered object.", 1.8 s |

**Other checks:**
- **`torch.load` without `weights_only=False` fails on the Lab 1 checkpoint with torch 2.8.0** (`UnpicklingError:
  Weights only load failed`, laptop). The checkpoint stores Python objects. torch 2.5.1 loads it either way.
  `export_torchscript.py` passes `weights_only=False`.
- **`torch.jit.trace` warns** about a data-dependent `if` (`TracerWarning: Converting a tensor to a Python boolean
  might cause the trace to be incorrect`), with torch 2.5.1 (ARCC) and 2.8.0 (laptop).

---

## 6 · Task 5 (optional) — Lab 3's plate, three methods

### 6.1 · Runs (published)

**The network on Lab 3's plate:**
- Setup: `plate_square_hole_reference.i` with the Lab 5 `[Materials]` block, `nt=48 nr=96`, u₁ = 1e-3 on the right
  edge, via `run_nn_simulation.sbatch`.
- Size: 9 216 QUAD9 elements, 140 361 degrees of freedom.

| job | initial guess | network run | elastic run (same mesh) | whole job |
|---|---|---|---|---|
| 22978828 (student route, final handout) | u_x = 1e-3·x (`FunctionIC`) | 78.2 s, 4 Newton iterations | 6.9 s, 2 iterations | 1 min 29 s |
| 22978009 (repeat, mbcpu-001) | u_x = 1e-3·x | 76.4 s, 4 iterations | — | — |
| 22977478 (student route, first handout draft) | zero | 357.0 s, 15 iterations | 7.3 s, 2 iterations | 6 min 6 s |
| 22978009 | zero, u₁ = 5e-4 | **not converged:** `DIVERGED_LINE_SEARCH` after 15 iterations, 555.6 s | — | — |

- **Why the zero initial guess fails:** the right-edge displacement is imposed on the first iterate while the interior
  is still at zero, so the elements next to that edge are strained far outside the training data. The elastic run
  needs 2 iterations either way; the input file's own comment explains the second one, which comes from the nodal
  stress output.
- **Same result either way:** the converged network results from both initial guesses agree to the printed digits
  (the values are held, §6.2).

**The PINN on the grid** (job 22969154, 17 s):
- The Lab 3 Task 2b snippet, with the Lab 5 Task 5c lines, on the course's seed-42 hard-BC PINN
  (`~/me5475-demo/L3-validate/optionA/pair_42_hard_bc/`).
- It reproduced `module_3/readings/measured_results.md` §6 exactly: u 9.81e-4, σ₁₁ 1.71e-3, σ₂₂ 2.15e-2,
  σ₁₂ 1.64e-2; σ₁₁(0, 0.1) = 3.2350.

**Grid extraction** (`extract_moose_reference.py`, 200 × 200, 39 671 points, 0 NaN): 4.1 s for the network run and
12.2 s for the shared reference, on the login node.
- Files: `nn_grid.csv` 6.9 MB, `reference_grid.csv` 6.3 MB, `pinn_grid.csv` 3.8 MB. Exodus: 6.6 MB per run.

### 6.2 · Error values — **held until Wed Nov 11**

The relative L2 errors of the network and the PINN against the reference, σ₁₁ at the top of the hole for each
method, and the share of nodes inside the full dataset hull are in the held file.

---

## 7 · The student route, literally (fresh folder, 2026-10-04 03:13–03:45)

**The mirror:** `/gscratch/xzhang16/me5475-demo/M5-validate/student/repo/`.
- It holds `module_5/examples/` (the final files; `code_md5_final.txt`) and a Lab 1 folder
  `labs/lab_1/testhandle/{training/checkpoints, data_generation/data}` with the files of §1.
- Every command was the handout's, with `<you>` = `testhandle`, run from a shell that had sourced `setup5475.sh`
  (CONDA_PREFIX set, as a student's would be). `git pull` was skipped (no git in the mirror).

**What ran:**
- **Task 1:** export and verify on mblog2 (§2).
- **Task 2:** `sbatch run_nn_simulation.sbatch`.
  - **The first submission (job 22977468) failed in 2 s at the pre-flight check, as did the four Task 4/5 jobs
    submitted with it (22977469–72).** The check piped `me5475-opt --registry` into `grep -q`. Under `set -o
    pipefail`, `grep -q` closes the pipe at the first match and the pipeline returns 141 (SIGPIPE), reproduced on the
    login node.
  - Fixed: `grep -c … > /dev/null` reads the whole list. Re-submitted as job 22977474 (§3.3).
- **Task 3:** the comparison, 6.2 s on mblog2.
- **Task 4:** jobs 22977475 (σ∞ = 1; failed as described in §4.1), 22977476 (2e-3; 1 min 8 s) and 22977477 (1e-4;
  41 s), the probe snippet, and the two comparisons.
- **Task 5:**
  - the hand edit of the `[Materials]` block (the `diff` against the reference input is in the evidence);
  - job 22977478, then 22978828 after the handout gained the initial guess;
  - the grid extraction and the three-way comparison.

**Final re-check (03:45):**
- The final `verify_torchscript.py` and `comparison_three_methods.py` were copied in by the handout's commands and
  re-run on the same files.
- Verify passed (§2). Every comparison metric was identical to the first pass; the scripts' later edits only changed
  figure edges and one printed label.

**`sacct`** for every Lab 5 job: `team/reviews/2026-10-03_M5_lab5_evidence/arcc_M5-validate/sacct_M5_jobs.txt`.

**Re-run after the October 4 review (job 23569018, 2026-10-05).** The revised `comparison_three_methods.py` was re-run
on the same student-route MOOSE outputs (default, 2e-3, 1e-4, Task 5 grid). `diff` against the first metrics shows
changed labels only, plus file paths and the new "points used" percentage: every number is identical. Grid-mode
controls:
- duplicate points: refused;
- no common points: refused;
- 10 infinite values: those 10 points left out (39 661 of 39 671 used, flagged);
- half the points missing: 50.00 % used, flagged.

The verify and export re-runs are in §2.

---

## 8 · Numbers stated before this validation, checked

C = confirmed, X = contradicted, U = not measured. Rows whose measured value is held say so.

| where (before) | claim | status | measured |
|---|---|---|---|
| `lab_5.md` Task 2b; L27 | the job "should complete in ~5 minutes" | X | 48–49 s for both runs at refine 3; 3 min 11 s at refine 4 (§3.3) |
| `lab_5.md` Task 1c; L26/L27 | a well-trained M1 MLP is within 5 % (L27: 1–5 %) of Hooke on the test strains | C | ≤ 1.49 % on all ten (§2) |
| `lab_5.md` Task 5.1; L27 §2 | L2 errors of M5 vs M0 ~1–5 %, "dominated by the MLP's training error" | held | §3.2 |
| `lab_5.md` Task 3; L27 | a three-way comparison M0 vs M3 PINN vs M5 on the *same* plate | X | the PINN solves Lab 3's plate (unit square, u₁ = 1); compared there only (§6) |
| starter prompts Task 3 | "peak von Mises 3.0 ± 0.1 for all three methods (Kirsch)" | X | Kirsch's 3σ∞ is the hoop stress σ_xx, not von Mises; MOOSE's peak element σ_xx/σ∞ is 2.679 at refine 3 and 2.876 at refine 4 (elastic run) |
| `moose_material_interface_reference.md` (Lab 5 draft) | PETSc's FD Jacobian test agrees with the autograd tangent, "1.4e-8 to 8.6e-7 over all Newton steps" | X (superseded) | that range came from an earlier, truncated test run; the final check is §9 (13 checks, max 8.99e-7, reduced load) |
| `moose_material_interface_reference.md` | ~50 ns (elastic) vs ~100 µs (network) per call, ~2000× slower | U | per-call cost (not measured); whole-run ratio about 13× at refine 3 (§3.3) |
| `moose_material_interface_reference.md` | "1500 Gauss points × 10 Newton iterations × ~5 calls = ~75 000 evaluations" | X | the plate has 49 152 quadrature points at refine 3 (12 288 QUAD4 × 4) and needed 3 Newton iterations and 4 residual evaluations (§3.3) |
| `torchscript_export_guide.md` | "A small MLP (~5000 parameters) exports to ~25 KB" | X (parameter count) | 2 339 parameters, 25.5–25.7 KB (§2) |
| `torchscript_export_guide.md` | "For MOOSE / libtorch: float32 is fine"; cast inputs with `.float()` | X | MOOSE runs the module in float64; a `.float()` in `forward` breaks the call or the Newton convergence (§5) |
| `export_torchscript.py` sanity print | Hooke σ_xx for (0.001, 0, 0) is 0.001346 | C | 1.346154e-3 |
| `lab_5.md` Task 4 (old stretch) | an LSTM can be deployed with `tangent_mode = 'autograd'` | X | the material is stateless (no history); not possible with this build |
| `README.md`, `lab_5.md` | a course C++ wrapper with input/output scaling and tangent-mode parameters | X | the material is `ComputeTorchScriptStress`, with one parameter, `filename` |

---

## 9 · The material's tangent (course build tests; `team/arcc_build/README.md`)

`ComputeTorchScriptStress` in `me5475-opt` `083a3c61…`, with a tanh MLP with fixed random weights (3-16-16-3, from
`team/arcc_build/tests/make_models.py`) on a 2 × 2 plane-strain mesh, `tests/single_element_mlp.i`, 4 MPI ranks:

- **PETSc's `-snes_test_jacobian` at every Newton step** (job 22969006, 2026-10-04): 13 checks; the largest
  ‖J − Jfd‖_F/‖J‖_F is 8.99488e-7. PETSc's own message says O(1e-8) means a hand-coded Jacobian is probably
  correct; the course gate is 1e-6. Two converged solves, no aborted solve, final time reached.
- **The network's autograd derivative against central finite differences** (`tests/check_tangent_fd.py`, 200 strains
  with standardized inputs up to ±4, into tanh saturation): max relative difference 1.641e-9. Job 22969006; re-run
  with the revised check in job 23563565 (2026-10-05), unchanged.
- **The load is deliberately small** (displacements 2.5e-4·t and −1.25e-4·t, two steps). This checks the
  derivative. It does not show that Newton converges at larger loads. At 4× this load (job 22960112) the first full
  step failed and was cut back; one of 39 checks gave 0.0113, at the 4th iterate of that failed step. The other 38
  were ≤ 8.6e-7. The cause of the 0.0113 was not isolated.
