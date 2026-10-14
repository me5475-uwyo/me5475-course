# Measured results for readings B, C, D — single source of truth
All produced 2026-09-01 on ARCC (/project/me5475/envs/ml4sm, torch 2.5.1+cu121; MOOSE rom_opt-opt via SLURM job 15666710, re-run as 15666711 from the canonical location ~/ME5475/examples, partition mb — identical results). Do not invent numbers; cite these.

## B · MLP fits sin(pi x)  (mlp_sine_reference.py, seed 0)
- parameters: 2209
- loss trajectory (MSE): epoch 1: 5.473e-01 · 1000: 1.147e-05 · 2000: 2.212e-05 · 3000: 1.711e-06 · 4000: 1.192e-06 · 5000: 2.871e-06
- final training loss: 6.707e-06 ; max |error| on 1000-pt grid: 5.335e-03
- BUGGY broadcasting variant (target shape (200,) vs prediction (200,1) -> (200,200) MSE):
  final loss 4.975e-01 (stuck ~= variance of sin), max |error| 1.000e+00 — the fit is DESTROYED, not degraded.
  PyTorch emits a UserWarning ("Using a target size (torch.Size([200])) different to the input size") — ignoring it is the bug.
- figure: figures/mlp_sine_fit.png

## C · 1-D FE bar  (fe1d_reference.py)
- problem: EA=1, L=1, b(x)=x, u(0)=0, end load F=0.3 at x=1; exact u(x)=0.8x - x^3/6, sigma(x)=0.8 - x^2/2
- convergence (N = elements) — **two stress measures, two different rates**:
  ```
  N    max|u_h-u|_nodes   L2(u)-err     midpt-stress   TRUE energy norm
  2    1.110e-16          1.278e-02     1.042e-02      8.122e-02
  4    4.441e-16          3.269e-03     2.604e-03      4.141e-02
  8    1.110e-16          8.220e-04     6.510e-04      2.080e-02
  16   1.110e-16          2.058e-04     1.628e-04      1.041e-02
  32   9.881e-15          5.146e-05     4.069e-05      5.208e-03
  64   8.327e-15          1.287e-05     1.017e-05      2.604e-03
  ```
- **CORRECTION 2026-09-01** (caught by the Implementer writing the 1-D FE primer, verified on ARCC): the
  earlier single "energy-err" column was midpoint-sampled stress, which **superconverges**. It falls
  4x per halving (O(h^2)) and is NOT the energy norm. The TRUE energy norm — (sigma_h - sigma_exact)
  integrated over each element by 20-pt Gauss — falls **2.00x per halving, i.e. O(h)**, which is the
  standard linear-FE result. Teach both: displacement O(h^2) in L2, stress O(h) in energy norm, with
  midpoint stress a superconvergent exception.
- nodal errors are machine zero — linear FE is NODALLY EXACT for this 1-D problem class (does not
  generalize to 2-D)
- figure: figures/fe1d_N4.png (N=4: piecewise-linear u through exact nodal values; staircase sigma vs parabola)

## D · MOOSE 2-D cantilever  (cantilever_beam.i + run_cantilever.sbatch + plot_beam_results.py)
- canonical ARCC location: `~/ME5475/examples/` (instructor account). Students copy the three files there
  (or into their own `~/ME5475/examples`) and run `sbatch run_cantilever.sbatch` from that directory.
- L x h = 1.0 x 0.1, E=1, nu=0.3, PLANE_STRAIN, q=1e-6 top pressure, 100x10 QUAD4 (1000 elements)
- SLURM job 15666710 on partition mb, account me5475; wall time seconds
- tip deflection at (1.0, 0.05): MOOSE 1.367180e-03
- Euler-Bernoulli with plane-strain modulus E*=E/(1-nu^2): 1.365000e-03 (+0.16% FE vs EB)
- Timoshenko (EB + shear qL^2/(2 kappa G A), kappa=5/6): 1.380600e-03 (-0.97% FE vs Timoshenko)
- interpretation: FE lies between EB and Timoshenko because the fully-clamped edge suppresses the
  cross-section shear rotation Timoshenko allows, while shear flexibility softens it relative to EB.
- max von Mises 2.2950e-04 at the clamped corners; zero near free end and neutral axis
- figure: figures/beam_contours.png (|u|, stress_xx antisymmetric about mid-plane, strain_xx, von Mises)

## E · The course MOOSE build change (2026-10): old vs new executable on the course inputs

`[fallback]` Co-Worker, 2026-10-03/04. The new course build is `/project/me5475/software/moose_torch/me5475/me5475-opt`
(SHA-256 `083a3c61…ec14`; MOOSE 437fbe5082, libMesh f8a1758, PETSc 0bd50eb, plus libtorch 2.5.1). The old one is
`/project/me5475/software/rom_opt_arcc/rom_opt-opt` (the same MOOSE, libMesh and PETSc versions).

- **Job:** 22969006 (2026-10-03, `mb`, 4 MPI ranks). Both executables ran with the course job-script runtime block, each
  with its own library-path line.
- **Inputs (8):**
  - `cantilever_beam.i`, `plate_with_hole.i` (M0);
  - `single_element_loadsweep.i` (M1);
  - `diffusion_1d.i`, `diffusion_2d.i`, `plate_square_hole_reference.i` (M3);
  - `j2_cyclic_loading.i` (M4);
  - `rve_homogenization.i` (M6).

  Each was run at its default settings.
- **Compared:** every CSV each run writes (postprocessor tables and line-sampler files): the same file names, the same
  columns and rows, finite values. Two line-sampler files written before the first solve are header-only in both runs.
- **Result:** all 8 inputs pass. Every compared value is identical, or within 1e-12 absolute (the comparison
  tolerance is 1e-9 relative, with a 1e-12 absolute floor for round-off near zero).
- **Not compared:** Exodus field files, non-default parameters, and anything outside these eight inputs. The full Lab 4
  dataset regeneration is a separate check (`module_4/readings/measured_results.md` §1.2, job 22941265).
- **Run time:** the new executable takes about 1 s longer to start (it loads the PyTorch library;
  `module_4/readings/measured_results.md` §1.2). Single short runs here took 2.3–3.9 s against 2.0–3.2 s; the
  4-rank plate reference took 161.9 s against 170.2 s.
- **Evidence:** `team/arcc_build/evidence/tests-22969006.out`, which also holds the neural-network material checks.
