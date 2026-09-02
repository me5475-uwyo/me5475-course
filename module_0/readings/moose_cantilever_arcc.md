# A complete HPC finite-element workflow: a 2-D cantilever beam with MOOSE on ARCC

**ME-5475 · Module 0 · read before Lecture 3 (Fri Sep 4).**
Prerequisite: the 1-D FE primer (`fe_1d_primer.md`) or prior FEA. No prior MOOSE or SLURM is assumed. Companions in this directory: `moose_input_file_anatomy.md`, `slurm_cheatsheet.md`.

One small, fully validated example, end to end: problem statement → cluster job → contours → validation against beam theory. The three files live in `module_0/examples/` and were run on ARCC MedicineBow as SLURM job 15666710:

| File | Role |
|---|---|
| `cantilever_beam.i` | MOOSE input file (the FE model) |
| `run_cantilever.sbatch` | SLURM submission script |
| `plot_beam_results.py` | Post-processing: contours + analytical check |

Every number quoted below is a measured result from that run.

---

## 1 · The problem

Take a slender rectangle in the x–y plane: length L = 1.0 along x, height h = 0.1 along y — the domain (x, y) ∈ [0, 1] × [0, 0.1], slenderness L/h = 10, unit out-of-plane thickness. Sketch it as a long thin bar lying flat:

- **left edge** (x = 0): fully clamped, u_x = u_y = 0 — a built-in support;
- **top face** (y = 0.1): uniform pressure q = 1×10⁻⁶ pushing **down**, i.e. traction t_y = −q;
- **right edge** (x = 1, the free tip) and **bottom face** (y = 0): traction-free — nothing prescribed, the *natural* condition.

Material: linear isotropic elastic, **E = 1, ν = 0.3**, in **plane strain** (ε_zz = 0). E = 1 is the course-wide dimensionless convention: `plate_with_hole.i` uses it, and so does Min Lin's PINN setup we compare against in Module 3, so FE and PINN results compare directly with no unit bookkeeping. The tiny q = 10⁻⁶ keeps small-strain linearity self-consistent.

Because the beam is slender, beam theory predicts the tip deflection — the check in Section 5, and the point: **never trust a simulation you have not validated against something computed another way.**

## 2 · The MOOSE input file, block by block

A MOOSE input file is a tree of named blocks. `cantilever_beam.i` has seven.

**`[Mesh]`** — MOOSE builds the mesh itself:

```text
[Mesh]
  [beam]
    type = GeneratedMeshGenerator
    dim  = 2
    nx   = 100
    ny   = 10
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 0.1
  []
[]
```

`GeneratedMeshGenerator` tiles the rectangle with a structured grid: 100 × 10 = **1000 four-node quadrilateral (QUAD4) elements** (101 × 11 = 1,111 nodes). It also auto-names the boundary sidesets `left`, `right`, `top`, `bottom` — exactly the names `[BCs]` refers to.

**`[Variables]`** — the unknowns:

```text
[Variables]
  [disp_x]
    family = LAGRANGE
    order  = FIRST
  []
  [disp_y]
    family = LAGRANGE
    order  = FIRST
  []
[]
```

Two displacement components, first-order Lagrange — the 2-D analogue of the 1-D FE primer's hat functions (bilinear on each quad).

**`[Modules/TensorMechanics/Master]`** — the physics, via an *action*:

```text
[Modules/TensorMechanics/Master]
  displacements = 'disp_x disp_y'
  [all]
    displacements      = 'disp_x disp_y'
    add_variables      = false
    strain             = SMALL
    incremental        = false
    generate_output    = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    planar_formulation = PLANE_STRAIN
  []
[]
```

An action is a macro expanding into objects you would otherwise write by hand: the **stress-divergence kernels** (one per displacement component — the weak form of ∇·σ = 0) plus a small-strain calculator consistent with `planar_formulation = PLANE_STRAIN` (ε_zz = 0). `generate_output` adds auxiliary fields for the listed stress/strain components and von Mises, with no extra blocks written.

**`[BCs]`** — where the 1-D FE primer pays off directly:

```text
[BCs]
  [clamp_x]
    type     = DirichletBC
    variable = disp_x
    boundary = 'left'
    value    = 0.0
  []
  [clamp_y]
    type     = DirichletBC
    variable = disp_y
    boundary = 'left'
    value    = 0.0
  []
  [top_pressure]
    type     = NeumannBC
    variable = disp_y
    boundary = 'top'
    value    = -1.0e-6
  []
[]
```

`DirichletBC` is the **essential** condition: it constrains the solution space itself — the 1-D FE primer's u(0) = 0. `NeumannBC` is the **natural** condition: it adds the boundary-traction integral ∫ t·v ds to the weak form's right-hand side — where the end load entered in 1-D. The value is **−1.0×10⁻⁶** because pressure q pushing down is traction t_y = −q. The free edges appear nowhere: absence of a BC *is* the traction-free natural condition.

**`[Materials]`** — two objects implementing σ = C : ε: `ComputeIsotropicElasticityTensor` builds C from `youngs_modulus = 1.0`, `poissons_ratio = 0.3`; `ComputeLinearElasticStress` applies it at each quadrature point.

**`[Postprocessors]`** — scalars extracted each solve:

- `tip_disp_y`: `PointValue` of `disp_y` at (1.0, 0.05, 0) — mid-height of the free end, the beam axis at the tip. The number we validate.
- `max_vonmises_stress`: `ElementExtremeValue` (max) of the von Mises field.
- `num_elements`: `NumElements` — a mesh sanity check (reports 1000).

**`[Executioner]`** — `type = Steady`, `solve_type = NEWTON`, PETSc `-pc_type lu`: a steady Newton solve with direct LU factorization — fine at 1000 elements (linear problem, essentially one Newton iteration), not scalable to millions.

**`[Outputs]`** — `exodus = true` writes `cantilever_beam_out.e` (mesh + all field data, for ParaView or our Python script); `csv = true` writes `cantilever_beam_out.csv` (the postprocessor table — the quickest post-run check).

## 3 · The SLURM script, key component by key component

MedicineBow is a shared cluster: you do not run programs on the login node; you describe a job and hand it to the **SLURM** scheduler. The description is `run_cantilever.sbatch` — a bash script whose specially formatted comments carry the resource request.

```bash
#!/bin/bash
```

The shebang: when the job starts, bash executes this file on a compute node.

```bash
#SBATCH --job-name=cantilever       # name shown in squeue
#SBATCH --account=me5475            # course allocation — billing/authorization
#SBATCH --partition=mb              # CPU partition on MedicineBow
#SBATCH --time=00:10:00             # wall-clock limit; this job needs seconds
#SBATCH --nodes=1                   # everything on one node
#SBATCH --ntasks=2                  # 2 MPI ranks — plenty for 1,000 elements
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=cantilever-%j.out  # %j = job id; stdout lands here
#SBATCH --error=cantilever-%j.err
```

To bash these are comments; `sbatch` parses them *before* the script runs. One by one:

- `--job-name`: cosmetic — the label you recognize in `squeue`.
- `--account=me5475`: charges the job to the course allocation. On a shared cluster every CPU-hour is charged to a project; no valid account, no job.
- `--partition=mb`: the CPU partition. (GPU work later uses `mb-l40s` / `mb-a30`.)
- `--time=00:10:00`: the wall-clock limit. If exceeded, **SLURM kills the job** (state `TIMEOUT`); output so far survives but the run ends mid-flight. Request margin, not excess — shorter jobs schedule sooner. This run needs only seconds.
- `--nodes=1 --ntasks=2 --cpus-per-task=1`: two MPI ranks on one node, one core each — plenty for 1000 elements; the point is to exercise the parallel machinery, not speed.
- `--mem=4G`: reserved memory; exceed it and the job is killed (`OUT_OF_MEMORY`).
- `--output` / `--error`: where stdout and stderr land; `%j` becomes the job id — here `cantilever-15666710.out` / `.err` — so runs never overwrite each other's logs.

```bash
set -euo pipefail
```

Three safety switches: `-e` aborts on the first failing command, `-u` makes an undefined variable an error, `-o pipefail` fails a pipeline if any stage fails. In a batch job nobody is watching; without this, a failed `module load` is silently ignored and the script barrels on. Always start batch scripts this way.

```bash
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
```

Batch shells are non-interactive, so we source Lmod's init to make `module` exist at all. `module purge` wipes the node's defaults; `module load` builds the *exact* stack the executable was compiled against — gcc 14.2.0 runtime, OpenMPI 5.0.5, matching HDF5. A binary linked against one MPI or C++ runtime will crash or misbehave under another; purge-then-load makes the job reproducible regardless of your login environment.

```bash
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"
```

`conda activate` brings in the shared course environment (`/project/me5475/envs/ml4sm`, the Lab 0 stack — also the Python you post-process with). The `LD_LIBRARY_PATH` export answers where the dynamic loader finds shared libraries at run time: the MOOSE executable needs the `libstdc++` matching gcc 14.2.0 (not the node default) and MOOSE's own `hit` input-parser library; the line prepends both. The `${LD_LIBRARY_PATH:-}` tail keeps `set -u` happy if the variable started unset.

```bash
MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt

srun --ntasks=$SLURM_NTASKS "$MOOSE_EXE" -i cantilever_beam.i
```

`MOOSE_EXE` points at the course's compiled MOOSE application (`-opt` = optimized build) in project space — you never build MOOSE yourself. `srun` launches the MPI ranks *inside* the allocation the `#SBATCH` lines reserved; `$SLURM_NTASKS` follows the `--ntasks` directive, so the rank count is changed in one place. The closing lines `echo` timestamps and `tail` the postprocessor CSV into the `.out` log.

**Operating it.** From the directory containing the files, on a login node:

```bash
sbatch run_cantilever.sbatch     # -> "Submitted batch job 15666710"
squeue -u $USER                  # your jobs: PD = pending, R = running
scancel <jobid>                  # kill a job you regret
```

`sbatch` returns immediately with the job id; the scheduler runs the job when resources free up. Once it vanishes from `squeue`, stdout/stderr sit in `cantilever-<jobid>.out` / `.err` in the submission directory, alongside `cantilever_beam_out.e` and `.csv`.

## 4 · Results

`python plot_beam_results.py` reads the Exodus file and produces `figures/beam_contours.png` — four panels on the undeformed geometry (displacements ~10⁻³ are invisible at scale):

- **|u| — displacement magnitude.** Zero at the clamp, growing monotonically toward the free end, largest at the tip: the beam droops.
- **stress_xx — bending stress.** **Antisymmetric about the mid-plane y = 0.05**: tension on the top face, compression on the bottom — the sign pattern for a downward-pressed cantilever, whose top fibers stretch as it curls down. Largest at the clamped end where the bending moment peaks; fading toward the tip.
- **strain_xx.** Mirrors stress_xx, as linear elasticity demands.
- **von Mises stress.** Sign-blind equivalent stress: it peaks at **both clamped corners** — measured maximum **2.2950×10⁻⁴** — and vanishes near the free end and along the neutral axis.

The **neutral axis** — the line y = 0.05 where stress_xx ≈ 0 — is visible in the second panel separating tension from compression: beam-theory structure emerging from a continuum model that was never told about beams.

## 5 · Validation against beam theory — the centerpiece

Beam theory predicts the tip deflection under uniform load per unit length q. Euler–Bernoulli (bending only):

> δ_EB = q L⁴ / (8 E\* I),  I = h³/12,

where in **plane strain** the correct bending modulus is E\* = E/(1 − ν²), not E. Timoshenko adds shear deformation:

> δ_T = δ_EB + q L² / (2 κ G A),  κ = 5/6, G = E/(2(1+ν)), A = h·1.

With E = 1, ν = 0.3, L = 1, h = 0.1, q = 10⁻⁶, against the measured MOOSE value:

| Quantity | Value |
|---|---|
| Euler–Bernoulli (plane-strain E\*) | 1.365000×10⁻³ |
| Timoshenko shear term | 1.560000×10⁻⁵ |
| Timoshenko total | 1.380600×10⁻³ |
| **MOOSE tip deflection (100×10)** | **1.367180×10⁻³** |
| FE vs EB | **+0.16%** |
| FE vs Timoshenko | **−0.97%** |

The FE result sits **between** the two theories — and that is exactly right. Timoshenko assumes every cross-section, including the one at the wall, freely develops a shear rotation; our clamp fixes the *entire* left edge pointwise, suppressing that rotation where it matters most, so the 2-D solution is stiffer than Timoshenko. Meanwhile shear flexibility, which Euler–Bernoulli ignores, softens it relative to EB. Landing between them at slenderness 10 is a *pass*: sub-percent agreement, residuals traceable to identifiable assumptions.

One trap worth naming: using E instead of E\*. The plane-strain constraint stiffens bending by 1/(1 − ν²) = 1/0.91 ≈ 1.099, so validating against the plain-E formula builds in a ~10% error and wrongly indicts the FE model. Match the theory to the formulation: plane strain ⇒ E\*.

## 6 · Run it yourself

1. Copy the three files into `~/ME5475/examples` on MedicineBow (the course's canonical location):
   `scp cantilever_beam.i run_cantilever.sbatch plot_beam_results.py <netid>@medicinebow.arcc.uwyo.edu:ME5475/examples/`
2. Log in, `cd` there, `sbatch run_cantilever.sbatch`.
3. Watch `squeue -u $USER` until the job disappears; skim `cantilever-<jobid>.out` — the postprocessor values are at the bottom.
4. Post-process (the `ml4sm` environment has the needed packages): `python plot_beam_results.py`. It writes `beam_contours.png` and prints the beam-theory comparison.

If you have ParaView, you can instead open `cantilever_beam_out.e` directly — warp by displacement, slice, probe any field. The Python script exists precisely so students *without* ParaView still get contours.

## 7 · Limitations — what to trust here

This model uses 100×10 **linear** quads. The tip deflection — an integrated, global quantity — is well converged, and it is what the validation rests on. The peak von Mises stress is not: a stress singularity lives at each clamped corner (fixed edge meeting a free face), so the reported maximum is *mesh-dependent* and would keep growing under refinement — a property of the mathematical problem, not a MOOSE defect. The rule: **validate with the global quantity; treat singular-point extrema as qualitative.**

---

## Questions for the instructor

1. Should this reading be added to `reading_list.md` (currently silent on Readings A–D), and under which lecture heading?
