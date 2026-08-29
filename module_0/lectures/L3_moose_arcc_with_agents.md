# Lecture 3 — MOOSE Verification + First ARCC Job

**Date:** Friday, September 4, 2026
**Module:** 0 — Setup, Agentic Workflow, ARCC + MOOSE Orientation
**Duration:** 50 minutes
**Format:** Live demos throughout — minimal slides, maximal terminal time

---

## Important: The install happened *before* this lecture

In response to feedback from earlier iterations of this course, we moved the MOOSE installation step out of class time and into a *pre-class graded homework assigned at the end of Lecture 2*. Students arrive at L3 with MOOSE already installed and verified. L3 is now what students were always going to do in class anyway: see a working FE simulation, learn ARCC, submit their first SLURM job.

The agentic-workflow lesson is unchanged — students still install MOOSE *with an agent* — they just do it as homework instead of live. The in-class portion of L3 spends 5 minutes verifying who succeeded, helps anyone who got stuck, and then moves on.

---

## Learning objectives

By the end of this lecture, students should be able to:

1. Run the provided `plate_with_hole.i` MOOSE input file locally, and visualize the von Mises stress field in ParaView.
2. Diagnose and recover from common MOOSE install / first-run failures.
3. Read the anatomy of a SLURM script (`#SBATCH` directives, module loading, srun invocation) and modify it for a Python job.
4. Submit a hello-world Python job to ARCC, monitor it with `squeue`, and retrieve the output; students whose access is still activating pair up today and submit as soon as access is active, without penalty.
5. Identify the takehome Lab 0 task — a mesh-refinement convergence study on the same plate-with-hole example, driven by agents, running on ARCC.

---

## Pre-class homework (assigned at the end of Lecture 2; due before L3)

Sent after L2 on Wednesday afternoon. Required of every student before walking into L3:

- **Install MOOSE locally** using the agent prompt provided in `module_0/examples/install_moose_prompt.md` (the same prompt we will discuss in class for those who got stuck). Use the conda/mamba INL-supported path. Commit your `install_log.txt` to your Lab 0 directory before L3.
- **Run `plate_with_hole.i` locally** with the default `uniform_refine = 0`. Commit a ParaView screenshot of the von Mises field.
- **Bring your laptop and your ARCC credentials if your account is active.** If not, pair up today — no penalty — and submit your own job as soon as access is active. Your account was requested through the instructor-submitted ARCC project change form; watch for the onboarding email and tell the instructor if you registered late.
- **Pre-class environment check.** Have at least 10 GB free disk; conda/mamba working; for Windows users, WSL2 Ubuntu 22.04+ already set up.

**If your install fails after two earnest attempts, come to L3 anyway.** The first 5 minutes of class are for troubleshooting. The homework can be completed on ARCC even if your laptop install never works — see Task 4 of Lab 0.

---

## Segment timing summary

| Time | Segment | Notes |
|---|---|---|
| 0–5 min | Install verification + troubleshooting | Show of hands; help stragglers |
| 5–10 min | What MOOSE is, in 5 minutes | The 60-sec version + a forward look |
| 10–25 min | Read `plate_with_hole.i` together; run it once on the projector | The mesh, BCs, postprocessors |
| 25–35 min | ARCC orientation | Cluster, filesystem, modules, SLURM essentials |
| 35–45 min | Live: submit hello-world Python job to ARCC | Watch it run, retrieve output |
| 45–50 min | Lab 0 preview + closing | What's due, when, how |

Net effect: students get the same MOOSE/ARCC content but the in-class fragile-network risk is gone.

---

## Segment 1 — Install verification + troubleshooting (5 min)

Open the lecture by asking: *who got MOOSE installed and got the screenshot for the homework?*

- **If 80%+ succeeded.** Acknowledge it, move on. Those who failed should pair with a successful neighbor for the in-class portions and use ARCC for the Lab 0 takehome.
- **If 50–80% succeeded.** Spend 5 minutes walking through the most common failure mode reported in the morning's homework submissions. Almost always one of: (a) macOS arm64 wheel not found → use `--platform osx-64` with Rosetta, (b) conda channel priority mismatch → add `--strict-channel-priority`, (c) WSL out of memory → bump `.wslconfig`.
- **If <50% succeeded.** Something is systematically wrong — most likely INL changed the install path. Pause class, work through it together as an extended Segment 1, and compress Segments 4 and 5 (ARCC) to compensate.

The point of this segment is *not* to fix every install. The point is to *normalize that some installs fail and recovery is part of the workflow*.

## Segment 2 — What MOOSE is, in 5 minutes (5 min)

**MOOSE** is the Multiphysics Object-Oriented Simulation Environment, an open-source finite-element framework from Idaho National Laboratory. Built on libMesh (mesh + element math) and PETSc (parallel solvers). Used widely for nuclear fuel modeling, geomechanics, multiphysics coupling, and increasingly for ML-augmented constitutive models.

Why MOOSE for this course:

- **Open source.** Installable on ARCC with no license fees, and easily on student laptops.
- **Parallel by default.** Same input file runs on 1 core or 1000.
- **Active and supported.** Funded by DOE, with a real user community.
- **Plays well with learned constitutive models.** In Module 5 we will couple a trained PyTorch model into a MOOSE Material object.
- **The right level of abstraction for this course.** We use input files; we do not write C++. Later in the course you will use agents to help prepare more complex input files and (much later) to scaffold custom Material classes.

What MOOSE is *not* for: rapid prototyping of new variational formulations (use FEniCS for that), highly specialized solid-mechanics features (use Abaqus or LS-DYNA for industrial contact), or pedagogical clarity in 100 lines (no FE code is clear in 100 lines). For an HPC-friendly, ML-coupling-friendly, open-source FE platform, MOOSE is the right choice.

Install paths INL supports:

1. **conda (mamba) packages** — recommended for end users on macOS and Linux. This is the path we use today.
2. **From source via INL's `mooseframework/moose-conda` recipe.** For users wanting to track development.
3. **Docker / Singularity containers.** For HPC and reproducibility.

We follow path 1 today. **For ARCC**, the course provides a pre-built MOOSE binary at `/project/me5475/software/rom_opt_arcc/rom_opt-opt` (SOLID_MECHANICS compiled in, hash `437fbe5082`), paired with the shared conda env at `/project/me5475/envs/ml4sm` — students do not need to build MOOSE themselves on ARCC.

## Segment 3 — Read `plate_with_hole.i` together on the projector + run it once (15 min)

The example is a quarter-annulus model of a large plate with a centered circular hole: inner radius 0.1, outer radius 1.0, symmetry edges at 0° and 90°, and the exact uniform-far-field traction projected onto the outer arc. We return to this geometry in Module 3 as the PINN lab anchor and again in Module 5 as the NN-augmented-MOOSE-material exemplar.

Open `module_0/examples/plate_with_hole.i` on the projector. Walk through it block by block while students follow on their laptops:

- `[Mesh]` — `AnnularMeshGenerator` with `rmin = 0.1` (hole radius), `rmax = 1.0` (outer edge), `dmin = 0`, `dmax = 90` (degrees). This produces a **quarter annulus** — we exploit the problem's symmetry to model only a quarter of the full plate-with-hole, which is enough thanks to the symmetric BCs below. The generator emits four named sidesets: `rmin` (the hole curve), `rmax` (the outer arc), `dmin` (the bottom radial edge, y=0), and `dmax` (the left radial edge, x=0). Mesh density: `nr = 8` radial × `nt = 24` azimuthal elements.
- `[Variables]` — `disp_x`, `disp_y`.
- `[Modules/TensorMechanics/Master]` — the legacy (still supported by the pinned course binary) master action that auto-generates the momentum equations and requested stress/strain outputs. `add_variables = false`, `incremental = false`, and `planar_formulation = PLANE_STRAIN` are load-bearing settings. A current local MOOSE install may print a TensorMechanics deprecation warning because the newer name is SolidMechanics; that warning is expected for this validated input.
- `[BCs]` — Symmetry: `disp_y = 0` on `dmin` (bottom radial edge), `disp_x = 0` on `dmax` (left radial edge). Far-field tension: `FunctionNeumannBC` on the outer arc `rmax` with `function = 'x / sqrt(x*x + y*y)'`, which evaluates to `cos(θ)` — the radial component of a uniform horizontal traction. The hole boundary `rmin` is traction-free by default (natural Neumann zero).
- `[Materials]` — `ComputeIsotropicElasticityTensor` with `youngs_modulus = 1` and `poissons_ratio = 0.3` to match Min Lin's PINN setup.
- `[Postprocessors]` — `ElementExtremeValue` reports both `max_vonmises_stress` and `max_stress_xx` (hoop stress at the hole top), while `NumElements` reports mesh size. At refine 4, expect ≈2.48 / ≈2.88; compare those measured values with the ≈2.67 / ≈3 infinite-plate references.
- `[Executioner]` — `Steady` with PETSc options.
- `[Outputs]` — Exodus and CSV.

**Why a quarter annulus?** A full plate-with-hole has two symmetry planes (x-axis and y-axis), so modeling only the first quadrant is equivalent. The boundary conditions on the symmetry edges are the symmetry BCs: zero normal displacement on each plane. This reduces mesh size by 4× without loss of accuracy. The outer-arc traction `cos(θ)` is the radial projection of a uniform horizontal far-field stress `σ_∞ = 1`.

After the walk-through (~10 min in), run the file live on the projector:

```bash
cd ~/MLCourse/module_0/examples
moose-opt -i plate_with_hole.i
```

It completes in ~5 seconds. Open `plate_with_hole_out.e` in ParaView (or `peacock`), apply *Warp by Vector* on displacement, color by `vonmises_stress`, and show the stress concentration around the hole. The coarse refine-0 peak von Mises value is ≈1.25 (the ≈1.62 value belongs to `stress_xx`). This is the "huh, it works" moment that anchors the rest of the lecture.

## Segment 4 — Brief ARCC orientation (10 min)

Live walkthrough on the projector while students follow on their own machines.

**Login.**

```bash
ssh <netid>@medicinebow.arcc.uwyo.edu
```

UW's HPC cluster for this course is **MedicineBow** (login host above; IP 10.198.64.227 for off-campus tunneling). Each student logs in with their UW NetID.

**Filesystem layout.**

- `$HOME` (`/home/<netid>`) — small quota, backed up. Use for code and per-user scripts.
- `/project/me5475/` — shared course storage (5 TB). Use for datasets, trained model checkpoints, and any group-shared artifacts. NOT backed up.
- `/scratch` — fast node-local storage during a job. Ephemeral; copy results back to `/project/me5475/` or `$HOME` before the job ends.

**The `module` system.** The course uses one module to bootstrap conda, then activates the course-provided env:

For ML-only jobs (GPU: PINN, CNN, FNO, LSTM; CPU: MLP, Optuna, GNN):
```bash
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
```

For MOOSE jobs (M0 convergence study, M6 RVE data generation) — additional modules and two `LD_LIBRARY_PATH` prepends are required:
```bash
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"
MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt
```

All SLURM scripts in this repo already include the correct stack; students do not need to invoke these lines manually except for interactive shells.

**SLURM essentials.**

- `sbatch script.sbatch` — submit a batch job
- `squeue -u $USER` — show your jobs
- `scancel <jobid>` — cancel a job
- `sacct -j <jobid>` — see resources used after a job finishes

**Partitions and accounts on MedicineBow for this course:**

| Use | Account | Partition | Hardware |
|-----|---------|-----------|----------|
| CPU work (MOOSE, light ML) | `me5475` | `mb` | Standard MedicineBow CPU nodes |
| GPU work (PINN, CNN, U-Net, FNO) | `me5475` | `mb-l40s,mb-a30` | L40S / A30 GPUs (`--gres=gpu:1`) |

**The `#SBATCH` directive cheat sheet.** Quick reference:

| Directive | What it sets | Example |
|-----------|--------------|---------|
| `--job-name` | Job name | `--job-name=hello` |
| `--account` | Allocation to charge | `--account=me5475` |
| `--partition` | Which queue | `--partition=mb` (CPU) or `mb-l40s,mb-a30` (GPU) |
| `--time` | Walltime | `--time=00:10:00` |
| `--nodes` | Number of nodes | `--nodes=1` |
| `--ntasks-per-node` | MPI ranks per node | `--ntasks-per-node=4` |
| `--cpus-per-task` | OpenMP threads per rank | `--cpus-per-task=1` |
| `--gres` | GPUs | `--gres=gpu:1` |
| `--mem` | Memory per node | `--mem=4G` |
| `--output` | stdout file | `--output=slurm-%j.out` |
| `--mail-type` | Email notifications | `--mail-type=END` |

More in `module_0/readings/slurm_cheatsheet.md`.

## Segment 5 — Submit hello-world job to ARCC (10 min)

Live demo on the projector. Students follow.

**The Python target** (`module_0/examples/hello.py`):

```python
import os, socket, sys
print(f"Hello from ARCC.")
print(f"Hostname: {socket.gethostname()}")
print(f"Python:   {sys.version}")
print(f"PWD:      {os.getcwd()}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'none')}")
```

**The SLURM script** (`module_0/examples/hello.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --account=me5475           # course allocation
#SBATCH --partition=mb             # MedicineBow CPU partition
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=hello-%j.out
#SBATCH --error=hello-%j.err

module purge
module load miniconda3/24.3.0      # conda base ships Python 3.12
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh

srun python hello.py
```

**Run it together:**

```bash
sbatch hello.sbatch
squeue -u $USER
# wait a moment...
cat hello-<jobid>.out
```

Students with active access follow the demo directly. Students whose access is still activating pair up for the in-class run and submit their own job as soon as access is active — without penalty. When the output file appears with the hello-world content, the end-to-end ARCC path is verified.

**Briefly preview Lab 0 (last ~2 min).** The takehome portion is a mesh-refinement convergence study on the plate-with-hole problem, running 5 jobs on ARCC via a SLURM array, with the modifications driven by your lead + review agents. Full instructions are in `module_0/homework/lab_0.md`. Due before L4 (Wed Sep 9).

---

## Assigned reading (before Module 1, Lecture 4)

**Primary.**

- MOOSE Framework, *Getting Started* page, mooseframework.inl.gov/getting_started. The current home page of installation and first-tutorial documentation.
- MOOSE Framework, *Tensor Mechanics* module overview, mooseframework.inl.gov/modules/tensor_mechanics. Read the *Introduction* and *Theory* sections.
- UW ARCC MedicineBow user guide, https://arccwiki.uwyo.edu/. The cluster overview, filesystem layout, and SLURM sections. For this course: account `me5475`; login `ssh <netid>@medicinebow.arcc.uwyo.edu`.
- `module_0/readings/slurm_cheatsheet.md` — keep this open during all your ARCC work this semester.

**Optional / reference.**

- Permann et al., *MOOSE: Enabling massively parallel multiphysics simulation*, SoftwareX 11, 2020. For students who want the design philosophy of MOOSE.
- `module_0/readings/moose_input_file_anatomy.md` — a section-by-section reference for the input file you ran today.

---

## Lab 0 deliverables (full Module 0 closure)

Combining the in-class portions of Lectures 2 and 3 with the takehome convergence study. See `module_0/homework/lab_0.md` for the canonical specification. In brief:

1. From Lecture 2: lead and review outputs for the weak-form derivation and the MLP-fits-sine code.
2. From Lecture 3: local MOOSE install verified + `plate_with_hole.i` running locally + screenshot of the von Mises field.
3. From Lecture 3 ARCC demo: `hello.sbatch` submitted, output committed.
4. **Takehome convergence study:** modify `plate_with_hole.i` to run at `uniform_refine ∈ {0, 1, 2, 3, 4}`, run all five via a SLURM array on ARCC, parse the CSV outputs, and plot both peak metrics vs. element count. Expect ≈2.48 (plane-strain von Mises) and ≈2.88 (hoop `max_stress_xx`) at refine 4; compare with the ≈2.67 / ≈3 infinite-plate references.
5. Reflection ≤ 400 words.

---

## Instructor notes (not for student view)

- This is the most time-pressured lecture of Module 0. Practice it twice before delivery to confirm the install completes in <15 minutes on your demo machine.
- Two failure modes to plan for: (a) the MOOSE install hits a real bug that the agent cannot resolve — keep a known-good environment open in a separate terminal as a fallback; (b) ARCC login is down — have the hello-world output pre-computed as a static file to show.
- The Module 0 → Module 1 transition assumes every student has the basic environment working. Use the days after Lecture 3 to debug stragglers individually.
- The plate-with-hole problem returns in Module 3 (PINN), Module 5 (NN-UMAT-in-MOOSE) and possibly in the final project. The fact that students implement it once in MOOSE here makes those later modules much smoother.
- When presenting the convergence plot, distinguish the assigned refine-4 expectations (≈2.48 von Mises / ≈2.88 hoop) from the infinite-plate references (≈2.67 / ≈3); the localized peaks are still rising at refine 4.
