# Module 3 — Physics-Informed Neural Networks for Solid Mechanics

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 7 lectures — L12 Mon Sep 28 → L18 Wed Oct 14 (Weeks 5–7; no class Mon Oct 12)
**Lab 3:** released Mon Sep 28, due Mon Oct 19, 11:59 PM Mountain Time

---

## What this module does

Module 3 is the longest module of the course and the meatiest single topic. By the end of M3, students have:

- Derived the PINN loss from first principles, and seen when a PINN is — and is not — a better tool than classical FE (L12: on linear, well-posed forward problems on simple geometries, FE wins).
- Solved 1-D and 2-D diffusion problems with DeepXDE end to end (L13).
- Read Min Lin's plate-with-hole PINN notebook and the course's PyTorch port of it, checked that the port imposes the same **ten** boundary conditions, and trained it as a GPU job on ARCC (Lab 3 Task 1).
- Compared the PINN quantitatively with a converged **MOOSE reference of exactly the same problem** — the square plate with a quarter hole, not Lab 0's quarter annulus (Task 2).
- Compared soft and hard displacement boundary conditions in a controlled experiment — same points, same starting weights, same schedule. Hard BCs guarantee the three displacement conditions exactly; in the course's release runs they were not faster, and neither method won on every error measure (Task 3; `readings/measured_results.md` §6).
- Tuned the PINN with an Optuna sweep on the GPUs, scored on a validation half of the reference grid (Task 4).
- Seen how a parametric PINN — inputs (x₁, x₂, E, ν) instead of (x₁, x₂) — lets one network represent the whole family of plate problems for varying (E, ν) (L16).
- *(stretch)* Recovered E and ν from sparse synthetic "DIC" displacement measurements **plus one measured reaction force** — under a prescribed displacement, displacements alone cannot identify E (L18; Task 5).

**Min Lin's plate-with-hole PINN notebook is the spine of this module.** A copy with outputs cleared is `examples/min_lin_2D_hole_example.ipynb` (TensorFlow backend; read it, you do not need to run it). The course scripts use DeepXDE's PyTorch backend (one line: `os.environ["DDE_BACKEND"] = "pytorch"`, before importing DeepXDE) so they are consistent with the rest of the course's stack.

**Module deliverable (Lab 3).** The same plate-with-hole problem solved five ways: port + run (Task 1), compare with MOOSE (Task 2), soft vs hard displacement BCs (Task 3), Optuna sweep of PINN hyperparameters (Task 4), and — as a stretch, +20 bonus — the inverse problem with a ten-seed ensemble (Task 5). Plus a reflection. Full handout: `homework/lab_3.md`.

**How the work runs.** Anything that trains a network runs as a submitted batch job, never on the login node: the plate PINNs on a GPU node (`pinn_train.sbatch`, `pinn_optuna_sweep.sbatch`, `pinn_inverse_ensemble.sbatch`), L13's small diffusion PINNs on a CPU node (`pinn_diffusion.sbatch`). MOOSE runs as a CPU job too (`run_moose_diffusion.sbatch`, `run_plate_reference.sbatch`). Course rule: **at most two of your GPU jobs at a time** — the array scripts enforce it with `%2`.

---

## File index

```
module_3/
├── README.md
├── examples/
│   ├── pinn_diffusion_1d.py                  L13 demo: 1-D steady diffusion
│   ├── pinn_diffusion_2d.py                  L13 demo: the same PINN in 2-D
│   ├── pinn_diffusion.sbatch                 SLURM job for the L13 demos: a CPU job (partition mb), no GPU
│   ├── diffusion_1d.i                        L13's 1-D diffusion problem in MOOSE (Diffusion + BodyForce kernels)
│   ├── diffusion_2d.i                        the same in 2-D
│   ├── run_moose_diffusion.sbatch            SLURM job for the two MOOSE diffusion inputs (CPU, seconds)
│   ├── min_lin_2D_hole_example.ipynb         Min Lin's original notebook (TF backend, outputs cleared)
│   ├── plate_with_hole_fixed.py              PyTorch port of Min's notebook, soft BCs (Task 1)
│   ├── plate_with_hole_hard_bc.py            hard-BC ansatz version (Task 3)
│   ├── plate_with_hole_parametric.py         course port of Min's parametric notebook (arrives with L16)
│   ├── plate_with_hole_inverse.py            inverse problem: E, nu from displacements + one force (Task 5)
│   ├── plate_square_hole_reference.i         MOOSE input for the shared reference solution
│   ├── run_plate_reference.sbatch            SLURM job to run it yourself (coarse mesh by default)
│   ├── extract_moose_reference.py            samples the shared MOOSE reference on a grid or at random points
│   ├── pinn_train.sbatch                     SLURM job for one PINN training (Tasks 1, 3)
│   ├── pinn_optuna_sweep.py                  Optuna objective for the PINN sweep (Task 4)
│   ├── pinn_optuna_sweep.sbatch              SLURM array: six sweep workers, at most two at a time
│   └── pinn_inverse_ensemble.sbatch          SLURM array: ten inverse seeds, at most two at a time
├── homework/
│   ├── lab_3.md                              the lab: Tasks 1–4, stretch Task 5, reflection
│   └── starter_prompts.md                    lead + review prompt templates for each task
└── readings/
    ├── reading_list.md                       annotated bibliography, lecture by lecture
    ├── deepxde_quickstart.md                 DeepXDE in two pages
    ├── pinn_failure_modes.md                 measured failures, the four tricks, and newer ones
    ├── plane_stress_and_plane_strain_primer.md  why Min's notebook (plane stress) and Lab 3 (plane strain) differ
    ├── moose_diffusion_walkthrough.md        L13's two diffusion problems in MOOSE, and how to run them
    ├── moose_plate_walkthrough.md            how Lab 3's MOOSE reference is built and run
    └── measured_results.md                   every computed number the readings and lab cite
```

**Lecture schedule.**

| Lecture | Date | Topic |
|---|---|---|
| L12 | Mon Sep 28 | PINN concept: the Raissi 2019 paper, the collocation viewpoint |
| L13 | Wed Sep 30 | 1-D / 2-D diffusion warm-up in DeepXDE |
| L14 | Fri Oct 2 | Min's fixed-(E, ν) notebook, walked through |
| L15 | Mon Oct 5 | Hard vs soft boundary conditions |
| L16 | Wed Oct 7 | Min's parametric notebook; operator-learning preview |
| L17 | Fri Oct 9 | Training pathologies: Steve Sun's four tricks, NTK and causal training |
| L18 | Wed Oct 14 | Inverse problems: identify E and ν from sparse measurements |

---

## How the module weaves Min's work into the lectures

| Lecture | Use of Min's material |
|---------|------------------------|
| L12 | Forward reference: "you'll see this geometry in two lectures" |
| L13 | None (diffusion warm-up is independent) |
| L14 | **Min's notebook (`examples/min_lin_2D_hole_example.ipynb`) on the projector**; walk through every code block; explain the mixed (u, σ) formulation |
| L15 | Min's geometry with the displacement BCs built into the network output (hard-BC ansatz); compare with soft BCs under the same schedule |
| L16 | **Min's `2D-Hole-Example-Various-E-Nu-1.ipynb`**; the (x₁, x₂, E, ν) → (u, σ) mapping |
| L17 | Diagnosis first — the course's two measured failures (missing BCs, too few points) — then the four tricks |
| L18 | Min's parametric setup reframed as an inverse problem: (E, ν) become unknowns, and one force measurement fixes E |

---

## Acknowledgments

Most of M3 builds directly on Min Lin's 2022 PINN notebooks produced during his graduate research with the instructor (CAMML Lab, UW Mechanical). Where this course extends Min's work — most notably the hard-BC ansatz, the comparison to MOOSE rather than ABAQUS, the Optuna sweep, and the inverse-problem formulation — the underlying PINN code structure is still recognizably Min's.

The four-tricks discussion in L17 adapts Steve Sun, *Geometric Learning for Solid Mechanics*, Lecture 7 (with cited modernizations).

---

## Environment and shared files

- **Python environment:** `/project/me5475/envs/ml4sm` on ARCC MedicineBow — DeepXDE 1.15.0 with the PyTorch 2.5.1 backend, Optuna 4.8.0, PyTorch Geometric. It does not contain TensorFlow; Min's TF notebook is for reading.
- **Shared MOOSE reference** (read-only; copy from, never write into): `/project/me5475/examples/lab3_reference/` — the converged solution of exactly the PINN's problem (`plate_square_hole_reference_out.e`), its input file, and its measured quantities (`plate_square_hole_reference_out.csv`). How it was built and checked: `readings/measured_results.md` §1.
- **Measured numbers:** every computed number in the M3 readings cites `readings/measured_results.md`; anything not measured says so.
