# Module 5 — NN-FEA Integration with MOOSE

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 3 lectures (L25 Fri Oct 30, L26 Mon Nov 2, L27 Wed Nov 4)
**Status:** Lab 5 package validated on ARCC, 2026-10-03 (`readings/measured_results.md`). Lectures not yet revised
to the validated design (see "Open items").

---

## What this module does

The deployment payoff for M1–M4: a trained network becomes the material of a finite-element simulation. By the end of
M5 students have:

- exported their Lab 1 MLP, σ(ε) for plane-strain linear elasticity, to a TorchScript file, and verified the file
  against Hooke's law, against the eager PyTorch model, and against the strains it was trained on;
- run Lab 0's plate with a hole in MOOSE with that network as the material, through the course material
  `ComputeTorchScriptStress`: MOOSE evaluates the network at every quadrature point and Newton uses its autograd
  tangent;
- compared the network run with the elastic run on the same mesh, field by field, and read the cost (Newton
  iterations, run time);
- loaded the plate beyond the strains in the network's data and seen what extrapolation does inside a solver.

Students do not write C++. They export a file, change one block of an input file, and run a job.

**Module deliverable (Lab 5):** released Tue Oct 27 (the day before L24), due Wed Nov 11 (L30). Lab 0's plate with the student's
own Lab 1 network as its material, compared with the elastic run; optional three-way comparison with the Lab 3 PINN
on Lab 3's plate.

---

## The course infrastructure

- **Executable:** `/project/me5475/software/moose_torch/me5475/me5475-opt`, the course MOOSE app built with libtorch
  2.5.1 (the PyTorch version of the course environment `/project/me5475/envs/ml4sm`, so files exported there load).
  Built and tested 2026-10-03; build record in `team/arcc_build/README.md`. The older `rom_opt-opt` does not have the
  network material.
- **Material:** `ComputeTorchScriptStress` (source in `team/arcc_build/app_src/`). Input block: `type` and `filename`.
  Plane Voigt strain (ε_xx, ε_yy, γ_xy) in, (σ_xx, σ_yy, σ_xy) out; float64; autograd tangent; σ_zz = 0; no state.
  Student-facing description: `readings/moose_material_interface_reference.md`.
- **Course reference network** (fallback for a Lab 1 network that cannot be used): staged on ARCC at
  `/project/me5475/examples/lab5_reference/` on 2026-10-26 (checkpoint, TorchScript file, training CSV, README,
  SHA256SUMS; read-only; check it with `sha256sum -c SHA256SUMS` in that folder).

---

## File index

```
module_5/
├── README.md                                  this file
├── lectures/
│   ├── L25_moose_primer_for_nn_materials.md   (Fri Oct 30) MOOSE material concept, no C++
│   ├── L26_torchscript_and_wrapper_pattern.md (Mon Nov 2) export pattern, the material's input block
│   └── L27_case_study_lab_walkthrough.md      (Wed Nov 4) Lab 5 walk-through and comparison
├── examples/
│   ├── export_torchscript.py                  Lab 1 checkpoint -> TorchScript file (standardization inside)
│   ├── verify_torchscript.py                  the file vs Hooke, sigma(0), autograd tangent, eager model, data hull
│   ├── plate_with_hole_nn_material.i          Lab 0's plate; `active` picks the network or the elastic material
│   ├── run_nn_simulation.sbatch               ARCC job: the network run and the elastic run, one after the other
│   └── comparison_three_methods.py            plate mode: elastic vs network (same mesh);
│                                              grid mode: Lab 3 reference vs network vs PINN (Lab 3's plate)
├── homework/
│   ├── lab_5.md                               Tue Oct 27 -> Wed Nov 11
│   └── starter_prompts.md
└── readings/
    ├── reading_list.md
    ├── measured_results.md                    every computed number in Lab 5 and the M5 readings
    ├── torchscript_export_guide.md            export idioms and the measured failure modes
    └── moose_material_interface_reference.md  what MOOSE asks of a material; what ComputeTorchScriptStress does
```

---

## Where this module connects

**Back to M1:** the Lab 1 MLP is the deployed artifact. Its training error and its training data both show up in the
plate (Lab 5 Tasks 3–4).

**Back to M0:** the plate, mesh, boundary conditions and solver are Lab 0's. The load is scaled to 1e-3 so that,
with the course reference network, almost all final element-average strains lie inside the hull of the Lab 1 data, a
coverage proxy (`examples/plate_with_hole_nn_material.i`, header).

**Back to M3:** Lab 3's PINN solves a different plate (unit square, prescribed displacement), so it is compared only on
that plate, against the shared MOOSE reference, in the optional Lab 5 Task 5.

**Back to M4:** Lab 4's LSTM and the ε^p-fed MLP cannot be deployed with this material: it keeps no state. Doing so is
a final-project-sized extension (a stateful material), not a Lab 5 task.

**Forward:** final projects that need a learned material inside a solver start from this pipeline.

---

## Open items

- **Lectures L25–L27 and their decks** still describe a hypothetical C++ wrapper with scaling and tangent-mode
  parameters, the old dates, a three-way comparison on Lab 0's plate, and a Python fallback. The list of claims the
  validated design contradicts is in `team/reviews/agent_reports/2026-10-03_coworker_subagent_M5_lab5.md` §7, for the
  deck audit.
- **The course MOOSE switch** (`module_0/readings/moose_build_switch_guide.md`) is phased; Lab 5 uses the new
  executable regardless, because only it has the network material.
