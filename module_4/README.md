# Module 4 — Data-Driven Constitutive Modeling

**Course:** Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang
**Module length:** 6 lectures (Wk 7 Fri → Wk 9 Wed)
**Status:** Draft v0.1

---

## What this module does

The cumulative application of everything we've built so far — MLP machinery from M1, hyperparameter sweeps from M2, physics-informed framing from M3 — to the deepest problem in solid mechanics: *what is the constitutive law?*

For elastic materials, the constitutive law is known (Hooke). For path-dependent materials — plasticity, damage, viscoplasticity — the law lives in textbooks but only as approximate forms (J2, Drucker-Prager, Mohr-Coulomb, Cam-Clay). Real polycrystalline materials, granular media, and shape-memory alloys typically need *something between* these textbook forms and what RVE simulations actually produce. ML lets us learn that something.

By the end of M4 students have:

- Generated J2 plasticity stress-strain trajectories from MOOSE single-element cyclic loading.
- Trained a stateless MLP baseline and watched it fail on path-dependent data.
- Trained an LSTM that carries the loading history, and compared it with the stateless baselines.
- Built physics constraints — frame indifference, material symmetry, polyconvexity — into neural-network constitutive models via Steve Sun's three-ways taxonomy.
- Implemented Sobolev H¹ training that fits energy and stress together (H², which adds stiffness, is discussed but not implemented in the course code).
- Understood Steve's level-set yield surface reformulation and trained a yield-surface neural network with autograd-managed gradients.

**Module deliverable (Lab 4)** is a J2 plasticity surrogate. **Midterm project spec** releases on L19 (first day of M4, Fri Oct 16); due Sun Nov 8, 11:59 PM MT (extended from Sun Nov 1 on Oct 26). No midterm presentation (decision 2026-10-03).

This module is the heaviest single citation of Steve Sun's *ML for Mechanics* lectures (L3, L4, L5, L6). The "three ways to enforce a constraint" taxonomy is L3; the RNN-for-plasticity content is L4; the level-set yield surface formulation is L5+L6.

---

## File index

```
module_4/
├── README.md
├── lectures/
│   ├── L19_constitutive_refresher.md                  (Fri Oct 16) refresher + data gen + midterm spec
│   ├── L20_physics_constraints_I.md                   (Mon Oct 19) frame indifference, symmetry
│   ├── L21_physics_constraints_II_icnn_sobolev.md     (Wed Oct 21) polyconvexity, ICNN, Sobolev
│   ├── L22_rnn_plasticity.md                          (Fri Oct 23) RNN, LSTM, path dependence
│   ├── L23_nn_yield_surface_I_levelset.md             (Mon Oct 26) level-set + signed-distance
│   └── L24_nn_yield_surface_II_sobolev_fem.md         (Wed Oct 28) Sobolev + algorithmic tangent
├── examples/
│   ├── j2_cyclic_loading.i                MOOSE: J2 plasticity single element, fully reversed cyclic BCs
│   ├── generate_plasticity_data.py        driver: 200 cyclic load paths -> trajectories.npz
│   ├── generate_data.sbatch               ARCC job: runs the driver with the course MOOSE build
│   ├── stateless_mlp_baseline.py          the "wrong way" baseline for L22
│   ├── lstm_plasticity.py                 LSTM that carries the loading history (Lab 4 Task 3; discussed in L22)
│   ├── icnn_hyperelastic.py               Input Convex NN for polyconvex hyperelasticity (L21)
│   ├── levelset_yield_surface.py          Steve's approach implemented (L23-L24, Lab 4 stretch)
│   └── train_constitutive.sbatch          ARCC SLURM template (CPU partition)
├── homework/
│   ├── lab_4.md                           J2 plasticity surrogate (Wed Oct 14 -> Fri Nov 6)
│   ├── midterm_project_spec.md            three flavors, due Sun Nov 8
│   └── starter_prompts.md
└── readings/
    ├── reading_list.md
    ├── measured_results.md                every computed number in Lab 4 and the M4 readings
    ├── sobolev_training_primer.md         H1/H2 derivation, worked examples
    └── icnn_primer.md                     Input Convex NN explained
```

---

## How this module weaves Steve Sun's material

| Steve's lecture | M4 use |
|---|---|
| L3 (physics constraints, three-ways taxonomy) | **L20** central organizing framework |
| L3 Sobolev training section | **L21** the H¹/H² training paradigm |
| L4 (RNN for plasticity) | **L22** condensed; frame-indifference repair via spectral decomposition |
| L5 (level-set yield surface) | **L23** the level-set/SDF formulation |
| L5 + L6 (Sobolev training of yield surface, multiply-layer, return mapping) | **L24** the algorithmic-tangent integration |

The intellectual debt to Steve is largest here. Citations are inline in each lecture; the reading list cites his 2021/2022 papers as primary throughout M4.

---

## Where this module connects forward and back

**Back to M1:** the constitutive MLP from Lab 1 is the stateless baseline of Lab 4 Task 2 — and the lesson is that stateless models fail on path-dependent data.

**Back to M3:** PINN's physics-informed loss is one example of constraint enforcement; M4 generalizes to *three* enforcement strategies and applies to constitutive maps rather than PDE solutions.

**Forward to M5 (Wk 10):** Module 5 evaluates trained constitutive models on states generated by MOOSE and, if the course's in-solver integration path is validated in time, inside a live MOOSE simulation (syllabus §5).

**Forward to the midterm:** Flavor B is "data-driven constitutive surrogate" — directly builds on Lab 4.

---

## Open items

- ARCC MOOSE binary: **`/project/me5475/software/moose_torch/me5475/me5475-opt`**: the course build from 2026-10, MOOSE `437fbe5082` with SolidMechanics and the TorchScript material Module 5 uses (`module_0/readings/moose_build_switch_guide.md`). Lab 4's data were first generated with the previous executable, `/project/me5475/software/rom_opt_arcc/rom_opt-opt` (same MOOSE version); the new build regenerates the identical file (`readings/measured_results.md` §1). **Lab 4 run end to end on ARCC 2026-10-03** — `j2_cyclic_loading.i` (after a boundary-condition fix), `generate_data.sbatch`, all four training scripts — and re-run the same day with **fully reversed cycles** (0 → +a → −a → 0, three times; instructor decision D3): `readings/measured_results.md` §1–§3. The radial-cycle results are kept there as superseded history (§5).
- **Reference `trajectories.npz` (decision D4, 2026-10-03):** students generate their own on ARCC (Lab 4 Task 1); a validated reference copy, for comparison by SHA-256 or for use if their generation fails, goes to `/project/me5475/examples/lab4_reference/` with `README.txt` and `SHA256SUMS`. It is prepared and checked (`readings/measured_results.md` §1.2) and is **staged by the Co-Worker only after the Reviewer gate** — until then that folder does not exist, and Lab 4 Task 1b already points to it.
- The level-set yield-surface example (`levelset_yield_surface.py`): validation run done 2026-10-03 (`readings/measured_results.md` §3).
