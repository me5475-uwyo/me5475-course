# Module 0 — Setup, Agentic Workflow, ARCC + MOOSE Orientation

**Course:** ME-5475-01 — Machine Learning for Computational Solid Mechanics (UW, Fall 2026)
**Instructor:** Prof. Xiang Zhang (CAMML Lab, UW Mechanical Engineering)
**Class meeting:** MWF 1:10–2:00 PM, Engineering Building Room 2070
**Office hours:** MWF 2:00–3:00 PM (right after class)
**Canvas:** https://uwyo.instructure.com/courses/620641
**Module length:** 3 lectures (Wk 1 Mon + Wed + Fri)
**Status:** Phase 1A.1 revision — two-metric ARCC validation complete

---

## What this module does

By the end of Module 0, each student has:

1. A working understanding of why ML matters in computational solid mechanics, with three concrete success stories memorized.
2. A working AI coding agent setup (Claude Code or Cursor) and hands-on practice with the 2-agent *lead + review* pattern.
3. A locally installed MOOSE finite-element framework, installed *with an agent before Lecture 3*, with one solid-mechanics problem (plate-with-hole) running.
4. An active ARCC account, a submitted hello-world SLURM job, and a successful mesh-refinement convergence study completed as Lab 0.

The plate-with-hole problem from this module is the same geometry that returns in Module 3 (PINN for elasticity), so students see a deliberate before/after — classical FE in Module 0, physics-informed neural network in Module 3, against the same problem and the same reference data.

---

## File index

```
module_0/
├── README.md                              this file
├── setup.md                               student setup checklist for L1
├── lectures/
│   ├── L1_why_ml_for_mechanics.md         (Mon Aug 31)
│   ├── L2_agents_lead_review.md           (Wed Sep 2)
│   └── L3_moose_arcc_with_agents.md       (Fri Sep 4)
├── examples/
│   ├── plate_with_hole.i                  MOOSE input — 2D plate with hole, uniaxial tension
│   ├── hello.py                           ARCC SLURM target — trivial Python
│   ├── hello.sbatch                       ARCC SLURM script for hello.py
│   ├── run_moose_local.sh                 one-liner to run plate_with_hole.i on laptop
│   ├── run_convergence_study.sbatch       SLURM array template for 5 refinement levels
│   └── extract_stress.py                  Python helper to read MOOSE CSV outputs and plot
├── homework/
│   ├── lab_0.md                           assignment instructions + rubric
│   └── starter_prompts.md                 example lead+review prompts students can adapt
└── readings/
    ├── reading_list.md                    annotated bibliography for Module 0
    ├── slurm_cheatsheet.md                one-page SLURM reference
    └── moose_input_file_anatomy.md        what each [Block] does
```

## How to use this module's files

- **Lectures.** The `lectures/` markdown files are the instructor's full speaking notes plus board work. They render in Jupyter Book as the live course textbook chapter.
- **Examples.** Every `examples/` artifact is referenced from at least one lecture and from Lab 0. They are versioned and treated as production code, not throwaway snippets.
- **Homework.** `lab_0.md` is the canonical assignment text; `starter_prompts.md` is a permitted aid that students may adapt.
- **Readings.** `reading_list.md` is what students are accountable for; the cheatsheets are reference material kept for the whole semester.

## Acknowledgments

Much of the conceptual scaffolding of this course traces to Prof. WaiChing (Steve) Sun's *ML for Mechanics* graduate course (Columbia / Stanford), generously shared with the instructor as 13 lecture decks. Specifically: the historical framing in L1 follows his Lecture 1; the three-ways-to-enforce-a-constraint taxonomy used throughout Modules 3–5 is his; the Sobolev (H¹/H²) training paradigm is his group's (Vlassis & Sun, CMAME 2021); the level-set / signed-distance-function reframing of yield surfaces in Module 4 is his (Vlassis & Sun, JAM 2021). Where we adapt his material we cite the underlying papers; where we depart from his approach (most notably the agentic workflow, the MOOSE-centric FE integration, and the de-emphasis of graph methods) the choices are the instructor's. Steve is also a collaborator on related research, and Module 4 in particular benefits directly from ongoing discussions.

The PINN labs in Module 3 are anchored on Min Lin's 2D plate-with-hole DeepXDE notebook (`PINN_Example/`), produced during his graduate research with the instructor (CAMML Lab, UW Mechanical). The Module 0 MOOSE example is the same geometry, deliberately, to enable cross-method comparison.

## Validation status

- `plate_with_hole.i` — the **base solve and two postprocessors were validated** on MedicineBow. Across the student range (refine 0–4), `max_vonmises_stress` is 1.25 → 1.56 → 1.93 → 2.25 → **2.48**, and `max_stress_xx` is 1.62 → 2.00 → 2.38 → 2.68 → **2.88**. Grade the refine-4 peaks against 2.48 / 2.88; the corresponding infinite-plate references are ≈2.67 (plane-strain von Mises) / ≈3 (Kirsch hoop). Instructor runs through refine 6 reach 2.72 / 3.10, confirming that the localized peaks are still rising at refine 4.

## Status — Phase 1A.1 revised; ARCC validation complete

The ARCC course allocation is provisioned, the software stack is installed, and the two-metric Lab 0 sequence has been validated through refine 6:

- **ARCC migration complete (as of 2026-06-09).** The course has a dedicated allocation `me5475` on the MedicineBow cluster, login `medicinebow.arcc.uwyo.edu`. Confirmed values: account `me5475`; CPU partition `mb`; GPU partition `mb-l40s,mb-a30` (with `--gres=gpu:1`). All SLURM scripts and the L3 SBATCH reference use these.
- **Software is installed under `/project/me5475/`** and the project is fully self-contained (no runtime dependency on the instructor's lab project): MOOSE binary `rom_opt-opt` (git hash `437fbe5082`, `SOLID_MECHANICS=yes`) at `/project/me5475/software/rom_opt_arcc/rom_opt-opt`; conda env `ml4sm` (PyTorch+CUDA, DeepXDE, Optuna, PyTorch Geometric) at `/project/me5475/envs/ml4sm`. Runtime preamble prepends both the GCC-14 `libstdc++` and `libhit-opt.so.0` paths to `LD_LIBRARY_PATH` (see `arcc_setup_notes.md`).
- **Two-metric validation complete (2026-08-25).** Student refine-4 expectations are `max_stress_xx ≈ 2.88` and `max_vonmises_stress ≈ 2.48`; use the infinite-plate references ≈3 and ≈2.67 as comparison lines, not as refine-4 grading targets.

### Inline-TODO resolutions (2026-05-20)

- 4-agent framework cited as **Zhang et al. (in prep., 2026)**; PDF distributed via Canvas at semester start.
- Reference textbook: **Belytschko, Liu, Moran & Elkhodary (2014)**, semester-wide consult resource for continuum mechanics and FEA implementation. No per-lecture chapter assignments.
- Lab 0 due **before L4 (Wed Sep 9)**.
- Office hours: **MWF 2:00–3:00 PM**, no TA.
