# Module 0 — Setup, Agentic Workflow, and ARCC + MOOSE Orientation

**Lectures:** L1 (Mon Aug 31) · L2 (Wed Sep 2) · L3 (Fri Sep 4)
**Lab 0 due:** Friday, September 11, 11:59 PM Mountain Time — see Canvas for the authoritative date.

By the end of this module you will have an ARCC account and a working local Python
environment, you will have used the two-agent lead + review workflow on both a derivation and
an implementation task, and you will have run a mesh-refinement convergence study on the
cluster for a plate-with-hole problem that returns in Module 3.

## Lectures

| | Topic | Notes |
|---|---|---|
| L1 | Why ML for computational solid mechanics — where it fits, and when it doesn't | [`lectures/L1_why_ml_for_mechanics.md`](lectures/L1_why_ml_for_mechanics.md) |
| L2 | AI agents and the lead + review workflow | [`lectures/L2_agents_lead_review.md`](lectures/L2_agents_lead_review.md) |
| L3 | MOOSE and ARCC, with agents | [`lectures/L3_moose_arcc_with_agents.md`](lectures/L3_moose_arcc_with_agents.md) |

## What's here

- **`setup.md`** — install Python, PyTorch, and your agent. Do this first.
- **`examples/`** — `hello.sbatch` (your first SLURM job), `plate_with_hole.i` (the MOOSE input
  file we walk through in L3), `run_convergence_study.sbatch` (the SLURM array for Lab 0),
  `extract_stress.py` (starter for parsing results), `install_moose_prompt.md` (the agent
  prompt for the local MOOSE install).
- **`readings/`** — MOOSE input-file anatomy, a SLURM cheat sheet, and the reading list.
- **`homework/lab_0.md`** — the lab handout and rubric.

## Two numbers to know for Lab 0

The plate-with-hole model is **plane strain** with ν = 0.3, on a quarter annulus. At the
refinement levels you run (0–4), expect the peaks to still be climbing:

| Metric | Expect at refine 4 | Infinite-plate reference |
|---|---|---|
| `max_stress_xx` (hoop at the hole) | ≈ **2.88** | ≈ 3 (Kirsch) |
| `max_vonmises_stress` | ≈ **2.48** | ≈ 2.67 = 3·√(1 − ν + ν²) |

The references are comparison lines, **not** targets for refine 4. Explaining the remaining gap
is part of the lab.
