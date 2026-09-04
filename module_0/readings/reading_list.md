# Module 0 — Reading List

Annotated bibliography of everything assigned across the three lectures of Module 0, plus optional / reference items. Items marked **[Primary]** are required reading; **[Optional]** items are recommended but not graded against.

---

## Before Lecture 2 (assigned in Lecture 1)

### [Primary] Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang, *Physics-informed machine learning*, Nature Reviews Physics 3, 422–440, 2021.

A 12-page survey that defines the modern landscape of ML for PDE-governed problems. Read fully.

- **What to take away.** A vocabulary for talking about the different ways to combine physics with ML: physics-informed loss functions, hard constraints in architecture, operator learning. The taxonomy in Figure 1 is the cleanest you will find in any one place.
- **What to skim.** The applications gallery — you will see specific cases in the relevant modules later.
- **DOI.** 10.1038/s42254-021-00314-5.

### [Primary] Vlassis & Sun, *Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity models with level set hardening*, CMAME 377, 113695, 2021.

The flagship paper of the level-set / Sobolev training framework you will meet at depth in Module 4. Read the introduction and Section 2 (the Sobolev training setup); algorithmic details can wait until Module 4. This is your collaborator Prof. Steve Sun's group's work; the conceptual scaffolding of much of this course traces to it.

- **What to take away.** The taxonomy of physics constraints (data, thermodynamic, frame-invariance) and how Sobolev (H¹) loss enforces *derivative-of-the-prediction* agreement, not just function-value agreement. The level-set reframing of yield surfaces (Section 4) is the most novel content in the paper; we return to it in Module 4.
- **DOI.** 10.1016/j.cma.2020.113695

### [Primary] Anthropic, *Building effective agents*, anthropic.com/engineering, December 2024.

The canonical introduction to agent design patterns. Open-access blog post (no DOI).

- **What to take away.** The distinction between *workflows* (predetermined sequences of LLM calls) and *agents* (LLMs that dynamically choose tools). The five pattern primitives — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. Lab 0 uses orchestrator-workers (the human is the orchestrator, the lead and review agents are the workers).
- **URL.** anthropic.com/engineering/building-effective-agents

---

## Before Lecture 3 (assigned in Lecture 2)

### [Primary] Anthropic, *How we built our multi-agent research system*, anthropic.com/engineering, June 2025.

A production multi-agent example aligned with the workflow we are teaching.

- **What to take away.** What multi-agent looks like when it works in production — and where it breaks down. Particularly relevant if the final-project capstone interests you.
- **URL.** anthropic.com/engineering/built-multi-agent-research-system

### [Optional] Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Hambro, Zettlemoyer, Cancedda & Scialom, *Toolformer: Language models can teach themselves to use tools*, NeurIPS 2023.

The academic precursor to tool-using LLMs. Read this only if you want the research-paper view of how tool use was operationalized before Claude / GPT-4 made it routine.

- **arXiv.** 2302.04761

### [Optional] Anthropic, *Model Context Protocol specification*, modelcontextprotocol.io.

Reference material for students who later want to write MCP servers exposing custom mechanics tools (e.g., a MOOSE-run server an agent can call). Not required for the course; useful for the final project.

---

## Before Module 1, Lecture 4 (assigned in Lecture 3)

### [Primary] MOOSE Framework, *Getting Started* page, mooseframework.inl.gov/getting_started.

The current INL documentation home for installation and first-tutorial guidance. Skim the *Installation* section even if our in-class agent demo handled it for you — the conceptual layout (conda / source / container paths) is worth knowing.

### [Primary] MOOSE Framework, *Tensor Mechanics module*, mooseframework.inl.gov/modules/tensor_mechanics.

Read the *Introduction* and *Theory* sections at minimum. These document the kernels and material classes used in `plate_with_hole.i`.

- **What to take away.** What `ComputeIsotropicElasticityTensor`, `ComputeLinearElasticStress`, and the `TensorMechanics/Master` action actually compute. Why `PLANE_STRAIN` is the right `planar_formulation` for the plate-with-hole problem (under uniaxial extension of a thin plate, `PLANE_STRESS` is the more conservative modeling choice — we use `PLANE_STRAIN` to match Min Lin's PINN setup for the comparison in Module 3).

### [Primary] UW ARCC MedicineBow user guide, https://arccwiki.uwyo.edu/

The cluster overview, filesystem layout, and SLURM sections. For this course, use the dedicated allocation `me5475` (partition `mb` for CPU, `mb-l40s,mb-a30` for GPU). Login: `ssh <netid>@medicinebow.arcc.uwyo.edu`.

- **What to take away.** The current partition names, walltime limits, default memory per node, and how MOOSE is provided (module vs. install-your-own conda env).

### [Primary] `module_0/readings/slurm_cheatsheet.md` (this directory).

A one-page reference. Print it out and keep it near your laptop for the whole semester.

### [Primary] `module_0/readings/moose_input_file_anatomy.md` (this directory).

Section-by-section reference for `plate_with_hole.i`. Read it alongside the file itself.

### [Optional] Permann, Gaston, Andrš, Carlsen, Kong, Lindsay, Miller, Peterson, Slaughter, Stogner & Martineau, *MOOSE: Enabling massively parallel multiphysics simulation*, SoftwareX 11, 100430, 2020.

The design philosophy and architecture of MOOSE in a single peer-reviewed paper. Worth reading if you want to understand *why* MOOSE looks the way it does. Not required to use MOOSE effectively.

- **DOI.** 10.1016/j.softx.2020.100430

---

## Historical / context (referenced in Lecture 1 but not assigned)

These are the founding papers of the field. Worth knowing they exist; not required to read in full.

- **Ghaboussi, Garrett & Wu, 1991** — *Knowledge-based modeling of material behavior with neural networks*, J. Engrg. Mech. 117(1), 132–153. The first NN constitutive law paper.
- **Kirchdoerfer & Ortiz, 2016** — *Data-driven computational mechanics*, CMAME 304, 81–101. The model-free / data-driven thread.
- **Raissi, Perdikaris & Karniadakis, 2019** — *Physics-informed neural networks*, J. Comp. Phys. 378, 686–707. Full reading deferred to Module 3.

---

## Reference textbook (semester-wide)

**Belytschko, Liu, Moran & Elkhodary, *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., Wiley, 2014.**

Used throughout the course as a *reference* (not a week-by-week assigned reading) for two topics:

1. **Solid mechanics background.** Continuum kinematics, strain measures, stress measures, balance laws, and constitutive frameworks. Useful when the lecture notes assume continuum-mechanics fluency that a given student may not yet have.
2. **FEA implementation.** The connection between the strong-form PDE, the weak form, element-level integration, and the assembled global system. Useful when we discuss MOOSE kernels (Module 0, Module 5), strain-energy formulations (Module 4), and J²-plasticity return mapping (Module 4).

No chapters are assigned to a specific lecture. Students are expected to consult the relevant section when a concept appears in lecture and they want a deeper treatment than the course notes provide. Alternatives if you already own a different graduate FEA text (Hughes; Reddy; Bathe; Borja for plasticity): use what you have — the topics overlap substantially.

Module 0 itself does not require this reference. The first place it becomes useful is Module 1 (tensors, Voigt notation) and Module 4 (plasticity).

## Module 0 supplementary readings (added 2026-09-01)

Written to support Lecture 2 (Wed Sep 2) and Lecture 3 (Fri Sep 4). A and B are the two worked
examples demonstrated live in L2; C is the FEA refresher promised in class for students who want to
review finite elements or have not taken an FEA course; D is a complete MOOSE + ARCC workflow.

| Reading | Read before | What it covers |
|---|---|---|
| [The weak form of linear elastostatics](weak_form_derivation.md) | L2, Wed Sep 2 | Strong → weak form, test functions, natural vs essential BCs, symmetry of a(u,v). The theory half of the L2 live demo. |
| [An MLP that fits sin(πx)](mlp_sine_walkthrough.md) | L2, Wed Sep 2 | PyTorch training loop end to end, tensor shapes, the silent broadcasting bug measured. The code half of the L2 live demo. |
| [Finite elements in 1-D: a primer](fe_1d_primer.md) | L3, Fri Sep 4 (optional but recommended) | The whole FE arc in the simplest setting: weak form, shape functions, element stiffness, assembly, BCs, solve, post-process, convergence. **Start here if you have not taken an FEA course.** |
| [A 2-D cantilever in MOOSE on ARCC](moose_cantilever_arcc.md) | L3, Fri Sep 4 | A complete run: MOOSE input file block by block, the SLURM submission script directive by directive, contour plots, and validation against beam theory. |

| [Submitting your work with Git and GitHub](git_submission_guide.md) | before your first submission | Step-by-step from accepting the repository invitation to opening the pull request, including authentication setup. Referenced by every lab handout. |

Runnable companions live in `module_0/examples/`: `mlp_sine_reference.py`, `fe1d_reference.py`,
`cantilever_beam.i`, `run_cantilever.sbatch`, `plot_beam_results.py`. Measured results that all four
readings cite: `module_0/readings/measured_results.md`.

The FEA primer also has a **printable PDF** (`module_0/readings/fe_1d_primer.pdf`, 5 pp, linked from its
Canvas page) for students who want to work through it with a pencil. The Canvas page remains the
accessible version; the PDF is generated from the same Markdown by `team/tools/md_to_pdf.py`, so the
two cannot drift.

Naming and layout convention for all readings: `module_0/readings/README.md`.
