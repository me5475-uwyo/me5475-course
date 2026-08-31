# Lecture 1 — Why ML for Computational Solid Mechanics

**Course:** ME-5475-01 — Machine Learning for Computational Solid Mechanics
**Date:** Monday, August 31, 2026 (MWF 1:10–2:00 PM, Engineering Bldg Room 2070)
**Module:** 0 — Setup, Agentic Workflow, ARCC + MOOSE Orientation
**Duration:** 50 minutes
**Format:** Lecture + course logistics + ARCC/GitHub setup during class
**Canvas:** https://uwyo.instructure.com/courses/620641

---

## Learning objectives

By the end of this lecture, students should be able to:

1. Trace the historical arc of ML in computational solid mechanics from Ghaboussi 1991 through modern operator-learning approaches.
2. Identify three concrete success stories of ML in solid mechanics — one in constitutive modeling, one in PDE solving, one in inverse identification.
3. Apply the four-condition feasibility test to a new problem to judge whether ML is the right tool.
4. Name the course toolchain (the course GitHub organization, Jupyter Book, PyTorch, DeepXDE, ARCC, MOOSE, AI coding agents) and explain what each is used for.

---

## Opening (5 min)

Where this course sits in your graduate program: between the classical computational solid mechanics courses you have already taken (continuum mechanics, finite element analysis, perhaps plasticity) and the rapidly maturing literature on machine learning for physics. The course is implementation-first. By the time you finish you will have working code for at least four distinct ML methods (MLP, PINN, CNN, and an operator-learning model) applied to mechanics problems, all running on the ARCC HPC cluster, all driven through GitHub with AI coding agents.

Two things this course is not. It is not a survey course — we go deep on a small number of methods rather than touching every architecture in the literature. And it is not a pure-ML course — every method we cover is motivated by a mechanics problem we care about, and every method is judged by whether it actually helps that problem.

## The history thread (15 min)

The story has two threads — the ML thread and the mechanics thread — and they only braid together around 1990.

**The ML thread, briefly.** McCulloch and Pitts wrote down the first mathematical neuron in 1943. Rosenblatt's perceptron in 1958 was the first trainable linear classifier. Minsky and Papert's *Perceptrons* book in 1969 showed perceptrons cannot solve XOR, and the field fell into the first AI winter. Backpropagation as we know it was popularized by Rumelhart, Hinton and Williams in 1986. Through the 1990s and 2000s most progress in mechanics-applicable ML was incremental until two events: AlexNet in 2012 (deep learning beats classical vision), and the broad availability of automatic differentiation through PyTorch (2017) and TensorFlow (2015).

**The mechanics thread.** Three early papers anchor the field:

- **Ghaboussi, Garrett & Wu, 1991** — *Knowledge-based modeling of material behavior with neural networks*. The first paper to put a neural network in the constitutive modeling role: stress as a learned function of strain history. Predates by 25 years almost everything in the modern literature.
- **Lefik & Schrefler, 2003** — frame indifference and material symmetry constraints on NN constitutive models. An early, explicit treatment of physics constraints in this setting.
- **Kirchdoerfer & Ortiz, 2016** — *Data-driven computational mechanics*. Reformulated the constitutive problem to bypass the model entirely, projecting onto a data manifold. A different philosophy than "train a NN" — and the start of the model-free thread.

**The modern era.** From 2019 onward the literature accelerates:

- **Raissi, Perdikaris & Karniadakis, 2019** — *Physics-Informed Neural Networks*. The PDE residual becomes the training loss. Both forward and inverse problems addressed in the same paper.
- **Mozaffar et al., 2019** — deep learning for path-dependent plasticity at scale.
- **Vlassis & Sun, 2021** — neural network yield surfaces with Sobolev training. The level-set / signed-distance-function reframing of plasticity. Steve Sun is your collaborator and we will return to his framework in Module 4.
- **Lu, Jin & Karniadakis, 2021** — DeepONet, operator learning at scale.
- **Li et al., 2020** — Fourier Neural Operator (FNO).
- **Pfaff et al., 2020** — MeshGraphNets, graph neural networks as fast surrogates for mesh-based simulations.
- **2023–present** — ML-augmented constitutive laws appearing in production FE codes (MOOSE, FEniCS, Abaqus user routines); operator learning competing with classical FE for select problems.

The cultural reference point: AlphaFold (2020, 2024) for protein structure. AlphaFold did not replace experimental structural biology, but it changed the question being asked at the bench. The question for this course is: what is the AlphaFold analog for solid mechanics — and what problems does it change?

## Where ML fits in the computational mechanics pipeline (10 min)

Five attachment points where ML plugs into a classical FE workflow. The course visits all five; each project follows its own specified option.

**(1) Surrogate constitutive models.** Replace ψ(F), σ(ε, q), or the yield function f(σ, q) with a learned function. Trained on either real experimental data or simulated data from a high-fidelity model (DEM, FFT on RVE, atomistic). Module 4 and Module 5 cover this.

**(2) PDE solvers.** Train a network whose output, evaluated at any (x, t), approximately satisfies a PDE. PINN is the original; operator-learning methods (DeepONet, FNO) generalize across families of PDEs. Modules 3 and 7.

**(3) Inverse problems.** Given sparse measurements, identify the material parameters, the boundary loads, or the geometry that produced them. PINN-based inverse, Bayesian inversion, ensemble methods. Module 3 (PINN inverse) and Module 9 (Bayesian + UQ).

**(4) Mesh-based surrogates.** Use a graph neural network to predict the next time-step of a full simulation in microseconds rather than the minutes a classical FE step takes. MeshGraphNets and successors. Module 8.

**(5) Data-driven discovery.** Given enough data, can we learn the form of a constitutive law we did not previously have a model for — micropolar, micromorphic, or beyond? Steve Sun's level-set / Sobolev approach (Module 4) is the most disciplined version of this we will see.

## The four-condition feasibility test (5 min)

Adapted from Steve Sun's L1, slide 21. Before reaching for ML, check that all four hold:

- **A smooth pattern exists.** If the underlying function is discontinuous or chaotic in the input space you care about, ML will struggle — and so will any other method.
- **The pattern is intractable to write down.** If you can write the analytical model, write it. ML is for the cases where the model is missing or the data is richer than any model.
- **Data exists, or can be generated.** Existence of data is non-negotiable. Generation from a higher-fidelity model is fine and often the norm in mechanics — FE on RVE, DEM, FFT, atomistic.
- **The manifold hypothesis applies.** The actual response surface lives on a much lower dimensional structure than the ambient space. Most of mechanics satisfies this — stress lives on a 6-dimensional manifold in 6-D, frame indifference cuts it further.

In-class discussion (3 min). Rate these three problems against the test:

- Predict the yield surface of a polycrystal from its grain microstructure image. (Strong yes.)
- Solve Laplace's equation on a unit square with constant boundary data. (Strong no — classical method is unbeatable.)
- Identify the elastic modulus of a part from a single full-field DIC measurement. (Yes — inverse problem with sparse data.)

## Course logistics and toolchain preview (10 min)

**Schedule.** M0 + ten technical modules (M1–M10), 40 lectures, MWF, August 31 through December 9. Thanksgiving classes are excused Wednesday–Friday only; there is no class September 7 or October 12. Friday, December 11 is the reserve/overflow day.

**Midterm project.** Due Sunday, November 1. Choose one of three options — constitutive surrogate, PINN inverse, or an approved open topic — and validate the result against an appropriate reference. FE-integration work follows in Lab 5; the midterm does not promise MOOSE deployment.

**Toolchain.** Briefly:

- **the course GitHub organization** — all assignments submitted as pull requests on a per-student repo. We use issues, branches, and PR reviews intentionally — it is part of what makes the agentic workflow tractable.
- **Jupyter Book** — the course textbook is a live, versioned site rendered from this repository. Every lecture page is markdown; every example is runnable.
- **Python 3.11+ and PyTorch 2.x** — main ML stack throughout the course.
- **DeepXDE** — used specifically for the PINN labs in Module 3, because Min Lin's existing plate-with-hole notebook is the lab anchor and it was written in DeepXDE/TensorFlow.
- **ARCC** — University of Wyoming's HPC cluster, used from Week 1 onward. The instructor submits the Project Change Request with student usernames; watch for the ARCC onboarding email and tell the instructor if you registered late. Course values are confirmed: login `medicinebow.arcc.uwyo.edu`, account `me5475`, CPU partition `mb`, and GPU partitions `mb-l40s,mb-a30` with `--gres=gpu:1`.
- **MOOSE** — open-source multiphysics finite element framework from Idaho National Lab. We use MOOSE through input files only; we will not write MOOSE C++ source code in this course (but agents will help us prepare input files in later modules).
- **AI coding agents** — Claude Code, Cursor, or equivalent. Encouraged for everything except exams. Mandatory for the final project. Every submission includes a prompt log.

**Grading.**

- 30% weekly labs
- 25% midterm project (constitutive surrogate, PINN inverse, or approved open topic; validated against an appropriate reference)
- 40% final project (open-ended, must use ARCC, must include an agentic-workflow component)
- 5% participation (in-class discussion, peer review)

**Office hours.** MWF 2:00–3:00 PM, right after class, in my office. No TA this semester — bring questions directly to office hours or post in the course Slack.

**Academic integrity with agents.** See syllabus §8. Short version: agent use is encouraged, prompts must be logged, but the final scientific judgments are yours.

## Closing — start the four setup actions today (5 min)

1. **ARCC account:** requested for you through the instructor-submitted Project Change Request. Watch for an email from ARCC, and tell the instructor if you registered late.
2. **the course GitHub organization:** sign in with your UW email and join the course assignment from Canvas.
3. **Local Python/PyTorch:** follow `module_0/setup.md` before Lecture 2.
4. **GitHub Copilot:** verify student status at <https://education.github.com/pack> with your UW email. Verification takes a few days, so start today.

---

## Assigned reading (before Lecture 2)

**Primary.**

- Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang, *Physics-informed machine learning*, Nature Reviews Physics 3, 2021. A survey of where the field is now. **Read fully.**
- Vlassis & Sun, *Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity models with level set hardening*, CMAME 377, 113695, 2021. The flagship paper of the level-set / Sobolev framework you will meet again in Module 4. **Read the introduction and Section 2 (the Sobolev training setup); the algorithmic details can wait until Module 4.** This is your collaborator Prof. Steve Sun's group's work — much of the conceptual scaffolding of this course traces to it.
- Anthropic, *Building effective agents* (blog post on anthropic.com/engineering, December 2024). Conceptual introduction to agents — preview for Lecture 2. **Read fully.**

**Optional / reference.**

- Ghaboussi, Garrett & Wu, *Knowledge-based modeling of material behavior with neural networks*, J. Engrg. Mech. 117(1), 1991. Skim for the historical context — see how it argues for the necessity of NN constitutive modeling on hardware much weaker than your laptop.
- Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comp. Phys. 378, 2019. Sections 1–3 only — full reading is Module 3.
- Vlassis, Ma & Sun, *Geometric deep learning for computational mechanics Part I: anisotropic hyperelasticity*, CMAME 371, 113299, 2020. Companion to the Sobolev paper, introducing the graph-based descriptor work that returns in Module 8.

## In-class deliverable

- ARCC onboarding status checked; tell the instructor if you registered late.
- the course GitHub organization account created (during class or before Lecture 2).
- GitHub Education student-status verification started with a UW email (today).
- Python 3.11+ and PyTorch 2.x installed locally (before next class).

---

## Instructor notes (not for student view)

- 5 minutes is too short for the four-condition test and the in-class discussion together. If running tight, drop one of the three discussion problems.
- The historical thread is the only part where I lean on Steve's L1 slides 10–19 directly. Reuse those figures with attribution.
- Watch the clock for the four setup actions; ARCC onboarding itself is instructor-submitted, not a student application.
