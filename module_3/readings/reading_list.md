# Module 3 — Reading List

*Module 3. The index for Lectures 12–18 (Mon Sep 28 – Wed Oct 14) and Lab 3 (released Mon Sep 28, due Mon Oct 19). Course readings: `deepxde_quickstart.md`, `pinn_failure_modes.md` and `measured_results.md`, in this folder; runnable companions are in `module_3/examples/`.*

The PINN module is the longest. Reading load reflects that. **[Primary]** required; **[Optional]** recommended.

**One symbol to watch.** Several of these papers use λ for unknown PDE coefficients. In this course a subscripted λ (λ_BC, λ_data) is a **loss weight** — not L5's Hessian eigenvalue, not L10's weight-decay coefficient — and Lab 3's unknowns are E and ν.

---

## Foundational papers (L12, Mon Sep 28)

### [Primary] Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. 378, 686–707, 2019.

The PINN paper. Read Sections 1–4 in full. The forward/inverse symmetry is the core insight: Section 3 solves PDEs from the equations, Section 4 ("data-driven discovery") identifies unknown coefficients from data — the kind of problem Lab 3 Task 5 solves for E and ν. The paper calls those coefficients λ₁, λ₂; see the note above.

### [Primary] Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang, *Physics-informed machine learning*, Nature Reviews Physics 3, 422–440, 2021.

You read this in M0. Re-skim it now that it has meaning. The review has no numbered sections: look for its three ways of putting physics into learning — observational, inductive and learning biases. The PINN loss is a learning bias; L15's hard-BC ansatz is an inductive one.

### [Optional] Lagaris, Likas & Fotiadis, *Artificial neural networks for solving ordinary and partial differential equations*, IEEE Trans. Neural Netw. 9(5), 987–1000, 1998.

An early forerunner of PINNs, before the name. Its trial solutions build the boundary conditions into the network output — the same construction as L15's hard-BC ansatz.

---

## DeepXDE library (L13, Wed Sep 30)

### [Primary] `module_3/readings/deepxde_quickstart.md`

Two-page intro you'll keep open while writing PINN scripts. Skim it after L12, before the L13 demo.

### [Primary] Lu, Meng, Mao & Karniadakis, *DeepXDE: A deep learning library for solving differential equations*, SIAM Review 63(1), 208–228, 2021.

The DeepXDE framework paper. The examples section is the most useful for practitioners.

### [Optional] DeepXDE documentation, for the course's version: deepxde.readthedocs.io/en/v1.15.0/

Skim 3–4 of the demos for variety. The linear-elasticity plate demo (`demos/pinn_forward/elasticity.plate`) uses the same mixed (u, σ) formulation as Lab 3.

### [Optional] `module_3/readings/moose_diffusion_walkthrough.md`

The same two diffusion problems, solved with finite elements in MOOSE. It covers:
- the weak form and its two kernels;
- the input files `examples/diffusion_1d.i` and `diffusion_2d.i`;
- running them as a CPU job (`run_moose_diffusion.sbatch`), and what the log should show.

Read it after L13. It shows how to run MOOSE; it is not a comparison with the PINN.

---

## Plate-with-hole and solid mechanics PINNs (L14, Fri Oct 2)

### [Primary] Min Lin's notebook, `module_3/examples/min_lin_2D_hole_example.ipynb` (TensorFlow backend, outputs cleared).

The lab anchor. Read every cell and find its ten boundary conditions; then read the course's PyTorch port, `module_3/examples/plate_with_hole_fixed.py`, and check it imposes the same ten (Lab 3 Tasks 1a–1b). You read the notebook; you do not need to run it.

### [Primary] `module_3/readings/plane_stress_and_plane_strain_primer.md`

Read it before Task 1a. Min's notebook uses plane-stress constants; the port, the MOOSE reference and Lab 3 are plane strain. The primer derives both constitutive laws, shows how to tell them apart in code, and measures the difference on Lab 3's plate: for this plate's boundary conditions, each in-plane stress component is 9 % lower in plane stress, and the sideways displacement u₂ differs by 29 % (relative L2) (`measured_results.md` §8).

### [Primary] Haghighat, Raissi, Moure, Gomez & Juanes, *A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics*, Comput. Methods Appl. Mech. Eng. 379, 113741, 2021.

*Skim.* The paper Min's notebook cites. It treats displacements and stresses as separate network outputs — the mixed (u, σ) formulation Lab 3 uses — and covers the inverse-problem framing you will meet again in L18.

### [Optional] Henkes, Wessels & Mahnken, *Physics informed neural networks for continuum micromechanics*, Comput. Methods Appl. Mech. Eng. 393, 114790, 2022.

Extension to micromechanics; useful breadth.

### [Optional] `module_3/readings/moose_plate_walkthrough.md`

How Lab 3's MOOSE reference is built and run:
- why its input file writes only three of the ten conditions (the rest are natural);
- the input file, block by block;
- a coarse run as a job (`run_plate_reference.sbatch`);
- extracting the fields with `extract_moose_reference.py`.

Read it before Task 2. You do not need to run MOOSE for the lab: the reference is shared.

---

## Hard-BC ansatz (L15, Mon Oct 5)

### [Primary] Lu, Pestourie, Yao, Wang, Verdugo & Johnson, *Physics-informed neural networks with hard constraints for inverse design*, SIAM J. Sci. Comput. 43(6), B1105–B1132, 2021.

The hard-constraint paper. Read the introduction and the construction of the hard constraints. Keep in mind what Lab 3 Task 3 measures: an output transform guarantees the conditions it encodes; it does not guarantee faster training or a more accurate interior (`measured_results.md` §6).

### [Primary] McClenny & Braga-Neto, *Self-adaptive physics-informed neural networks*, J. Comput. Phys. 474, 111722, 2023.

The soft-BC alternative: instead of building the conditions into the output, learn a weight for each collocation point, trained by gradient ascent while the network descends. It changes how the penalty trade-off is struck; like any finite weight, it does not *guarantee* that a boundary condition holds exactly. Read the introduction and the method.

### [Optional] Berg & Nyström, *A unified deep artificial neural network approach to partial differential equations in complex geometries*, Neurocomputing 317, 28–41, 2018.

Builds boundary-condition-satisfying outputs from distance functions, for complex geometries — the general form of the construction L15 applies to the plate's straight edges.

---

## Parametric PINN and the operator-learning bridge (L16, Wed Oct 7)

### [Primary] Min Lin's parametric notebook, `2D-Hole-Example-Various-E-Nu-1.ipynb`, walked through in L16.

The notebook itself is not in your repository; the course's PyTorch port, `module_3/examples/plate_with_hole_parametric.py`, arrives with L16 (Wed Oct 7). Re-read the port after L16: adding E and ν as network inputs — (x₁, x₂, E, ν) instead of (x₁, x₂) — is the conceptual key.

### [Primary] Lu, Jin, Pang, Zhang & Karniadakis, *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, Nature Machine Intelligence 3, 218–229, 2021.

Skim the problem set-up and the DeepONet architecture — the operator-learning preview we go deep on in M7.

---

## Training pathologies and tricks (L17, Fri Oct 9)

### [Primary] `module_3/readings/pinn_failure_modes.md`

The two failures this course measured on the plate, then all four tricks with code patterns.

### [Primary] van der Meer, Oosterlee & Borovykh, *Optimally weighted loss functions for solving PDEs with neural networks*, J. Comput. Appl. Math. 405, 2022.

How to choose the weight between the PDE and boundary terms, with an optimal choice for linear, well-posed problems. Background for Trick 2 (loss weights).

### [Primary] Chen, Badrinarayanan, Lee & Rabinovich, *GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks*, ICML 2018.

Trick 2 (adaptive loss weights).

### [Primary] Yu, Kumar, Gupta, Levine, Hausman & Finn, *Gradient surgery for multi-task learning*, NeurIPS 2020.

Trick 3 (PCGrad).

### [Optional] Wang, Teng & Perdikaris, *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*, SIAM J. Sci. Comput. 43(5), A3055–A3081, 2021.

Why the PDE and boundary terms of a PINN fight each other, and adaptive loss weights to balance them — Trick 2, specialised to PINNs.

### [Optional] Wang, Yu & Perdikaris, *When and why PINNs fail to train: A neural tangent kernel perspective*, J. Comput. Phys. 449, 110768, 2022.

NTK analysis — more rigorous but harder. Read if you want the theoretical perspective.

### [Optional] Wang, Sankaran & Perdikaris, *Respecting causality for training physics-informed neural networks*, Comput. Methods Appl. Mech. Eng. 421, 116813, 2024.

Causal training for time-dependent PINNs (the 2022 arXiv version is titled *Respecting causality is all you need for training physics-informed neural networks*). Not used in Lab 3 (steady problem) but useful for the final project.

### [Optional] Chen, Goodfellow & Shlens, *Net2Net: Accelerating learning via knowledge transfer*, ICLR 2016.

Trick 4 (curriculum learning via network growth).

### [Optional] Steve Sun, *Geometric Learning for Solid Mechanics*, Lecture 7 (2023), shown in L17.

The original framing of the four tricks, and the source of Trick 1 (FEM-guided warm-up). Worth it for the historical and cultural context of the field.

---

## Inverse problems (L18, Wed Oct 14)

### [Primary] Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 378, 2019 — Section 4 again.

You read Sections 1–4 for L12; now read Section 4 again with the inverse-problem framing in mind. Then read the docstring of `module_3/examples/plate_with_hole_inverse.py`: under a prescribed displacement the displacement field does not depend on E, so the plate needs one force measurement besides the displacements.

### [Optional] Yang, Meng & Karniadakis, *B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data*, J. Comput. Phys. 425, 109913, 2021.

Bayesian extension. Useful if you want proper uncertainty quantification rather than an ensemble spread (Lab 3 Task 5e) for the final project.

### [Optional] Tartakovsky, Marrero, Perdikaris, Tartakovsky & Barajas-Solano, *Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems*, Water Resources Research 56(5), e2019WR026731, 2020.

A different application (subsurface flow) with the same idea: the unknowns — there, a whole constitutive function — are trained jointly with the solution network.

---

## Reference (no due date)

### Surveys

- Cuomo, Schiano di Cola, Giampaolo, Rozza, Raissi & Piccialli, *Scientific machine learning through physics-informed neural networks: Where we are and what's next*, J. Sci. Comput. 92(3), 88, 2022. A broad survey of PINN variants and applications.

### PINN libraries beyond DeepXDE (for the final project)

- **SciANN** — Keras-based PINN library from the Haghighat–Juanes group. Lower-level than DeepXDE.
- **NVIDIA PhysicsNeMo** (formerly Modulus; its symbolic PINN part was Modulus-Sym) — NVIDIA's GPU-oriented physics-ML framework. developer.nvidia.com/physicsnemo
- **NeuralPDE.jl** — Julia ecosystem. Worth knowing it exists.

### A note on DeepXDE backend selection

DeepXDE 1.15 supports the backends `tensorflow.compat.v1`, `tensorflow`, `pytorch`, `jax` and `paddle`. We use PyTorch throughout the course for consistency with M1/M2; the course environment does not include TensorFlow. Min's original notebooks use TF. Switching is one environment-variable line plus any backend-specific calls (`tf.concat` → `torch.cat`); see `deepxde_quickstart.md`.
