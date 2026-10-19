# Module 4 — Reading List

The most Steve-Sun-heavy module. **[Primary]** required; **[Optional]** recommended.

Every computed number in Lab 4 and in these readings — run times, data statistics, test errors — is recorded, with what was run where, in [Module 4 measured results](https://github.com/me5475-uwyo/me5475-course/blob/main/module_4/readings/measured_results.md) (`module_4/readings/measured_results.md`). References such as §1.3 mean Section 1.3 of that document. The Lab 4 Task 2–3 test errors and their analysis are added there after Lab 4 closes (Fri Nov 6).

---

## After Lecture 19 (Fri Oct 16)

### [Optional · midterm Flavor B] `module_4/readings/constitutive_models_moose_walkthrough.md`

Brief theory and four complete MOOSE input examples: Drucker-Prager pressure sensitivity, kinematic hardening, Voce saturation and Hill anisotropy. Includes the ARCC job procedure and checks to perform before generating data. Course-build checks and measured timestep sensitivity are recorded in [M4 measured results](https://github.com/me5475-uwyo/me5475-course/blob/main/module_4/readings/measured_results.md), Section 7. These examples are starting points for further validation, not a complete midterm dataset.

---

## Before Lecture 20 (assigned in Lecture 19)

### [Primary] Belytschko, Liu, Moran & Elkhodary, *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., 2014. Chapter 5 (Constitutive Models).

Sections 5.1, 5.2, 5.6. The plasticity formalism this module uses.

### [Primary] Mozaffar, Bostanabad, Chen, Ehmann, Cao & Bessa, *Deep learning predicts path-dependent plasticity*, PNAS 116(52), 26414–26420, 2019.

The canonical RNN-for-plasticity paper. Skim — the experimental setup is the most relevant part.

### [Optional] Ghaboussi, Garrett & Wu, *Knowledge-based modeling of material behavior with neural networks*, J. Engrg. Mech. 117(1), 1991.

You read this in M0. Re-skim now that you understand the formalism.

---

## Before Lecture 21 (assigned in Lecture 20)

### [Primary] Steve Sun's L3 slides 9–20 (in `ML4Mechanics/Lecture_3_supervised_ML_physics_constraints.pdf`).

The three-ways-to-enforce-a-constraint taxonomy that anchors L20.

### [Primary] Lefik & Schrefler, *Artificial neural network as an incremental nonlinear constitutive model for a finite element code*, CMAME 192, 2003.

An early neural network used as an incremental constitutive model inside a finite element code.

### [Primary] `module_4/readings/frame_indifference_primer.md`

The background for L20 slides 3–5: rotating the axes versus rotating the body, how each deformation and stress measure behaves under a superposed rotation, and why an energy model fed C satisfies frame indifference for any weights (a statement about the scalar energy, not about a network that outputs stress directly). It recaps *Index notation and coordinate transformation* (M1) in four lines and covers only what is new: strain and stiffness in a rotated basis, and the active rotation. Companion: `module_4/examples/frame_indifference_checks.py`.

### [Optional] Heider, Wang & Sun, *SO(3)-invariance of informed-graph-based deep neural network for anisotropic elastoplastic materials*, CMAME 363, 2020.

Architectural frame invariance for anisotropic plasticity.

### [Optional] Cohen & Welling, *Group Equivariant Convolutional Networks*, ICML 2016.

The mathematical foundation for Way-3 (architectural) constraints.

---

## Before Lecture 22 (assigned in Lecture 21)

### [Primary] Amos, Xu & Kolter, *Input Convex Neural Networks*, ICML 2017.

ICNN paper. Read Sections 1–3.

### [Primary] Klein, Fernandez, Martin, Neff & Weeger, *Polyconvex anisotropic hyperelasticity with neural networks*, JMPS 159, 2022.

Mechanics application of ICNN.

### [Primary] `module_4/readings/sobolev_training_primer.md`

Synthesized H¹/H² derivation and code.

### [Primary] `module_4/readings/icnn_primer.md`

ICNN explained in 2 pages.

### [Optional] Linka & Kuhl, *A new family of constitutive artificial neural networks towards automated model discovery*, CMAME 403, 2023.

Constitutive Artificial Neural Networks (CANN) as alternative to ICNN.

### [Optional] Ball, *Convexity conditions and existence theorems in nonlinear elasticity*, ARMA 63, 1977.

The classical reference on polyconvexity.

---

## Before Lecture 23 (assigned in Lecture 22)

### [Primary] Steve Sun's L4 slides 14–33 (in `ML4Mechanics/Lecture_4_recurrent_neural_network.pdf`).

The RNN-for-plasticity material this lecture condenses.

### [Primary] Mozaffar et al. PNAS 2019 (re-read).

Now read more carefully.

### [Optional] Wang & Sun, *A multiscale multi-permeability poroplasticity model linked by recursive homogenizations and deep learning*, CMAME 334, 2018.

Steve's plasticity-with-RNN application.

### [Optional] Olah, *Understanding LSTM Networks*, colah.github.io/posts/2015-08-Understanding-LSTMs.

The best LSTM explainer; Steve also cites it.

---

## Before Lecture 24 (assigned in Lecture 23)

### [Primary] Vlassis & Sun, *Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity models with level set hardening*, CMAME 377, 113695, 2021. **§2.3 (training the evolving yield function as a level set) and §3 (the return-mapping implementation).**

The level-set framework's central paper. §2.1–2.2 cover Sobolev training of a smooth functional; now read §2.3 and §3. (Section numbers as in the arXiv preprint, 2010.11265.)

### [Primary] Steve Sun's L5 slides 29–48.

The theoretical core of the level-set / Eikonal framework.

### [Optional] Osher & Fedkiw, *Level Set Methods and Dynamic Implicit Surfaces*, Springer 2003.

The classical reference for level-set methods.

### [Optional] Vlassis & Sun, *Component-based machine learning paradigm for discovering rate-dependent and pressure-sensitive level-set plasticity models*, J. Appl. Mech. 89(2), 021003, 2022 (online Nov 2021).

Application of the level-set framework to viscoplasticity.

---

## Reference (no due date)

### Books / surveys

- Simo & Hughes, *Computational Inelasticity*, Springer 1998. The classical reference on rate-independent plasticity, return mapping, and algorithmic tangents.
- Borja, *Plasticity: Modeling and Computation*, Springer 2013. Modern friendly alternative to Simo-Hughes.
- Cuomo et al., *Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next*, J. Scientific Computing 92(3), 2022.

### Papers anchoring the course's M4-relevant lineage

- **Ghaboussi 1991** — first NN constitutive law.
- **Lefik-Schrefler 2003** — frame indifference enforcement.
- **Mozaffar 2019** — RNN-for-plasticity.
- **Vlassis-Sun 2021** — level-set + Sobolev training, the Steve framework.
- **Klein-Fernandez-Martin-Neff-Weeger 2022** — polyconvex ICNN for hyperelasticity.
- **Linka-Kuhl 2023** — CANN architecture.

For the midterm and final, these papers are the methodological references most likely to be cited.
