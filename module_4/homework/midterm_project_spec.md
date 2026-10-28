# Midterm Project — Specification

**Released:** Fri Oct 16, 2026
**Due:** Sun Nov 8, 2026, 11:59 PM MT (extended from Sun Nov 1 on Oct 26)
**Weight:** 25% of total course grade
**Format:** GitHub repo + 4–6 page IMRaD report (no presentation)
**Submission:** Pull request from branch `midterm` in your private course repository (`me5475-uwyo/me5475-<your-github-username>`), with your work in `labs/midterm/<your-github-handle>/`.

---

## Project framing

Pick one of three flavors and execute it end-to-end on ARCC. The project's goal is to demonstrate that you can take an ML-for-mechanics problem from *idea* to *trained, documented, validated artifact* — exactly what a publishable paper or a research milestone requires.

You are graded not on novelty but on *execution quality*: clean code, defensible reference solution, sound HP tuning, honest discussion of limitations.

---

## Three flavors

### Flavor A — PINN Inverse Problem with Spatially-Varying Material Parameters

Extend Lab 3 Task 5 from a constant E to a *spatially varying* E(x, y) on a non-trivial 2-D geometry. Recover the field from sparse synthetic DIC measurements via PINN. (This is the syllabus's option 2, the PINN inverse problem. If you did not attempt Lab 3's stretch Task 5, start from `module_3/examples/plate_with_hole_inverse.py` and L18.)

**Required components.**

- Geometry of your choice that isn't the plate-with-hole. Examples: L-shape, notched bar, complex multi-material plate. Must be 2-D plane strain or plane stress.
- A "true" E(x,y) field of your choice. Examples: smooth Gaussian-bump-shaped E variation, two-domain piecewise-constant E with an interface, sinusoidal stiffness pattern. Keep ν constant: either known, or recovered as one unknown number, as in Lab 3 Task 5. A spatially varying ν(x,y) is an optional extension, not a requirement.
- **Remove the global E-scale ambiguity.** L18 showed that when a linear-elastic plate is loaded only by prescribed displacements (the other edges traction-free, no body force), the displacement field does not depend on the overall scale of E. That stays true when E varies in space: multiply E(x,y) by any positive constant and the displacements do not change; only the stresses and reactions scale. So either load at least one boundary by a **known, nonzero** traction, or add a measured **nonzero** reaction force on a **specified loaded boundary** to the loss, as Lab 3 Task 5 does. (A force summed over the whole boundary is zero by equilibrium and carries no scale information.) Say in your Methods which one you used and why it fixes the scale.
- **That fixes the scale, not everything.** Sparse, noisy data can still leave parts of the E field poorly determined. In your Methods, describe how you parameterize E(x,y) and any regularization; in your Discussion, assess where the recovery is and is not reliable. A smooth, positive E is the simplest start. A piecewise-constant E needs proper treatment at the interface, not the constant-E strong-form residual reused blindly.
- Synthetic DIC measurements: 50–100 random interior points with 1% Gaussian noise. State what the 1% is relative to (Lab 3 Task 5 used 1% of the largest displacement component, in magnitude, among the measured points).
- PINN with E(x,y) parameterized by a small NN (or an extra network output) that is *jointly* trained with the displacement-field NN.
- Reference solution from MOOSE (with the true E(x,y)) for comparison. In MOOSE a spatially varying E is a material property: `GenericFunctionMaterial` can supply E(x, y) from a function to `ComputeVariableIsotropicElasticityTensor`. Both objects are in the course MOOSE build; the course has not run that combination, so verify your reference as in Lab 3 Task 2.
- Recovery quality: pointwise relative error in the recovered E field; pass/fail on whether the recovery captures the main features (location of stiffness variations, magnitude of contrast).
- HP sweep: minimum 15 Optuna trials. For planning: one Lab 3 inverse run took about 9 minutes on an A30 GPU (`module_3/readings/measured_results.md` §4); your geometry and network will differ.

**Skills you reuse from earlier modules.** Lab 3 entire (PINN setup), Lab 2 (Optuna), Lab 0 (MOOSE generation of ground truth).

---

### Flavor B — Data-Driven Constitutive Surrogate

Take a more complex constitutive model than the linear-isotropic-elastic and J2-linear-hardening of Labs 1 and 4. Examples: Drucker-Prager (pressure-dependent), J2 with kinematic hardening (Bauschinger effect), Hill anisotropic yield, isotropic hardening with nonlinear (Voce) hardening law. Train a physics-constrained NN constitutive model that captures it from MOOSE simulation data. (This is the syllabus's option 1, the constitutive surrogate.)

The course MOOSE build includes the objects these need, among them `CappedDruckerPragerStressUpdate` (Drucker-Prager), `CombinedNonlinearHardeningPlasticity` (kinematic hardening), `HillElastoPlasticityStressUpdate` (Hill), and `IsotropicPlasticityStressUpdate` with a `hardening_function` (Voce). The course has run only the Lab 4 J2 input, so start from `module_4/examples/j2_cyclic_loading.i` and check your new model on a single run before generating data.

**Required components.**

- A specific constitutive model selected and named.
- A data generation strategy producing >= 100 trajectories of cyclic loading covering the relevant input space (strain magnitudes, principal directions, loading paths). Lab 4's loading is proportional: one strain direction per trajectory, with only its sign reversing. Non-proportional paths need strain components whose time histories are not proportional to one another; separate but proportional time functions would not change the direction.
- Split train / validation / test by trajectory before any tuning, and state whether internal-state inputs (such as ε^p) are supplied from MOOSE or predicted by your model.
- An ML architecture choice with rationale: stateless MLP-with-q, LSTM, level-set yield surface, or transformer over time. **At least one physics constraint** built in (frame indifference, material symmetry, thermodynamic consistency, etc.) — Way 1, 2, or 3 from L20.
- A trained model evaluated on a held-out set with a quantitative metric (MSE on stress, hysteresis loop matching, or yield surface fitting).
- HP sweep with at least 15 Optuna trials.
- For full credit: comparison to a *stateless baseline* showing why the architecture choice matters.

**Skills you reuse.** Lab 4 entire, Lab 2 (Optuna), Lab 1 (data-loading patterns).

---

### Flavor C — Open Option (proposal by L20, Mon Oct 19; instructor approval by L21, Wed Oct 21)

Propose your own ML-for-mechanics problem. Must:

- Be in the broad scope of "ML for computational solid mechanics" (NOT pure ML or pure mechanics).
- Use ARCC for at least one substantial training run.
- Validate against an appropriate reference that the instructor approves with your proposal (for example, an FE solution or a published benchmark).
- Include at least one HP sweep with >= 15 Optuna trials.
- Produce a deliverable that could conceivably be a chapter of a thesis (a clean repo + report + figures, not a one-off script).

**Examples of suitable Open Option projects:**

- Operator learning (DeepONet or FNO) applied to a custom mechanics PDE.
- A graph neural network surrogate for a specific FE problem.
- A diffusion model for microstructure generation.
- A differentiable simulator coupled to a PINN.
- A specific industrial application (composite delamination, granular flow, additive manufacturing residual stresses) selected with the instructor.

**Pick a flavor by L20 (Mon Oct 19).** Email the instructor your choice (and for Flavor C, a 1-paragraph proposal).

---

## Required deliverables (all flavors)

```
labs/midterm/<your-github-handle>/
├── README.md                                     project overview, how to run
├── report.pdf                                    IMRaD report, 4-6 pages, includes figures
├── code/                                          all .py and .i files needed to reproduce
│   ├── data_generation/                          MOOSE inputs, drivers
│   ├── training/                                  training scripts, Optuna sweep
│   ├── analysis/                                  comparison/validation scripts
│   └── checkpoints/                                trained model state
├── data/                                          dataset (or instructions to regenerate)
├── figures/                                       all figures used in the report
├── slurm_logs/                                    one or two representative SLURM outputs
└── prompts.md                                     agent prompt log + reflection paragraph
```

The repo is the artifact; the report explains it.

**`prompts.md`** follows the syllabus's log rules: which tool and model you used, approximate dates, your lead and review prompts, and which outputs you actually used. If you used no AI, say so in one line; that is a complete log. **It also holds the required reflection paragraph:** what your review pass caught in the lead agent's output, and what you caught yourself. (If you used no AI, the reflection is about what your own checking caught.) The reflection does not count toward the report's page limit.

**Keep the repository small.** GitHub rejects any file over 100 MB. Commit checkpoints and data only if they are small (a few MB); otherwise put regeneration instructions in `data/` and keep the files on ARCC.

**Compute.** Every training run and sweep is a submitted job on ARCC, never on the login node, and never more than two of your GPU jobs at once — the same rule as Lab 3.

---

## Report format (4–6 pages, IMRaD)

**Introduction (~0.5 page).** What's the problem? Why is it interesting/important? What's already known? What's your specific contribution (which flavor, which choices)?

**Methods (~1.5 pages).** The PDE/constitutive model, the data generation, the network architecture (with diagram), the loss function, the training procedure, the HP search space.

**Results (~1.5 pages).** Quantitative metrics, plots of trained behavior vs reference, ablation studies (with vs without the physics constraint, e.g.).

**Discussion (~0.5 page).** Limitations, what would you do with another month, what does this say about the choice of method?

**References.** At least 5 citations from the course readings + the broader literature.

**Citation style.** APA or IEEE; consistent throughout.

---

## Grading rubric (100 points)

| Component | Points |
|-----------|--------|
| Code: runs, reproducible, documented | 20 |
| Data: appropriate quantity and coverage; defensible reference | 15 |
| Method: ML architecture choice well-justified | 15 |
| Physics constraint(s): at least one in place, explained, verified | 15 |
| HP sweep: >= 15 trials, results analyzed | 10 |
| Report: clear writing, all sections present, ≤ 6 pages | 25 |

For Flavor A, the physics constraint is the PDE residual together with your boundary-condition treatment and the identifiability fix; for Flavor C, it is the physics content agreed in your proposal.

---

## Important dates

- **Mon Oct 19 (L20):** flavor choice locked.
- **Wed Oct 21 (L21):** if Flavor C, proposal approved.
- **Sun Nov 8, 11:59 PM MT:** project due (extended from Sun Nov 1).

Reach out to the instructor throughout the window (about three weeks, Oct 16 – Nov 8). The Office Hours times during Wks 8–10 are dedicated to midterm consultation.

---

## Some honest advice

- **Choose the flavor that excites you most.** About three weeks is long enough to do a good job, short enough that low motivation will kill you.
- **Start early. The Optuna sweep + retraining is the slow step**, not the writing. Get the sweep running by the end of Week 8 (Fri Oct 23) if possible.
- **Use agents extensively. Log the prompts.** The lead+review pattern saves you from your own bugs.
- **Compare to a baseline, honestly.** A single error number means little on its own. Report the metric, the evaluation set, how accurate your reference is and how you verified it, the method's limitations, and measured costs where they matter. Then explain what the method actually bought you, or why it fell short: a well-documented negative result is a good result here. The comparison is the contribution.
- **Talk to the instructor when stuck.** Don't burn 3 days debugging in isolation.

---

## Connecting your midterm and final projects

You may choose a challenging mechanics problem and address a manageable version of it for the midterm using a method relevant to what we have learned so far. Your midterm must stand on its own: meet the requirements of your chosen flavor and present a working, validated result. You do not need to solve the most complex version of the problem at this stage.

For the final project, you may continue with the same topic by investigating a different method and a more complex situation, such as a richer material model, a different loading history, or a more demanding geometry. Compare the new method with your midterm approach, and clearly distinguish reused work from substantial new contributions. A better result is not guaranteed; a careful comparison and an honest explanation of what improves or fails are valuable.

Alternatively, you may choose different topics for the midterm and final projects. Continuing the same topic is optional. Discuss your final-project scope with the instructor during the proposal process; the final project retains its own requirements and deliverables.
