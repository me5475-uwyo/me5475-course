# Lab 1 — Single-Element Constitutive Regression

**Module:** 1 — Math + Mechanics Foundations
**Released:** Fri Sep 4, 2026 (the day Lecture 3 is taught — hold this lab while Module 1 is taught)
**Due:** Wed Sep 23, 2026, 11:59 PM Mountain Time (the day Lecture 10 is taught)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_1` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

---

## What this lab does

Lab 1 builds the bridge from M0 (MOOSE) to M4 (data-driven constitutive modeling). You will use MOOSE to generate stress-strain pairs on a single unit element under random load directions, then train a PyTorch MLP to learn σ(ε) and verify that it recovers the analytical Hooke's-law stiffness matrix.

This is the simplest possible constitutive-modeling-via-ML problem. The analytical answer is known exactly. Three things will happen during this lab that will recur all semester:

1. You'll generate data from a high-fidelity FE simulation. The same pattern recurs in M4 (real constitutive surrogates), M6 (CNN field surrogates), and M7 (operator learning).
2. You'll train a small MLP with Adam + reverse-mode autograd on ARCC. The five-line training loop recurs in every subsequent module.
3. You'll verify the learned model against the analytical answer. This sanity check is what separates a working ML pipeline from a plausible-looking one.

---

## Deliverables

```
labs/lab_1/<your-github-handle>/
├── preliminaries/                          (from Lecture 8 in-class lab)
│   ├── sine_mlp.py                         your version of the fit y = sin(pi x)
│   ├── sine_mlp.png                        local-run output figure
│   ├── sine_mlp_on_arcc-<jobid>.out        ARCC SLURM output of the same training
│   └── prompts_l8.md                       any prompts you used in L8
├── data_generation/
│   ├── single_element_loadsweep.i          your (possibly modified) copy
│   ├── generate_data.py                    your (possibly modified) driver
│   ├── data/single_element.csv             ~200 stress-strain rows
│   └── prompts_data_gen.md                 prompts for data generation
├── training/
│   ├── mlp_constitutive.py                 your training script
│   ├── train_mlp.sbatch                    your SLURM script (customized for ARCC)
│   ├── checkpoints/
│   │   ├── mlp_constitutive.pt             trained model + scalers
│   │   ├── mlp_constitutive.png            loss curves
│   ├── mlp-<jobid>.out                     ARCC SLURM stdout
│   └── prompts_training.md                 prompts for training
├── analysis/
│   ├── stiffness_comparison.txt            the implied C vs analytical (copy of stdout)
│   ├── (optional) extra plots / experiments
│   └── prompts_analysis.md
└── reflection.md                           <= 500 words
```

---

## Tasks

> **If your ARCC account is not active yet:** tell me. The ARCC portions of Tasks 1
> and 3 require cluster access, and no deadline or grade penalty attaches to work you
> could not do for lack of a working account (syllabus §4). We will re-time those
> portions for you individually.

### Task 1 — In-class preliminaries (from L8)

You completed this in Lecture 8: train an MLP locally on y = sin(πx), then submit the same training as a SLURM job on ARCC. Commit to `preliminaries/`.

### Task 2 — Data generation

Generate ~200 stress-strain pairs on a single 2-D unit element using `single_element_loadsweep.i` (provided) and the driver `generate_data.py` (provided). Either run locally on your laptop (5 min total) or on an ARCC login node (do not submit hundreds of tiny MOOSE jobs to the cluster — each is < 1 second, batch overhead dominates).

The strain components are sampled uniformly in [-0.005, +0.005]. This is well within the small-strain regime where linear elasticity is exactly valid, so the labels are noise-free.

Use lead+review agents on at least one of:

- Verify the MOOSE input file enforces a *uniform* strain field on the single element. (Hint: the four-edge displacement BCs should be consistent with a constant strain.)
- Verify the Voigt convention in the driver: the column you read as `gamma_xy` should be `2 * strain_xy_avg` from MOOSE's tensor output. Getting this wrong by a factor of 2 will silently make your learned stiffness's (3,3) entry off by 2.

Commit `data/single_element.csv` (a CSV with 6 columns and ~200 rows).

### Task 3 — Train the MLP on ARCC

Modify `mlp_constitutive.py` if you want to change the architecture; the default is a 4-layer MLP of width 32 with tanh activations, which has ~2300 parameters for a 2-parameter problem. (Why does that work? Reflection question 2.)

Modify `train_mlp.sbatch` for your ARCC account, partition, and PyTorch availability. Submit via `sbatch`. The job should complete in well under 30 minutes (typically 2–5 minutes on a single CPU).

Commit the trained checkpoint `checkpoints/mlp_constitutive.pt` and the loss-curve plot.

### Task 4 — Verification: does the trained MLP recover Hooke's law?

The training script computes the *implied stiffness matrix* by taking the Jacobian of the trained MLP at zero strain (autograd) and un-standardizing. Compare to the analytical 3×3 plane-strain stiffness for E = 1.0, ν = 0.3:

```
       E            [[ 1 - nu,    nu,           0      ]
C = ----------------  [   nu,    1 - nu,         0      ]    ≈  [[1.346  0.577  0     ]
    (1+nu)(1-2nu)    [   0,        0,      (1-2nu)/2  ]]         [ 0.577  1.346  0     ]
                                                                  [ 0      0      0.385]]
```

Save the comparison to `analysis/stiffness_comparison.txt`. You can copy stdout from the training script — it prints both matrices.

Verify the off-diagonal entries (3,1), (3,2), (1,3), (2,3) are small (the trained MLP should have learned that shear strain is decoupled from normal stress).

If the (3,3) entry is off by a factor of 2, you have a Voigt-factor-of-2 bug somewhere — debug it.

### Task 5 — Reflection (≤ 500 words)

Write `reflection.md` answering all four:

1. **Numerical accuracy.** What was the maximum element-wise relative error of your learned stiffness vs. the analytical answer? Is that error consistent with the test-set MSE you observed?
2. **Overparameterization.** The MLP has ~2300 parameters; the analytical Hooke's law for linear isotropic plane-strain elasticity has 2 parameters (E and ν). Why does the trained MLP not overfit, given that it has 1000× more parameters than the underlying physics?
3. **Voigt factor of 2.** Did you encounter the factor-of-2 issue at any point in this lab? Describe where. If you did not — verify explicitly that your code handles it correctly (e.g., paste the line where you handle gamma_xy vs eps_xy and explain it).
4. **Forward look.** This trained network is, in some sense, an "ML constitutive model." What changes if (a) the material is not linear elastic (becomes path-dependent — preview of M4) or (b) the material parameters E, ν vary across samples in the dataset (preview of Min Lin's parametric notebook in M3)?

---

## Grading rubric (out of 100)

| Component | Points |
|-----------|--------|
| Preliminaries from L8 (Task 1) | 10 |
| Data generation: CSV exists with ~200 rows, columns correct (Task 2) | 15 |
| Training runs to completion on ARCC (Task 3) | 20 |
| Verification: stiffness comparison printed and reasonable (Task 4) | 20 |
| Verification: off-diagonals are zero or near-zero (Task 4 sanity) | 10 |
| Reflection: all four questions answered concretely (Task 5) | 20 |
| Prompts logged in all sub-folders | 5 |

**Penalty conditions.**

- Training did not actually run on ARCC: -10.
- Voigt factor-of-2 bug present and unfixed: -10.
- Reflection generic or boilerplate: -10.

---

## Hints

- **MOOSE silently runs forever** if you give it inconsistent BCs that imply zero solution. Verify the strain components you computed back from MOOSE match what you imposed (within 1e-6 relative).
- **Standardization is crucial.** With raw strain at magnitude ~1e-3 and raw stress at magnitude ~1e-3, the MSE loss starts at ~1e-6 and you'll mistake it for "well converged." Standardization to zero mean / unit variance puts everyone on a scale where you can see what's happening.
- **The Jacobian at zero strain might not be exactly the underlying stiffness** if the standardization is asymmetric or the MLP's behavior at the origin is biased. Spot check by also computing the Jacobian at a small nonzero strain and confirming it agrees.
- **Reproducibility.** Set `torch.manual_seed(42)` and `np.random.seed(42)` somewhere prominent.

---

## What this lab leads into

- **Module 3, Week 6 (Lab 3).** You'll see the *same* plate-with-hole geometry (from M0) solved by a PINN that uses *no* training data — only the PDE residual. You will compare the PINN's stress field at the hole edge to (a) your M0 MOOSE convergence study, (b) Min Lin's ABAQUS reference, and (c) a baseline where you use your Lab 1 MLP as the constitutive model inside a finite-element simulation (sort of — Module 5 makes this rigorous).
- **Module 4, Week 7 (Lab 4).** You'll repeat this lab but with a *path-dependent* material (J2 plasticity), and discover why the stateless MLP architecture you used here is no longer sufficient.
- **Module 5, Week 9.** You'll take your trained MLP from this lab, export it via TorchScript, and plug it into MOOSE as a custom Material — running the M0 plate-with-hole with your learned linear-elastic model and verifying the result matches the analytical-stiffness MOOSE run from M0.

The pieces compound. Keep your code clean.

---

## Academic integrity reminder

Agents are encouraged. Logged prompts in each subdirectory are mandatory. The numerical results you submit must be your own — copying a classmate's CSV or checkpoint is plagiarism. Discussing strategy with classmates is fine and encouraged.
