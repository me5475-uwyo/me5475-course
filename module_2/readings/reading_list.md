# Module 2 — Reading List

Annotated bibliography. **[Primary]** items are required; **[Optional]** are recommended but ungraded.

---

## Before Lecture 10

*Lecture 9 (Mon Sep 21) was run as a Lab 1 working session rather than a lecture, so this material
is assigned as reading instead. Start with the primer — it is the deck in written form.*

### [Primary] `module_2/readings/losses_and_optimizers_primer.md`

The L9 material: choosing a loss (MSE / MAE / Huber and the three-question diagnostic), what `g`
and `θ` are, AdamW versus Adam+L2, where gradient clipping goes, learning-rate schedules, and what
an LR-range test can and cannot tell you. **Read fully**, including §4 — the measured seed
dependence is the part most people get wrong. Six self-check questions at the end.

Slides: `M2_L9_loss_optimizers_in_practice.pdf` (Canvas, Module 2). Numbers:
`module_2/readings/measured_results.md`.

### [Primary] Smith, *Cyclical Learning Rates for Training Neural Networks*, WACV 2017.

arXiv:1506.01186. The LR-range-test paper. Read sections 1–3 — the practical recipe (run a few hundred iterations with geometrically increasing LR, plot loss, pick the LR one decade below divergence) is the contribution.

### [Primary] Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.

arXiv:1711.05101. The AdamW paper. Skim Sections 1–2 for the L2-vs-weight-decay distinction; Section 4 for the practical recipe.

### [Optional] Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 8 sections 8.3–8.5.

Review of optimizers from a deep learning practitioner's perspective.

### [Optional] Karpathy, *A Recipe for Training Neural Networks*, 2019 blog post.

karpathy.github.io/2019/04/25/recipe. The most-cited practitioner's guide to training. Read once for the patterns; it changes how you debug.

---

## Before Lecture 11 (assigned in Lecture 10)

### [Primary] `module_2/readings/hyperparameter_tuning_guide.md`

A short Optuna + LR-range guide. Read fully.

### [Primary] Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13, 2012.

The "random search beats grid search in high dimensions" paper. Short, readable, foundational. Read intro + Section 2.

### [Primary] Optuna documentation, *Quickstart* tutorial.

optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html. The 5-minute introduction. You will use this in L11 live.

### [Optional] Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 7 (Regularization).

Sections 7.1 (parameter norm penalties), 7.4 (dataset augmentation), 7.8 (early stopping), 7.12 (dropout). Read 7.8 carefully.

---

## Before Lab 2 (assigned in Lecture 11)

### [Primary] `module_2/readings/slurm_sweep_patterns.md`

The two patterns (array job vs persistent worker) plus the decision rules. Read fully before submitting your first sweep.

### [Primary] Optuna documentation, *Pruning Unpromising Trials*.

optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html. The Median pruner is what Lab 2 uses; this is the official tutorial.

### [Primary] Akiba, Sano, Yanase, Ohta & Koyama, *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.

arXiv:1907.10902. Skim — the algorithm description in Section 3 (TPE sampler) is the contribution.

### [Optional] Falkner, Klein & Hutter, *BOHB: Robust and Efficient Hyperparameter Optimization at Scale*, ICML 2018.

The other widely-used HPO method (Bayesian Optimization + Hyperband). Worth knowing the name; conceptually similar to Optuna's TPE+MedianPruner.

### [Optional] Liaw et al., *Tune: A Research Platform for Distributed Model Selection and Training*, 2018.

The Ray Tune paper. Useful if your final project needs distributed multi-GPU HP search.

---

## Reference (no due date)

### Books

- **Goodfellow, Bengio & Courville (2016).** *Deep Learning*. MIT Press. Free online at deeplearningbook.org. Chapter 7 (regularization), Chapter 8 (optimization) are the M2-relevant chapters.

### Blogs and tutorials with long shelf lives

- Karpathy, *A Recipe for Training Neural Networks* — best practitioner's checklist.
- Distill.pub, *Why Momentum Really Works* (Goh, 2017) — best visualization of optimizer behavior.
- Sebastian Ruder, *An overview of gradient descent optimization algorithms* — encyclopedia of optimizers.

### Reproducibility resources

- The ML Reproducibility Checklist (Joelle Pineau, NeurIPS 2018) — the field-standard checklist for what should accompany any published ML result.
- *PapersWithCode* — pre-checked reproducibility for many ML papers.
