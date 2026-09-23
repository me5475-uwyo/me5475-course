# Lecture 10 — Regularization, Generalization, and Hyperparameter Tuning

**Date:** Wednesday, September 23, 2026
**Module:** 2 — DL Practice + ARCC at Scale
**Duration:** 50 minutes
**Format:** 10-minute recap of the L9 reading, then a 40-minute lecture with the loss-curve gallery

> **Delivered structure (instructor decision, 2026-09-23).** L9 (Mon Sep 21) ran as a Lab 1 working
> session, so its slides were posted as reading. L10 therefore opens with a **10-minute recap of the
> L9 reading** (the loss as a model of the noise; θ and g, AdamW versus Adam+L2, clipping; the
> learning rate and the range test). The time comes out of Section 5 — **TPE and pruning are deferred
> to L11** — and Section 6, which becomes a 2-minute checklist. Delivered clock:
> recap 0–10 · S1 10–15 · S2 15–22 · S3 22–28 · S4 28–40 · S5 40–43 · S6 43–45 · closing 45–50.

---

## Learning objectives

By the end of this lecture, students should be able to:

1. State the bias-variance decomposition and identify which regime a given training run is in from a single loss curve.
2. Apply three regularization techniques (L2/weight-decay, dropout, early stopping) and explain when each helps.
3. Construct a train/val/test split that does not leak — and recognize the common leakage patterns.
4. Diagnose six canonical loss-curve shapes (healthy, underfit, overfit, unstable, dead, leaky).
5. Choose between grid, random, and Bayesian (TPE) hyperparameter search given a problem and a compute budget.
6. Apply a reproducibility checklist (seeds, deterministic ops, env pinning) to any new training script.

---

## Why this lecture exists

L9 (posted as reading and recapped at the start of this class) was about *making models train*. L10 is about *making them train well and being able to prove it*. Without the discipline taught here, students will produce results they cannot defend — to a reviewer, a collaborator, or themselves six months later.

---

## Section 1 — Bias-variance decomposition (5 min)

The classical picture. For a regression model `f̂` trained on a finite dataset, the expected test error decomposes:

```
E[(y - f_hat(x))^2]  =  (E[f_hat(x)] - f_true(x))^2   +   Var[f_hat(x)]   +   sigma_noise^2
                       |---- bias^2 ----|              |--variance--|     |--irreducible--|
```

- **Bias** = systematic error from a model class too restrictive to capture the truth. High bias = underfitting.
- **Variance** = sensitivity of the learned model to the specific training set. High variance = overfitting.
- **Irreducible noise** = labels have noise σ²; you cannot do better.

For neural networks, the picture is murkier than the textbook says — modern deep learning often operates in a regime where the model has enough capacity that bias is essentially zero, and variance is controlled by implicit regularization from SGD and overparameterization. But the diagnostic intuition still works: a model that underfits the *training set* has high bias; a model that fits training perfectly and fails on validation has high variance.

## Section 2 — Three regularization techniques (7 min)

**Weight decay.** Shrink the parameters toward zero every step:

```
theta  <-  theta - eta * (gradient step)  - eta * lambda * theta
```

With **plain SGD** this is the same as adding an L2 penalty to the loss — with PyTorch's convention,

```
L_total = L_data + (1/2) * lambda * ||theta||^2        # gradient: lambda * theta
```

— so `weight_decay=lambda` and the penalty are interchangeable. **With Adam they are not.** Putting
the penalty in the loss (Adam + L2) sends `lambda * theta` through the adaptive denominator, so
parameters with large gradients get *less* decay. `AdamW` applies the decay separately, outside the
denominator — use it whenever `weight_decay > 0` (L9 slide 6). Typical values: 1e-5 to 1e-3.

`lambda` here is **L9's weight decay** — not L5's Hessian eigenvalue (`team/NOTATION.md`).

What it does: pushes the optimizer toward flatter regions of the loss landscape — empirically improves generalization, and is cheap. For mechanics ML, almost always worth a small amount (`weight_decay = 1e-4` is a safe default).

**Dropout.** During training, randomly zero out a fraction `p` of activations in each forward pass. During eval, use all activations (and PyTorch scales them automatically so the expected value is unchanged).

```python
self.net = nn.Sequential(
    nn.Linear(...), nn.Tanh(), nn.Dropout(p=0.1),
    nn.Linear(...), nn.Tanh(), nn.Dropout(p=0.1),
    nn.Linear(...),
)
```

What it does: prevents the network from relying on any single neuron, forcing redundancy. Effective for large overparameterized image classifiers; **rarely useful** for the small networks we use in this course because (a) overfitting is not the bottleneck and (b) it adds noise to gradients in ways that interact badly with PINN/Sobolev training. Name it, but do not reach for it unless you can show it helps.

**Early stopping.** Train until validation loss stops improving, then take the model from the best-validation epoch. Trivially implementable; almost always helps.

```python
best_val = float('inf')
patience = 50
counter = 0
for epoch in range(max_epochs):
    # ... train ...
    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break
model.load_state_dict(best_state)
```

The most universally useful regularizer. Free, model-agnostic, no hyperparameter except `patience`.

## Section 3 — Train/val/test splits done right (6 min)

Three sets:

- **Train** — used for gradient updates.
- **Val** — used for tuning hyperparameters and early stopping.
- **Test** — used *once* at the end to report the final number.

**The cardinal rule.** Anything you choose based on val performance — the architecture, the learning rate, the number of epochs — has *leaked information about val into your model*. You must report final numbers on data the model has never been judged against. That is the test set.

**Common leakage patterns** students fall into:

1. *Tuning hyperparameters by watching test performance.* Now test isn't test — it's a second val.
2. *Sampling val/test from a different distribution than train.* Leaks information about train (e.g., normalizing val using all-of-data statistics).
3. *Re-using the same test set across many papers / runs.* The test set is becoming a val set.
4. *Splitting time-series naively.* Random splits leak future into past for sequential data.

**For Lab 2** the simplest split is 70/15/15 random with a fixed seed, which is what the starter `mlp_constitutive.py` does. The HP search uses val; final reported number is on test.

**K-fold CV.** When data is precious (sometimes the case in scientific ML), split into K folds, train K models with each fold held out for validation, average. Optuna supports this transparently via the `suggest_*` API + a custom objective that loops over folds.

## Section 4 — The loss-curve gallery (12 min)

Six canonical loss-curve shapes. The instructor walks through `module_2/examples/loss_curve_gallery.py` on the projector — six training runs that intentionally produce each pathology. Students should be able to identify each by sight.

**(a) Healthy training.**
```
Train loss: decreasing, monotonic-ish, sublinear in log space, reaches a stable plateau.
Val loss:   tracks train with small gap; reaches a stable plateau slightly above.
Diagnosis:  great. Stop early if val plateaus.
```

**(b) Underfitting / high bias.**
```
Train loss: high, slow decrease, plateaus way above zero.
Val loss:   tracks train (no gap), also high.
Diagnosis:  model too small, learning rate too low, or wrong loss. Try bigger model first.
```

**(c) Overfitting / high variance.**
```
Train loss: drops near zero.
Val loss:   drops, then rises again as train continues to drop.
Diagnosis:  classic. Apply early stopping at val-loss minimum; consider regularization.
```

**(d) Unstable / LR too high.**
```
Train loss: oscillates wildly, may spike upward; sometimes diverges to NaN.
Val loss:   similar, possibly even worse.
Diagnosis:  drop learning rate by 10x. Or apply gradient clipping.
```

**(e) Dead training.**
```
Train loss: flat from the start, never decreases.
Val loss:   flat at the same value.
Diagnosis:  optimizer.step() not called? Loss function returning constant? Data not flowing? CHECK BASIC PLUMBING before tuning anything.
```

**(f) Leaky split.**
```
Train loss: drops to zero.
Val loss:   drops below train loss. (Suspiciously good.)
Diagnosis:  suspicious — most often a leak (val examples in the training set). Check the split and
            the evaluation mode: dropout or augmentation active only in training can legitimately
            make val loss lower. In this course's split exercise, treat it as a warning sign.
```

Students will encounter all six during Lab 2. The gallery is the muscle memory.

## Section 5 — Hyperparameter search strategies (3 min — random vs grid only; TPE and pruning deferred to L11)

Three strategies, in increasing order of sophistication:

**Grid search.** Define a grid `{lr: [1e-4, 1e-3, 1e-2], width: [16, 32, 64]}`. Run all combinations. Pros: complete, reproducible. Cons: exponential in dimensions — 5 hyperparameters × 5 values each = 3125 runs.

**Random search.** Sample each hyperparameter independently from a distribution. Bergstra & Bengio (2012) showed random search is better than grid for high dimensions because most hyperparameters don't matter, and random search explores the few that do without wasting trials on irrelevant axes. Pros: no exponential blow-up. Cons: needs more trials than necessary because it doesn't learn from previous trials.

**Bayesian / TPE (Tree-structured Parzen Estimator).** Optuna's default. Maintains a probabilistic model of (hyperparameters → objective) based on completed trials, samples the next trial from where the model thinks the objective is best. Pros: often quoted as 5–10× fewer trials than random for the same final performance — a literature rule of thumb, not measured in this course. Cons: implementation complexity, but Optuna handles it for you.

**Choosing.** For 1–2 hyperparameters with cheap trials: grid. For 5+ hyperparameters with expensive trials: TPE. For everything in between: random.

**Pruning.** Optuna's "median pruner" tracks the trajectory of each trial's validation loss and *kills* trials that are obviously worse than the median at the same epoch. Often quoted as roughly a 3× speed-up — a rule of thumb, not measured here. Recommended default.

## Section 6 — Reproducibility checklist (2 min, as a handout)

Before submitting any artifact, run through:

```
[ ] torch.manual_seed(N) set, BEFORE any model or data is created
[ ] numpy.random.seed(N) set, same N
[ ] CUDA determinism if using GPU: torch.use_deterministic_algorithms(True)
[ ] Train/val/test split uses a fixed seed for the shuffle
[ ] Data loading uses num_workers=0 or fixed worker seeds
[ ] Saved checkpoints include: model state, optimizer state, scheduler state, RNG state, hyperparameters
[ ] Environment locked: requirements.txt or conda env.yml frozen
[ ] Code is committed in git with the run-producing SHA recorded in the output
```

The script `module_2/examples/reproducibility_checklist.py` **tests the seeding mechanically, on
CPU**: same seed → identical weights, different seed → different weights. **GPU determinism is a
manual check** — on a GPU the script only prints the instruction; it does not run a GPU comparison.
The rest of the list is yours. Run it in your CI / pre-commit.

---

## Assigned reading (before L11)

**Primary.**

- `module_2/readings/hyperparameter_tuning_guide.md` — one-page Optuna + LR-range cheatsheet.
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13, 2012. Read the introduction and Section 2.
- Optuna documentation, *Quickstart* page (optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html). Skim — you will use this in L11.

**Optional.**

- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 7 (Regularization for Deep Learning), sections 7.1, 7.4, 7.8, 7.12.
- Pedregosa et al., *Hyperparameter selection for deep learning*, JMLR 22, 2021. Short formal paper.

---

## Instructor notes

- The loss-curve gallery is the highest-value content of the lecture. Practice the 6-figure walkthrough; aim for 90 seconds per figure.
- Drop the dropout section if running short — for our class context it is mostly a "named but not used" topic.
- Section 6 (reproducibility) is short but essential. End with: "if you cannot reproduce your own result, you do not have a result."
