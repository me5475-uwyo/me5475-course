# Losses and Optimizers in Practice — a primer

*Module 2. Read before Lecture 10 (Wed Sep 23). This is the written form of the L9 slide deck,
`M2_L9_loss_optimizers_in_practice.pdf`, which was posted rather than lectured — Monday Sep 21 was
used as a Lab 1 working session instead.*

**Companions you can run:** `module_2/examples/lr_finder.py`,
`module_2/examples/loss_curve_gallery.py`, `module_1/examples/generate_data_analytic.py`.
**Every number here is measured** and recorded in `module_2/readings/measured_results.md`.

---

## 1 · Choosing a loss

The loss is a **model of your noise**. Choosing one is not a matter of taste; it is a claim about
where your labels came from and what went wrong on the way.

| loss | formula | assumes | costs |
|---|---|---|---|
| **MSE** | ½(ŷ − y)² | Gaussian noise | one bad label drags the whole fit |
| **MAE** | \|ŷ − y\| | Laplace noise | gradient never shrinks near the minimum |
| **Huber** | quadratic inside δ_H, linear outside | Gaussian core, heavy tails | one more knob |

δ_H is in the units of your residual. Set it at the boundary between *errors you want fitted* and
*errors you want tolerated*.

### The three-question diagnostic

**Q1 — where did the labels come from?** A simulator has *no label noise*. Lab 1's elasticity is
exact to solver tolerance — MOOSE agrees with the closed-form stiffness to seven digits
(`measured_results.md` §2). **So the Lab 1 answer is MSE, and Q1 is the reason.**

**Q2 — do outliers matter?** If a few bad points should be tolerated rather than chased, move to
Huber. Otherwise quadratic is the better use of the gradient.

**Q3 — do derivatives matter?** In mechanics, usually yes, because **stress is a derivative of
energy and stiffness is a derivative of stress.** Then you add a term that penalises the error in
the *slope* as well as in the *value*:

```
L_H1 = MSE(ŷ, y)  +  α · MSE(dŷ/dx, dy/dx)
        └ values ┘         └ slopes ┘
```

**That is all "Sobolev" means: fit the slopes as well as the values.** The name is a surname —
Sergei Sobolev — attached to the function spaces that require derivatives to be well-behaved too.
`H¹` = values + first derivatives; `H²` adds second derivatives.

Why it matters: a wiggly curve can pass through every data point and still have the wrong slope
everywhere between them. MSE cannot tell the difference. In mechanics that is fatal rather than
cosmetic — a value-only fit can get the energy right and the stress wrong. **The Sobolev loss
returns in M4, at L21.**

---

## 2 · What moves, and what moves it

Two symbols appear in every optimizer equation, and it is worth being exact about both.

- **θ** — the parameter vector: *every* `W` and `b`, flattened into one vector. For Lab 1's
  `[3, 32, 32, 32, 3]` MLP that is **2339** entries.
- **g** — the gradient of the loss with respect to θ. L8's `δ⁽ˡ⁾ ≡ ∂ℒ/∂z⁽ˡ⁾` travels backward and
  becomes `∂ℒ/∂W⁽ˡ⁾`; stack those in the same order and you have **g**.

**They are the same shape, index for index.** Entry *j* of **g** is the derivative of the loss with
respect to entry *j* of **θ**. Once that lands, `θ ← θ − ηg` is arithmetic rather than notation.

> **Two collisions worth naming.** On L5, momentum was written `v ← βv + (1−β)g` — a **first**
> moment. Adam's `v̂` is the **second** moment, the thing under the square root. **L5's `v` is
> Adam's `m̂`.** Separately, L5's `λ` was a Hessian eigenvalue; from here `λ` is **weight decay**.
> Both spellings are standard; what is wrong is reading one and calling the other.

### AdamW: one word, and it should be your default

Adam with L2 folds the decay into the gradient:

```
g ← g + λθ        then   θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)
```

The decay then **rides through the adaptive denominator**, so a parameter with large gradients gets
*less* decay — backwards from what you wanted. AdamW applies it separately:

```
θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε) − ηλθ_t
```

In code that is `torch.optim.Adam` → `torch.optim.AdamW`. **Use it whenever `weight_decay > 0`.**

Symbols: `η` learning rate · `λ` weight decay · `ε = 10⁻⁸` guarding the division ·
`m̂_t, v̂_t` the bias-corrected first and second moments of **g**.

### Gradient clipping: for the one that explodes

AdamW rescales the **typical** gradient. It gives you no protection from a single pathological one.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Between `backward()` and `step()`, and nowhere else.** Before `backward()` there is nothing to
clip; after `step()` the step has already been taken. It rescales the whole vector if its norm
exceeds `max_norm`, leaving the **direction** unchanged. Reach for it when the loss goes `NaN`.
**PINN training in M3 almost always needs it.**

---

## 3 · The learning rate

**Adaptive optimizers rescale η. They never choose it.** Everything above acts on the gradient;
η is still a number you pick, and it should not be constant.

| schedule | when |
|---|---|
| **constant** | small problems with an adaptive optimizer — Lab 1's default |
| **cosine annealing** | widely used, pairs well with Adam |
| **ReduceLROnPlateau** | watch the validation loss, cut by a factor when it stalls. Robust, few knobs — **the course default from M3 onward** |
| **warm-up** | a *modifier*, not a schedule: start near zero for a few hundred steps, then ramp. Close to mandatory for large PINNs, usually as *warm-up then cosine* |

### The LR-range test (Smith 2017)

Three steps, about thirty seconds of compute: start at η = 10⁻⁷, ramp geometrically (×1.1 per
step), plot loss against rate on log axes. You get a bowl. **Take the rate about one order of
magnitude below where the loss turns up** — read the *wall*, not the bottom, because the minimum
is already near the edge and one unlucky batch tips you over.

---

## 4 · What the range test actually tells you

This is the part that is easy to over-read, so the course measured it.

Running `lr_finder.py` on the same 200-row dataset, varying only the seed
(`measured_results.md` §3):

| seed | suggested LR |
|---|---|
| 0 | 1.07e-03 |
| 1 | 1.00e-02 |
| **42** (the script default) | **1.00e-02** |
| 7 | *no suggestion printed* |
| 123 | *no suggestion printed* |

**Two of five seeds produce no answer at all.** And where it prints `1.00e-02`, the "divergence" it
backs off from sits at `1e-1` — which is exactly `--lr-max`, the top of the sweep. The test ran out
of range rather than finding a wall.

It gives the same `1.00e-02` on a laptop and on ARCC, across two different versions of torch. That
looks like reproducibility. **It is the artifact being stable, not the measurement being
reproducible** — the top of the sweep does not move when torch does.

*Why so weak here:* this implementation takes **one optimizer step per learning rate**, on a single
model whose rate ramps continuously. On a target as easy as linear elasticity the loss barely moves
at small rates, so the minimum always lands near the top. A fuller Smith range test uses several
steps per rate and a fresh model.

**The honest reading: an LR-range test points you at a *decade*, not at a value.** That is still far
better than guessing. To get a defensible number, take the decade from here and tune inside it on a
validation set — which is Lab 2, with Optuna (`hyperparameter_tuning_guide.md`).

---

## 5 · Defaults for this course

- **MSE** for simulator labels; Huber when outliers should be tolerated; add a Sobolev term when
  slopes matter (M4).
- **AdamW** whenever `weight_decay > 0`.
- **`clip_grad_norm_`** between `backward()` and `step()` — and expect to need it in M3.
- **Constant η** for Lab 1; **ReduceLROnPlateau** from M3 onward.
- **A range test beats guessing**, and it gives you a decade rather than a number.

---

## Self-check

1. Your labels come from a strain gauge on a real specimen, and three of two hundred readings are
   obviously bad. Which loss, and why does Q1 alone not settle it?
2. Write down the shapes of **g** and **θ** for the Lab 1 MLP. What is entry 1847 of **g**?
3. Why does folding weight decay into the gradient interact badly with Adam specifically, and not
   with plain SGD?
4. You clip gradients before calling `backward()`. What happens, and why is there no error?
5. A classmate reports that the LR finder gave them `1.00e-02` on two different machines and
   concludes the value is robust. What would you ask them to check first?
6. The L9 deck says "ReduceLROnPlateau is the course default from M3 onward" but Lab 1 uses a
   constant rate. Give the argument for each.

---

## References

- **Smith**, *Cyclical Learning Rates for Training Neural Networks*, **WACV 2017** (arXiv:1506.01186)
  — §§1–3, the range-test procedure.
- **Loshchilov & Hutter**, *Decoupled Weight Decay Regularization*, **ICLR 2019** (arXiv:1711.05101)
  — §4, the AdamW results.
- **Goodfellow, Bengio & Courville**, *Deep Learning*, Ch. 8.3–8.5 — optimizers in depth.
- **Karpathy**, *A Recipe for Training Neural Networks*, 2019 — the practitioner's patterns.

*Course convention: papers are cited by **publication venue year**, not arXiv year
(`team/NOTATION.md`).*
