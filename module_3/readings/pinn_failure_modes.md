# PINN Failure Modes — The Four Tricks Reference

*Module 3. The reference for Lecture 17 (Fri Oct 9), and for Lab 3 (due Mon Oct 19) whenever a PINN will not train — or trains beautifully and is wrong. Runnable companions: `module_3/examples/plate_with_hole_fixed.py` and `plate_with_hole_hard_bc.py`; the trick snippets below are patterns to adapt, not course scripts.*

A consolidated reference for L17. Measured numbers cite `module_3/readings/measured_results.md` by section; anything marked *rule of thumb* was not measured in this course, and **none of the four tricks has been measured on the plate problem (not measured).**

**Notation.** ℒ is the loss, a weighted sum ℒ = Σ λ_k ℒ_k of terms such as ℒ_PDE and ℒ_BC, each the **mean** of its squared residuals over its own points (DeepXDE's `"MSE"`). λ with a subscript (λ_BC, λ_FEM, λ_k) is always a **loss weight** — not L5's Hessian eigenvalue, not L10's weight-decay coefficient. η is the learning rate; P is a parameter count.

---

## First: two failures this course has measured

Before reaching for a trick, rule these out. Both happened on the Lab 3 plate, and neither shows up in the training loss.

**1. The problem was posed wrong.** An earlier port of Min's notebook dropped the three σ₁₂ = 0 conditions on the left, bottom and right edges. With the same training budget it reached essentially the same training loss as the corrected script (7.3e-6 against 6.6e-6) — and a displacement error of 0.597 against 1.20e-3, with σ₁₁ near the top of the hole at 0.063 instead of 3.100 (`measured_results.md` §2). **A small PINN loss does not certify that the problem was posed right.** Check: count your boundary conditions (the plate has ten — two per edge, two on the hole), and compare with an independent reference (Lab 3 Task 2).

**2. Too few collocation points.** With 1 500 + 300 points the hard-BC PINN fitted its fixed training points and oscillated between them: training loss 8.3e-7, test loss 7.7, displacement error 4.8e-2. Resampling the points every 1 000 Adam steps helped but was not enough (test loss 0.93, error 1.8e-2 — L-BFGS then refits one fixed set). With 4 000 + 600 points the same script reached test loss 1.9e-5 and error 1.04e-3 (`measured_results.md` §3). That is what worked in the measured run, not a proof of the minimum you need. **Symptom: test loss far above training loss.** The PDE part of DeepXDE's test loss uses separately sampled points; its BC part reuses the training boundary points, so check boundary conditions on fresh points yourself.

---

## Quick decision tree

```
Loss not going down at all?
   YES -> Mode E (Dead). Check basic plumbing -- not PINN-specific.

Test loss far above training loss?
   YES -> Too few collocation points (measured failure 2). Add points first.

Loss low, but the answer disagrees with an independent reference?
   YES -> Count the BCs (measured failure 1); check the PDE residual line by line;
          run the physics-police checks in module_3/homework/starter_prompts.md.

Loss plateaus high; ℒ_BC dominates?
   YES -> Mode B (Solid BC violation). Apply: hard-BC ansatz (L15) if the violated
          terms are displacement conditions; Trick 2 (loss weights); Trick 3 (PCGrad).

Loss plateaus high; ℒ_PDE dominates?
   YES -> Mode A (Stalled). Apply: Trick 1 (FEM-guided warm-up) OR Trick 4 (Net2Net).

Loss plateaus at a small but non-negligible value, terms balanced?
   YES -> Mode D (Capacity bottleneck). Bigger network + Trick 4 to train it.

Gradient inner products g_i . g_j consistently negative?
   YES -> Mode C (Gradient interference). Apply: Trick 3 (PCGrad).
```

On the hard-BC ansatz: it makes the three displacement conditions exact for any network output. It does nothing for the seven traction conditions, which stay in the loss, and under the same schedule the course's hard-BC runs were not faster than soft (`measured_results.md` §6). Use it for the guarantee, not as a convergence cure.

For the hole check, use the reference for *this* finite plate, σ₁₁ = 3.2418 at (0, 0.1) (`measured_results.md` §1) — not the infinite-plate Kirsch factor of 3, which describes a different problem.

---

## Trick 1 — FEM-guided fading supervision

**The intuition.** PINN training starts from a random network. An auxiliary supervision — a coarse FE solution of the same problem — acts as a warm-up that puts the network near the right basin, then fades out so the final solution is driven by the physics.

**Implementation pattern.**

```python
def total_loss(step, fade_steps):
    loss_pde = compute_pde_loss()            # ℒ_PDE: mean squared residual
    loss_bc = compute_bc_loss()              # ℒ_BC
    loss_fem = compute_fem_data_loss()       # mean squared mismatch with the FE solution at FE nodes

    fade = max(0.0, 1.0 - step / fade_steps)     # 1 -> 0 over the first fade_steps steps
    lambda_fem = 100.0 * fade                    # λ_FEM: a loss weight (rule-of-thumb values)
    return loss_pde + loss_bc + lambda_fem * loss_fem
```

In DeepXDE the FE data enters as `PointSetBC` terms, and loss weights are fixed at `compile`. To fade them, train in chunks and re-`compile` with a smaller λ_FEM between chunks — each `compile` builds a fresh optimizer, so Adam's running averages restart.

**When to use.** Mode A (stalled convergence), when a coarse FE solve is cheap.

**Two cautions for Lab 3.** Lab 0's MOOSE model is a *different problem* (a quarter annulus with a traction load), so its output cannot warm up the plate PINN; the plate's own FE input is `module_3/examples/plate_square_hole_reference.i`. And if you warm up from the same reference you then score against (Task 2), the comparison is no longer independent — use a coarser mesh, and say so. `plate_with_hole_fixed.py --use-fem-warmup` is a stub: it prints a message and trains normally.

**Source.** Steve Sun's *ML for Mechanics* L7, as adapted in L17. We have not verified a journal paper for this exact fading recipe.

---

## Trick 2 — Balancing the loss weights

**The intuition.** A static λ_BC = 1 is a guess. The right value depends on the physics and on the network's state, both of which change during training. Adjust the λ_k during training so that no term's gradient swamps the others.

**Implementation pattern** — a simplified gradient-norm balancing rule. It is the idea behind GradNorm and behind Wang, Teng & Perdikaris's adaptive loss weights for PINNs, not either algorithm exactly.

```python
# Do NOT make the λ_k an nn.Parameter trained by the optimizer: dℒ/dλ_k = ℒ_k >= 0, so
# gradient descent drives every weight to zero and then negative. Update them by a rule.
lambdas = torch.ones(n_terms)                       # λ_k, one per loss term

def weighted_loss(losses):                          # losses: 1-D tensor of the ℒ_k
    return (lambdas * losses).sum()

def rebalance(losses, params, alpha=0.5):           # call every ~100 steps (rule of thumb)
    norms = torch.stack([
        torch.cat([g.flatten() for g in torch.autograd.grad(L, params, retain_graph=True)]).norm()
        for L in losses
    ])                                              # gradient norm of each term
    target = norms.mean() / (norms + 1e-12)         # small-gradient terms get larger weights
    lambdas.mul_(1 - alpha).add_(alpha * target)    # smoothed update
```

The static alternative is to tune one weight: Lab 3 Task 4's sweep searches `bc_loss_weight` (log-uniform in [1, 100], applied to the seven traction terms) as a hyperparameter.

**When to use.** Mode A or Mode B, when several loss terms compete — a common first choice (*rule of thumb*). A larger λ_BC shifts the trade-off; no finite weight *guarantees* that a boundary condition holds exactly.

**References.** Chen, Badrinarayanan, Lee & Rabinovich, *GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks*, ICML 2018. Wang, Teng & Perdikaris, *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*, SIAM J. Sci. Comput. 43(5), A3055–A3081, 2021 — the PINN-specific version. van der Meer, Oosterlee & Borovykh, *Optimally weighted loss functions for solving PDEs with neural networks*, J. Comput. Appl. Math. 405, 2022 — how to choose the weight for linear, well-posed problems.

**Related — Self-Adaptive PINN.** McClenny & Braga-Neto, J. Comput. Phys. 474, 2023, put a learnable weight on each *collocation point* (not each loss term) and train those weights by gradient **ascent** while the network descends — the network is pushed to fix the points it fits worst.

---

## Trick 3 — Gradient surgery / PCGrad

**The intuition.** Two loss terms may have gradients that conflict — improving one drives the other up — and the summed gradient is partly cancelled. PCGrad removes the conflicting component: whenever gᵢ · gⱼ < 0, project gᵢ onto the plane normal to gⱼ.

**Implementation pattern** (needs your own PyTorch training loop around the network: DeepXDE's `train` sums the terms before `backward`).

```python
def pcgrad_step(losses, params, optimizer):
    # one flattened gradient per loss term
    grads = [torch.cat([g.flatten() for g in torch.autograd.grad(L, params, retain_graph=True)])
             for L in losses]
    projected = []
    for i, g_i in enumerate(grads):
        g = g_i.clone()
        for j in torch.randperm(len(grads)).tolist():    # random order, as in the paper
            if j == i:
                continue
            dot = torch.dot(g, grads[j])                  # against the ORIGINAL g_j
            if dot < 0:
                g -= dot / grads[j].pow(2).sum() * grads[j]
        projected.append(g)
    total = torch.stack(projected).sum(dim=0)
    offset = 0                                            # write it back into .grad
    for p in params:
        p.grad = total[offset:offset + p.numel()].view_as(p).clone()
        offset += p.numel()
    optimizer.step()
```

**Cost.** One backward pass and one stored full gradient *per loss term* per step. The soft plate has 15 terms (5 PDE + 10 BC), the hard plate 12 (5 + 7); grouping them — all PDE residuals as one term, all BCs as another — keeps it to two (*rule of thumb*; cost not measured).

**When to use.** Mode C (gradient interference). Diagnose by computing gᵢ · gⱼ periodically; if consistently negative, try PCGrad.

**Reference.** Yu, Kumar, Gupta, Levine, Hausman & Finn, *Gradient surgery for multi-task learning*, NeurIPS 2020.

---

## Trick 4 — Net2Net curriculum

**The intuition.** The plate network — 6 hidden layers of width 50, 2 inputs, 5 outputs — has P = 13 155 parameters (arithmetic, not a measurement). From a random start the optimizer may not find a good basin. Start small (L17's example: 2 hidden layers of width 25, P = 855), train, then *inject* capacity (Net2WiderNet or Net2DeeperNet) while preserving the function, and continue training.

**Implementation pattern.** Net2WiderNet, doubling a hidden layer of width n. In PyTorch an `nn.Linear` weight has shape (out, in) — rows are where you are going, as in L7's W^(ℓ) ∈ ℝ^{n_ℓ × n_{ℓ−1}}.

```python
# layer ℓ:   W (n, n_in), b (n,)        layer ℓ+1:  W_next (n_next, n), b_next unchanged
W_new = torch.cat([W, W], dim=0)                            # (2n, n_in): every unit duplicated
b_new = torch.cat([b, b], dim=0)                            # (2n,)
W_next_new = torch.cat([W_next / 2, W_next / 2], dim=1)     # (n_next, 2n): each copy carries half
```

The function is exactly preserved: the widened layer outputs every activation twice, and the next layer adds the two halves back together. Exact copies receive identical gradients and would stay identical, so add a little noise to the new weights to break the symmetry (*rule of thumb*).

**When to use.** Mode D (capacity bottleneck), when you've tried Tricks 1–3 and still can't get below some accuracy floor.

**Reference.** Chen, Goodfellow & Shlens, *Net2Net: Accelerating learning via knowledge transfer*, ICLR 2016.

---

## Modern tricks (briefly)

**NTK reweighting** (Wang, Yu & Perdikaris, J. Comput. Phys. 449, 2022). Analyse the PINN's gradient flow through the neural tangent kernel and reweight the loss terms to balance its eigenvalues. Mathematically principled; harder to implement. Worth knowing it exists.

**Causal training** (Wang, Sankaran & Perdikaris, Comput. Methods Appl. Mech. Eng. 421, 2024; arXiv 2022 as *Respecting causality is all you need…*). For *time-dependent* PINNs: weight the residual at time t only once the network has converged at earlier times. Not applicable to Lab 3 (steady plate-with-hole), but relevant to transient final projects.

**Random Fourier feature embedding** (Tancik et al., NeurIPS 2020; Wang, Wang & Perdikaris, Comput. Methods Appl. Mech. Eng. 384, 2021). Feed the network `[sin(Bx), cos(Bx)]` instead of x, where B is a fixed random matrix. Helps a PINN represent high-frequency or multi-scale solutions. Easy to implement; consider it for a final project with multi-scale physics.

---

## Reading PINN loss curves — what each shape means

Plot the loss terms separately (DeepXDE stores one column per term in `losshistory.loss_train`); the total hides which term is stuck.

| Shape | Diagnosis | Try |
|-------|-----------|-----|
| All terms dropping smoothly to small values | Healthy — so far | Verify against a reference and on fresh boundary points |
| Test loss far above training loss | Too few collocation points (§3) | More points |
| ℒ_PDE plateaus high while ℒ_BC drops near zero | Mode A: optimizer giving up on the PDE | Trick 1 or 2 |
| ℒ_BC plateaus high while ℒ_PDE drops near zero | Mode B: BCs neglected | Hard-BC ansatz (displacement terms only); Trick 2 |
| Both plateau at non-trivial values | Mode A+B or Mode D | Trick 1, 2, then 4 |
| Loss oscillates / spikes | η too high (*rule of thumb*) | Lower η; gradient clipping |
| Loss NaN | Numerical issue | Check for log(0), sqrt(negative), division by 0 in the PDE |
| Loss small but predictions wrong | Mis-posed problem (§2) or mis-written PDE | Count the BCs; physics-police checks; compare with MOOSE |

---

## One final piece of advice

**Rule out the two measured failures first** — count the boundary conditions and compare with the reference; check test loss against training loss. They are cheap to check, and in this course each one hid behind a small training loss. Then match the trick to the symptom. Trick 1 is the natural next step for a stalled plate PINN, because a coarse FE solve of this problem is easy to get; whether it helps here has not been measured.
