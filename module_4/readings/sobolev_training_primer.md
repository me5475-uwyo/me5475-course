# Sobolev Training in Two Pages

A practitioner's reference for Sobolev H¹ and H² loss in PyTorch.

---

## What Sobolev training is

Standard MSE training fits *function values*:

```
L_MSE = mean over i of (psi_NN(x_i) - psi_true(x_i))^2
```

Sobolev H¹ training adds *derivative values*:

```
L_H1 = L_MSE + alpha * mean over i of ||d psi_NN / d x|_{x_i}  -  grad_psi_true_i||^2
```

Sobolev H² adds *second derivative values*:

```
L_H2 = L_H1 + beta * mean over i of ||d^2 psi_NN / d x dx|_{x_i}  -  Hessian_psi_true_i||^2
```

Why do this in mechanics? Because the derivative of ψ *is the stress* (for hyperelasticity), and the second derivative *is the corresponding constitutive tangent* (∂P/∂F for hyperelasticity) that FE solvers need. Training ψ to predict stress and stiffness *consistently* with itself is the whole point.

---

## PyTorch implementation

The chain: forward pass → backward pass for ∇ψ → backward pass again for ∇²ψ.

```python
import torch

def sobolev_h1_loss(model, x, y_true, dy_true, alpha=1.0):
    """
    model:    callable y = f(x)
    x:        input tensor, shape (N, in_dim), with requires_grad=True
    y_true:   target values, shape (N,) or (N, 1)
    dy_true:  target gradients, shape (N, in_dim)
    alpha:    Sobolev weight on the gradient term
    """
    x.requires_grad_(True)
    y_pred = model(x)
    L_data = ((y_pred.squeeze() - y_true.squeeze()) ** 2).mean()

    # Gradient via autograd
    dy_pred = torch.autograd.grad(
        y_pred.sum(),
        x,
        create_graph=True,    # need this for backprop through the gradient
    )[0]
    L_grad = ((dy_pred - dy_true) ** 2).mean()

    return L_data + alpha * L_grad

def sobolev_h2_loss(model, x, y_true, dy_true, d2y_true, alpha=1.0, beta=0.1):
    x.requires_grad_(True)
    y_pred = model(x)
    dy_pred = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0]

    # Hessian (or its components) via differentiating dy_pred row by row
    d2y_pred = []
    for i in range(dy_pred.shape[-1]):
        row = torch.autograd.grad(
            dy_pred[..., i].sum(), x, create_graph=True
        )[0]
        d2y_pred.append(row)
    d2y_pred = torch.stack(d2y_pred, dim=-1)
    # d2y_pred[..., i, j] = d^2 y / dx_i dx_j

    L_data = ((y_pred.squeeze() - y_true.squeeze()) ** 2).mean()
    L_grad = ((dy_pred - dy_true) ** 2).mean()
    L_hess = ((d2y_pred - d2y_true) ** 2).mean()
    return L_data + alpha * L_grad + beta * L_hess
```

---

## Computational cost

- **Standard MSE:** 1 forward pass, 1 backward pass for gradient w.r.t. parameters.
- **H¹ Sobolev:** 1 forward + 1 backward (for ∇y_pred), then 1 more backward (for gradient of L_H1 w.r.t. parameters). **Roughly 2× the passes of MSE training** — a pass-count heuristic, not a measured or derived compute or memory ratio.
- **H² Sobolev:** 1 forward + 1 backward (∇y) + N_input more backward passes (one per component of the gradient, for the Hessian), then a final backward (for gradient of L_H2 w.r.t. params). **Roughly (2 + N_input)× the passes** — again a pass-count heuristic, not a measured or derived ratio.

The extra passes grow with the input dimension, so H² is cheap for small inputs such as the 6 Voigt components and gets more expensive as the input grows; where it becomes too expensive depends on the model and the hardware (not measured here). For Lab 4 and the midterm, the inputs are small.

---

## When to use each level

- **MSE (no Sobolev).** Use for problems where you only care about the function value (regression on stress fields, classification).
- **H¹ Sobolev.** Use when the gradient of the output has physical meaning (stress is the gradient of energy; outward normal to a yield surface is the gradient of the yield function; flow direction is the gradient of the SDF).
- **H² Sobolev.** Use when you also need the second derivative (algorithmic tangent for FE return mapping; ellipticity check of an elastic energy; smoothness regularization).

---

## Activation function caveat

ReLU's first derivative is a step: 0 for negative inputs, 1 for positive ones, undefined at the kink. Its second derivative is zero everywhere except at the kink, where, in the distributional sense, it is a delta. Pointwise autodiff never sees that delta: it returns 0 for the second derivative almost everywhere.

**So H² Sobolev with ReLU cannot work, for a network built from affine layers and ReLUs and differentiated with respect to its direct input.** The predicted Hessian is zero almost everywhere, so the Hessian term of the loss is just the size of the true Hessian. That term can be large, and no parameter update can reduce it, because its gradient through the network is zero.

This conclusion depends on that setting. If the network's input is a **nonlinear feature** of the physical variable, as in `icnn_hyperelastic.py`, where the inputs are invariants of F, then the Hessian with respect to F picks up the curvature of the features themselves: D²_F(g∘y) = Dyᵀ (D²_y g) Dy + Σⱼ (∂g/∂yⱼ) D²_F yⱼ. The second term is generally nonzero even when D²_y g = 0. Multiply layers (below) change it as well.

For H¹ Sobolev, ReLU is usable. A ReLU network's gradient is piecewise constant, so the derivative fit is piecewise constant too, and the function fits are less smooth.

For H² Sobolev in an ordinary network, use activations with continuous second derivatives: tanh, GELU or softplus. Continuous does not mean nonzero everywhere; for example, tanh'' is 0 at 0. **For an ICNN, only softplus of these keeps the convexity proof**, because the activation must be convex and non-decreasing; tanh and GELU are not (`icnn_primer.md`).

Steve Sun's "multiply layer" architecture (L6 slide 25) squares a layer's output element by element. With ReLU, as Vlassis & Sun use it, each squaring doubles the degree of every nonconstant polynomial piece, which makes nonzero second derivatives possible (it does not guarantee them for every piece or weight setting, nor make them continuous across the kinks). `module_4/examples/levelset_yield_surface.py` puts multiply layers after tanh, whose second derivatives are already continuous.

---

## A worked example (worked through in the icnn_hyperelastic.py demo)

**Neo-Hookean hyperelasticity:**

```
psi(F) = (mu/2)(I_1 - 3 - 2 ln J) + (lambda/2)(ln J)^2   # energy per reference volume
P_iJ   = d psi / d F_iJ  = mu (F - F^{-T}) + lambda (ln J) F^{-T}   # first Piola-Kirchhoff stress
A_iJkL = d P_iJ / d F_kL                                            # its tangent, dP/dF
```

The derivative of an energy per reference volume with respect to F is the **first Piola–Kirchhoff stress P**, under the course convention P = J σ F⁻ᵀ, where σ is the Cauchy stress. Its tangent ∂P/∂F is the material tangent of this hyperelastic model. The "algorithmic tangent" is the corresponding quantity for a return-mapping plasticity update. The current demo names this tensor `P` and prints "Test P rel-err"; older logs use `sigma`. Every stress number below is an error in P, not in the Cauchy stress.

Training a neural network ψ_NN(F) with:

- **MSE only:** the loss fits ψ_NN(F) to ψ_true(F) at the training points (a finite loss does not make them match exactly); P derived from autograd is not fitted at all, and can be far off, even at the training points.
- **H¹ Sobolev:** the loss fits both ψ_NN(F) and ∂ψ_NN/∂F = P at the training points; P-recovery is much better.
- **H² Sobolev:** also fits the tangent ∂P/∂F.

Measured with `icnn_hyperelastic.py` (`measured_results.md` §4), on held-out F drawn from the training distribution: after 20 000 epochs the relative error in P was **42.6 % with MSE only and 10.2 % with H¹** (H² is not implemented there). The script's current default is 20 000 epochs; earlier runs used a default of 5 000. At 5 000 epochs neither had converged (1 614 % and 657 %), so do not compare at that length.

The visual signature to expect in parity plots of P_true vs P_pred (only MSE and H¹ were measured, §4):
- MSE-only training: scatter, but with substantial spread.
- H¹ training: tighter clustering on the diagonal.
- H² training (not implemented or measured here): adds tangent-error supervision; stress and tangent accuracy must be evaluated.

For Lab 4 and the midterm Flavor B, H¹ training is the recommended default.

---

## Common Sobolev-training bugs

1. **Forgetting `create_graph=True` on the inner backward.** Without it, you cannot backprop through the gradient to update parameters; the optimizer sees zero gradient on the Sobolev term.
2. **Forgetting `requires_grad_(True)` on the input.** Then `torch.autograd.grad(y, x)` raises a `RuntimeError`; it does not return None.
3. **Reusing tensors across iterations.** Reusing the same *leaf* input x, a tensor you created and marked `requires_grad=True`, in fresh forward/backward passes is fine: each iteration builds a new graph, and the old one is freed. Two things do go wrong. If you call `.backward()` on a loss involving x, gradients **accumulate in `x.grad`** across iterations; zero it, or use `torch.autograd.grad`, which does not write `.grad`. And if x is a **non-leaf**, the output of earlier computation still attached to a graph, every iteration builds on that old history, and memory can grow; detach it first: `x = x.detach().requires_grad_(True)`.
4. **Computing the Hessian column-by-column when row-by-row is cheaper.** It depends on dimensions; profile if in doubt.
5. **Trying H² with ReLU.** See above.

---

## A useful normalization trick

Standardize y and dy *separately*, so the Sobolev weight α does not have to absorb the scale difference. Then compare like with like:
- If the network output z is the standardized value, the physical prediction is ŷ = μ_y + s_y z.
- Its derivative is ∂ŷ/∂x = s_y ∂z/∂x.
- Standardize that derivative with the *derivative* statistics before comparing it to the standardized derivative target.

```python
# Once, from the whole training set:
mu_y, s_y   = y_train.mean(), y_train.std()
mu_dy, s_dy = dy_train.mean(0), dy_train.std(0)        # per input component, shape (d,)

# Every step: x_batch, y_batch and dy_batch hold the SAME samples, in the same order.
x  = x_batch.detach().requires_grad_(True)             # (B, d) raw, unscaled inputs
z  = model(x).reshape(-1)                              # (B,)   predicts standardized y
y_std  = (y_batch.reshape(-1) - mu_y) / s_y            # (B,)   whether y_batch is (B,) or (B, 1)
dy_std = (dy_batch - mu_dy) / s_dy                     # (B, d)
dz = torch.autograd.grad(z.sum(), x, create_graph=True)[0]    # (B, d)
dy_hat = s_y * dz                                      # d(y_hat)/dx, since y_hat = mu_y + s_y * z
loss = ((z - y_std) ** 2).mean() \
     + alpha * (((dy_hat - mu_dy) / s_dy - dy_std) ** 2).mean()
```

Two details matter here. First, the targets must be the batch's own targets: standardize `y_batch` and `dy_batch`, not the whole training set. Second, reshape the prediction and the value target to the same `(B,)` shape. A `(B, 1)` prediction minus a `(B,)` target broadcasts silently to `(B, B)` and compares every prediction with every target. The snippet assumes a scalar output, a model that treats samples independently (so `z.sum()` gives each sample its own gradient; no batch normalization), and nonzero standard deviations.

Multiplying the output by s_y alone neither restores the mean nor turns the raw derivative into a standardized one. If the inputs are standardized too (the model takes x_s = (x − μ_x) / s_x), the chain rule adds the input scale: ∂ŷ/∂x = s_y (∂z/∂x_s) / s_x.

With both terms on the same scale, `alpha = 1` is a reasonable starting point. Without standardization, α would also have to absorb the scale difference between the derivative term and the value term.
