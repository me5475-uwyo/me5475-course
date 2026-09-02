# An MLP fits sin(πx): the training loop you will be asked to judge

*Read before Lecture 2 (Wed Sep 2). In class the instructor will paste the lead prompt below into a coding agent, get a PyTorch file back, and route it to a review agent. Your job as the human in the loop is to judge both agents' output — this reading gives you the mechanics to do that. All measured numbers here come from a run of `module_0/examples/mlp_sine_reference.py` on ARCC (torch 2.5.1, seed 0), recorded in `measured_results.md`.*

## 1. The task

Fit y = sin(πx) on [−1, 1] with a small neural network: 200 noise-free training points, a 4-layer, width-32, tanh MLP, Adam optimizer, 5000 epochs of full-batch training, mean-squared-error loss.

Why this toy problem? Because it is the smallest thing that exercises *every* part of a training loop — data tensors, a model, a loss, an optimizer, an evaluation pass, a plot — and because you know the right answer. When something goes wrong (and in this reading something will go quantifiably wrong), you can see it immediately instead of wondering whether the physics is at fault.

## 2. What an MLP actually is

For this problem, the network is a pipeline that takes one number in and produces one number out:

```
x (1 number) → Linear(1,32) → tanh → Linear(32,32) → tanh
             → Linear(32,32) → tanh → Linear(32,1) → output
```

A `Linear(m, n)` layer computes **Wx + b**: it multiplies its input by an n×m weight matrix and adds a bias vector. That is all it is — an affine map. The `tanh` in between is what makes the network nonlinear. This matters more than it sounds: without the activations, the stack of four Linear layers would collapse algebraically into a single affine map W′x + b′, and no amount of training could make one affine map look like a sine wave. Say this to yourself once, explicitly: *the composition of affine maps is affine; the activations are the entire source of expressive power.*

The measured parameter count is **2209**. You can account for every one of them: Linear(1,32) holds 32 weights + 32 biases = 64; each Linear(32,32) holds 1024 + 32 = 1056 (there are two of them); Linear(32,1) holds 32 + 1 = 33. Training means adjusting these 2209 numbers so the pipeline's output matches sin(πx) at the training points.

## 3. The training loop, line by line

```python
torch.manual_seed(0)
x = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)   # shape (200, 1)
y = torch.sin(torch.pi * x)                        # shape (200, 1)
```

The `.unsqueeze(1)` deserves a sentence. PyTorch's convention is that data tensors are (batch, features): rows are samples, columns are features — *even when there is only one feature*. `torch.linspace` gives a flat (200,) vector; `.unsqueeze(1)` adds the trailing feature dimension to make it a (200, 1) column. `nn.Linear` expects that layout, and — as Section 5 shows — the loss function silently misbehaves without it.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1, 5001):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
```

MSE loss is the mean of (prediction − target)² over all 200 points. The three optimizer calls must appear in exactly this order, and each does one specific thing:

- **`zero_grad()`** clears the gradient buffers. PyTorch's `backward()` *accumulates* gradients into `.grad` rather than overwriting them (a deliberate design, useful elsewhere). Omit `zero_grad()` and every step uses the sum of all past gradients — the optimizer is steered by stale history and training degrades or diverges.
- **`backward()`** computes, by automatic differentiation, the gradient of the loss with respect to all 2209 parameters. Omit it and `.grad` stays at whatever it was — `step()` then changes nothing (or repeats the last update).
- **`step()`** lets Adam update the parameters using the freshly computed gradients. Call it *before* `backward()` and you update with the previous iteration's gradient — a one-step-stale bug subtle enough that training often still limps along, which is exactly why a reviewer has to check the order rather than the output.

After training, the reference switches to `model.eval()` and wraps the prediction pass in `torch.no_grad()` before plotting on a dense 1000-point grid.

## 4. The measured run

With seed 0, the recorded loss trajectory is:

| epoch | 1 | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|---|
| MSE | 5.473e-01 | 1.147e-05 | 2.212e-05 | 1.711e-06 | 1.192e-06 | 2.871e-06 |

Final training loss: **6.707e-06**. Maximum absolute error on the 1000-point grid: **5.335e-03**. The fit is visually indistinguishable from the true curve — see `figures/mlp_sine_fit.png`.

Look at that table honestly: the loss is **not monotone**. Epoch 2000 is *worse* than epoch 1000, and the final recomputed loss is higher than the epoch-5000 checkpoint (which is recorded before that epoch's parameter update). This is normal for Adam, even in full-batch training with no noise in the data: Adam carries momentum and per-parameter adaptive step sizes, so it routinely overshoots a narrowing valley and re-descends. A trajectory that wobbles across four orders of magnitude and lands at 10⁻⁶ is a healthy run. Do not let an agent — or your own instincts — tell you that a non-decreasing checkpoint means the code is broken.

## 5. The broadcasting bug, quantified

This is the most valuable section of the reading. Suppose the lead agent writes the target as a flat vector:

```python
y_flat = y.squeeze()                      # shape (200,) — the bug
loss = loss_fn(model(x), y_flat)          # (200,1) vs (200,)
```

Nothing crashes. The loss is a scalar, training runs 5000 epochs, a plot appears. But the measured outcome is: final loss **4.975e-01** — stuck at roughly the variance of the target — and max |error| **1.000e+00**. The fit is *destroyed*, not merely degraded: the network has learned essentially nothing about the sine wave.

Why? When PyTorch subtracts a (200,) tensor from a (200, 1) tensor, its broadcasting rules align trailing dimensions: the (200,) target is treated as a 1×200 row, the prediction as a 200×1 column, and each size-1 dimension is stretched to match the other — producing a (200, 200) difference matrix in which *every prediction is compared against every target*. The mean of that 40,000-entry square is still a scalar, so the loop runs happily while optimizing a meaningless objective (whose best answer is a near-constant output).

PyTorch does try to help: it emits a `UserWarning` — "Using a target size (torch.Size([200])) different to the input size…". The real failure is not the broadcast; it is **ignoring the warning**. Warnings scrolling past in agent output are exactly the kind of thing a lead glosses over and a shape-focused review prompt catches.

## 6. What to check — the six review items

The review prompt below asks for a PASS/FAIL on six specific items. Here is what each one catches:

1. **Tensor shapes.** The (200,)-vs-(200,1) broadcast above: silent, scalar-producing, fit-destroying.
2. **train()/eval() handling.** This network has no dropout or batch norm, so the modes happen to be numerically identical here — but the habit is the point: the surrogates in later modules will have mode-dependent layers, and evaluating in the wrong mode gives wrong predictions with no error message.
3. **Loss correctness.** All 200 points included, mean not sum (a summed loss silently multiplies the effective learning rate by 200), no sign error.
4. **Gradient-flow order.** `zero_grad → backward → step`, for the reasons in Section 3. Leads sometimes drop `zero_grad()` or put `backward()` after `step()`.
5. **Capacity and learning-rate sanity.** Four layers of width 32 with tanh is ample for one smooth wave, and 1e-3 is a standard Adam rate — the reviewer should confirm the hyperparameters are *plausible for this problem*, not just syntactically present.
6. **Seed set.** The lead almost always omits `torch.manual_seed(...)`. Without it, every run initializes differently, and you cannot compare a "fixed" version against a buggy one — the numbers in this reading are only reproducible because the seed is 0.

## 7. Where this goes

The architecture you just read about is not a throwaway. In Modules 2 and 4 this same 4×32 tanh MLP — same training loop, same three-call gradient dance — becomes the constitutive surrogate: input strain, output stress, trained on real (ε, σ) data instead of a sine wave. Learn the loop here, where you know the answer.

## The two prompts (verbatim)

These are copied exactly from `module_0/lectures/L2_agents_lead_review.md` (Worked Example B). The full runnable reference implementation is `module_0/examples/mlp_sine_reference.py`, which contains **both** the correct version and the instrumented buggy broadcasting variant.

**Lead prompt** (fresh session):

```
Write a PyTorch implementation of a 4-layer MLP with width 32 and tanh activations
that fits the function y = sin(πx) on [-1, 1]. Specifications:

- Generate 200 training points uniformly on [-1, 1] with no noise.
- Use Adam optimizer, learning rate 1e-3, 5000 epochs, full-batch (not mini-batch).
- Loss is MSE.
- After training, plot the network prediction against the true function on a
  dense grid of 1000 points, overlay the training data.
- Print the final training loss.

Return a single Python file that runs end-to-end.
```

**Review prompt** (fresh session):

```
Read the following PyTorch code and check the following items independently:

1. Data tensor shapes — does the input have shape (N, 1) as PyTorch expects?
2. Training mode handling — is .train() called appropriately? Does .eval() get
   called before the plotting phase?
3. Loss correctness — does the MSE include all training points, no missing
   factor, no sign error?
4. Gradient flow — does the code zero gradients, call backward, step the
   optimizer, in the right order?
5. Hyperparameter sanity — width 32 + 4 layers + tanh is enough capacity for
   sin(πx); does the learning rate seem reasonable?
6. Reproducibility — is the random seed set?

Reply with PASS / FAIL per item with one sentence of justification.

Code to review:
<paste lead's output here>
```
