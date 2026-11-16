# DeepONet Primer

A two-page reference for L31 and Lab 7.

---

## The setup

Learn an operator `G: u -> s` where:

- `u` is an input function (e.g., initial condition, source term, microstructure).
- `s` is an output function (e.g., solution at later time, stress field).

You have n training pairs `(u_i, s_i)`. You want a model that, given a new input function u_new, predicts s_new ≈ G[u_new].

---

## The DeepONet architecture

A DeepONet has three parts:

1. **Branch network.** Takes the input function u sampled at m fixed *sensor locations*: `[u(x_1), u(x_2), ..., u(x_m)]`. Outputs K numbers: the *weights* (coefficients) for this input function.
2. **Trunk network.** Takes a query location y. Outputs K numbers: the values at y of K learned expansion functions, the *shapes*.
3. **Inner product.** Compute `s(y) ≈ branch(u_sensors) . trunk(y) + bias`.

```
Input function u (sampled at m fixed sensors x_1, ..., x_m)
   |
   v
[Branch MLP]  --> K weights c_1, ..., c_K                (one set per input u)

Query location y (where we want s(y))
   |
   v
[Trunk MLP]   --> K shape values phi_1(y), ..., phi_K(y)  (the same shapes for every u)

Output:
   s(y) ≈ sum_k c_k * phi_k(y)  + bias
```

The trunk learns the shapes; the branch predicts their weights. The branch is a learned map from the m sensor values to the K weights, not a projection of u onto the shapes, and nothing in training makes the shapes orthogonal or even linearly independent.

**Why this form is reasonable.** Chen & Chen (1995) proved an existence result for this form (with one-hidden-layer networks; DeepONet uses deep ones). Let G be a continuous operator defined on a *compact* set V of continuous input functions, with outputs on a compact domain. Then for any tolerance ε there are sensors x_1, ..., x_m, a number of terms K and networks large enough that |G_θ[u](y) − G[u](y)| < ε for every u in V and every y in the output domain: the approximation is uniform. The theorem says such a network exists; it does not promise that training finds it, or that a model fitted to finite data generalizes.

---

## PyTorch implementation skeleton

```python
import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, m_sensors, query_dim, K, branch_hidden=128, trunk_hidden=128, depth=4):
        super().__init__()
        # Branch: m_sensors -> K
        branch_layers = [nn.Linear(m_sensors, branch_hidden), nn.Tanh()]
        for _ in range(depth - 2):
            branch_layers += [nn.Linear(branch_hidden, branch_hidden), nn.Tanh()]
        branch_layers.append(nn.Linear(branch_hidden, K))
        self.branch = nn.Sequential(*branch_layers)
        # Trunk: query_dim -> K
        trunk_layers = [nn.Linear(query_dim, trunk_hidden), nn.Tanh()]
        for _ in range(depth - 2):
            trunk_layers += [nn.Linear(trunk_hidden, trunk_hidden), nn.Tanh()]
        trunk_layers.append(nn.Linear(trunk_hidden, K))
        self.trunk = nn.Sequential(*trunk_layers)
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y):
        # u_sensors: (B, m)
        # y: (B, Q, query_dim)
        b = self.branch(u_sensors)       # (B, K)
        t = self.trunk(y)                 # (B, Q, K)
        return (b.unsqueeze(1) * t).sum(dim=-1) + self.bias  # (B, Q)
```

With m_sensors = 128, query_dim = 1, K = 128 and the defaults above, this is the model in `module_7/examples/deeponet_burgers.py`: branch 128→128→128→128→K, trunk 1→128→128→128→K, tanh; 115,841 parameters.

---

## Training

Loss: MSE between predicted and true output at query points. One epoch is a full pass over the training functions in shuffled mini-batches, with a fresh random set of query points for each batch (as in `deeponet_burgers.py`):

```python
# u0_tr: (n_train, m) input functions at the sensors; uT_tr: (n_train, N_grid) outputs on the grid
# x_t: (N_grid, 1) grid positions, used as the query locations y
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn = nn.MSELoss()
for epoch in range(epochs):
    # One epoch = one pass over all training functions, in shuffled mini-batches
    model.train()
    perm = torch.randperm(len(u0_tr))
    for start in range(0, len(u0_tr), batch_size):
        idx_b = perm[start : start + batch_size]
        u_batch = u0_tr[idx_b]                                         # (B, m)
        # Sub-sample Q query positions for this batch
        q_idx = torch.randint(0, N_grid, (n_query,))
        y_batch = x_t[q_idx].unsqueeze(0).expand(len(idx_b), -1, -1)  # (B, Q, 1)
        s_batch = uT_tr[idx_b][:, q_idx]                               # (B, Q)
        optimizer.zero_grad()
        pred = model(u_batch, y_batch)
        loss = loss_fn(pred, s_batch)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

In Lab 7 (`deeponet_burgers.py` defaults): 800 training functions (the other 200 of the 1000 pairs are the test set), `batch_size = 32`, `n_query = 64`, 500 epochs, so each epoch is 25 optimizer steps.

---

## Key hyperparameters

| HP | Typical | What it controls |
|----|---------|------------------|
| m_sensors | input grid resolution (Lab 7: 128) | how the input function is discretized; fixed once the model is built |
| K | 64–256 (Lab 7: 128) | how many shapes each output is a weighted sum of |
| branch / trunk depth | 3–5 layers (Lab 7: 4) | network expressivity |
| branch / trunk width | 64–256 (Lab 7: 128) | network expressivity |
| learning rate | 1e-3 to 1e-4 (Lab 7: 1e-3, cosine schedule) | as usual |

---

## Strengths

- **Universal, in principle.** For a continuous operator on a compact set of input functions, a network of this form with small uniform error exists (Chen & Chen 1995); training is not guaranteed to find it.
- **Flexible input types.** Branch can be an MLP (function-on-grid), CNN (image input), GNN (graph input).
- **Per-query evaluation.** The trunk takes ONE query position; you can query at arbitrary y.
- **Bonus: per-query is parallelizable.** Predict at 10000 y's in parallel via batching.

---

## Weaknesses

- **Fixed sensors.** The branch expects the input at the m sensors it was trained with (128 in Lab 7). Inputs on other grids must be resampled to the 128 sensors; Lab 7 Task 4 measures the cost. (The output side is not tied to a grid: the trunk can be queried at any y.)
- **No spatial structure in the branch.** A vanilla branch MLP doesn't exploit spatial locality (use a CNN-branch for image inputs).
- **One shared set of shapes.** The trunk learns *one* set of K shapes that every output is built from; only the weights change with u. For outputs with moving sharp features (such as Burgers fronts) this may be limiting.

---

## When to use DeepONet over FNO

- When the input function is irregular (not on a uniform grid).
- When the output domain is irregular (e.g., scattered query points).
- Which of the two is more accurate, or cheaper, on a given problem is not settled by the architecture alone; Lab 7 Tasks 3 and 4 measure it on Burgers.

---

## Physics-Informed DeepONet

Physics-informed DeepONet (Wang, Wang & Perdikaris 2021) adds the PDE residual, plus the initial/boundary-condition losses, to the training loss. The condition terms matter: a PDE residual alone does not pin down a unique solution. An example for the operator f -> s with s_yy = f on [0, 1] and prescribed boundary values:

```python
def physics_informed_loss(model, f_sensors, y_query, f_query, y_bc, s_bc,
                          lambda_pde=1.0, lambda_bc=1.0):
    # f_sensors: (B, m)     the input function f at the m sensors (branch input)
    # y_query:   (B, Q, 1)  collocation points;  f_query: (B, Q)  f at those points
    # y_bc:      (B, 2, 1)  the boundary points y = 0 and y = 1;  s_bc: (B, 2)  prescribed s there
    y_query = y_query.detach().clone().requires_grad_(True)   # autograd needs d/dy
    s_pred = model(f_sensors, y_query)                                              # (B, Q)
    s_y = torch.autograd.grad(s_pred.sum(), y_query, create_graph=True)[0]         # (B, Q, 1)
    s_yy = torch.autograd.grad(s_y.sum(), y_query, create_graph=True)[0][..., 0]   # (B, Q)
    L_pde = ((s_yy - f_query) ** 2).mean()                  # PDE residual
    L_bc = ((model(f_sensors, y_bc) - s_bc) ** 2).mean()    # boundary-condition loss
    return lambda_pde * L_pde + lambda_bc * L_bc
```

For a time-dependent problem such as Lab 7's Burgers operator, the trunk would take (x, t), the residual would be s_t + s s_x − ν s_xx, and an initial-condition term would match s(x, 0) to u_0 (plus a periodic boundary term). Useful when paired solution data are scarce and you trust the PDE and its conditions. Out of scope for Lab 7's main tasks.

---

## Common pitfalls

| Problem | Possible cause | Fix |
|---------|----------------|-----|
| Trunk shapes stuck near zero | Trunk learning rate too small | Equal LR; warm-up |
| Predictions are noisy/jagged | Too few shapes (K too small), or too little training | Try a larger K (Lab 7: 128); check that the training loss has leveled off |
| Branch saturates (all outputs same) | Branch input not standardized | Standardize u to zero-mean, unit-var |
| Test error >> train error | Overfitting | Add weight decay; reduce K; more data |
| Predictions at boundary look bad | Trunk has not seen boundary y's in training | Include boundary y's in training queries |
