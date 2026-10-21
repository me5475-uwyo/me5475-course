"""
icnn_hyperelastic.py
====================

Module 4 / Lecture 21 demo -- an Input-Convex Neural Network (ICNN) learns the
compressible neo-Hookean energy, trained with a Sobolev H1 loss.

Architecture: ICNN with non-negative weights on the internal (z) layers, softplus
activations, and unconstrained skip connections from the input. It is provably
convex in its INPUTS. It is NOT guaranteed polyconvex in F: the standard sufficient
construction would also make psi non-decreasing in I_1, I_2 and -ln J, which the
unconstrained skip weights do not enforce. Check it: icnn_rank_one_check.py.

Input: invariants (I_1 - 3, I_2 - 3, J - 1, -ln J) -- frame-indifferent by
       construction (Way 3 from L20).
Output: strain energy psi (scalar), per unit reference volume.

Stress P = dpsi/dF, the first Piola-Kirchhoff stress, computed via autograd
(it rotates correctly, P(QF) = Q P(F), because psi is frame-indifferent).

Sobolev H1 training: fit psi AND its gradient w.r.t. F (the stress P).
--alpha 0 turns the stress term off (plain MSE on psi), for comparison.

Usage
-----
    python icnn_hyperelastic.py                  # H1, 20000 epochs (about 40 s on a laptop CPU)
    python icnn_hyperelastic.py --alpha 0 --out icnn_mse.png --save-model icnn_mse.pt
    python icnn_rank_one_check.py --model icnn_hyperelastic.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Neo-Hookean: the "truth" we are trying to learn
# -----------------------------------------------------------------------------
def neo_hookean(F_batch, mu=0.5, lam=1.0):
    """
    Compressible neo-Hookean strain energy.
    psi(F) = (mu/2)(I_1 - 3 - 2 ln J) + (lam/2)(ln J)^2

    F_batch: (N, 3, 3)
    Returns psi: (N,) and P = dpsi/dF, the first Piola-Kirchhoff stress: (N, 3, 3)
    """
    F_batch = F_batch.detach().clone().requires_grad_(True)
    C = torch.einsum("nij,nik->njk", F_batch, F_batch)
    I_1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    J = torch.linalg.det(F_batch)
    psi = (mu / 2) * (I_1 - 3 - 2 * torch.log(J)) + (lam / 2) * torch.log(J) ** 2
    P = torch.autograd.grad(psi.sum(), F_batch, create_graph=False)[0]
    return psi.detach(), P.detach()


# -----------------------------------------------------------------------------
# Invariants
# -----------------------------------------------------------------------------
def invariants_of_F(F_batch):
    """
    Compute (I_1, I_2, J, ln J) for batch of F. Returns shape (N, 4).
    I_1 = tr(C); I_2 = 0.5(I_1^2 - tr(C^2)); J = det F
    """
    C = torch.einsum("nij,nik->njk", F_batch, F_batch)
    I_1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    C_sq = torch.einsum("nij,njk->nik", C, C)
    tr_C_sq = torch.diagonal(C_sq, dim1=-2, dim2=-1).sum(-1)
    I_2 = 0.5 * (I_1 ** 2 - tr_C_sq)
    J = torch.linalg.det(F_batch)
    # Use centered invariants for ICNN: at F = I, (I_1, I_2, J, -ln J) = (3, 3, 1, 0)
    return torch.stack([I_1 - 3, I_2 - 3, J - 1, -torch.log(J)], dim=-1)


# -----------------------------------------------------------------------------
# ICNN: Input-Convex Neural Network
# -----------------------------------------------------------------------------
class NonNegLinear(nn.Module):
    """Linear layer with non-negative weights (via exp parameterization)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.log_w = nn.Parameter(torch.randn(out_features, in_features) - 2.0)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, torch.exp(self.log_w), self.bias)


class ICNN(nn.Module):
    """
    ICNN for psi(invariants). Convex by construction:
      - Internal-state path uses non-negative weights + monotone non-decreasing
        convex activation (softplus).
      - Skip connections from input use unconstrained linear layers (their
        contribution is convex because it is affine in the input).
    """
    def __init__(self, n_in=4, hidden=32, depth=3):
        super().__init__()
        # Input layer (unconstrained -- the result will be the input to the convex chain)
        self.W_y_first = nn.Linear(n_in, hidden)
        # Internal layers: non-negative weights on z, unconstrained on y skip
        self.W_z = nn.ModuleList([NonNegLinear(hidden, hidden) for _ in range(depth - 1)])
        self.W_y = nn.ModuleList([nn.Linear(n_in, hidden) for _ in range(depth - 1)])
        # Output layer
        self.W_z_out = NonNegLinear(hidden, 1)
        self.W_y_out = nn.Linear(n_in, 1)

    def forward(self, y):
        z = F.softplus(self.W_y_first(y))
        for W_z, W_y in zip(self.W_z, self.W_y):
            z = F.softplus(W_z(z) + W_y(y))
        return (self.W_z_out(z) + self.W_y_out(y)).squeeze(-1)


# -----------------------------------------------------------------------------
# Hyperelastic ICNN that takes F directly
# -----------------------------------------------------------------------------
class HyperelasticICNN(nn.Module):
    """psi_NN(F) = ICNN(invariants(F))."""
    def __init__(self, hidden=32, depth=3):
        super().__init__()
        self.icnn = ICNN(n_in=4, hidden=hidden, depth=depth)

    def forward(self, F):
        I = invariants_of_F(F)
        return self.icnn(I)


# -----------------------------------------------------------------------------
# Sobolev H1 loss
# -----------------------------------------------------------------------------
def sobolev_h1_loss(model, F_batch, psi_true, P_true, alpha=1.0):
    """Loss = MSE on psi + alpha * MSE on P = dpsi/dF (first Piola-Kirchhoff)."""
    F_batch = F_batch.clone().requires_grad_(True)
    psi_pred = model(F_batch)
    L_data = ((psi_pred - psi_true) ** 2).mean()

    P_pred = torch.autograd.grad(psi_pred.sum(), F_batch, create_graph=True)[0]
    L_stress = ((P_pred - P_true) ** 2).mean()
    return L_data + alpha * L_stress, L_data, L_stress


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20000,
                        help="5000 is far from converged: test P error ~657%% (M4 ledger, section 4)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0, help="Sobolev weight on stress")
    parser.add_argument("--out", type=Path, default=Path("icnn_hyperelastic.png"))
    parser.add_argument("--save-model", type=Path, default=Path("icnn_hyperelastic.pt"),
                        help="trained weights, for icnn_rank_one_check.py")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Generate training data: F's in a window around identity ---
    n = args.n_samples
    eta = 0.3 * torch.randn(n, 3, 3)
    eta = 0.5 * (eta + eta.transpose(-1, -2))    # symmetric perturbation
    F_train = torch.eye(3).unsqueeze(0) + eta
    # Reject samples with det F too small or large
    J = torch.linalg.det(F_train)
    keep = (J > 0.5) & (J < 1.8)
    F_train = F_train[keep]
    print(f"Kept {len(F_train)} of {n} samples after rejecting extreme deformations")

    psi_true, P_true = neo_hookean(F_train)

    # --- 2. Build model ---
    model = HyperelasticICNN(hidden=args.hidden, depth=args.depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 3. Train with H1 Sobolev loss ---
    history = {"data": [], "stress": [], "total": []}
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss, L_d, L_s = sobolev_h1_loss(model, F_train, psi_true, P_true, args.alpha)
        loss.backward()
        optimizer.step()
        history["data"].append(L_d.item())
        history["stress"].append(L_s.item())
        history["total"].append(loss.item())
        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}  total={loss.item():.4e}  data={L_d.item():.4e}  stress={L_s.item():.4e}")

    # --- 4. Test set: NEW Fs not seen during training ---
    eta_te = 0.3 * torch.randn(200, 3, 3)
    eta_te = 0.5 * (eta_te + eta_te.transpose(-1, -2))
    F_test = torch.eye(3).unsqueeze(0) + eta_te
    keep_te = (torch.linalg.det(F_test) > 0.5) & (torch.linalg.det(F_test) < 1.8)
    F_test = F_test[keep_te]
    psi_te_true, P_te_true = neo_hookean(F_test)
    F_te_grad = F_test.clone().requires_grad_(True)
    psi_te_pred = model(F_te_grad)
    P_te_pred = torch.autograd.grad(psi_te_pred.sum(), F_te_grad, create_graph=False)[0]
    print(f"\nTest psi MSE:    {((psi_te_pred - psi_te_true)**2).mean():.4e}")
    print(f"Test P MSE:      {((P_te_pred - P_te_true)**2).mean():.4e}")
    print(f"Test P rel-err:  {(P_te_pred - P_te_true).norm() / P_te_true.norm():.4%}")
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth}, args.save_model)
    print(f"Wrote {args.save_model}")

    # --- 5. Visualize ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(history["data"], color="#1f3a68", linestyle="-", label="psi MSE")
    axes[0].semilogy(history["stress"], color="#8a4b00", linestyle="--", label="P MSE")
    axes[0].semilogy(history["total"], color="#2d6a2d", linestyle=":", linewidth=2, label="total loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, which="both")

    axes[1].scatter(psi_te_true.detach(), psi_te_pred.detach(), s=8, color="#1f3a68", alpha=0.9)
    lo, hi = float(psi_te_true.min()), float(psi_te_true.max())
    axes[1].plot([lo, hi], [lo, hi], color="#b00000", linestyle="--", linewidth=1)
    axes[1].set_xlabel("psi truth"); axes[1].set_ylabel("psi predicted")
    axes[1].set_title("Test set: energy")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(P_te_true.flatten(), P_te_pred.detach().flatten(), s=4, color="#1f3a68", alpha=0.9)
    lo, hi = float(P_te_true.min()), float(P_te_true.max())
    axes[2].plot([lo, hi], [lo, hi], color="#b00000", linestyle="--", linewidth=1)
    axes[2].set_xlabel("P truth (all 9 components)"); axes[2].set_ylabel("P predicted")
    axes[2].set_title("Test set: stress")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"ICNN, neo-Hookean, alpha = {args.alpha} ({'H1: energy + stress' if args.alpha else 'MSE on energy only'}), "
                 f"{args.epochs} epochs")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
