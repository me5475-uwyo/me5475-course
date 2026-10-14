"""
levelset_yield_surface.py
=========================

Module 4 / Lectures 23-24 + Lab 4 stretch -- Train a neural network to represent
a yield surface as a signed-distance function (SDF).

Following Vlassis & Sun, CMAME 2021. The yield surface is the zero level set
of a learned function phi(sigma) -- a signed distance function (SDF) in stress
space.

Training data: points sampled in stress space, with labels giving the signed
distance to a circle of radius kappa in a toy (sigma_x, sigma_y) plane, a stand-in
for J2's deviatoric circle. We use it because the true SDF is known exactly.

Architecture: a "multiply" MLP (Vlassis-Sun L6 slide 25) with two multiply
layers, following their design for yield functions with useful second
derivatives (an algorithmic tangent needs them; the course computes none).

Loss: Sobolev H1 -- function value AND outward-normal direction at training
points.

Usage
-----
    python levelset_yield_surface.py
    python levelset_yield_surface.py --kappa 1.0 --n-samples 1000 --epochs 5000
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


# -----------------------------------------------------------------------------
# Toy J2-like yield surface: a circle in the (sigma_x, sigma_y) plane
# -----------------------------------------------------------------------------
def analytical_j2_sdf(sigma, kappa=1.0):
    """
    An artificial test case, not J2 itself: a circle of radius kappa in the
    Euclidean (sigma_x, sigma_y) plane, chosen because its signed distance is
    known exactly. (J2's true yield surface is a circle in the deviatoric plane.)

    Here we work in a simplified 2-D stress space (sigma_x, sigma_y) and define
    the J2 surface as sqrt(sigma_x^2 + sigma_y^2) = kappa. This is a circle
    in stress space and a reasonable testbed for the SDF idea. (In real
    plasticity, the right object is the deviatoric stress -- but the structural
    pattern is identical.)

    SDF: phi(sigma) = sqrt(sigma_x^2 + sigma_y^2) - kappa
    """
    return torch.sqrt((sigma ** 2).sum(-1)) - kappa


def analytical_j2_flow_direction(sigma, kappa=1.0):
    """Outward normal of the J2 yield surface = sigma / |sigma|."""
    norm = sigma.norm(dim=-1, keepdim=True)
    return sigma / (norm + 1e-12)


# -----------------------------------------------------------------------------
# Multiply-layer MLP (Vlassis-Sun L6 slide 25)
# -----------------------------------------------------------------------------
class MultiplyLayer(nn.Module):
    """Element-wise self-multiplication: h^(n) = h^(n-1) * h^(n-1)."""
    def forward(self, x):
        return x * x


class MultiplyMLP(nn.Module):
    """
    Architecture: dense -> tanh -> multiply -> dense -> tanh -> multiply -> dense
    The multiply layers follow Vlassis & Sun's design for yield functions with
    useful second derivatives; the course does not compute a tangent from them.
    """
    def __init__(self, in_dim=2, hidden=32, out_dim=1):
        super().__init__()
        self.d1 = nn.Linear(in_dim, hidden)
        self.m1 = MultiplyLayer()
        self.d2 = nn.Linear(hidden, hidden)
        self.m2 = MultiplyLayer()
        self.d3 = nn.Linear(hidden, out_dim)

    def forward(self, sigma):
        x = self.d1(sigma); x = torch.tanh(x)
        x = self.m1(x)
        x = self.d2(x);    x = torch.tanh(x)
        x = self.m2(x)
        return self.d3(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Sobolev H1 loss for the SDF
# -----------------------------------------------------------------------------
def sobolev_loss(model, sigma, phi_true, flow_true, alpha=1.0):
    sigma = sigma.clone().requires_grad_(True)
    phi_pred = model(sigma)
    L_data = ((phi_pred - phi_true) ** 2).mean()

    grad_phi = torch.autograd.grad(phi_pred.sum(), sigma, create_graph=True)[0]
    grad_phi_norm = grad_phi / (grad_phi.norm(dim=-1, keepdim=True) + 1e-8)
    L_grad = ((grad_phi_norm - flow_true) ** 2).mean()
    return L_data + alpha * L_grad, L_data, L_grad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa", type=float, default=1.0, help="Yield stress (J2 SDF radius)")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("levelset_yield_surface.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Sample points in stress space ---
    # Mix of: on-surface, near-surface, far-from-surface
    n = args.n_samples
    sigma_on = torch.zeros(n // 4, 2)
    theta_on = torch.rand(n // 4) * 2 * np.pi
    sigma_on[:, 0] = args.kappa * torch.cos(theta_on)
    sigma_on[:, 1] = args.kappa * torch.sin(theta_on)

    sigma_near = (torch.rand(n // 2, 2) - 0.5) * 2 * 1.5 * args.kappa
    sigma_far = (torch.rand(n // 4, 2) - 0.5) * 2 * 3 * args.kappa
    sigma_all = torch.cat([sigma_on, sigma_near, sigma_far], dim=0)

    phi_true = analytical_j2_sdf(sigma_all, kappa=args.kappa)
    flow_true = analytical_j2_flow_direction(sigma_all, kappa=args.kappa)

    # --- 2. Build model ---
    model = MultiplyMLP(in_dim=2, hidden=args.hidden, out_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 3. Train ---
    history = {"data": [], "grad": [], "total": []}
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss, L_d, L_g = sobolev_loss(model, sigma_all, phi_true, flow_true, args.alpha)
        loss.backward()
        optimizer.step()
        history["data"].append(L_d.item())
        history["grad"].append(L_g.item())
        history["total"].append(loss.item())
        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}  total={loss.item():.4e}  data={L_d.item():.4e}  grad={L_g.item():.4e}")

    # --- 4. Visualize on a 2-D grid ---
    grid = np.linspace(-2 * args.kappa, 2 * args.kappa, 200)
    XX, YY = np.meshgrid(grid, grid)
    sigma_grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32))
    with torch.no_grad():
        phi_grid = model(sigma_grid).numpy().reshape(XX.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cs = axes[0].contourf(XX, YY, phi_grid, levels=20, cmap="RdBu_r", vmin=-args.kappa, vmax=args.kappa)
    axes[0].contour(XX, YY, phi_grid, levels=[0], colors="k", linewidths=2)        # learned yield surface
    # Overlay analytical surface as dashed
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(args.kappa * np.cos(theta_circle), args.kappa * np.sin(theta_circle), "g--", linewidth=2, label="analytical")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("sigma_x"); axes[0].set_ylabel("sigma_y")
    axes[0].set_title("Learned SDF (color) and yield surface (black contour)\nvs analytical (green dashed)")
    axes[0].legend()
    fig.colorbar(cs, ax=axes[0], fraction=0.046, pad=0.04)

    # Error map
    phi_true_grid = np.sqrt(XX ** 2 + YY ** 2) - args.kappa
    err = phi_grid - phi_true_grid
    cs = axes[1].contourf(XX, YY, err, levels=20, cmap="seismic", vmin=-0.1, vmax=0.1)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("sigma_x"); axes[1].set_ylabel("sigma_y")
    axes[1].set_title("SDF error (learned - analytical)")
    fig.colorbar(cs, ax=axes[1], fraction=0.046, pad=0.04)

    # Loss curve
    axes[2].semilogy(history["data"], label="L_data")
    axes[2].semilogy(history["grad"], label="L_grad")
    axes[2].semilogy(history["total"], label="L_total")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("loss")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, which="both")
    axes[2].set_title("Sobolev H1 loss components")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nMax SDF error: {np.abs(err).max():.4e}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
