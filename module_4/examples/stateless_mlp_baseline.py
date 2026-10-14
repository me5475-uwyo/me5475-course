"""
stateless_mlp_baseline.py
==========================

Module 4 / Lab 4 Task 2 -- The "wrong way" baseline.

Train a stateless MLP that takes only the current strain epsilon and predicts
the current stress sigma. Demonstrates -- experimentally -- that path-dependent
materials cannot be captured by stateless models, regardless of capacity.

This script ALSO trains a variant with explicit plastic strain input:
sigma = MLP(epsilon, eps_p). That variant DOES work, because eps_p IS the
internal variable. This shows that "stateless" only fails when q is hidden.

Usage
-----
    python stateless_mlp_baseline.py --data trajectories.npz
    python stateless_mlp_baseline.py --data trajectories.npz --include-plastic-strain
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="trajectories.npz from generate_plasticity_data.py")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--include-plastic-strain", action="store_true",
                   help="Feed eps_p in addition to epsilon (the 'correct' baseline that works)")
    p.add_argument("--out", type=Path, default=None,
                   help="Figure file (default: stateless_baseline.png; with --include-plastic-strain, "
                        "stateless_with_q.png)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers):
        super().__init__()
        widths = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        modules = []
        for i in range(len(widths) - 1):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def flatten_trajectories(eps_traj, sigma_traj, eps_p_traj, include_plastic):
    """Flatten the (N_traj, T, 3) arrays into (N_traj * T, *) for stateless training."""
    valid = ~np.isnan(eps_traj[..., 0])
    eps_flat = eps_traj[valid]                       # (M, 3)
    sigma_flat = sigma_traj[valid]
    if include_plastic:
        eps_p_flat = eps_p_traj[valid]
        X = np.concatenate([eps_flat, eps_p_flat], axis=-1)   # (M, 6)
    else:
        X = eps_flat                                          # (M, 3)
    Y = sigma_flat                                            # (M, 3)
    return X.astype(np.float32), Y.astype(np.float32)


def main():
    args = parse_args()
    if args.out is None:
        args.out = Path("stateless_with_q.png" if args.include_plastic_strain else "stateless_baseline.png")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    epsilon, sigma, eps_p = data["epsilon"], data["sigma"], data["eps_p"]
    print(f"Loaded {epsilon.shape[0]} trajectories of length up to {epsilon.shape[1]}")

    # Train/val split (by trajectory, not by step -- avoids leakage)
    n_traj = epsilon.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_traj)
    n_test = int(0.2 * n_traj)
    test_traj_idx = idx[:n_test]
    train_traj_idx = idx[n_test:]

    X_tr, Y_tr = flatten_trajectories(
        epsilon[train_traj_idx], sigma[train_traj_idx], eps_p[train_traj_idx],
        include_plastic=args.include_plastic_strain,
    )
    X_te, Y_te = flatten_trajectories(
        epsilon[test_traj_idx], sigma[test_traj_idx], eps_p[test_traj_idx],
        include_plastic=args.include_plastic_strain,
    )
    print(f"Train: {X_tr.shape[0]} (eps, sigma) pairs")
    print(f"Test:  {X_te.shape[0]} (eps, sigma) pairs")
    print(f"Input dim: {X_tr.shape[-1]}")

    # Standardize
    mu_x, sigma_x = X_tr.mean(0), X_tr.std(0); sigma_x[sigma_x == 0] = 1
    mu_y, sigma_y = Y_tr.mean(0), Y_tr.std(0); sigma_y[sigma_y == 0] = 1
    X_tr_s = (X_tr - mu_x) / sigma_x
    X_te_s = (X_te - mu_x) / sigma_x
    Y_tr_s = (Y_tr - mu_y) / sigma_y
    Y_te_s = (Y_te - mu_y) / sigma_y

    model = MLP(in_dim=X_tr.shape[-1], out_dim=3, hidden=args.hidden, layers=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr_s); Y_tr_t = torch.tensor(Y_tr_s)
    X_te_t = torch.tensor(X_te_s); Y_te_t = torch.tensor(Y_te_s)

    history = {"train": [], "test": []}
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        Y_pred = model(X_tr_t)
        loss = loss_fn(Y_pred, Y_tr_t)
        loss.backward()
        optimizer.step()
        history["train"].append(loss.item())

        model.eval()
        with torch.no_grad():
            te_loss = loss_fn(model(X_te_t), Y_te_t).item()
        history["test"].append(te_loss)
        if epoch % 200 == 0:
            print(f"epoch {epoch:5d}  train={loss.item():.4e}  test={te_loss:.4e}")

    # Plot: predicted vs true on the three held-out trajectories with the most
    # plastic flow (an elastic trajectory has no loop to miss)
    model.eval()
    flow = [np.nansum(np.abs(np.diff(eps_p[i], axis=0))) for i in test_traj_idx]
    plot_idx = test_traj_idx[np.argsort(flow)[::-1][:3]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_i, ax in enumerate(axes):
        traj_i = plot_idx[ax_i]
        eps_traj = epsilon[traj_i]
        sigma_traj = sigma[traj_i]
        eps_p_traj = eps_p[traj_i]
        # Get valid timesteps
        valid = ~np.isnan(eps_traj[..., 0])
        if args.include_plastic_strain:
            X_t = np.concatenate([eps_traj[valid], eps_p_traj[valid]], axis=-1)
        else:
            X_t = eps_traj[valid]
        X_t_s = (X_t - mu_x) / sigma_x
        with torch.no_grad():
            Y_p_s = model(torch.tensor(X_t_s.astype(np.float32))).numpy()
        Y_p = Y_p_s * sigma_y + mu_y
        # Plot sigma_xx vs eps_xx (hysteresis loop)
        ax.plot(eps_traj[valid, 0], sigma_traj[valid, 0], "k-", label="MOOSE truth", linewidth=2)
        ax.plot(eps_traj[valid, 0], Y_p[:, 0], "r--", label="MLP prediction", linewidth=1.5)
        ax.set_xlabel("epsilon_xx"); ax.set_ylabel("sigma_xx")
        ax.set_title(f"Trajectory {traj_i}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    title = "Stateless MLP {} -- {}".format(
        "WITH eps_p input" if args.include_plastic_strain else "(no eps_p)",
        "should track the cyclic loops" if args.include_plastic_strain else "FAILS to track cycles",
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nWrote {args.out}")
    final_test_rel_err = np.sqrt(history["test"][-1])
    print(f"Final test MSE (standardized): {history['test'][-1]:.4e}")
    print(f"Roughly: RMS error = {final_test_rel_err*100:.1f}% of the spread (std) of each stress component")


if __name__ == "__main__":
    main()
