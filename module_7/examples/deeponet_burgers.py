"""
deeponet_burgers.py
====================

Module 7 / Lab 7 -- DeepONet for 1-D Burgers' equation.

Learns the operator G: u_0(x) -> u(x, T=1) for the 1-D viscous Burgers' PDE.

Architecture:
  Branch network: takes u_0 sampled at N grid points -> K coefficients
  Trunk network: takes a query position y -> K basis functions
  Output: dot product of branch and trunk + bias

Usage
-----
    python deeponet_burgers.py --data burgers_data.npz --epochs 500 --out deeponet.pt
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
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=500, help="Passes over all training functions")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--K", type=int, default=128, help="Number of branch/trunk basis modes")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-query", type=int, default=64, help="Query points per training pair")
    p.add_argument("--out", type=Path, default=Path("deeponet_burgers.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class DeepONet(nn.Module):
    def __init__(self, m_sensors: int, K: int, hidden: int, depth: int):
        super().__init__()
        # Branch
        branch_layers = [nn.Linear(m_sensors, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            branch_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        branch_layers.append(nn.Linear(hidden, K))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk (1-D input: y)
        trunk_layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            trunk_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        trunk_layers.append(nn.Linear(hidden, K))
        self.trunk = nn.Sequential(*trunk_layers)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y):
        # u_sensors: (B, m)
        # y: (B, Q, 1)
        b = self.branch(u_sensors)        # (B, K)
        t = self.trunk(y)                  # (B, Q, K)
        return (b.unsqueeze(1) * t).sum(dim=-1) + self.bias


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    u0 = data["u0"]      # (N_samples, N_grid)
    uT = data["uT"]      # (N_samples, N_grid)
    x_grid = data["x_grid"]
    N_grid = u0.shape[-1]

    # Train/test split
    n_total = len(u0)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_total)
    n_test = int(0.2 * n_total)
    test_idx = idx[:n_test]; train_idx = idx[n_test:]

    u0_tr = torch.tensor(u0[train_idx])
    uT_tr = torch.tensor(uT[train_idx])
    u0_te = torch.tensor(u0[test_idx])
    uT_te = torch.tensor(uT[test_idx])
    x_t = torch.tensor(x_grid).unsqueeze(-1)         # (N_grid, 1)

    model = DeepONet(m_sensors=N_grid, K=args.K, hidden=args.hidden, depth=args.depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepONet parameters: {n_params}")

    history = {"train": [], "test": []}
    for epoch in range(args.epochs):
        # One epoch = one pass over all training functions, in shuffled mini-batches
        model.train()
        perm = torch.randperm(len(u0_tr))
        for start in range(0, len(u0_tr), args.batch_size):
            idx_b = perm[start : start + args.batch_size]
            u0_batch = u0_tr[idx_b]                                        # (B, N_grid)
            uT_batch = uT_tr[idx_b]                                        # (B, N_grid)
            # Sub-sample query positions for this batch
            q_idx = torch.randint(0, N_grid, (args.n_query,))
            y_batch = x_t[q_idx].unsqueeze(0).expand(len(idx_b), -1, -1)  # (B, Q, 1)
            target = uT_batch[:, q_idx]                                    # (B, Q)

            optimizer.zero_grad()
            pred = model(u0_batch, y_batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        history["train"].append(loss.item())

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                y_full = x_t.unsqueeze(0).expand(len(u0_te), -1, -1)
                pred_te = model(u0_te, y_full)
                test_loss = loss_fn(pred_te, uT_te).item()
            history["test"].append((epoch, test_loss))
            # Relative L2
            rel_l2 = ((pred_te - uT_te) ** 2).sum().sqrt() / (uT_te ** 2).sum().sqrt()
            print(f"epoch {epoch:4d}  train={loss.item():.4e}  test={test_loss:.4e}  rel-L2={rel_l2.item():.4%}")

    # Final eval
    model.eval()
    with torch.no_grad():
        y_full = x_t.unsqueeze(0).expand(len(u0_te), -1, -1)
        pred_te = model(u0_te, y_full)
    rel_l2 = ((pred_te - uT_te) ** 2).sum().sqrt() / (uT_te ** 2).sum().sqrt()
    print(f"\nFinal test relative L2 error: {rel_l2.item():.4%}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "args": vars(args), "rel_l2": float(rel_l2)}, args.out)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax_i, ax in enumerate(axes.flat):
        sample_i = ax_i * (len(u0_te) // 4)
        ax.plot(x_grid, u0_te[sample_i].numpy(), "k--", alpha=0.6, label="initial u_0(x)")
        ax.plot(x_grid, uT_te[sample_i].numpy(), "b-", linewidth=2, label="true u(x, T=1) (spectral solver)")
        ax.plot(x_grid, pred_te[sample_i].numpy(), "r--", linewidth=1.5, label="DeepONet prediction")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Test sample {sample_i}")
    fig.suptitle(f"DeepONet 1-D Burgers — rel L2 = {rel_l2.item():.4%}")
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=120)
    print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
