"""
lr_finder.py
============

Module 2 / Lecture 9 demo --- the LR-range-test (Smith, WACV 2017).

Sweeps the learning rate from a small value (1e-7) to a large value (1e-1)
geometrically across N training steps, recording the loss at each step.
Plots loss vs. learning rate on a log-x axis. The optimal LR is roughly
one order of magnitude below the rate at which loss first diverges.

Usage
-----
    python lr_finder.py --data ../../module_1/examples/data/single_element.csv \
                        --steps 100 \
                        --out lr_finder.png

The example runs on Lab 1's constitutive dataset by default but works on any
(X, Y) CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("../../module_1/examples/data/single_element.csv"))
    p.add_argument("--steps", type=int, default=100, help="Number of LRs to test (default: 100)")
    p.add_argument("--lr-min", type=float, default=1e-7)
    p.add_argument("--lr-max", type=float, default=1e-1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("lr_finder.png"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        widths = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        modules: list[nn.Module] = []
        for i in range(len(widths) - 1):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load data ---
    df = pd.read_csv(args.data)
    X = df[["eps_xx", "eps_yy", "gamma_xy"]].to_numpy(dtype=np.float32)
    Y = df[["sigma_xx", "sigma_yy", "sigma_xy"]].to_numpy(dtype=np.float32)
    # Standardize
    mu_x, sigma_x = X.mean(0, keepdims=True), X.std(0, keepdims=True); sigma_x[sigma_x == 0] = 1
    mu_y, sigma_y = Y.mean(0, keepdims=True), Y.std(0, keepdims=True); sigma_y[sigma_y == 0] = 1
    X = (X - mu_x) / sigma_x
    Y = (Y - mu_y) / sigma_y
    X_t = torch.tensor(X)
    Y_t = torch.tensor(Y)

    # --- LR schedule (geometric) ---
    lrs = np.geomspace(args.lr_min, args.lr_max, args.steps)
    losses: list[float] = []

    # --- Sweep ---
    model = MLP(in_dim=3, out_dim=3, hidden=args.hidden, layers=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[0])
    loss_fn = nn.MSELoss()
    best = float("inf")

    for step, lr in enumerate(lrs):
        # Set the LR for this step.
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        Y_pred = model(X_t)
        loss = loss_fn(Y_pred, Y_t)
        if not np.isfinite(loss.item()):
            print(f"Loss became NaN/Inf at LR = {lr:.2e}, step {step}")
            losses.append(float("inf"))
            break
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < best:
            best = loss.item()
        # Heuristic early stop: loss has blown up by 10x from the best -- we're past divergence.
        if loss.item() > 10 * best and step > 10:
            print(f"Loss diverging at LR = {lr:.2e}, step {step}; stopping sweep.")
            break

    losses_arr = np.array(losses[: len(losses)])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(lrs[: len(losses_arr)], losses_arr, "o-", markersize=3)
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("training loss after one step")
    ax.set_title("LR-range-test (Smith 2017)")
    ax.grid(True, alpha=0.3, which="both")

    # Suggest a value -- the LR ~10x smaller than where loss first goes above 2x its min.
    min_idx = int(np.argmin(losses_arr))
    if min_idx < len(losses_arr) - 1:
        diverge_idx = np.where(losses_arr > 2 * losses_arr[min_idx])[0]
        diverge_idx = diverge_idx[diverge_idx > min_idx]
        if len(diverge_idx) > 0:
            recommended_lr = lrs[diverge_idx[0]] / 10.0
            ax.axvline(recommended_lr, color="C2", linestyle="--", label=f"Suggested LR ≈ {recommended_lr:.2e}")
            ax.axvline(lrs[diverge_idx[0]], color="C3", linestyle=":", label=f"Divergence LR ≈ {lrs[diverge_idx[0]]:.2e}")
            ax.legend()
            print(f"\nSuggested learning rate: {recommended_lr:.3e}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
