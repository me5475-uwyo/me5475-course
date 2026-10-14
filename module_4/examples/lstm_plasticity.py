"""
lstm_plasticity.py
==================

Module 4 / Lab 4 Task 3 -- LSTM for path-dependent J2 plasticity.

Takes a strain trajectory epsilon(t) as input; outputs the corresponding stress
trajectory sigma(t). The LSTM's hidden state can carry the loading history that
the plastic internal variable q encodes; it is not q itself.

Compared to stateless_mlp_baseline.py: this architecture has a memory, so the
same strain can map to different stresses on different loading branches.

Usage
-----
    python lstm_plasticity.py --data trajectories.npz
    python lstm_plasticity.py --data trajectories.npz --hidden 64 --layers 3
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
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Trajectories per optimizer step (0 = all training trajectories at once)")
    p.add_argument("--out", type=Path, default=Path("lstm_plasticity.png"))
    p.add_argument("--save", type=Path, default=Path("lstm_plasticity.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class LSTMPlasticity(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, eps_seq):
        h, _ = self.lstm(eps_seq)
        return self.head(h)


def build_padded_batches(eps, sigma, batch_size, valid_mask):
    """
    Build batches where each trajectory has the same length (the trajectories
    in trajectories.npz are all roughly the same length, but we mask out
    NaN-padded steps).

    Returns dataloader-friendly tensors with shape (B, T, 3) for both eps and sigma,
    plus a mask (B, T) marking valid steps.
    """
    # eps, sigma: (N_traj, T_max, 3)
    n_traj = eps.shape[0]
    T_max = eps.shape[1]
    # Just convert to tensors and let the loss mask out invalid steps
    eps_t = torch.tensor(np.nan_to_num(eps, nan=0.0).astype(np.float32))
    sigma_t = torch.tensor(np.nan_to_num(sigma, nan=0.0).astype(np.float32))
    mask = torch.tensor(valid_mask.astype(np.float32))     # (N_traj, T_max)
    return eps_t, sigma_t, mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    epsilon, sigma = data["epsilon"], data["sigma"]
    valid = ~np.isnan(epsilon[..., 0])

    # Split trajectories into train/test
    n_traj = epsilon.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_traj)
    n_test = int(0.2 * n_traj)
    test_idx = idx[:n_test]; train_idx = idx[n_test:]

    # Standardize using train statistics on the valid (non-NaN) entries
    eps_train_valid = epsilon[train_idx][valid[train_idx]]
    sigma_train_valid = sigma[train_idx][valid[train_idx]]
    mu_e, sd_e = eps_train_valid.mean(0), eps_train_valid.std(0); sd_e[sd_e == 0] = 1
    mu_s, sd_s = sigma_train_valid.mean(0), sigma_train_valid.std(0); sd_s[sd_s == 0] = 1

    eps_std = (epsilon - mu_e) / sd_e
    sigma_std = (sigma - mu_s) / sd_s

    eps_tr, sigma_tr, mask_tr = build_padded_batches(
        eps_std[train_idx], sigma_std[train_idx], args.batch_size, valid[train_idx]
    )
    eps_te, sigma_te, mask_te = build_padded_batches(
        eps_std[test_idx], sigma_std[test_idx], args.batch_size, valid[test_idx]
    )

    model = LSTMPlasticity(in_dim=3, out_dim=3, hidden=args.hidden, num_layers=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def masked_mse(pred, target, mask):
        diff = (pred - target) ** 2
        diff = diff * mask.unsqueeze(-1)
        return diff.sum() / (mask.sum() * pred.shape[-1])

    # Mini-batches of whole trajectories (never of time steps), reshuffled each epoch.
    n_tr = eps_tr.shape[0]
    bs = n_tr if args.batch_size <= 0 else min(args.batch_size, n_tr)
    gen = torch.Generator().manual_seed(args.seed)
    history = {"train": [], "test": []}
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n_tr, generator=gen) if bs < n_tr else torch.arange(n_tr)
        ep_loss = 0.0
        for start in range(0, n_tr, bs):
            j = order[start:start + bs]
            optimizer.zero_grad()
            loss = masked_mse(model(eps_tr[j]), sigma_tr[j], mask_tr[j])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(j) / n_tr
        history["train"].append(ep_loss)

        model.eval()
        with torch.no_grad():
            sigma_pred_te = model(eps_te)
            te_loss = masked_mse(sigma_pred_te, sigma_te, mask_te).item()
        history["test"].append(te_loss)
        if epoch % 200 == 0:
            print(f"epoch {epoch:5d}  train={ep_loss:.4e}  test={te_loss:.4e}")

    # Visualize: predicted hysteresis vs truth for the three test trajectories
    # with the most plastic flow (the same three stateless_mlp_baseline.py plots)
    model.eval()
    n_plot = min(3, len(test_idx))
    eps_p = data["eps_p"]
    flow = [np.nansum(np.abs(np.diff(eps_p[i], axis=0))) for i in test_idx]
    plot_pos = np.argsort(flow)[::-1][:n_plot]           # positions within test_idx
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), squeeze=False)
    axes = axes[0]
    with torch.no_grad():
        sigma_pred = model(eps_te).numpy() * sd_s + mu_s
    for ax_i, ax in enumerate(axes):
        traj_i = test_idx[plot_pos[ax_i]]
        valid_i = valid[traj_i]
        # Pull the actual (un-standardized) trajectory
        eps_real = epsilon[traj_i, valid_i]
        sigma_real = sigma[traj_i, valid_i]
        sigma_p_i = sigma_pred[plot_pos[ax_i], valid_i]
        ax.plot(eps_real[:, 0], sigma_real[:, 0], "k-", label="MOOSE truth", linewidth=2)
        ax.plot(eps_real[:, 0], sigma_p_i[:, 0], "g--", label="LSTM prediction", linewidth=1.5)
        ax.set_xlabel("epsilon_xx"); ax.set_ylabel("sigma_xx")
        ax.set_title(f"Trajectory {traj_i}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("LSTM plasticity: hysteresis loops on held-out trajectories")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nWrote {args.out}")
    print(f"Final test MSE (standardized): {history['test'][-1]:.4e}")

    torch.save({"model_state": model.state_dict(),
                "mu_eps": mu_e.tolist(), "sd_eps": sd_e.tolist(),
                "mu_sigma": mu_s.tolist(), "sd_sigma": sd_s.tolist(),
                "args": vars(args)}, args.save)
    print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
