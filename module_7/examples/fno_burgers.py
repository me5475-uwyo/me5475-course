"""
fno_burgers.py
==============

Module 7 / Lab 7 -- Fourier Neural Operator for 1-D Burgers' equation.

Following Li et al. ICLR 2021. The architecture:
  lift -> [Fourier-conv + spatial-conv + GELU] x n_layers -> project

Key feature: the same weights can be applied on a grid of any size N (the spectral
layer acts on the first `modes` Fourier coefficients, which mean the same thing at any
N). Whether accuracy carries over when you train at N=128 and evaluate at N=256 or 512
is what the Lab 7 cross-resolution test measures.

Usage
-----
    python fno_burgers.py --data burgers_data.npz --epochs 500 --out fno.pt
    python fno_burgers.py --data burgers_data.npz --modes 16 --width 64
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
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("fno_burgers.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SpectralConv1d(nn.Module):
    """1-D spectral convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Complex-valued learnable weights for each Fourier mode
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, in_c, N) -- spatial
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                      # (B, in_c, N//2 + 1)
        out_ft = torch.zeros(B, self.out_c, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        # Multiply only the first `modes` Fourier modes
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = torch.einsum(
            "bik,iok->bok", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, modes: int = 16,
                 width: int = 64, n_layers: int = 4):
        super().__init__()
        # Lift: (in_dim + 1 for position) -> width
        self.lift = nn.Linear(in_dim + 1, width)
        # Spectral + spatial mix
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.spatial = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        # Project
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, N) -- input function values at N grid points
        B, N = x.shape
        # grid positions x_j = j / N on [0, 1), as in the data's x_grid
        pos = (torch.arange(N, device=x.device, dtype=x.dtype) / N).unsqueeze(0).expand(B, -1)
        x = torch.stack([x, pos], dim=-1)                      # (B, N, 2)
        x = self.lift(x)                                       # (B, N, width)
        x = x.permute(0, 2, 1)                                  # (B, width, N)
        for sp, sl in zip(self.spectral, self.spatial):
            x = self.activation(sp(x) + sl(x))
        x = x.permute(0, 2, 1)                                  # (B, N, width)
        x = self.activation(self.project1(x))
        return self.project2(x).squeeze(-1)                     # (B, N)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    u0 = data["u0"]
    uT = data["uT"]
    x_grid = data["x_grid"]
    N_grid = u0.shape[-1]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(u0))
    n_test = int(0.2 * len(u0))
    test_idx = idx[:n_test]; train_idx = idx[n_test:]

    u0_tr = torch.tensor(u0[train_idx])
    uT_tr = torch.tensor(uT[train_idx])
    u0_te = torch.tensor(u0[test_idx])
    uT_te = torch.tensor(uT[test_idx])

    model = FNO1d(in_dim=1, out_dim=1, modes=args.modes, width=args.width, n_layers=args.n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Count real numbers, as for the DeepONet: each complex spectral weight is two (real and imaginary parts).
    n_entries = sum(p.numel() for p in model.parameters())
    n_params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(f"FNO parameters: {n_params} real numbers ({n_entries} tensor entries; a complex weight counts twice)")

    history = {"train": [], "test": []}
    for epoch in range(args.epochs):
        # One epoch = one pass over all training functions, in shuffled mini-batches
        model.train()
        perm = torch.randperm(len(u0_tr))
        for start in range(0, len(u0_tr), args.batch_size):
            idx_b = perm[start : start + args.batch_size]
            u0_batch = u0_tr[idx_b]
            uT_batch = uT_tr[idx_b]

            optimizer.zero_grad()
            pred = model(u0_batch)
            loss = loss_fn(pred, uT_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        history["train"].append(loss.item())

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred_te = model(u0_te)
                test_loss = loss_fn(pred_te, uT_te).item()
                rel_l2 = ((pred_te - uT_te) ** 2).sum().sqrt() / (uT_te ** 2).sum().sqrt()
            history["test"].append((epoch, test_loss))
            print(f"epoch {epoch:4d}  train={loss.item():.4e}  test={test_loss:.4e}  rel-L2={rel_l2.item():.4%}")

    # Final eval
    model.eval()
    with torch.no_grad():
        pred_te = model(u0_te)
    rel_l2 = ((pred_te - uT_te) ** 2).sum().sqrt() / (uT_te ** 2).sum().sqrt()
    print(f"\nFinal test relative L2 error: {rel_l2.item():.4%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # plain types only (paths as strings), so torch.load works with its default weights_only=True
    saved_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    torch.save({"model_state": model.state_dict(), "args": saved_args, "rel_l2": float(rel_l2)}, args.out)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax_i, ax in enumerate(axes.flat):
        sample_i = ax_i * (len(u0_te) // 4)
        ax.plot(x_grid, u0_te[sample_i].numpy(), "k--", alpha=0.6, label="initial u_0(x)")
        ax.plot(x_grid, uT_te[sample_i].numpy(), "b-", linewidth=2, label="true u(x, T=1)")
        ax.plot(x_grid, pred_te[sample_i].numpy(), "g--", linewidth=1.5, label="FNO prediction")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title(f"Test sample {sample_i}")
    fig.suptitle(f"FNO 1-D Burgers — rel L2 = {rel_l2.item():.4%}")
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=120)
    print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
