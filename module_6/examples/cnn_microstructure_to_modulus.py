"""
cnn_microstructure_to_modulus.py
=================================

Module 6 / Lab 6 main task -- train a CNN that predicts the apparent
modulus from a 64x64 binary microstructure image.

Architecture: 4 conv blocks (16, 32, 64, 128 channels; conv, batch norm, ReLU, max pool)
+ global average pool + linear regression head. 97,761 parameters at the defaults.

Usage
-----
    python cnn_microstructure_to_modulus.py \\
        --data rve_dataset.npz \\
        --epochs 200 \\
        --batch-size 32 \\
        --out checkpoints/cnn_modulus.pt
"""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("checkpoints/cnn_modulus.pt"))
    p.add_argument("--seed", type=int, default=42,
                   help="model initialization and minibatch order only")
    p.add_argument("--split-seed", type=int, default=42,
                   help="the train/val/test partition; keep it fixed for every run you compare")
    return p.parse_args()


class MicrostructureCNN(nn.Module):
    def __init__(self, base_channels: int = 16, depth: int = 4):
        super().__init__()
        layers = []
        in_c = 1
        for i in range(depth):
            out_c = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_c, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_data(images, moduli, seed=42, val_frac=0.15, test_frac=0.15):
    n = len(images)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return {
        "train": (images[train_idx], moduli[train_idx]),
        "val":   (images[val_idx],   moduli[val_idx]),
        "test":  (images[test_idx],  moduli[test_idx]),
        "indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load data ---
    data = np.load(args.data, allow_pickle=True)
    images = data["images"].astype(np.float32)        # (N, 64, 64)
    moduli = data["moduli"].astype(np.float32)        # (N,)
    print(f"Loaded {len(images)} microstructures")
    print(f"Modulus range: {moduli.min():.3f} -- {moduli.max():.3f}")

    # --- Split ---
    # One fixed partition (--split-seed) for every run you compare; --seed only changes initialization
    # and minibatch order. Rerunning with another --seed keeps the same test rows.
    splits = split_data(images, moduli, seed=args.split_seed)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]
    print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}  (split seed {args.split_seed})")
    test_ids = splits["indices"]["test"]
    print(f"Test rows: first five {test_ids[:5].tolist()} (identical for every run with this split seed)")

    # --- Normalize moduli (zero mean, unit std on training) ---
    mu_y = Y_tr.mean()
    sd_y = Y_tr.std()
    Y_tr_n = (Y_tr - mu_y) / sd_y
    Y_va_n = (Y_va - mu_y) / sd_y
    Y_te_n = (Y_te - mu_y) / sd_y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    X_tr_t = torch.tensor(X_tr).unsqueeze(1).to(device)  # (N, 1, 64, 64)
    X_va_t = torch.tensor(X_va).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_te).unsqueeze(1).to(device)
    Y_tr_t = torch.tensor(Y_tr_n).to(device)
    Y_va_t = torch.tensor(Y_va_n).to(device)
    Y_te_t = torch.tensor(Y_te_n).to(device)

    # --- Model + optimizer ---
    model = MicrostructureCNN(base_channels=args.base_channels, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # --- Train ---
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        # Mini-batch training
        idx = torch.randperm(len(X_tr_t), device=device)
        epoch_loss = 0.0
        for i in range(0, len(idx), args.batch_size):
            batch_idx = idx[i : i + args.batch_size]
            optimizer.zero_grad()
            Y_pred = model(X_tr_t[batch_idx])
            loss = loss_fn(Y_pred, Y_tr_t[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= len(X_tr_t)
        history["train"].append(epoch_loss)

        # Val
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), Y_va_t).item()
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train={epoch_loss:.4e}  val={val_loss:.4e}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test set evaluation ---
    model.eval()
    with torch.no_grad():
        Y_te_pred_n = model(X_te_t).cpu().numpy()
    Y_te_pred = Y_te_pred_n * sd_y + mu_y
    mae = np.abs(Y_te_pred - Y_te).mean()
    mean_modulus = Y_te.mean()
    relative_mae = mae / mean_modulus
    print(f"\nTest set MAE:           {mae:.4e}")
    print(f"Test set MAE / mean(E): {relative_mae:.4%}")

    # --- Save ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        # CPU copies, so the checkpoint loads on a login node or CPU job as well as on a GPU
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": {"seed": args.split_seed, "test_idx": test_ids.tolist()},
        "device": str(device),
        "scalers": {"mu_y": float(mu_y), "sd_y": float(sd_y)},
        "history": history,
        "test_mae": float(mae),
        "test_relative_mae": float(relative_mae),
    }, args.out)
    print(f"Saved {args.out}")

    # --- Parity plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(Y_te, Y_te_pred, s=20, alpha=0.6)
    lo, hi = Y_te.min(), Y_te.max()
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("MOOSE E_app")
    axes[0].set_ylabel("CNN-predicted E_app")
    axes[0].set_title(f"Test parity (MAE = {mae:.4f}, rel = {relative_mae:.2%})")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(history["train"], label="train")
    axes[1].semilogy(history["val"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE (standardized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].set_title("Training history")

    fig.tight_layout()
    fig_path = args.out.with_suffix(".png")
    fig.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
