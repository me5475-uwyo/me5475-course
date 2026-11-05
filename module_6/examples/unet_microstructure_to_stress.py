"""
unet_microstructure_to_stress.py
=================================

Module 6 / Lab 6 stretch task -- U-Net that predicts the full 64x64 stress
field from a microstructure image.

Requires:
  - Dataset generated with --save-stress-fields (so stress_fields exists in .npz)
  - GPU recommended (the script uses one if the job has it; it prints the device)

Architecture: 3-level U-Net with skip connections. ~1.9M parameters (base=32).

Usage
-----
    python unet_microstructure_to_stress.py \\
        --data rve_dataset_with_fields.npz \\
        --epochs 200 \\
        --out checkpoints/unet_stress.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # headless: save figures without a display (SLURM nodes)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("checkpoints/unet_stress.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    3-level U-Net. Encoder: enc1 (64x64) -> enc2 (32x32) -> enc3 (16x16) -> bottleneck (8x8)
    Decoder: bottleneck -> up3 -> dec3 (16x16) -> up2 -> dec2 (32x32) -> up1 -> dec1 (64x64)
    Output: 3 channels for (sigma_xx, sigma_yy, sigma_xy)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = double_conv(in_channels, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.bottleneck = double_conv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def split_data(images, fields, seed=42, val_frac=0.15, test_frac=0.15):
    n = len(images)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return {
        "train": (images[train_idx], fields[train_idx]),
        "val":   (images[val_idx],   fields[val_idx]),
        "test":  (images[test_idx],  fields[test_idx]),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load
    data = np.load(args.data, allow_pickle=True)
    if "stress_fields" not in data.files:
        raise RuntimeError(
            "rve_dataset doesn't contain stress_fields. Re-generate with "
            "--save-stress-fields."
        )
    images = data["images"].astype(np.float32)             # (N, 64, 64)
    fields = data["stress_fields"].astype(np.float32)      # (N, 3, 64, 64)
    print(f"Loaded {len(images)} samples; field shape {fields.shape}")

    splits = split_data(images, fields, seed=args.seed)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]

    # Normalize per-channel stress
    mu_y = Y_tr.mean(axis=(0, 2, 3), keepdims=True)
    sd_y = Y_tr.std(axis=(0, 2, 3), keepdims=True); sd_y[sd_y == 0] = 1
    Y_tr_n = (Y_tr - mu_y) / sd_y
    Y_va_n = (Y_va - mu_y) / sd_y

    X_tr_t = torch.tensor(X_tr).unsqueeze(1)
    X_va_t = torch.tensor(X_va).unsqueeze(1)
    X_te_t = torch.tensor(X_te).unsqueeze(1)
    Y_tr_t = torch.tensor(Y_tr_n)
    Y_va_t = torch.tensor(Y_va_n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    X_tr_t = X_tr_t.to(device); Y_tr_t = Y_tr_t.to(device)
    X_va_t = X_va_t.to(device); Y_va_t = Y_va_t.to(device)
    X_te_t = X_te_t.to(device)

    model = UNet(base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
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

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), Y_va_t).item()
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train={epoch_loss:.4e}  val={val_loss:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Quantitative test-set metric: relative L2 error over the full field ---
    # (the Lab 6 deliverable; computed on the held-out test set in physical units)
    model.eval()
    with torch.no_grad():
        Y_te_pred = model(X_te_t).cpu().numpy() * sd_y + mu_y    # denormalize
    err = Y_te_pred - Y_te
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(Y_te) + 1e-12))
    per_channel = {
        name: float(np.linalg.norm(err[:, k]) / (np.linalg.norm(Y_te[:, k]) + 1e-12))
        for k, name in enumerate(["sigma_xx", "sigma_yy", "sigma_xy"])
    }
    print(f"Test relative L2 error (full field): {100 * rel_l2:.2f}%")
    for name, rk in per_channel.items():
        print(f"  {name}: {100 * rk:.2f}%")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        # CPU copies, so the checkpoint loads on a login node or CPU job as well as on a GPU
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "args": vars(args),
        "scalers": {"mu_y": mu_y.tolist(), "sd_y": sd_y.tolist()},
        "history": history,
        "test_rel_l2": rel_l2,
        "test_rel_l2_per_channel": per_channel,
    }, args.out)
    print(f"Saved {args.out}")

    # --- Visualize a few test predictions ---
    n_show = min(3, len(X_te_t))
    if n_show == 0:
        print(f"Skipping visualization: test set is empty")
    else:
        _viz_loop(model, X_te_t, Y_te, n_show, sd_y, mu_y, args)


def _viz_loop(model, X_te_t, Y_te, n_show, sd_y, mu_y, args):
    model.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(X_te_t), n_show, replace=False)
        X_samp = X_te_t[sample_idx]
        Y_pred_n = model(X_samp).cpu().numpy()
        Y_pred = Y_pred_n * sd_y + mu_y
    Y_truth = Y_te[sample_idx]

    fig, axes = plt.subplots(n_show, 4, figsize=(14, 10 * n_show // 3 + 1),
                             squeeze=False)
    for row in range(n_show):
        # Plot microstructure, true sigma_xx, predicted sigma_xx, error
        axes[row, 0].imshow(X_samp[row, 0].cpu().numpy(), cmap="gray")
        axes[row, 0].set_title(f"Microstructure (sample {sample_idx[row]})")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        true_sxx = Y_truth[row, 0]
        pred_sxx = Y_pred[row, 0]
        vmin, vmax = true_sxx.min(), true_sxx.max()
        cs = axes[row, 1].imshow(true_sxx, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("MOOSE sigma_xx (truth)")
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        plt.colorbar(cs, ax=axes[row, 1], fraction=0.046)
        cs = axes[row, 2].imshow(pred_sxx, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 2].set_title("U-Net sigma_xx (predicted)")
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        plt.colorbar(cs, ax=axes[row, 2], fraction=0.046)
        diff = pred_sxx - true_sxx
        d_max = max(abs(diff.min()), abs(diff.max()))
        cs = axes[row, 3].imshow(diff, cmap="seismic", vmin=-d_max, vmax=d_max)
        axes[row, 3].set_title("Predicted - Truth (sigma_xx)")
        axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])
        plt.colorbar(cs, ax=axes[row, 3], fraction=0.046)
    fig.tight_layout()
    fig_path = args.out.with_suffix(".png")
    fig.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()

