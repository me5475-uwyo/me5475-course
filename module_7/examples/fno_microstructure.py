"""
fno_microstructure.py
=====================

Module 7 / Lab 7 stretch task (Task 5) -- a 2-D Fourier Neural Operator for the Lab 6
microstructure-to-stress-field problem, set up so its result can be compared with the Lab 6
U-Net on the same cells.

Comparable by construction with module_6/examples/unet_microstructure_to_stress.py:
  - same data file and keys: images (N, 64, 64), stress_fields (N, 3, 64, 64)
  - same split: split_data(seed=42) -> 70 % train, 15 % validation, 15 % test (identical copy below);
    this holds for the same dataset file and the same --seed as the U-Net run (the script warns if the
    U-Net checkpoint was trained with another seed; --seed changes the split as well as the initialization)
  - same per-component standardization of the stresses (training-set mean and std)
  - same loss (MSE on standardized stresses), optimizer (AdamW), epochs (200), batch size (16)
  - the weights of the best validation epoch are kept
  - the same test metric in physical units: relative L2 = ||pred - true|| / ||true|| over the whole
    test set, for the full field and per component (sigma_xx, sigma_yy, sigma_xy)

FNO details: the cells are NOT periodic (Lab 6 uses mixed boundary conditions), so the lifted fields
are zero-padded by --pad points on the right and bottom before the Fourier layers and cropped back
after them (Li et al. 2021 pad non-periodic problems the same way). Two coordinate channels (cell
centers) are appended to the image. The spectral layer keeps the low positive and the low negative
row frequencies (rfft2 halves the column frequencies already).

Usage (from a folder holding this script, the Lab 6 field dataset and, optionally, your U-Net):
    python fno_microstructure.py --data rve_dataset.npz --out fno_micro.pt
    python fno_microstructure.py --data rve_dataset.npz --out fno_micro.pt \\
        --unet-ckpt unet_stress.pt          # also evaluates the U-Net and draws it in the figure
On ARCC:  sbatch operator_train.sbatch fno-micro --data rve_dataset.npz --unet-ckpt unet_stress.pt

The U-Net comparison imports UNet from unet_microstructure_to_stress.py, so copy that file next to
this one. Uses a GPU when the job has one (the printout's "Device:" line says which ran).
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

COMPONENTS = ["sigma_xx", "sigma_yy", "sigma_xy"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="Lab 6 dataset with stress_fields")
    p.add_argument("--epochs", type=int, default=200, help="Passes over all training cells")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes", type=int, default=12, help="Fourier modes kept in each direction")
    p.add_argument("--width", type=int, default=32, help="channels inside the Fourier layers")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--pad", type=int, default=8, help="zero padding (points) for the non-periodic cell")
    p.add_argument("--unet-ckpt", type=Path, default=None, help="optional Lab 6 U-Net checkpoint to compare")
    p.add_argument("--out", type=Path, default=Path("fno_micro.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_data(images, fields, seed=42, val_frac=0.15, test_frac=0.15):
    """Identical to split_data in unet_microstructure_to_stress.py: same seed, same cells."""
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


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: keep the low (positive and negative) row frequencies and the
    low column frequencies, multiply them by learned complex weights, transform back."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.out_c = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(   # positive row frequencies
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(   # negative row frequencies
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, in_c, H, W)
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                                     # (B, in_c, H, W//2 + 1)
        out_ft = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m_x = min(self.modes_x, H // 2)
        m_y = min(self.modes_y, W // 2 + 1)
        out_ft[:, :, :m_x, :m_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m_x, :m_y], self.weights1[:, :, :m_x, :m_y])
        out_ft[:, :, -m_x:, :m_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m_x:, :m_y], self.weights2[:, :, :m_x, :m_y])
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, modes=12, width=32, n_layers=4, pad=8):
        super().__init__()
        self.pad = pad
        self.lift = nn.Linear(in_channels + 2, width)   # +2 coordinate channels
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.spatial = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, 1, H, W) -- the microstructure image
        B, _, H, W = x.shape
        rows = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H   # cell centers
        cols = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W
        g_row, g_col = torch.meshgrid(rows, cols, indexing="ij")
        grid = torch.stack([g_row, g_col], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)                 # (B, 3, H, W)
        x = self.lift(x.permute(0, 2, 3, 1))            # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)                       # (B, width, H, W)
        if self.pad > 0:
            x = F.pad(x, [0, self.pad, 0, self.pad])    # the cell is not periodic: pad, then crop
        for i, (sp, sl) in enumerate(zip(self.spectral, self.spatial)):
            x = sp(x) + sl(x)
            if i < len(self.spectral) - 1:
                x = self.activation(x)
        if self.pad > 0:
            x = x[..., :H, :W]
        x = x.permute(0, 2, 3, 1)                       # (B, H, W, width)
        x = self.activation(self.project1(x))
        return self.project2(x).permute(0, 3, 1, 2)     # (B, 3, H, W)


def n_real_parameters(model):
    """Count real numbers: a complex weight is two (its real and imaginary parts)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def rel_l2(pred, true):
    """Relative L2 over the whole test set (full field and per component), physical units."""
    err = pred - true
    full = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12))
    per = {name: float(np.linalg.norm(err[:, k]) / (np.linalg.norm(true[:, k]) + 1e-12))
           for k, name in enumerate(COMPONENTS)}
    return full, per


def unet_predictions(ckpt_path, X_te_t, device, split_seed):
    """Load a Lab 6 U-Net checkpoint and predict the test cells in physical units."""
    from unet_microstructure_to_stress import UNet    # copy that file next to this one
    # weights_only=False: the Lab 6 checkpoint stores its argparse values (Path objects) as well
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(ck["args"].get("seed", 42)) != split_seed:
        print(f"WARNING: the U-Net was trained with --seed {ck['args'].get('seed')}, this run uses "
              f"--seed {split_seed}. The split seeds differ: these evaluation cells may have been used to train "
              f"or validate the U-Net, so this is not a comparison on a common held-out test set. Use the same "
              f"dataset ordering and split seed for both models.")
    base = int(ck["args"].get("base_channels", 32))
    unet = UNet(base=base).to(device)
    unet.load_state_dict(ck["model_state"])
    unet.eval()
    mu = np.array(ck["scalers"]["mu_y"], dtype=np.float32)
    sd = np.array(ck["scalers"]["sd_y"], dtype=np.float32)
    with torch.no_grad():
        return unet(X_te_t).cpu().numpy() * sd + mu


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    if "stress_fields" not in data.files:
        raise RuntimeError("This dataset has no stress_fields. Generate it with --save-stress-fields (Lab 6 Task 5a).")
    images = data["images"].astype(np.float32)             # (N, 64, 64)
    fields = data["stress_fields"].astype(np.float32)      # (N, 3, 64, 64)
    print(f"Loaded {len(images)} samples; field shape {fields.shape}")

    splits = split_data(images, fields, seed=args.seed)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]
    print(f"Split: {len(X_tr)} train, {len(X_va)} validation, {len(X_te)} test (the U-Net's split)")

    mu_y = Y_tr.mean(axis=(0, 2, 3), keepdims=True)
    sd_y = Y_tr.std(axis=(0, 2, 3), keepdims=True); sd_y[sd_y == 0] = 1
    Y_tr_n = (Y_tr - mu_y) / sd_y
    Y_va_n = (Y_va - mu_y) / sd_y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    X_tr_t = torch.tensor(X_tr).unsqueeze(1).to(device); Y_tr_t = torch.tensor(Y_tr_n).to(device)
    X_va_t = torch.tensor(X_va).unsqueeze(1).to(device); Y_va_t = torch.tensor(Y_va_n).to(device)
    X_te_t = torch.tensor(X_te).unsqueeze(1).to(device)

    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.n_layers, pad=args.pad).to(device)
    print(f"FNO2d parameters: {n_real_parameters(model)} (real numbers; a complex weight counts twice)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        # One epoch = one pass over all training cells, in shuffled mini-batches
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        epoch_loss = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr_t[b]), Y_tr_t[b])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b)
        epoch_loss /= len(X_tr_t)
        history["train"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), Y_va_t).item()
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train={epoch_loss:.4e} (epoch mean)  val={val_loss:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test metric in physical units, the same definition as the U-Net script
    model.eval()
    with torch.no_grad():
        Y_te_pred = model(X_te_t).cpu().numpy() * sd_y + mu_y
    fno_full, fno_per = rel_l2(Y_te_pred, Y_te)
    print(f"FNO test relative L2 error (full field): {100 * fno_full:.2f}%")
    for name, v in fno_per.items():
        print(f"  {name}: {100 * v:.2f}%")

    unet_pred, unet_full, unet_per = None, None, None
    if args.unet_ckpt is not None:
        unet_pred = unet_predictions(args.unet_ckpt, X_te_t, device, args.seed)
        unet_full, unet_per = rel_l2(unet_pred, Y_te)
        print(f"U-Net test relative L2 error on the same test cells (full field): {100 * unet_full:.2f}%")
        for name, v in unet_per.items():
            print(f"  {name}: {100 * v:.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        # CPU copies and plain types, so the checkpoint loads anywhere with torch.load
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "scalers": {"mu_y": mu_y.tolist(), "sd_y": sd_y.tolist()},
        "history": history,
        "test_rel_l2": fno_full,
        "test_rel_l2_per_channel": fno_per,
        "unet_test_rel_l2": unet_full,
        "unet_test_rel_l2_per_channel": unet_per,
    }, args.out)
    print(f"Saved {args.out}")

    # Figure: three test cells (the first three of the test split), sigma_xx
    n_show = min(3, len(X_te))
    if n_show == 0:
        print("Skipping the figure: the test set is empty")
        return
    cols = ["Microstructure", "MOOSE sigma_xx (truth)", "FNO sigma_xx"]
    cols.append("U-Net sigma_xx" if unet_pred is not None else "FNO - truth (sigma_xx)")
    fig, axes = plt.subplots(n_show, 4, figsize=(15, 3.6 * n_show), squeeze=False)
    for r in range(n_show):
        truth = Y_te[r, 0]
        vmin, vmax = truth.min(), truth.max()
        panels = [(X_te[r], "gray", None, None), (truth, "viridis", vmin, vmax),
                  (Y_te_pred[r, 0], "viridis", vmin, vmax)]
        if unet_pred is not None:
            panels.append((unet_pred[r, 0], "viridis", vmin, vmax))
        else:
            d = Y_te_pred[r, 0] - truth
            dm = float(np.abs(d).max()) or 1.0
            panels.append((d, "seismic", -dm, dm))
        for c, (img, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(cols[c] if r == 0 else "", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c > 0:
                plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=120)
    print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
