"""
mlp_constitutive.py
====================

Module 1 / Lab 1 — Train a PyTorch MLP to learn sigma(epsilon) for 2D plane-strain
linear isotropic elasticity. Input: (eps_xx, eps_yy, gamma_xy) in Voigt form.
Output: (sigma_xx, sigma_yy, sigma_xy) in Voigt form.

Usage
-----
    python mlp_constitutive.py \
        --data data/single_element.csv \
        --epochs 5000 \
        --hidden 32 \
        --layers 4 \
        --out checkpoints/mlp_constitutive.pt

The script:
  1. Loads the CSV produced by generate_data.py.
  2. Splits into train/val/test by random shuffle (with fixed seed).
  3. Standardizes inputs and outputs (mean 0, std 1) using train-set statistics.
  4. Trains an MLP with Adam.
  5. Reports MSE on each split + the implied 3x3 stiffness matrix at zero
     strain (numerical Jacobian by finite differences).
  6. Saves the model + scalers to a single .pt checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """A simple feed-forward network with configurable depth and width."""

    def __init__(self, in_dim: int = 3, out_dim: int = 3, hidden: int = 32, layers: int = 4):
        super().__init__()
        modules: list[nn.Module] = []
        widths = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for i in range(len(widths) - 1):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                modules.append(nn.Tanh())  # tanh has nonzero 2nd derivative -- useful for Sobolev training later
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
INPUT_COLS = ["eps_xx", "eps_yy", "gamma_xy"]
OUTPUT_COLS = ["sigma_xx", "sigma_yy", "sigma_xy"]


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df[INPUT_COLS].to_numpy(dtype=np.float32)
    Y = df[OUTPUT_COLS].to_numpy(dtype=np.float32)
    return X, Y


def split_data(
    X: np.ndarray, Y: np.ndarray, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return {
        "train": (X[train_idx], Y[train_idx]),
        "val": (X[val_idx], Y[val_idx]),
        "test": (X[test_idx], Y[test_idx]),
    }


def standardize(
    X_train: np.ndarray, X_other: list[np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray], list[np.ndarray]]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_other_std = [(X - mu) / sigma for X in X_other]
    return (mu, sigma), [X_train_std, *X_other_std]


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    epochs: int,
    lr: float,
    log_every: int = 100,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred, Y_train)
        loss.backward()
        optimizer.step()
        history["train"].append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()
        history["val"].append(val_loss)

        if epoch % log_every == 0:
            print(f"  epoch {epoch:5d}: train={loss.item():.4e}  val={val_loss:.4e}")

    return history


# -----------------------------------------------------------------------------
# Analysis: extract the linear part of the trained map
# -----------------------------------------------------------------------------
def implied_stiffness(model: nn.Module, scalers: dict) -> np.ndarray:
    """At eps = 0, the Jacobian d sigma / d eps is the small-strain stiffness matrix.

    The trained MLP works on standardized inputs / outputs, so we have to
    un-standardize the Jacobian.
    """
    model.eval()
    mu_x, sigma_x = scalers["mu_x"], scalers["sigma_x"]
    mu_y, sigma_y = scalers["mu_y"], scalers["sigma_y"]

    x_zero_std = torch.tensor(((0.0 - mu_x) / sigma_x).astype(np.float32), requires_grad=True)
    y_std = model(x_zero_std)

    # Compute Jacobian d y_std / d x_std via autograd, then un-standardize.
    jacobian_std = torch.zeros(3, 3)
    for i in range(3):
        grad_i = torch.autograd.grad(y_std[0, i], x_zero_std, retain_graph=True, create_graph=False)[0]
        jacobian_std[i] = grad_i

    # d y / d x = d y_std / d x_std * (sigma_y / sigma_x)  (elementwise scaling)
    sigma_y_t = torch.tensor(sigma_y.astype(np.float32)).squeeze()    # shape (3,)
    sigma_x_t = torch.tensor(sigma_x.astype(np.float32)).squeeze()    # shape (3,)
    C = jacobian_std * sigma_y_t.unsqueeze(1) / sigma_x_t.unsqueeze(0)
    return C.detach().numpy()


def kirsch_analytical_C(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """Plane-strain isotropic elasticity 3x3 stiffness matrix in Voigt form (gamma_xy)."""
    factor = E / ((1 + nu) * (1 - 2 * nu))
    return factor * np.array(
        [
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2],
        ],
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV from generate_data.py")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mlp_constitutive.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load + split + standardize ---
    X, Y = load_data(args.data)
    print(f"Loaded {len(X)} samples from {args.data}")
    splits = split_data(X, Y, seed=args.seed)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]

    (mu_x, sigma_x), (X_tr_s, X_va_s, X_te_s) = standardize(X_tr, [X_va, X_te])
    (mu_y, sigma_y), (Y_tr_s, Y_va_s, Y_te_s) = standardize(Y_tr, [Y_va, Y_te])

    # --- Build + train ---
    model = MLP(in_dim=3, out_dim=3, hidden=args.hidden, layers=args.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params} parameters (architecture: {args.layers} layers, width {args.hidden})")

    history = train(
        model,
        torch.tensor(X_tr_s), torch.tensor(Y_tr_s),
        torch.tensor(X_va_s), torch.tensor(Y_va_s),
        epochs=args.epochs, lr=args.lr,
    )

    # --- Test-set evaluation ---
    model.eval()
    with torch.no_grad():
        Y_te_pred_std = model(torch.tensor(X_te_s)).numpy()
    Y_te_pred = Y_te_pred_std * sigma_y + mu_y      # un-standardize
    abs_err = np.abs(Y_te_pred - Y_te)
    print("\nTest-set MAE per component:")
    for col, e in zip(OUTPUT_COLS, abs_err.mean(axis=0)):
        print(f"  {col}: {e:.4e}")

    # --- Implied stiffness vs analytical ---
    scalers = {"mu_x": mu_x, "sigma_x": sigma_x, "mu_y": mu_y, "sigma_y": sigma_y}
    C_learned = implied_stiffness(model, scalers)
    C_truth = kirsch_analytical_C(E=1.0, nu=0.3)
    print("\nLearned stiffness (Voigt 3x3):")
    print(np.array2string(C_learned, precision=4))
    print("\nAnalytical Hooke's law stiffness:")
    print(np.array2string(C_truth, precision=4))
    print("\nElement-wise relative error:")
    rel_err = np.abs(C_learned - C_truth) / np.maximum(np.abs(C_truth), 1e-8)
    print(np.array2string(rel_err, precision=4))

    # --- Save checkpoint ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scalers": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scalers.items()},
            "history": history,
            "args": vars(args),
        },
        args.out,
    )
    print(f"\nCheckpoint written to {args.out}")

    # --- Plot loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history["train"], label="train")
    ax.semilogy(history["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (standardized)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = args.out.with_suffix(".png")
    fig.savefig(plot_path, dpi=120)
    print(f"Loss curve written to {plot_path}")


if __name__ == "__main__":
    main()
