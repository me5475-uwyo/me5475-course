"""
sine_mlp.py
===========

Module 1 / Lecture 8 hands-on lab — train a small MLP to fit y = sin(pi x)
on [-1, 1]. The canonical "first PyTorch training run" of the course.

This is the script that students will type along with the instructor during
Lecture 8 and then submit as their first ARCC PyTorch job via train_mlp.sbatch
(suitably modified -- you can also use this script as a target for that
SLURM template by adjusting the srun line).

Usage
-----
    python sine_mlp.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(42)

    # 1. Generate training data: N points uniformly on [-1, 1], y = sin(pi x).
    N = 200
    x = torch.linspace(-1.0, 1.0, N).unsqueeze(1)
    y = torch.sin(torch.pi * x)

    # 2. Define a 4-layer MLP with width 32 and tanh activations.
    model = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )

    # 3. Optimizer and loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4. Train.
    losses: list[float] = []
    for epoch in range(5000):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}: loss = {loss.item():.6e}")

    # 5. Evaluate on a dense grid.
    x_dense = torch.linspace(-1.0, 1.0, 1000).unsqueeze(1)
    with torch.no_grad():
        y_dense = model(x_dense)
    y_truth = torch.sin(torch.pi * x_dense)

    # 6. Plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x.numpy(), y.numpy(), "o", label="training data", markersize=3, alpha=0.6)
    axes[0].plot(x_dense.numpy(), y_dense.numpy(), "-", label="MLP prediction", linewidth=2)
    axes[0].plot(x_dense.numpy(), y_truth.numpy(), "--", label=r"$\sin(\pi x)$", alpha=0.7)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Fit")
    axes[1].semilogy(losses)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE loss")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].set_title("Training loss")
    fig.tight_layout()
    fig.savefig("sine_mlp.png", dpi=120)
    print("\nWrote sine_mlp.png")

    # 7. Final test MSE on uniform grid -- should be tiny.
    final_mse = ((y_dense - y_truth) ** 2).mean().item()
    print(f"Final dense-grid MSE: {final_mse:.4e}")


if __name__ == "__main__":
    main()
