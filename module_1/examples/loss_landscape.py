"""
loss_landscape.py
=================

Module 1 / Lecture 5 live demo — visualize gradient descent on a 1-D loss
landscape. Used in class to show the four learning-rate regimes
(too small, optimal, overshooting-but-converging, diverging).

Usage
-----
    python loss_landscape.py

Produces two figures:
  - loss_landscape_eta_sweep.png : trajectory of theta_k for four learning rates
  - loss_landscape_convergence.png : loss vs. iteration for the same four rates
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # save-only: no display needed, so this works over SSH
import matplotlib.pyplot as plt
import numpy as np


def loss(theta: float) -> float:
    """A simple quadratic: L(theta) = (theta - 2)^2 + 1, minimum at theta = 2."""
    return (theta - 2.0) ** 2 + 1.0


def grad(theta: float) -> float:
    """Gradient of L at theta."""
    return 2.0 * (theta - 2.0)


def gradient_descent(
    theta_0: float, eta: float, n_steps: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Run gradient descent from theta_0 with learning rate eta."""
    thetas = np.zeros(n_steps + 1)
    losses = np.zeros(n_steps + 1)
    thetas[0] = theta_0
    losses[0] = loss(theta_0)
    for k in range(n_steps):
        thetas[k + 1] = thetas[k] - eta * grad(thetas[k])
        losses[k + 1] = loss(thetas[k + 1])
    return thetas, losses


def main() -> None:
    theta_0 = 6.0           # start far from minimum
    n_steps = 30
    learning_rates = {
        "eta = 0.05 (too small)": 0.05,
        "eta = 0.5 (optimal-ish)": 0.5,
        "eta = 0.9 (overshooting but converging)": 0.9,
        "eta = 1.05 (diverging)": 1.05,
    }

    # --- Figure 1: trajectories on the loss curve ---
    theta_grid = np.linspace(-3, 9, 400)
    L_grid = (theta_grid - 2.0) ** 2 + 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_grid, L_grid, "k-", label="loss curve", linewidth=1.5)

    for (name, eta), color in zip(learning_rates.items(), ["C0", "C2", "C1", "C3"]):
        thetas, losses = gradient_descent(theta_0, eta, n_steps=n_steps)
        ax.plot(thetas, losses, "o-", color=color, label=name, markersize=4, alpha=0.85)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$L(\theta)$")
    ax.set_title("Gradient descent on $L(\\theta) = (\\theta - 2)^2 + 1$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_landscape_eta_sweep.png", dpi=120)
    print("Wrote loss_landscape_eta_sweep.png")

    # --- Figure 2: convergence curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, eta), color in zip(learning_rates.items(), ["C0", "C2", "C1", "C3"]):
        thetas, losses = gradient_descent(theta_0, eta, n_steps=n_steps)
        ax.semilogy(np.arange(n_steps + 1), losses - 1.0 + 1e-30,  # subtract minimum, add eps for log
                    "o-", color=color, label=name, markersize=4)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$L(\theta_k) - L^*$")
    ax.set_title("Convergence (semilog) at four learning rates")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("loss_landscape_convergence.png", dpi=120)
    print("Wrote loss_landscape_convergence.png")


if __name__ == "__main__":
    main()
