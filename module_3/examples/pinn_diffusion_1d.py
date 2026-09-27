"""
pinn_diffusion_1d.py
====================

Module 3 / Lecture 13 demo --- a PINN for 1-D steady diffusion, in DeepXDE
(PyTorch backend).

Problem:
    -d^2 u / dx^2 = pi^2 sin(pi x)        on (0, 1)
    u(0) = 0, u(1) = 0

Exact solution: u(x) = sin(pi x). It is used only to SCORE the network (the test
metric and the error plot); it never enters the loss.

pinn_diffusion_2d.py is this file one dimension up -- `diff` the two.

Usage
-----
On ARCC this trains, so it runs as a job, never on the login node:
    sbatch pinn_diffusion.sbatch 1d
Anywhere DeepXDE is installed:
    python pinn_diffusion_1d.py [--iterations 5000] [--num-domain 50] [--seed 42]

Prints DeepXDE's loss table, then a summary (the two loss terms and the test
error at the end of each stage), and writes pinn_diffusion_1d.png.
About half a minute as a 4-core CPU job on ARCC, of which training is about
10 s (module_3/readings/measured_results.md, section 7).
"""

from __future__ import annotations

import argparse
import os
os.environ["DDE_BACKEND"] = "pytorch"  # set BEFORE importing dde

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepxde as dde

BROWN, GOLD, GREY = "#492F24", "#8A5E0D", "#6B5D4F"   # course palette, all >= 5.7:1 on white


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L13: a PINN for steady diffusion, 1-D")
    p.add_argument("--num-domain", type=int, default=50, help="interior collocation points")
    p.add_argument("--iterations", type=int, default=5000, help="Adam iterations before L-BFGS")
    p.add_argument("--seed", type=int, default=42, help="fixes the initial weights")
    p.add_argument("--out", default="pinn_diffusion_1d.png")
    return p.parse_args()


def exact(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def main() -> None:
    args = parse_args()
    # PINN training is non-convex (L12): different seeds land on different
    # solutions. Fix it before anything is created -- L10's checklist, item 1.
    dde.config.set_random_seed(args.seed)

    # --- 1. Geometry ---
    geom = dde.geometry.Interval(0, 1)

    # --- 2. PDE residual ---
    def pde(x, u):
        du_xx = dde.grad.hessian(u, x)
        return -du_xx - np.pi**2 * torch.sin(np.pi * x)

    # --- 3. Boundary conditions ---
    bc = dde.icbc.DirichletBC(
        geom,
        lambda x: 0.0,                        # what it demands: u = 0
        lambda _, on_boundary: on_boundary,   # where: all of the boundary
    )

    # --- 4. Data: collocation points ---
    data = dde.data.PDE(
        geom,
        pde,
        [bc],
        num_domain=args.num_domain,
        num_boundary=2,
        solution=exact,               # scores the network; never in the loss
        num_test=100,
    )

    # --- 5. Network ---
    net = dde.nn.FNN([1] + [32] * 4 + [1], "tanh", "Glorot uniform")

    # --- 6. Model ---
    model = dde.Model(data, net)

    # --- 7. Train: Adam, then L-BFGS ---
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(iterations=args.iterations)
    model.compile("L-BFGS", metrics=["l2 relative error"])
    losshistory, train_state = model.train()

    # --- 8. Report, and plot against the exact solution ---
    summary(losshistory, args.iterations, "1-D")
    x = np.linspace(0, 1, 1001).reshape(-1, 1)
    u_nn, u = model.predict(x), exact(x)
    print(f"max |u_NN - u| on 1001 evenly spaced points: {np.abs(u_nn - u).max():.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(x, u, "-", color=BROWN, lw=2, label="exact  sin(pi x)")
    ax.plot(x, u_nn, "--", color=GOLD, lw=2, label="PINN  u_NN(x)")
    xc = data.train_x_all.ravel()
    ax.plot(xc, np.full_like(xc, -0.06), "|", color=GREY, ms=9,
            label=f"collocation points ({xc.size})")
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("The solution")

    ax = axes[0, 1]
    ax.semilogy(x, np.abs(u_nn - u), "-", color=BROWN, lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("|u_NN - u|")
    ax.set_title("The error, everywhere (log axis)")

    plot_history(axes[1, 0], axes[1, 1], losshistory, args.iterations)
    for a in axes.flat:
        a.grid(True, which="major", color="#DDD5CA", lw=0.8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


# ----------------------------------------------------------------------------
# Reporting helpers -- identical in pinn_diffusion_1d.py and pinn_diffusion_2d.py
# ----------------------------------------------------------------------------
def summary(history, adam_iterations: int, label: str) -> None:
    """The two loss terms and the test error at the end of each training stage."""
    steps = np.array(history.steps)
    loss = np.array(history.loss_train)          # columns: L_PDE, L_BC (training points)
    err = np.array(history.metrics_test)[:, 0]   # relative L2 error on the test points
    i_adam = int(np.flatnonzero(steps == adam_iterations)[0])
    print(f"\n=== L13 summary, {label} ===")
    print(f"{'':26s}{'L_PDE':>10s}{'L_BC':>10s}{'test rel. L2 error':>20s}")
    for name, i in [(f"end of Adam ({adam_iterations} its)", i_adam),
                    (f"end of L-BFGS (+{steps[-1] - adam_iterations} its)", len(steps) - 1)]:
        print(f"{name:26s}{loss[i, 0]:10.2e}{loss[i, 1]:10.2e}{err[i]:20.2e}")


def plot_history(ax_loss, ax_err, history, adam_iterations: int) -> None:
    """Loss terms and test error against step, with the Adam -> L-BFGS switch marked."""
    steps = np.array(history.steps)
    loss = np.array(history.loss_train)
    err = np.array(history.metrics_test)[:, 0]
    ax_loss.semilogy(steps, loss[:, 0], "-o", color=BROWN, lw=2, ms=4, label="L_PDE")
    ax_loss.semilogy(steps, loss[:, 1], "--s", color=GOLD, lw=2, ms=4, label="L_BC")
    ax_loss.set_title("What the optimizer sees: the loss terms")
    ax_loss.legend(loc="upper center", fontsize=8)
    ax_err.semilogy(steps, err, "-o", color=BROWN, lw=2, ms=4)
    ax_err.set_title("What we care about: the test error (relative L2)")
    for ax in (ax_loss, ax_err):
        ax.axvline(adam_iterations, color=GREY, ls=":", lw=1.2)
        ax.text(0.02, 0.04, "Adam", transform=ax.transAxes, color="#222222", fontsize=8)
        ax.text(adam_iterations, ax.get_ylim()[1], "L-BFGS from here ", rotation=90,
                ha="right", va="top", color="#222222", fontsize=8)
        ax.set_xlabel("step")


if __name__ == "__main__":
    main()
