"""
pinn_diffusion_2d.py
====================

Module 3 / Lecture 13 demo, part 2 --- the same PINN, one dimension up.

Problem:
    -(d^2u/dx1^2 + d^2u/dx2^2) = 2 pi^2 sin(pi x1) sin(pi x2)    on (0, 1)^2
    u = 0                                                       on the boundary

Exact solution: u = sin(pi x1) sin(pi x2) -- again used only to score the network.

Deliberately parallel to pinn_diffusion_1d.py; `diff` the two. Steps 1-7 change
in five places, and nowhere else:

    geometry    Interval(0, 1)          ->  Rectangle([0, 0], [1, 1])
    residual    one hessian             ->  two, indexed i=j=0 and i=j=1
    forcing     pi^2 sin(pi x)          ->  2 pi^2 sin(pi x1) sin(pi x2)
    points      50 + 2,  test 100       ->  2000 + 200,  test 1024 (a 32 x 32 grid)
    network     [1] + [32]*4 + [1]      ->  [2] + [32]*6 + [1]

The BC, the model and the Adam-then-L-BFGS train are the same lines. Outside
steps 1-7, only the exact solution, the defaults and the plots (contours) differ.

Usage
-----
On ARCC this trains, so it runs as a job, never on the login node:
    sbatch pinn_diffusion.sbatch 2d
Anywhere DeepXDE is installed:
    python pinn_diffusion_2d.py [--iterations 5000] [--num-domain 2000] [--seed 42]

Two to three minutes as a 4-core CPU job on ARCC (module_3/readings/measured_results.md,
section 7). --iterations 2000 shortens Adam, but L-BFGS then runs longer: little saved.
"""

from __future__ import annotations

import argparse
import os
os.environ["DDE_BACKEND"] = "pytorch"  # set BEFORE importing dde

import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import deepxde as dde

BROWN, GOLD, GREY = "#492F24", "#8A5E0D", "#6B5D4F"   # course palette, all >= 5.7:1 on white
SEQ = LinearSegmentedColormap.from_list("light_to_brown", ["#FBF7F0", "#C9A36A", BROWN])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L13: a PINN for steady diffusion, 2-D")
    p.add_argument("--num-domain", type=int, default=2000, help="interior collocation points")
    p.add_argument("--iterations", type=int, default=5000, help="Adam iterations before L-BFGS")
    p.add_argument("--seed", type=int, default=42, help="fixes the initial weights")
    p.add_argument("--out", default="pinn_diffusion_2d.png")
    return p.parse_args()


def exact(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])


def main() -> None:
    args = parse_args()
    # PINN training is non-convex (L12): different seeds land on different
    # solutions. Fix it before anything is created -- L10's checklist, item 1.
    dde.config.set_random_seed(args.seed)

    # --- 1. Geometry ---
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    # --- 2. PDE residual ---
    def pde(x, u):
        du_x1x1 = dde.grad.hessian(u, x, i=0, j=0)
        du_x2x2 = dde.grad.hessian(u, x, i=1, j=1)
        x1, x2 = x[:, 0:1], x[:, 1:2]
        f = 2 * np.pi**2 * torch.sin(np.pi * x1) * torch.sin(np.pi * x2)
        return -du_x1x1 - du_x2x2 - f

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
        num_boundary=200,
        solution=exact,               # scores the network; never in the loss
        num_test=1024,
    )

    # --- 5. Network ---
    net = dde.nn.FNN([2] + [32] * 6 + [1], "tanh", "Glorot uniform")

    # --- 6. Model ---
    model = dde.Model(data, net)

    # --- 7. Train: Adam, then L-BFGS ---
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(iterations=args.iterations)
    model.compile("L-BFGS", metrics=["l2 relative error"])
    losshistory, train_state = model.train()

    # --- 8. Report, and plot against the exact solution ---
    summary(losshistory, args.iterations, "2-D")
    n = 101
    xs = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    u_nn, u = model.predict(pts).reshape(n, n), exact(pts).reshape(n, n)
    print(f"max |u_NN - u| on a {n} x {n} grid: {np.abs(u_nn - u).max():.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    levels = np.linspace(min(u.min(), u_nn.min()), max(u.max(), u_nn.max()), 21)
    for ax, field, title in [(axes[0, 0], u, "exact  sin(pi x1) sin(pi x2)"),
                             (axes[0, 1], u_nn, "PINN  u_NN(x1, x2)")]:
        c = ax.contourf(xx, yy, field, levels=levels, cmap=SEQ)   # one shared scale
        fig.colorbar(c, ax=ax, ticks=np.linspace(0, 1, 6))
        ax.set_title(title)
    c = axes[0, 2].contourf(xx, yy, np.abs(u_nn - u), levels=20, cmap=SEQ)
    fig.colorbar(c, ax=axes[0, 2])
    axes[0, 2].set_title("|u_NN - u|  (its own colour scale)")
    for ax in axes[0]:
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal")

    plot_history(axes[1, 0], axes[1, 1], losshistory, args.iterations)
    for a in axes[1, :2]:
        a.grid(True, which="major", color="#DDD5CA", lw=0.8)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.5, f"{data.train_x_all.shape[0]} collocation points\n"
                    f"({args.num_domain} inside + 200 on the edges)\n\n"
                    f"max |u_NN - u| = {np.abs(u_nn - u).max():.1e}",
                    va="center", fontsize=11, color="#222222")
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
