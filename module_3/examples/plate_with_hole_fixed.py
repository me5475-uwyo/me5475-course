"""
plate_with_hole_fixed.py
========================

Module 3 / Lab 3 Task 1 -- 2D plate with circular hole under uniaxial extension,
solved with PINN. PyTorch-backend port of Min Lin's original DeepXDE notebook
(TF backend, 2022; course copy module_3/examples/min_lin_2D_hole_example.ipynb).

This is the SOFT-BC version. See plate_with_hole_hard_bc.py for the hard-BC
ansatz version (L15 / Lab 3 Task 3).

Geometry: square plate [0, 1] x [0, 1] minus a quarter-disk of radius 0.1
          centered at the origin (i.e., the bottom-left corner). The full
          plate-with-centered-hole symmetry is encoded via the left edge
          u_1=0 and bottom edge u_2=0 boundary conditions.

PDE: 2-D linear isotropic plane-strain elasticity.
     E = 1.0, nu = 0.3 (as the shared MOOSE reference of this same problem,
     /project/me5475/examples/lab3_reference/ -- not Lab 0's annulus).
     One deliberate difference from Min's notebook: its constants
     C11 = E/(1-nu^2), C12 = E*nu/(1-nu^2) are PLANE STRESS; this port and the
     MOOSE reference are PLANE STRAIN (Lame lambda and mu, below).

Mixed formulation: network outputs (u_1, u_2, sigma_11, sigma_22, sigma_12).
Five outputs avoid second derivatives in the PDE residual.

Boundary conditions (as in Min Lin's notebook):
  Left  edge (x_1 = 0):  u_1 = 0, sigma_12 = 0   (symmetry: no normal motion, no shear)
  Bottom edge (x_2 = 0): u_2 = 0, sigma_12 = 0   (symmetry)
  Right edge (x_1 = 1):  u_1 = 1, sigma_12 = 0   (applied displacement, no shear)
  Top   edge (x_2 = 1):  sigma_22 = 0, sigma_12 = 0  (traction-free)
  Hole  boundary:        sigma_n = 0              (traction-free)
Every edge carries TWO conditions -- one per displacement direction. In FEM the
sigma_12 = 0 conditions come for free (natural BCs); in this mixed (u, sigma)
PINN they do not, and must be in the loss. (Restored 2026-09-25: an earlier port
dropped the three sigma_12 conditions on left/bottom/right, leaving the problem
under-constrained -- team/reviews/agent_reports/2026-09-24_...lab3_reference.md.)

Training: Adam(lr=1e-3, 50000 epochs) + L-BFGS finetune: about 10-11 minutes on an
A30 GPU as a submitted job (module_3/readings/measured_results.md sec. 6).

Usage
-----
    python plate_with_hole_fixed.py
    python plate_with_hole_fixed.py --epochs 10000        # quick test
    python plate_with_hole_fixed.py --save model.pt       # checkpoint
    python plate_with_hole_fixed.py --use-fem-warmup      # L17 Trick 1 -- NOT implemented (stub)

ARCC SLURM submission: use pinn_train.sbatch.

Acknowledgments: PINN structure follows Min Lin's 2022 notebook; the
TF -> PyTorch port + modularization is for this course.
"""

from __future__ import annotations

import argparse
import os
os.environ["DDE_BACKEND"] = "pytorch"   # set BEFORE importing dde

from pathlib import Path

import deepxde as dde
import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Problem constants
# -----------------------------------------------------------------------------
E_VALUE = 1.0
NU_VALUE = 0.3
LAMBDA_LAME = E_VALUE * NU_VALUE / ((1 + NU_VALUE) * (1 - 2 * NU_VALUE))
MU_LAME = E_VALUE / (2 * (1 + NU_VALUE))
HOLE_RADIUS = 0.1


# -----------------------------------------------------------------------------
# PDE residual (mixed formulation)
# -----------------------------------------------------------------------------
def pde(x, u):
    """
    Residual of plane-strain linear elasticity in mixed (u, sigma) form.

    Inputs:
        x : (N, 2) collocation points (x_1, x_2)
        u : (N, 5) network outputs (u_1, u_2, sigma_11, sigma_22, sigma_12)

    Returns a list of 5 residual tensors, each (N, 1), to be driven to zero
    in the PINN loss.
    """
    u1, u2 = u[:, 0:1], u[:, 1:2]
    S11, S22, S12 = u[:, 2:3], u[:, 3:4], u[:, 4:5]

    # First derivatives of network outputs
    u1_x = dde.grad.jacobian(u, x, i=0, j=0)
    u1_y = dde.grad.jacobian(u, x, i=0, j=1)
    u2_x = dde.grad.jacobian(u, x, i=1, j=0)
    u2_y = dde.grad.jacobian(u, x, i=1, j=1)
    S11_x = dde.grad.jacobian(u, x, i=2, j=0)
    S22_y = dde.grad.jacobian(u, x, i=3, j=1)
    S12_x = dde.grad.jacobian(u, x, i=4, j=0)
    S12_y = dde.grad.jacobian(u, x, i=4, j=1)

    # Strain (Voigt-tensor convention: eps_12 is the symmetric component)
    eps_11 = u1_x
    eps_22 = u2_y
    eps_12 = 0.5 * (u1_y + u2_x)

    # Momentum balance (no body force)
    r_mom_1 = S11_x + S12_y
    r_mom_2 = S12_x + S22_y

    # Constitutive (plane-strain Hooke's law)
    trace_eps = eps_11 + eps_22
    r_cst_1 = S11 - (LAMBDA_LAME * trace_eps + 2 * MU_LAME * eps_11)
    r_cst_2 = S22 - (LAMBDA_LAME * trace_eps + 2 * MU_LAME * eps_22)
    r_cst_3 = S12 - 2 * MU_LAME * eps_12

    return [r_mom_1, r_mom_2, r_cst_1, r_cst_2, r_cst_3]


# -----------------------------------------------------------------------------
# Boundary classifiers
# -----------------------------------------------------------------------------
def is_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def is_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)


def is_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)


def is_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1.0)


def is_hole(x, on_boundary):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return on_boundary and np.isclose(r, HOLE_RADIUS, atol=1e-3)


# -----------------------------------------------------------------------------
# Build the DeepXDE problem
# -----------------------------------------------------------------------------
# Same collocation budget as the hard-BC version (4000/600), so Task 3 compares like with like.
def build_problem(num_domain: int = 4000, num_boundary: int = 600, num_test: int = 2000):
    rect = dde.geometry.Rectangle(xmin=[0.0, 0.0], xmax=[1.0, 1.0])
    hole = dde.geometry.Disk(center=[0.0, 0.0], radius=HOLE_RADIUS)
    geom = rect - (rect & hole)

    # Dirichlet BCs
    bc_left_u1 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_left, component=0)
    bc_right_u1 = dde.icbc.DirichletBC(geom, lambda x: 1.0, is_right, component=0)
    bc_bottom_u2 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_bottom, component=1)

    # No shear on the symmetry edges and the loaded edge (sigma_12 is output 4)
    bc_left_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_left, component=4)
    bc_bottom_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_bottom, component=4)
    bc_right_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_right, component=4)

    # Top traction BCs, imposed on stress outputs
    bc_top_S22 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=3)
    bc_top_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=4)

    def hole_normal_traction(x, u_pred):
        # n = x/r is disk-outward (the plate's is -x/r); sign moot for t = 0
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        S11 = u_pred[:, 2:3]; S22 = u_pred[:, 3:4]; S12 = u_pred[:, 4:5]
        t_x = S11 * nx + S12 * ny
        t_y = S12 * nx + S22 * ny
        return t_x, t_y

    # For hole, we enforce traction = 0 via two operator BCs (one for t_x, one for t_y).
    bc_hole_tx = dde.icbc.OperatorBC(
        geom,
        lambda x, u, _: hole_normal_traction(x, u)[0],
        is_hole,
    )
    bc_hole_ty = dde.icbc.OperatorBC(
        geom,
        lambda x, u, _: hole_normal_traction(x, u)[1],
        is_hole,
    )

    data = dde.data.PDE(
        geom,
        pde,
        [bc_left_u1, bc_right_u1, bc_bottom_u2,
         bc_left_S12, bc_bottom_S12, bc_right_S12,
         bc_top_S22, bc_top_S12, bc_hole_tx, bc_hole_ty],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test=num_test,
    )
    net = dde.maps.FNN([2] + [50] * 6 + [5], "tanh", "Glorot uniform")
    return data, net


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(model: dde.Model, adam_epochs: int, lr: float = 1e-3):
    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(iterations=adam_epochs, display_every=1000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)
    return losshistory, train_state


# -----------------------------------------------------------------------------
# Visualize the trained PINN
# -----------------------------------------------------------------------------
def plot_solution(model: dde.Model, out_path: Path, label: str = "soft BCs"):
    # Build a fine grid for plotting; mask out the hole.
    n_grid = 100
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    mask = XX ** 2 + YY ** 2 >= HOLE_RADIUS ** 2
    X_test = np.stack([XX[mask], YY[mask]], axis=1)

    pred = model.predict(X_test)
    u1, u2, S11, S22, S12 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    # von Mises for PLANE STRAIN: the out-of-plane stress is not zero, S33 = nu * (S11 + S22).
    S33 = NU_VALUE * (S11 + S22)
    vm = np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - S33) ** 2 + (S33 - S11) ** 2) + 3 * S12 ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    titles = ["u_1", "u_2", "sigma_11", "sigma_22", "sigma_12", "von Mises stress"]
    fields = [u1, u2, S11, S22, S12, vm]
    for ax, title, field in zip(axes.flat, titles, fields):
        Z = np.full(XX.shape, np.nan)
        Z[mask] = field
        cs = ax.contourf(XX, YY, Z, levels=20, cmap="viridis")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Plate-with-hole PINN ({label}, fixed E=1, nu=0.3)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50000, help="Adam epochs (default: 50000)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-domain", type=int, default=4000)
    parser.add_argument("--num-boundary", type=int, default=600)
    parser.add_argument("--save", type=Path, default=Path("plate_with_hole_fixed.pt"))
    parser.add_argument("--out", type=Path, default=Path("plate_with_hole_fixed.png"))
    parser.add_argument("--use-fem-warmup", action="store_true",
                        help="L17 Trick 1: include FEM-computed displacements as auxiliary supervision early in training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data, net = build_problem(
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
    )

    if args.use_fem_warmup:
        # Not implemented (optional L17 idea): add displacement samples from the
        # shared MOOSE reference as auxiliary data; weight high initially, fade
        # over first 20% of training.
        print("FEM warm-up requested -- see L17 reading for the trick.")
        print("This option is a stub: nothing is added to training. Lab 3 does not require it.")

    model = dde.Model(data, net)
    losshistory, train_state = train(model, adam_epochs=args.epochs, lr=args.lr)

    # Save model + loss history
    model.save(str(args.save))
    np.savez(
        args.save.with_suffix(".loss.npz"),
        loss_train=losshistory.loss_train,
        loss_test=losshistory.loss_test,
        steps=losshistory.steps,
    )

    plot_solution(model, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
