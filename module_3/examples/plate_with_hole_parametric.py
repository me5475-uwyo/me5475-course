"""
plate_with_hole_parametric.py
=============================

Module 3 / Lecture 16 -- Parametric PINN. One trained network
solves the plate-with-hole problem for any (E, nu) in a specified range.

PyTorch-backend port of Min Lin's 2022 parametric notebook (2D-Hole-Example-Various-E-Nu-1,
TF backend; not in the student repository).

Differences from plate_with_hole_fixed.py:
  - Network input is (x_1, x_2, E, nu) -- 4 inputs, not 2.
  - E and nu are sampled per collocation point (uniform random in a specified range).
  - Lambda and mu are computed per point (not constants).
  - Hard BCs on (x_1, x_2) are independent of (E, nu) -- the same ansatz works.
  - DeepXDE gets a 4-D Hypercube over (x_1, x_2, E, nu) as its geometry; the points
    themselves are sampled on the 2-D plate (with the hole) and given as anchors, and
    the boundary tests look only at (x_1, x_2). This is how Min's notebook does it.
    (A 2-D geometry with 4-D anchors fails: DeepXDE stacks them together.)

Differences from Min's parametric notebook: plane STRAIN (the notebook is plane stress, like the
fixed-(E, nu) notebook -- see module_3/readings/plane_stress_and_plane_strain_primer.md); hard
displacement BCs (the notebook's are soft); E in [0.5, 2.0], nu in [0.2, 0.4] (notebook: [1, 3], [0.25, 0.35]).

Search ranges for (E, nu):
  E  in [0.5, 2.0]
  nu in [0.2, 0.4]

Acknowledgments: PINN structure follows Min Lin's 2022 notebook with TF -> PyTorch port.

Usage
-----
    python plate_with_hole_parametric.py
"""

from __future__ import annotations

import argparse
import os
os.environ["DDE_BACKEND"] = "pytorch"

from pathlib import Path

import deepxde as dde
import matplotlib
matplotlib.use("Agg")          # save-only: pins the renderer to a file, never a window
import matplotlib.pyplot as plt
import numpy as np
import torch

HOLE_RADIUS = 0.1
E_MIN, E_MAX = 0.5, 2.0
NU_MIN, NU_MAX = 0.2, 0.4


def pde(x, u):
    """
    Parametric PDE residual. E and nu are inputs (x[:, 2], x[:, 3]),
    not constants.
    """
    E = x[:, 2:3]
    nu = x[:, 3:4]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    u1, u2 = u[:, 0:1], u[:, 1:2]
    S11, S22, S12 = u[:, 2:3], u[:, 3:4], u[:, 4:5]

    # First derivatives. dde.grad.jacobian(u, x, i, j) is partial of u[:, i] w.r.t. x[:, j].
    # We only need derivatives w.r.t. spatial coords (j = 0 or 1), NOT (E, nu).
    u1_x = dde.grad.jacobian(u, x, i=0, j=0)
    u1_y = dde.grad.jacobian(u, x, i=0, j=1)
    u2_x = dde.grad.jacobian(u, x, i=1, j=0)
    u2_y = dde.grad.jacobian(u, x, i=1, j=1)
    S11_x = dde.grad.jacobian(u, x, i=2, j=0)
    S22_y = dde.grad.jacobian(u, x, i=3, j=1)
    S12_x = dde.grad.jacobian(u, x, i=4, j=0)
    S12_y = dde.grad.jacobian(u, x, i=4, j=1)

    eps_11 = u1_x
    eps_22 = u2_y
    eps_12 = 0.5 * (u1_y + u2_x)

    r_mom_1 = S11_x + S12_y
    r_mom_2 = S12_x + S22_y

    trace_eps = eps_11 + eps_22
    r_cst_1 = S11 - (lam * trace_eps + 2 * mu * eps_11)
    r_cst_2 = S22 - (lam * trace_eps + 2 * mu * eps_22)
    r_cst_3 = S12 - 2 * mu * eps_12

    return [r_mom_1, r_mom_2, r_cst_1, r_cst_2, r_cst_3]


def sample_collocation_points(geom_2d, n_domain=8000, n_boundary=1500, rng=None):
    """
    Sample 2-D points from the plate-with-hole geometry, then augment each with
    independently sampled (E, nu). Returns the (n, 4) anchor points.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X_2d_dom = geom_2d.random_points(n_domain, random="pseudo")
    # DeepXDE's boundary sampler for rect - (rect & disk) also returns points on the straight
    # edges of the removed quarter disk -- INSIDE the hole. Drop those and
    # sample again until there are n_boundary points on the plate's real boundary. The points are
    # float32, so points on the arc can sit ~1e-8 below r = 0.1: the tolerance is 1e-6, not 1e-9.
    B = geom_2d.random_boundary_points(n_boundary, random="pseudo")
    X_2d_bnd = np.empty((0, 2), dtype=B.dtype)
    while True:
        r = np.hypot(B[:, 0].astype(np.float64), B[:, 1].astype(np.float64))
        X_2d_bnd = np.vstack([X_2d_bnd, B[r >= HOLE_RADIUS - 1e-6]])
        if len(X_2d_bnd) >= n_boundary:
            break
        B = geom_2d.random_boundary_points(n_boundary, random="pseudo")
    X_2d_bnd = X_2d_bnd[:n_boundary]
    X_2d = np.vstack([X_2d_dom, X_2d_bnd])

    E_samples = rng.uniform(E_MIN, E_MAX, (X_2d.shape[0], 1))
    nu_samples = rng.uniform(NU_MIN, NU_MAX, (X_2d.shape[0], 1))

    return np.concatenate([X_2d, E_samples, nu_samples], axis=1)


def output_transform(x, y):
    """
    Hard-BC ansatz on (u_1, u_2). (E, nu) inputs do NOT affect the BC structure.
    """
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    u1_hard = x1 + x1 * (1.0 - x1) * y[:, 0:1]
    u2_hard = x2 * y[:, 1:2]
    return torch.cat([u1_hard, u2_hard, y[:, 2:5]], dim=1)


def build_problem_parametric(n_domain=8000, n_boundary=1500, seed=42):
    rect = dde.geometry.Rectangle(xmin=[0.0, 0.0], xmax=[1.0, 1.0])
    hole = dde.geometry.Disk(center=[0.0, 0.0], radius=HOLE_RADIUS)
    geom_2d = rect - (rect & hole)

    # Points are sampled on the 2-D plate (with its hole), then given (E, nu) -- the anchors.
    rng = np.random.default_rng(seed)
    X_anchors = sample_collocation_points(geom_2d, n_domain, n_boundary, rng)

    # DeepXDE's own geometry must have the anchors' dimension (4), so it gets a Hypercube over
    # (x_1, x_2, E, nu); it is used only to hold the problem, never to sample. Its idea of
    # "on the boundary" is the Hypercube's, not the plate's -- so the boundary tests below ignore
    # on_boundary and look at (x_1, x_2) directly. Anchors from random_boundary_points lie exactly
    # on their edge or on the hole; any point within np.isclose tolerance of one passes too.
    geom_4d = dde.geometry.Hypercube(xmin=[0.0, 0.0, E_MIN, NU_MIN], xmax=[1.0, 1.0, E_MAX, NU_MAX])

    def is_left(x, on_boundary):
        return np.isclose(x[0], 0.0)

    def is_right(x, on_boundary):
        return np.isclose(x[0], 1.0)

    def is_bottom(x, on_boundary):
        return np.isclose(x[1], 0.0)

    def is_top(x, on_boundary):
        return np.isclose(x[1], 1.0)

    def is_hole(x, on_boundary):
        return np.isclose(np.hypot(x[0], x[1]), HOLE_RADIUS, atol=1e-6)

    # Displacement BCs are hard (output_transform below); the loss keeps the seven traction terms.

    # sigma_12 = 0 on left, bottom and right (restored 2026-09-26, as in the other three plate
    # scripts; the hard-BC ansatz fixes only u_1 and u_2). Validated on ARCC against MOOSE at nine
    # (E, nu) points, 2026-09-27: module_3/readings/measured_results.md, section 9.
    bc_left_S12 = dde.icbc.DirichletBC(geom_4d, lambda x: 0.0, is_left, component=4)
    bc_bottom_S12 = dde.icbc.DirichletBC(geom_4d, lambda x: 0.0, is_bottom, component=4)
    bc_right_S12 = dde.icbc.DirichletBC(geom_4d, lambda x: 0.0, is_right, component=4)
    bc_top_S22 = dde.icbc.DirichletBC(geom_4d, lambda x: 0.0, is_top, component=3)
    bc_top_S12 = dde.icbc.DirichletBC(geom_4d, lambda x: 0.0, is_top, component=4)

    def hole_normal_traction_x(x, u, _):
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        return u[:, 2:3] * nx + u[:, 4:5] * ny

    def hole_normal_traction_y(x, u, _):
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        return u[:, 4:5] * nx + u[:, 3:4] * ny

    bc_hole_tx = dde.icbc.OperatorBC(geom_4d, hole_normal_traction_x, is_hole)
    bc_hole_ty = dde.icbc.OperatorBC(geom_4d, hole_normal_traction_y, is_hole)

    # num_test=None: the "test" loss is evaluated on the training anchors. A separate test set
    # sampled from the Hypercube would put points inside the hole. Check accuracy against MOOSE.
    data = dde.data.PDE(
        geom_4d,
        pde,
        [bc_left_S12, bc_bottom_S12, bc_right_S12, bc_top_S22, bc_top_S12, bc_hole_tx, bc_hole_ty],
        num_domain=0,
        num_boundary=0,
        anchors=X_anchors,
        num_test=None,
    )

    # 4 inputs (x_1, x_2, E, nu), 5 outputs (u_1, u_2, S_11, S_22, S_12)
    net = dde.maps.FNN([4] + [50] * 6 + [5], "tanh", "Glorot uniform")
    net.apply_output_transform(output_transform)
    return data, net


def plot_parametric_slice(model, out_path: Path):
    """
    Visualize: fix (x_1, x_2) = (0.5, 0.5), sweep (E, nu), plot u_1.
    """
    E_grid = np.linspace(E_MIN, E_MAX, 30)
    nu_grid = np.linspace(NU_MIN, NU_MAX, 20)
    EE, NN = np.meshgrid(E_grid, nu_grid)
    X = np.zeros((EE.size, 4))
    X[:, 0] = 0.5
    X[:, 1] = 0.5
    X[:, 2] = EE.ravel()
    X[:, 3] = NN.ravel()
    pred = model.predict(X)
    u1 = pred[:, 0].reshape(EE.shape)
    s11 = pred[:, 2].reshape(EE.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cs1 = axes[0].contourf(EE, NN, u1, levels=20, cmap="viridis")
    axes[0].set_xlabel("E"); axes[0].set_ylabel(r"$\nu$")
    axes[0].set_title("u_1 at (x_1, x_2) = (0.5, 0.5)")
    plt.colorbar(cs1, ax=axes[0])

    cs2 = axes[1].contourf(EE, NN, s11, levels=20, cmap="plasma")
    axes[1].set_xlabel("E"); axes[1].set_ylabel(r"$\nu$")
    axes[1].set_title("sigma_11 at (x_1, x_2) = (0.5, 0.5)")
    plt.colorbar(cs2, ax=axes[1])

    fig.suptitle("Parametric PINN: solution slice across (E, nu) at fixed (x_1, x_2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100000, help="Adam epochs (default: 100000 -- parametric is harder)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-domain", type=int, default=8000)
    parser.add_argument("--n-boundary", type=int, default=1500)
    parser.add_argument("--save", type=Path, default=Path("plate_with_hole_parametric.pt"))
    parser.add_argument("--out", type=Path, default=Path("plate_with_hole_parametric.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data, net = build_problem_parametric(
        n_domain=args.n_domain, n_boundary=args.n_boundary, seed=args.seed,
    )
    model = dde.Model(data, net)

    model.compile("adam", lr=args.lr)
    model.train(iterations=args.epochs, display_every=2000)
    model.compile("L-BFGS")
    model.train(display_every=100)

    model.save(str(args.save))
    plot_parametric_slice(model, args.out)


if __name__ == "__main__":
    main()
