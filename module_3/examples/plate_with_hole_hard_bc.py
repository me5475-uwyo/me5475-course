"""
plate_with_hole_hard_bc.py
==========================

Module 3 / Lab 3 Task 3 -- Plate-with-hole PINN with HARD-BC ansatz on
the Dirichlet boundaries.

Compared to plate_with_hole_fixed.py (soft BCs everywhere):
  - The three Dirichlet BCs (u_1 = 0 on left, u_2 = 0 on bottom, u_1 = 1 on right)
    are STRUCTURALLY enforced via the output_transform mechanism.
  - The PINN loss therefore omits these three terms; the PDE residual and the SEVEN
    traction conditions remain (sigma_12 = 0 on left, bottom and right; sigma_22 and
    sigma_12 = 0 on top; zero traction on the hole) -- ten soft BCs become seven.
  - What the ansatz GUARANTEES: the three prescribed displacements hold exactly, for
    any network output. Whether training is faster or ends more accurate is NOT
    guaranteed -- Lab 3 Task 3 measures it, at the same collocation points and the
    same optimizer schedule as plate_with_hole_fixed.py.

The hard-BC ansatz used:

    u_1_hard(x_1, x_2) = x_1 + x_1 * (1 - x_1) * N_1(x_1, x_2)
    u_2_hard(x_1, x_2) = x_2 * N_2(x_1, x_2)

This satisfies:
    at x_1 = 0:  u_1_hard = 0                   ✓
    at x_1 = 1:  u_1_hard = 1 + 0 * N_1 = 1     ✓
    at x_2 = 0:  u_2_hard = 0                   ✓

Stress components (network outputs 2, 3, 4: sigma_11, sigma_22, sigma_12) are
NOT transformed -- they remain soft constraints driven by the constitutive PDE
residual.

Acknowledgments: Hard-BC formulation for elasticity following Lu et al. 2021
SIAM J. Sci. Comp. The PINN architecture (mixed (u, sigma)) is unchanged from
Min Lin's 2022 notebook.

Usage
-----
    python plate_with_hole_hard_bc.py                 # submit via pinn_train.sbatch hard_bc
    python plate_with_hole_hard_bc.py --seed 43       # same seed as the soft run you compare with
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

# Reuse the PDE, boundary classifiers, and constants from the soft-BC version.
from plate_with_hole_fixed import (
    pde,
    is_left, is_right, is_bottom, is_top, is_hole,
    HOLE_RADIUS,
    plot_solution,
    train,
)


def output_transform(x, y):
    """
    Apply the hard-BC ansatz to the first two network outputs (u_1, u_2).
    Stress outputs (indices 2, 3, 4) are passed through unchanged.

    x: (N, 2) collocation points
    y: (N, 5) raw network outputs
    """
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    # u_1 ansatz: satisfies u_1(0, x_2) = 0 and u_1(1, x_2) = 1
    u1_hard = x1 + x1 * (1.0 - x1) * y[:, 0:1]
    # u_2 ansatz: satisfies u_2(x_1, 0) = 0
    u2_hard = x2 * y[:, 1:2]
    # Stresses are unchanged
    return torch.cat([u1_hard, u2_hard, y[:, 2:5]], dim=1)


# 4000/600 collocation points (was 1500/300): with 1500 the hard-BC PINN fitted the fixed
# points and oscillated between them (test loss 7.7); module_3/readings/measured_results.md §3.
def build_problem_hard_bc(num_domain=4000, num_boundary=600, num_test=2000):
    rect = dde.geometry.Rectangle(xmin=[0.0, 0.0], xmax=[1.0, 1.0])
    hole = dde.geometry.Disk(center=[0.0, 0.0], radius=HOLE_RADIUS)
    geom = rect - (rect & hole)

    # Dirichlet BCs are now STRUCTURAL. Only the traction (sigma) BCs remain in the loss --
    # including sigma_12 = 0 on the left, bottom and right edges, which the ansatz does NOT
    # enforce (it only fixes u_1 and u_2).
    bc_left_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_left, component=4)
    bc_bottom_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_bottom, component=4)
    bc_right_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_right, component=4)
    bc_top_S22 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=3)
    bc_top_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=4)

    def hole_normal_traction(x, u_pred):
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        S11 = u_pred[:, 2:3]; S22 = u_pred[:, 3:4]; S12 = u_pred[:, 4:5]
        t_x = S11 * nx + S12 * ny
        t_y = S12 * nx + S22 * ny
        return t_x, t_y

    bc_hole_tx = dde.icbc.OperatorBC(
        geom, lambda x, u, _: hole_normal_traction(x, u)[0], is_hole,
    )
    bc_hole_ty = dde.icbc.OperatorBC(
        geom, lambda x, u, _: hole_normal_traction(x, u)[1], is_hole,
    )

    data = dde.data.PDE(
        geom,
        pde,
        [bc_left_S12, bc_bottom_S12, bc_right_S12,
         bc_top_S22, bc_top_S12, bc_hole_tx, bc_hole_ty],   # only traction BCs
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test=num_test,
    )

    net = dde.maps.FNN([2] + [50] * 6 + [5], "tanh", "Glorot uniform")
    # The crucial line: apply the hard-BC output transform.
    net.apply_output_transform(output_transform)
    return data, net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50000,
                        help="Adam epochs (default: 50000 -- the SAME as plate_with_hole_fixed.py, so Task 3 compares like with like)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=Path, default=Path("plate_with_hole_hard_bc.pt"))
    parser.add_argument("--out", type=Path, default=Path("plate_with_hole_hard_bc.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data, net = build_problem_hard_bc()
    model = dde.Model(data, net)
    losshistory, _ = train(model, adam_epochs=args.epochs, lr=args.lr)   # same schedule as soft

    model.save(str(args.save))
    np.savez(   # same loss-history file as the soft script, for Task 3's comparison plot
        args.save.with_suffix(".loss.npz"),
        loss_train=losshistory.loss_train,
        loss_test=losshistory.loss_test,
        steps=losshistory.steps,
    )
    plot_solution(model, args.out, label="hard BCs")
    print(f"Done. Plot at {args.out}, model at {args.save}")


if __name__ == "__main__":
    main()
