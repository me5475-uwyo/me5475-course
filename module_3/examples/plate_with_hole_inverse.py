"""
plate_with_hole_inverse.py
==========================

Module 3 / Lecture 18 + Lab 3 Task 5 -- Inverse PINN. From sparse synthetic
"DIC measurements" of u_1 and u_2 PLUS one measured reaction force, recover the
material parameters E and nu jointly with the displacement and stress fields.

Why the force is needed
-----------------------
The plate is loaded by a PRESCRIBED DISPLACEMENT (u_1 = 1 on the right edge).
For linear elasticity with displacement loading, the displacement field does
not depend on E at all -- only on nu. Stress, on the other hand, scales with E.
So displacement data alone can pin down nu but can never identify E: every E
fits them equally well. One force measurement fixes the scale: the reaction on
the loaded edge, F = integral of sigma_11(1, x_2) dx_2 over 0 <= x_2 <= 1 (per
unit thickness, plane strain), is proportional to E. That is what a real test
records too -- DIC gives displacements, the load cell gives the force.

Setup:
  - n_measurements (default 20) random points from the shared MOOSE reference
    (extract_moose_reference.py --mode random), 1% Gaussian noise on u_1, u_2.
  - One reaction force: reaction_right_x from the reference's postprocessor CSV
    (1.0734 for E = 1), with the same relative noise.
  - E and nu are learnable parameters, initialized at (0.5, 0.25).
    True values: E = 1.0, nu = 0.3.
  - Hard BCs for u_1, u_2 on the Dirichlet edges (L15); soft BCs for all
    tractions, including sigma_12 = 0 on left, bottom and right.
  - Loss = PDE residuals + traction BCs + displacement data + force mismatch.

Output:
  - Recovered (E, nu), printed and written to --save (one CSV row).
  - For uncertainty: run several seeds (pinn_inverse_ensemble.sbatch runs
    seeds 0-9 as a SLURM array) and aggregate the rows yourself (Task 5c).

Usage
-----
    # One run (submit through a job script -- training does not run on the login node):
    python plate_with_hole_inverse.py --measurements measurements.csv --seed 0

    # Ensemble of 10 seeds on ARCC:
    sbatch pinn_inverse_ensemble.sbatch
"""

from __future__ import annotations

import argparse
import os
os.environ["DDE_BACKEND"] = "pytorch"

from pathlib import Path

import deepxde as dde
import numpy as np
import pandas as pd
import torch

HOLE_RADIUS = 0.1
SHARED_FORCE_CSV = "/project/me5475/examples/lab3_reference/plate_square_hole_reference_out.csv"
N_EDGE = 101   # points on the right edge used to integrate the reaction force


def pde_with_learnable_params(E_param, nu_param, net, F_meas):
    """
    Returns a PDE residual function that uses the learnable parameters E_param
    and nu_param (closure pattern). In DeepXDE 1.15's PyTorch backend,
    dde.Variable IS a torch tensor -- use it directly (it has no .value).

    A sixth residual carries the force measurement: the network's sigma_11 is
    evaluated on N_EDGE points of the right edge, integrated (trapezoid rule),
    and compared with F_meas. It is the same number at every collocation point,
    so its mean-squared value is exactly (F_pred - F_meas)^2.
    """
    edge_y = torch.linspace(0.0, 1.0, N_EDGE).reshape(-1, 1)
    edge_x = torch.cat([torch.ones_like(edge_y), edge_y], dim=1)

    def pde(x, u):
        E = E_param
        nu = nu_param
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        u1, u2 = u[:, 0:1], u[:, 1:2]
        S11, S22, S12 = u[:, 2:3], u[:, 3:4], u[:, 4:5]

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

        # Force mismatch on the loaded edge
        S11_edge = net(edge_x.to(x.device))[:, 2]
        F_pred = torch.trapezoid(S11_edge, edge_y.to(x.device).squeeze())
        r_force = (F_pred - F_meas) * torch.ones_like(u1)

        return [r_mom_1, r_mom_2, r_cst_1, r_cst_2, r_cst_3, r_force]

    return pde


def output_transform(x, y):
    """Hard-BC ansatz on (u_1, u_2)."""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    u1_hard = x1 + x1 * (1.0 - x1) * y[:, 0:1]
    u2_hard = x2 * y[:, 1:2]
    return torch.cat([u1_hard, u2_hard, y[:, 2:5]], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True,
                        help="CSV with columns x_1, x_2, u_1, u_2 (from extract_moose_reference.py)")
    parser.add_argument("--n-measurements", type=int, default=20)
    parser.add_argument("--noise-level", type=float, default=0.01,
                        help="Fractional Gaussian noise on displacements AND force (default: 0.01)")
    parser.add_argument("--force-csv", type=Path, default=Path(SHARED_FORCE_CSV),
                        help="MOOSE postprocessor CSV with reaction_right_x (default: the shared reference)")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=Path("inverse_result.txt"))
    args = parser.parse_args()

    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Load measurements from MOOSE; subsample; add noise ---
    df = pd.read_csv(args.measurements)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), args.n_measurements, replace=False)
    meas = df.iloc[idx].reset_index(drop=True)

    sigma_noise = args.noise_level * np.abs(meas[["u_1", "u_2"]].values).max()
    meas["u_1_noisy"] = meas["u_1"] + rng.normal(0, sigma_noise, args.n_measurements)
    meas["u_2_noisy"] = meas["u_2"] + rng.normal(0, sigma_noise, args.n_measurements)

    # The force measurement: the reference's reaction on the loaded edge, plus noise
    F_true = float(pd.read_csv(args.force_csv)["reaction_right_x"].iloc[-1])
    F_meas = F_true * (1.0 + args.noise_level * rng.normal())
    print(f"Force measurement: {F_meas:.5f} (reference {F_true:.5f}, {args.noise_level*100:.1f}% noise)")

    # --- 2. Set up the inverse PINN ---
    rect = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
    hole = dde.geometry.Disk(center=[0, 0], radius=HOLE_RADIUS)
    geom = rect - (rect & hole)

    # Learnable parameters
    E_param = dde.Variable(0.5, dtype=torch.float32)
    nu_param = dde.Variable(0.25, dtype=torch.float32)

    # The network is built first: the force residual evaluates it on the right edge.
    net = dde.maps.FNN([2] + [50] * 6 + [5], "tanh", "Glorot uniform")
    net.apply_output_transform(output_transform)

    pde_fn = pde_with_learnable_params(E_param, nu_param, net, F_meas)

    # --- 3. Traction BCs (u_1, u_2 on the Dirichlet edges are handled by the hard-BC ansatz) ---
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

    bc_left_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_left, component=4)
    bc_bottom_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_bottom, component=4)
    bc_right_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_right, component=4)
    bc_top_S22 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=3)
    bc_top_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=4)

    def hole_traction_x(x, u, _):
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        return u[:, 2:3] * nx + u[:, 4:5] * ny

    def hole_traction_y(x, u, _):
        nx = x[:, 0:1] / HOLE_RADIUS
        ny = x[:, 1:2] / HOLE_RADIUS
        return u[:, 4:5] * nx + u[:, 3:4] * ny

    bc_hole_tx = dde.icbc.OperatorBC(geom, hole_traction_x, is_hole)
    bc_hole_ty = dde.icbc.OperatorBC(geom, hole_traction_y, is_hole)

    # --- 4. Measurement constraints (PointSetBC) ---
    measurement_pts = meas[["x_1", "x_2"]].values
    bc_meas_u1 = dde.icbc.PointSetBC(
        measurement_pts, meas["u_1_noisy"].values.reshape(-1, 1), component=0,
    )
    bc_meas_u2 = dde.icbc.PointSetBC(
        measurement_pts, meas["u_2_noisy"].values.reshape(-1, 1), component=1,
    )

    # --- 5. Build PDE data + network ---
    data = dde.data.PDE(
        geom,
        pde_fn,
        [bc_left_S12, bc_bottom_S12, bc_right_S12, bc_top_S22, bc_top_S12,
         bc_hole_tx, bc_hole_ty, bc_meas_u1, bc_meas_u2],
        num_domain=4000,     # as plate_with_hole_hard_bc.py (measured_results.md §3)
        num_boundary=600,
        num_test=2000,
    )

    model = dde.Model(data, net)

    # --- 6. Train (E_param and nu_param are tracked via external_trainable_variables) ---
    model.compile(
        "adam",
        lr=args.lr,
        external_trainable_variables=[E_param, nu_param],
    )

    # Callback to log the learned parameters as they evolve
    variable_log = dde.callbacks.VariableValue(
        [E_param, nu_param], period=500, filename=f"inverse_variables_seed_{args.seed}.dat",
    )
    model.train(iterations=args.epochs, callbacks=[variable_log], display_every=2000)

    model.compile("L-BFGS", external_trainable_variables=[E_param, nu_param])
    model.train(display_every=100)

    # --- 7. Report ---
    E_recovered = float(E_param.detach().cpu())
    nu_recovered = float(nu_param.detach().cpu())

    print("\n" + "=" * 50)
    print("Inverse PINN result")
    print(f"  Seed:            {args.seed}")
    print(f"  Measurements:    {args.n_measurements} displacement points + 1 reaction force, "
          f"{args.noise_level*100:.1f}% noise")
    print(f"  E   recovered:   {E_recovered:.4f}   (truth: 1.0000)")
    print(f"  nu  recovered:   {nu_recovered:.4f}  (truth: 0.3000)")
    print(f"  E   rel error:   {abs(E_recovered - 1.0):.4%}")
    print(f"  nu  rel error:   {abs(nu_recovered - 0.3) / 0.3:.4%}")
    print("=" * 50)

    with args.save.open("w") as f:
        f.write(f"seed,E_recovered,nu_recovered,E_true,nu_true,n_measurements,noise_level\n")
        f.write(f"{args.seed},{E_recovered},{nu_recovered},1.0,0.3,{args.n_measurements},{args.noise_level}\n")
    print(f"Saved single-run result to {args.save}")


if __name__ == "__main__":
    main()
