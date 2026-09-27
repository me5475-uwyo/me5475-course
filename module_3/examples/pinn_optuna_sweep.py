"""
pinn_optuna_sweep.py
====================

Module 3 / Lab 3 Task 4 -- Optuna hyperparameter sweep for the plate-with-hole
PINN. Reuses the M2 sweep pattern; the objective trains a hard-BC PINN with
the trial's hyperparameters and reports validation L2 error against the
MOOSE reference.

Hyperparameters searched:
  - lr             : log-uniform in [1e-4, 1e-2]
  - hidden_width   : categorical {32, 50, 64}
  - hidden_layers  : integer in [4, 8]
  - adam_epochs    : categorical {10000, 25000, 50000}
  - bc_loss_weight : log-uniform in [1, 100]

Each trial trains a PINN and reports the relative L2 error of (u_1, u_2)
against the MOOSE reference -- on the VALIDATION half of the reference grid only.

Validation vs held-out half
---------------------------
The reference grid is split ONCE, by --split-seed (default 42, the same for every
worker), into a validation half and a second half (split_reference below). The
sweep tunes on the validation half; Task 4e reports the retrained best model on
the second half, which the Optuna objective never used. It is not an untouched
end-to-end test set: Lab 3 Tasks 2-3 already looked at the whole grid (lab_3.md).
Every worker logs a split fingerprint.

Usage
-----
    # Single worker (development):
    python pinn_optuna_sweep.py --study-name pinn_lab3 --storage sqlite:///pinn_opt.db \
        --reference reference_grid.csv --n-trials 5

    # Many workers (sbatch pinn_optuna_sweep.sbatch)
"""

from __future__ import annotations

import argparse
import hashlib
import os
os.environ["DDE_BACKEND"] = "pytorch"

from pathlib import Path

import deepxde as dde
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from plate_with_hole_hard_bc import (
    build_problem_hard_bc,
    output_transform,
    pde,
    HOLE_RADIUS,
    is_top, is_hole,
)
from plate_with_hole_fixed import is_left, is_right, is_bottom


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-name", default="pinn_lab3_sweep")
    p.add_argument("--storage", default="sqlite:///pinn_opt.db")
    p.add_argument("--reference", type=Path, required=True,
                   help="Reference CSV (from extract_moose_reference.py --mode grid)")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42,
                   help="Sampler + initialisation seed; differs per worker (the sbatch adds the task ID)")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                   help="Validation/test split of the reference grid -- the SAME for every worker")
    return p.parse_args()


SPLIT_SEED = 42
SQLITE_TIMEOUT_S = 60      # wait for the SQLite lock longer than the 5 s default (as in Lab 2)


def split_reference(ref_df: pd.DataFrame, split_seed: int = SPLIT_SEED):
    """Split the reference grid once into (validation, test) halves; returns them plus a fingerprint.

    Task 4e: call this with the same split_seed and report on the TEST half.
    """
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(ref_df))
    half = len(ref_df) // 2
    val_idx, test_idx = np.sort(idx[:half]), np.sort(idx[half:])
    fp = hashlib.sha1(val_idx.tobytes() + b"|" + test_idx.tobytes()).hexdigest()[:10]
    return ref_df.iloc[val_idx].reset_index(drop=True), ref_df.iloc[test_idx].reset_index(drop=True), fp


def l2_error_vs_reference(model, ref_df: pd.DataFrame) -> float:
    """
    Compute L2-norm relative error of (u_1, u_2) prediction vs. MOOSE reference.
    """
    X = ref_df[["x_1", "x_2"]].values
    pred = model.predict(X)
    u1_pred, u2_pred = pred[:, 0], pred[:, 1]
    u1_true = ref_df["u_1"].values
    u2_true = ref_df["u_2"].values
    err = np.sqrt(np.sum((u1_pred - u1_true) ** 2 + (u2_pred - u2_true) ** 2))
    ref_norm = np.sqrt(np.sum(u1_true ** 2 + u2_true ** 2))
    return float(err / ref_norm)


def make_objective(reference_csv: Path, seed: int, split_seed: int):
    ref_val, _ref_test, fp = split_reference(pd.read_csv(reference_csv), split_seed)
    print(f"Split seed {split_seed} -> reference split fingerprint {fp} "
          f"({len(ref_val)} validation points; must match in every worker's log)")

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_width = trial.suggest_categorical("hidden_width", [32, 50, 64])
        hidden_layers = trial.suggest_int("hidden_layers", 4, 8)
        adam_epochs = trial.suggest_categorical("adam_epochs", [10000, 25000, 50000])
        # bc_loss_weight: keep simple by exposing only the Neumann BC weight
        bc_loss_weight = trial.suggest_float("bc_loss_weight", 1.0, 100.0, log=True)

        dde.config.set_random_seed(seed + trial.number)
        torch.manual_seed(seed + trial.number)
        np.random.seed(seed + trial.number)

        # Build data with this trial's network width/depth
        rect = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
        hole = dde.geometry.Disk(center=[0, 0], radius=HOLE_RADIUS)
        geom = rect - (rect & hole)

        # sigma_12 = 0 on left, bottom, right (the hard-BC ansatz fixes only u_1, u_2)
        bc_left_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_left, component=4)
        bc_bottom_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_bottom, component=4)
        bc_right_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_right, component=4)
        bc_top_S22 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=3)
        bc_top_S12 = dde.icbc.DirichletBC(geom, lambda x: 0.0, is_top, component=4)

        def hole_tx(x, u, _):
            nx = x[:, 0:1] / HOLE_RADIUS
            ny = x[:, 1:2] / HOLE_RADIUS
            return u[:, 2:3] * nx + u[:, 4:5] * ny
        def hole_ty(x, u, _):
            nx = x[:, 0:1] / HOLE_RADIUS
            ny = x[:, 1:2] / HOLE_RADIUS
            return u[:, 4:5] * nx + u[:, 3:4] * ny
        bc_hole_tx = dde.icbc.OperatorBC(geom, hole_tx, is_hole)
        bc_hole_ty = dde.icbc.OperatorBC(geom, hole_ty, is_hole)

        data = dde.data.PDE(
            geom,
            pde,
            [bc_left_S12, bc_bottom_S12, bc_right_S12,
             bc_top_S22, bc_top_S12, bc_hole_tx, bc_hole_ty],
            num_domain=4000,     # as plate_with_hole_hard_bc.py (measured_results.md §3)
            num_boundary=600,
            num_test=2000,
        )

        net = dde.maps.FNN(
            [2] + [hidden_width] * hidden_layers + [5],
            "tanh",
            "Glorot uniform",
        )
        net.apply_output_transform(output_transform)

        # Loss weights: PDE = 1 by default; multiply BC terms by bc_loss_weight.
        n_pde, n_bc = 5, 7   # 5 PDE residuals; 7 traction BCs (3 sigma_12 + top 2 + hole 2)
        loss_weights = [1] * n_pde + [bc_loss_weight] * n_bc

        model = dde.Model(data, net)
        model.compile("adam", lr=lr, loss_weights=loss_weights)

        # Train with periodic L2-error reports for pruning
        report_every = max(adam_epochs // 10, 500)
        n_chunks = adam_epochs // report_every
        for chunk in range(n_chunks):
            model.train(iterations=report_every, display_every=report_every)
            err = l2_error_vs_reference(model, ref_val)
            trial.report(err, step=chunk)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final L-BFGS finetune
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train(display_every=100)

        final_err = l2_error_vs_reference(model, ref_val)
        return final_err

    return objective


def main():
    args = parse_args()

    storage = args.storage
    if storage.startswith("sqlite"):
        storage = optuna.storages.RDBStorage(
            storage, engine_kwargs={"connect_args": {"timeout": SQLITE_TIMEOUT_S}}
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        direction="minimize",
        load_if_exists=True,
    )

    objective = make_objective(args.reference, args.seed, args.split_seed)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print(f"\nWorker done. This worker ran {args.n_trials} trials.")
    print(f"Current best L2 relative error (validation half): {study.best_value:.4e}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
