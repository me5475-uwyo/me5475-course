"""
optuna_sweep.py
===============

Module 2 / Lab 2 --- Optuna sweep worker for the Lab 1 constitutive MLP.

This script is a SINGLE WORKER. The SLURM array submits N copies that all
pull trials from the same shared Optuna database. Together they form a
parallel hyperparameter search.

Usage
-----
    # Single-worker local (for development):
    python optuna_sweep.py --study-name lab2_sweep --storage sqlite:///optuna.db --n-trials 10

    # Many workers on ARCC (driven by optuna_sweep.sbatch):
    sbatch optuna_sweep.sbatch

The search space:
    lr                : log-uniform in [1e-5, 1e-1]
    width             : categorical {16, 32, 64, 128}
    n_layers          : integer in [2, 6]
    weight_decay      : log-uniform in [1e-7, 1e-2]
    batch_size        : categorical {16, 32, 64, 'full'}

The objective: validation MSE on a held-out 15% of the Lab 1 dataset, after
training for a fixed number of epochs with the trial's hyperparameters. The
training loop calls trial.report() every REPORT_EVERY epochs so MedianPruner
can kill bad trials early.

Why not every epoch: each report is a write to the shared database. Ten
workers writing every epoch to one SQLite file on ARCC's network home
directory made workers crash with "database is locked" (measured 2026-09-24).
Reporting every 10 epochs cuts the writes tenfold, and a 60 s lock timeout
makes a worker wait longer for the lock -- it can still time out. Optuna
advises against parallel SQLite on network file systems; journal storage is
the fallback (module_2/readings/slurm_sweep_patterns.md).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


INPUT_COLS = ["eps_xx", "eps_yy", "gamma_xy"]
OUTPUT_COLS = ["sigma_xx", "sigma_yy", "sigma_xy"]
REPORT_EVERY = 10          # epochs between trial.report() calls (database writes)
SQLITE_TIMEOUT_S = 60      # how long a worker waits for the SQLite lock
SPLIT_SEED = 42            # ONE split for the whole study -- see load_and_split()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("../../module_1/examples/data/single_element.csv"))
    p.add_argument("--study-name", default="lab2_sweep")
    p.add_argument("--storage", default="sqlite:///optuna.db")
    p.add_argument("--n-trials", type=int, default=5, help="Trials per worker")
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42,
                   help="Sampler + model-initialisation seed. Differs per worker (the sbatch adds the task ID).")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                   help="Train/val/test split seed. Must be THE SAME for every worker in a study, "
                        "and for Task 4's retraining -- otherwise trials are scored on different rows.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_layers: int):
        super().__init__()
        widths = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Data: shared across all trials (each worker loads once)
# -----------------------------------------------------------------------------
def load_and_split(data_path: Path, split_seed: int = SPLIT_SEED):
    """70/15/15 split and train-only standardisation, fixed by split_seed.

    Every worker in a study must call this with the SAME split_seed, so every
    trial is trained and scored on the same rows with the same scaling. (Until
    2026-09-25 the per-worker --seed was passed here, and each worker scored its
    trials on a different validation set -- found by the Reviewer.) Task 4
    reuses this function, with the same split_seed, so the test rows are ones no
    trial ever saw.
    """
    df = pd.read_csv(data_path)
    X = df[INPUT_COLS].to_numpy(dtype=np.float32)
    Y = df[OUTPUT_COLS].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(X))
    n_val = int(0.15 * len(X))
    n_test = int(0.15 * len(X))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    # Standardize using train statistics
    mu_x = X[train_idx].mean(0, keepdims=True)
    sigma_x = X[train_idx].std(0, keepdims=True); sigma_x[sigma_x == 0] = 1
    mu_y = Y[train_idx].mean(0, keepdims=True)
    sigma_y = Y[train_idx].std(0, keepdims=True); sigma_y[sigma_y == 0] = 1

    def std(arr, mu, sig):
        return (arr - mu) / sig

    return {
        "X_train": torch.tensor(std(X[train_idx], mu_x, sigma_x)),
        "Y_train": torch.tensor(std(Y[train_idx], mu_y, sigma_y)),
        "X_val":   torch.tensor(std(X[val_idx], mu_x, sigma_x)),
        "Y_val":   torch.tensor(std(Y[val_idx], mu_y, sigma_y)),
        "X_test":  torch.tensor(std(X[test_idx], mu_x, sigma_x)),
        "Y_test":  torch.tensor(std(Y[test_idx], mu_y, sigma_y)),
        "split": {"seed": split_seed, "train_idx": train_idx, "val_idx": val_idx,
                  "test_idx": test_idx, "fingerprint": split_fingerprint(val_idx, test_idx)},
    }


def split_fingerprint(val_idx, test_idx) -> str:
    """Short hash of the validation and test rows: identical in every worker's log iff the split is shared."""
    h = hashlib.sha1(np.sort(val_idx).tobytes() + b"|" + np.sort(test_idx).tobytes())
    return h.hexdigest()[:10]


def make_objective(data: dict, max_epochs: int, seed: int):

    def objective(trial: optuna.trial.Trial) -> float:
        # --- Sample hyperparameters ---
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        width = trial.suggest_categorical("width", [16, 32, 64, 128])
        n_layers = trial.suggest_int("n_layers", 2, 6)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        batch_choice = trial.suggest_categorical("batch_size", [16, 32, 64, "full"])

        # Seed for reproducibility (different per trial, deterministic within trial)
        torch.manual_seed(seed + trial.number)
        np.random.seed(seed + trial.number)

        model = MLP(in_dim=3, out_dim=3, hidden=width, n_layers=n_layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        X_tr, Y_tr = data["X_train"], data["Y_train"]
        X_va, Y_va = data["X_val"], data["Y_val"]
        N = X_tr.shape[0]

        # Convert batch choice to int (or full batch).
        if batch_choice == "full":
            B = N
        else:
            B = int(batch_choice)

        best_val = float("inf")
        patience_counter = 0
        patience = 200

        for epoch in range(max_epochs):
            # --- Training pass ---
            model.train()
            idx = torch.randperm(N)
            for i in range(0, N, B):
                batch = idx[i : i + B]
                optimizer.zero_grad()
                y_pred = model(X_tr[batch])
                loss = loss_fn(y_pred, Y_tr[batch])
                loss.backward()
                optimizer.step()

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_va), Y_va).item()

            # --- Pruning report: every REPORT_EVERY epochs (each is a DB write) ---
            if epoch % REPORT_EVERY == 0:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # --- Early stopping ---
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val

    return objective


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data = load_and_split(args.data, args.split_seed)
    print(f"Split seed {args.split_seed} -> split fingerprint {data['split']['fingerprint']} "
          f"(must match in every worker's log)")

    storage = args.storage
    if storage.startswith("sqlite"):
        # Wait up to SQLITE_TIMEOUT_S for the lock instead of SQLite's default 5 s.
        storage = optuna.storages.RDBStorage(
            storage, engine_kwargs={"connect_args": {"timeout": SQLITE_TIMEOUT_S}}
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100),
        direction="minimize",
        load_if_exists=True,
    )

    objective = make_objective(data, args.max_epochs, args.seed)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print(f"\nWorker done. Trials completed by this worker: {args.n_trials}")
    print(f"Current best across all workers: {study.best_value:.4e}")
    print(f"Best params so far: {study.best_params}")


if __name__ == "__main__":
    main()
