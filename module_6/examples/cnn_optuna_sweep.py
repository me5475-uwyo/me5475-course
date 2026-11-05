"""
cnn_optuna_sweep.py
====================

Module 6 / Lab 6 -- Optuna sweep for the microstructure-to-modulus CNN.

Reuses the M2 pattern. Each trial trains a CNN with sampled hyperparameters
and returns validation MSE on the held-out set.

Search space:
    lr             : log-uniform [1e-4, 1e-2]
    base_channels  : categorical {8, 16, 32}
    depth          : integer [3, 5]
    weight_decay   : log-uniform [1e-6, 1e-3]
    batch_size     : categorical {16, 32, 64}

Usage
-----
    # Local single-worker:
    python cnn_optuna_sweep.py --data rve_dataset.npz --n-trials 5

    # Parallel ARCC sweep (ten CPU workers, one shared split):
    sbatch cnn_optuna_sweep.sbatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from cnn_microstructure_to_modulus import MicrostructureCNN, split_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--study-name", default="lab6_cnn_sweep")
    p.add_argument("--storage", default="sqlite:///cnn_opt.db")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42,
                   help="sampler and model initialization; differs per worker")
    p.add_argument("--split-seed", type=int, default=42,
                   help="the train/val/test partition: the SAME for every worker and every retrain")
    return p.parse_args()


def make_objective(images, moduli, max_epochs, seed, split_seed):
    # Every worker uses one fixed partition (split_seed), so all trials are judged on the same validation rows
    # and the test rows never take part in the sweep.
    splits = split_data(images, moduli, seed=split_seed)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]

    mu_y = Y_tr.mean(); sd_y = Y_tr.std()
    Y_tr_n = (Y_tr - mu_y) / sd_y
    Y_va_n = (Y_va - mu_y) / sd_y

    X_tr_t = torch.tensor(X_tr).unsqueeze(1)
    X_va_t = torch.tensor(X_va).unsqueeze(1)
    Y_tr_t = torch.tensor(Y_tr_n)
    Y_va_t = torch.tensor(Y_va_n)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        base_channels = trial.suggest_categorical("base_channels", [8, 16, 32])
        depth = trial.suggest_int("depth", 3, 5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        torch.manual_seed(seed + trial.number)
        model = MicrostructureCNN(base_channels=base_channels, depth=depth)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        for epoch in range(max_epochs):
            model.train()
            idx = torch.randperm(len(X_tr_t))
            for i in range(0, len(idx), batch_size):
                bi = idx[i : i + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(X_tr_t[bi]), Y_tr_t[bi])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_va_t), Y_va_t).item()
            if val_loss < best_val:
                best_val = val_loss

            # Report every 10 epochs, not every epoch: ten workers writing to one SQLite file on
            # ARCC's network filesystem can hit "database is locked" (Lab 2, 2026-09-24).
            if (epoch + 1) % 10 == 0:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_val

    return objective


def main():
    args = parse_args()
    data = np.load(args.data, allow_pickle=True)
    images = data["images"].astype(np.float32)
    moduli = data["moduli"].astype(np.float32)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        direction="minimize",
        load_if_exists=True,
    )

    va_ids = split_data(images, moduli, seed=args.split_seed)["indices"]["val"]
    print(f"Split seed {args.split_seed}; validation rows, first five: {va_ids[:5].tolist()} "
          f"(the same in every worker); worker seed {args.seed}; CPU threads {torch.get_num_threads()}")
    objective = make_objective(images, moduli, args.max_epochs, args.seed, args.split_seed)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print(f"\nWorker done. This worker ran up to {args.n_trials} trials.")
    print(f"Current best val MSE: {study.best_value:.4e}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
