"""
analyze_study.py
================

Module 2 / Lab 2 --- reads a completed (or in-progress) Optuna study and
produces the analysis artifacts the homework asks for:

  - Console summary: best value, best params, top-3 trials, hyperparameter
    importance (fANOVA-based).
  - parallel_coords.html : interactive parallel-coordinates plot.
  - optimization_history.html : objective vs trial number.
  - param_importances.html : bar chart of hyperparameter importance.
  - slice.html : each hyperparameter against the objective, one dot per trial.

Usage
-----
    python analyze_study.py --study-name lab2_sweep --storage sqlite:///optuna.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import optuna.visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-name", default="lab2_sweep")
    p.add_argument("--storage", default="sqlite:///optuna.db")
    p.add_argument("--out-dir", type=Path, default=Path("study_analysis"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    n_running = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING)

    print("Optuna study summary")
    print("=" * 50)
    print(f"Study name:   {args.study_name}")
    print(f"Storage:      {args.storage}")
    print(f"Trials total: {len(study.trials)}  (complete: {n_complete}, pruned: {n_pruned}, "
          f"failed: {n_failed}, running: {n_running})")
    print(f"Best value:   {study.best_value:.6e}")
    print(f"Best params:  {study.best_params}")

    # Top 3 trials by objective value
    print("\nTop 3 completed trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    for rank, t in enumerate(completed[:3], start=1):
        print(f"  {rank}. trial #{t.number:3d}  value={t.value:.4e}  params={t.params}")

    # Hyperparameter importance (fANOVA-based)
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance (fANOVA):")
        for param, imp in sorted(importance.items(), key=lambda kv: -kv[1]):
            print(f"  {param:14s}: {imp:.4f}")
    except Exception as e:
        print(f"\nCouldn't compute hyperparameter importance: {e}")

    # Generate visualizations
    print("\nGenerating visualizations:")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        path = args.out_dir / "parallel_coords.html"
        fig.write_html(str(path))
        print(f"  -> {path}")
    except Exception as e:
        print(f"  parallel_coords FAILED: {e}")

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        path = args.out_dir / "optimization_history.html"
        fig.write_html(str(path))
        print(f"  -> {path}")
    except Exception as e:
        print(f"  optimization_history FAILED: {e}")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        path = args.out_dir / "param_importances.html"
        fig.write_html(str(path))
        print(f"  -> {path}")
    except Exception as e:
        print(f"  param_importances FAILED: {e}")

    try:
        fig = optuna.visualization.plot_slice(study)
        path = args.out_dir / "slice.html"
        fig.write_html(str(path))
        print(f"  -> {path}")
    except Exception as e:
        print(f"  slice plot FAILED: {e}")

    print(f"\nDone. Open the .html files in your browser.")


if __name__ == "__main__":
    main()
