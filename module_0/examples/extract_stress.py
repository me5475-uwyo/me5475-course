"""
extract_stress.py
=================

Lab 0 helper script (a starting point -- students should evolve it via their
lead+review agent workflow).

Reads the per-refinement-level MOOSE CSV outputs produced by
run_convergence_study.sbatch, extracts the steady-state values of
max_vonmises_stress, max_stress_xx, and num_elements, and produces a log-log
convergence plot with both infinite-plate reference values overlaid.

Usage
-----
    python extract_stress.py [--pattern 'plate_with_hole_refine_*_out.csv']
                             [--output convergence.png]
                             [--sigma-inf 1.0]
                             [--vm-ref 2.67]
                             [--hoop-ref 3.0]

Expected CSV format (one row per simulation; MOOSE writes header + data row)
    time, max_stress_xx, max_vonmises_stress, num_elements
    0,    <peak hoop>,   <peak vM>,           <element count>
"""

from __future__ import annotations

import argparse
import glob
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


REFINE_RE = re.compile(r"refine_(\d+)_out\.csv$")
REFINE4_VM_EXPECTED = 2.48
REFINE4_HOOP_EXPECTED = 2.88


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="plate_with_hole_refine_*_out.csv",
        help="Glob pattern for MOOSE CSV outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="convergence.png",
        help="Path to write the convergence plot (default: %(default)s)",
    )
    parser.add_argument(
        "--sigma-inf",
        type=float,
        default=1.0,
        help="Far-field applied stress used in the input file (default: %(default)s)",
    )
    parser.add_argument(
        "--vm-ref",
        "--scf-ref",
        dest="vm_ref",
        type=float,
        default=2.67,
        help="Infinite-plate plane-strain von Mises reference (default: %(default)s)",
    )
    parser.add_argument(
        "--hoop-ref",
        type=float,
        default=3.0,
        help="Infinite-plate Kirsch hoop-stress reference (default: %(default)s)",
    )
    return parser.parse_args()


def load_results(pattern: str) -> pd.DataFrame:
    """Read all matching CSVs and return a tidy DataFrame indexed by refinement level."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched pattern: {pattern}")

    rows = []
    for path in files:
        m = REFINE_RE.search(path)
        if not m:
            print(f"Skipping {path}: filename does not match refine_<N>_out.csv", file=sys.stderr)
            continue
        level = int(m.group(1))
        df = pd.read_csv(path)
        required = {"max_vonmises_stress", "max_stress_xx", "num_elements"}
        missing = required.difference(df.columns)
        if missing:
            sys.exit(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
        # MOOSE writes a header row + one data row per timestep; for Steady we want the last.
        last = df.iloc[-1]
        rows.append(
            {
                "refine": level,
                "num_elements": int(last["num_elements"]),
                "max_vonmises_stress": float(last["max_vonmises_stress"]),
                "max_stress_xx": float(last["max_stress_xx"]),
            }
        )

    if not rows:
        sys.exit("No usable CSV files found.")
    return pd.DataFrame(rows).sort_values("refine").reset_index(drop=True)


def plot_convergence(
    df: pd.DataFrame,
    sigma_inf: float,
    vm_ref: float,
    hoop_ref: float,
    output: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        df["num_elements"],
        df["max_vonmises_stress"],
        "o-",
        label=r"MOOSE peak $\sigma_\mathrm{vM}$",
    )
    ax.loglog(
        df["num_elements"],
        df["max_stress_xx"],
        "s-",
        label=r"MOOSE peak $\sigma_{xx}$ (hoop at hole top)",
    )
    ax.axhline(
        vm_ref * sigma_inf,
        color="C3",
        linestyle="--",
        label=rf"plane-strain $\sigma_{{\mathrm{{vM}}}}$ reference ≈ {vm_ref:g}$\sigma_\infty$",
    )
    ax.axhline(
        hoop_ref * sigma_inf,
        color="C4",
        linestyle=":",
        label=rf"Kirsch hoop reference ≈ {hoop_ref:g}$\sigma_\infty$",
    )
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Peak stress")
    ax.set_title("Plate-with-hole: mesh-refinement convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")

    # Print a small table to stdout so students can paste into their reflection
    print("\nRefinement summary")
    print(df.to_string(index=False))
    print(f"\nRelative error vs plane-strain von Mises reference ({vm_ref:g}):")
    vm_error = (df["max_vonmises_stress"] - vm_ref * sigma_inf) / (vm_ref * sigma_inf)
    for level, error in zip(df["refine"], vm_error):
        print(f"  refine={level}: {error:+.4%}")

    print(f"\nRelative error vs Kirsch hoop reference ({hoop_ref:g}):")
    hoop_error = (df["max_stress_xx"] - hoop_ref * sigma_inf) / (hoop_ref * sigma_inf)
    for level, error in zip(df["refine"], hoop_error):
        print(f"  refine={level}: {error:+.4%}")

    refine4 = df.loc[df["refine"] == 4]
    if not refine4.empty:
        row = refine4.iloc[-1]
        vm_expected = REFINE4_VM_EXPECTED * sigma_inf
        hoop_expected = REFINE4_HOOP_EXPECTED * sigma_inf
        vm_delta = (row["max_vonmises_stress"] - vm_expected) / vm_expected
        hoop_delta = (row["max_stress_xx"] - hoop_expected) / hoop_expected
        print("\nRefine-4 checks against MedicineBow course expectations (±2%):")
        print(
            f"  von Mises: {row['max_vonmises_stress']:.4g} vs {vm_expected:.4g} "
            f"({vm_delta:+.2%})"
        )
        print(
            f"  hoop stress: {row['max_stress_xx']:.4g} vs {hoop_expected:.4g} "
            f"({hoop_delta:+.2%})"
        )


def main() -> None:
    args = parse_args()
    df = load_results(args.pattern)
    plot_convergence(df, args.sigma_inf, args.vm_ref, args.hoop_ref, args.output)


if __name__ == "__main__":
    main()
