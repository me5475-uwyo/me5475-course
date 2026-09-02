"""
generate_data.py
================

Module 1 / Lab 1 — driver that builds the constitutive training dataset by
running single_element_loadsweep.i hundreds of times with random strain inputs.

Usage
-----
    python generate_data.py --n 200 --out data/single_element.csv
    python generate_data.py --n 200 --out data/single_element.csv --moose-exe moose-opt
    python generate_data.py --n 200 --out data/single_element.csv --max-strain 0.01

Outputs
-------
A CSV with columns:
    eps_xx, eps_yy, gamma_xy, sigma_xx, sigma_yy, sigma_xy
(All in Voigt notation with gamma_xy = 2 * eps_xy. Sigma_xy is the symmetric
component, not multiplied by 2.)

Notes
-----
- This script does NOT need to be parallel-clever; each MOOSE run is < 1 second.
  We just loop. For 200 cases, expect ~3 minutes on a laptop.
- The driver does its own random-sampling so the dataset is reproducible from
  the --seed argument.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="Number of training samples (default: 200)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/single_element.csv"),
        help="Output CSV path (default: data/single_element.csv)",
    )
    p.add_argument(
        "--moose-exe",
        default="moose-opt",
        help="Name or path of MOOSE executable (default: moose-opt)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "single_element_loadsweep.i",
        help="MOOSE input file (default: single_element_loadsweep.i in this directory)",
    )
    p.add_argument(
        "--max-strain",
        type=float,
        default=0.005,
        help="Sample epsilon components uniformly in [-max-strain, +max-strain] (default: 0.005)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def sample_strain(rng: np.random.Generator, max_strain: float) -> tuple[float, float, float]:
    """Sample a single (eps_xx, eps_yy, gamma_xy) triple uniformly in a cube."""
    return tuple(rng.uniform(-max_strain, max_strain, size=3))   # type: ignore


def run_moose_once(
    moose_exe: str,
    input_file: Path,
    workdir: Path,
    eps_xx: float,
    eps_yy: float,
    gamma_xy: float,
    file_base: str,
) -> dict[str, float]:
    """Run MOOSE once with the given strain BCs; return averaged stress/strain from the CSV."""
    cmd = [
        moose_exe,
        "-i",
        str(input_file),
        f"Functions/right_x_fn/symbol_values={eps_xx}",
        f"Functions/top_y_fn/symbol_values={eps_yy}",
        f"Functions/right_y_fn/symbol_values={gamma_xy}",
        f"Functions/top_x_fn/symbol_values={gamma_xy}",
        f"Outputs/file_base={file_base}",
    ]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"MOOSE failed for ({eps_xx}, {eps_yy}, {gamma_xy}):", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("MOOSE returned non-zero exit code")

    csv_path = workdir / f"{file_base}.csv"
    df = pd.read_csv(csv_path)
    row = df.iloc[-1]    # Steady executioner writes one data row plus header
    return {
        "eps_xx": float(row["strain_xx_avg"]),
        "eps_yy": float(row["strain_yy_avg"]),
        "gamma_xy": 2.0 * float(row["strain_xy_avg"]),   # Voigt: gamma_xy = 2 eps_xy
        "sigma_xx": float(row["stress_xx_avg"]),
        "sigma_yy": float(row["stress_yy_avg"]),
        "sigma_xy": float(row["stress_xy_avg"]),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not shutil.which(args.moose_exe):
        sys.exit(
            f"MOOSE executable not found: {args.moose_exe}\n"
            "Activate your MOOSE conda environment first."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="moose_lab1_") as tmpdir:
        tmppath = Path(tmpdir)
        # Copy the input file into the workdir so MOOSE writes outputs alongside it.
        local_input = tmppath / args.input.name
        shutil.copy(args.input, local_input)

        rows: list[dict[str, float]] = []
        for i in range(args.n):
            eps_xx, eps_yy, gamma_xy = sample_strain(rng, args.max_strain)
            row = run_moose_once(
                args.moose_exe, local_input, tmppath, eps_xx, eps_yy, gamma_xy, file_base=f"run_{i:04d}"
            )
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  done {i+1}/{args.n}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
