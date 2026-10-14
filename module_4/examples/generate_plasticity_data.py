"""
generate_plasticity_data.py
============================

Module 4 / Lab 4 -- driver that runs j2_cyclic_loading.i many times with
randomized load-path amplitudes, collects the (strain, stress, plastic-strain)
trajectories, and packages them into trajectories.npz for ML training.

Each trajectory is one MOOSE simulation:
  - 3 fully reversed cycles: 0 -> +amplitude -> -amplitude -> 0, three times
  - 60 time steps (dt = 0.05), so 61 rows with t = 0
  - random amplitudes for (eps_xx_max, eps_yy_max, gamma_xy_max)

Usage
-----
    python generate_plasticity_data.py --n 200 --out trajectories.npz
    python generate_plasticity_data.py --n 200 --out trajectories.npz --max-strain 0.02

On ARCC (MOOSE is not on PATH; 200 runs belong in a job, not on the login node):
    sbatch generate_data.sbatch
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="Number of trajectories (default: 200)")
    p.add_argument("--out", type=Path, default=Path("trajectories.npz"))
    p.add_argument("--moose-exe", default="moose-opt")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "j2_cyclic_loading.i",
        help="MOOSE input file (default: j2_cyclic_loading.i in this directory)",
    )
    p.add_argument(
        "--max-strain",
        type=float,
        default=0.02,
        help="Max amplitude for any strain component (default: 0.02 = 2%)",
    )
    p.add_argument(
        "--proportional-fraction",
        type=float,
        default=0.5,
        help="Fraction of trajectories whose amplitudes are a random direction times 0.5-1 x max-strain "
             "(the rest draw each component independently); default 0.5. Every path is radial either way: "
             "all three components follow the same cyclic factor f(t), so the strain stays on one line "
             "through the origin (f(t) changes sign, the direction does not rotate).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_amplitudes(rng: np.random.Generator, max_s: float, proportional: bool):
    """Sample (eps_xx_amp, eps_yy_amp, gamma_xy_amp) for one trajectory."""
    if proportional:
        # Proportional: all three components proportional to a single 'amplitude'
        a = rng.uniform(0.5, 1.0) * max_s
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        return tuple(a * direction)
    else:
        # "Non-proportional" here means only that each component is drawn independently.
        # The strain path is still radial -- eps(t) = f(t) * amplitudes, one f(t) for
        # all three components -- so the strain stays on one line through the origin:
        # f(t) reverses its sign, but the loading direction never rotates.
        return tuple(rng.uniform(-max_s, max_s, size=3))


def run_one_trajectory(
    moose_exe: str,
    input_file: Path,
    workdir: Path,
    amplitudes: tuple[float, float, float],
    file_base: str,
) -> pd.DataFrame:
    """Run MOOSE once with the given amplitudes; return the time-series DataFrame."""
    eps_xx, eps_yy, gamma_xy = amplitudes
    cmd = [
        moose_exe,
        "-i",
        str(input_file),
        f"Functions/eps_xx_amp/expression={eps_xx}",
        f"Functions/eps_yy_amp/expression={eps_yy}",
        f"Functions/gamma_xy_amp/expression={gamma_xy}",
        f"Outputs/file_base={file_base}",
    ]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"MOOSE failed for amplitudes ({eps_xx}, {eps_yy}, {gamma_xy}):", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("MOOSE returned non-zero exit code")

    csv_path = workdir / f"{file_base}.csv"
    df = pd.read_csv(csv_path)
    return df


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not shutil.which(args.moose_exe):
        sys.exit(
            f"MOOSE executable not found: {args.moose_exe}\n"
            "Activate your MOOSE conda environment first. On ARCC, MOOSE is not on PATH:\n"
            "submit generate_data.sbatch, which passes the course build as --moose-exe."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_proportional = int(args.n * args.proportional_fraction)

    all_trajectories = []
    with tempfile.TemporaryDirectory(prefix="moose_m4_") as tmpdir:
        tmppath = Path(tmpdir)
        local_input = tmppath / args.input.name
        shutil.copy(args.input, local_input)

        for i in range(args.n):
            proportional = i < n_proportional
            amps = sample_amplitudes(rng, args.max_strain, proportional)
            try:
                df = run_one_trajectory(
                    args.moose_exe, local_input, tmppath, amps, file_base=f"run_{i:04d}",
                )
            except RuntimeError:
                print(f"Trajectory {i} failed; skipping.")
                continue
            all_trajectories.append(df)
            if (i + 1) % 20 == 0:
                print(f"  done {i+1}/{args.n}")

    if not all_trajectories:
        sys.exit("No trajectories succeeded.")

    # Pad to common length and stack
    max_T = max(len(t) for t in all_trajectories)
    epsilon = np.full((len(all_trajectories), max_T, 3), np.nan, dtype=np.float32)
    sigma = np.full((len(all_trajectories), max_T, 3), np.nan, dtype=np.float32)
    eps_p = np.full((len(all_trajectories), max_T, 3), np.nan, dtype=np.float32)

    for traj_i, df in enumerate(all_trajectories):
        T = len(df)
        epsilon[traj_i, :T, 0] = df["strain_xx_avg"].values
        epsilon[traj_i, :T, 1] = df["strain_yy_avg"].values
        epsilon[traj_i, :T, 2] = 2.0 * df["strain_xy_avg"].values    # Voigt gamma_xy = 2 * eps_xy
        sigma[traj_i, :T, 0] = df["stress_xx_avg"].values
        sigma[traj_i, :T, 1] = df["stress_yy_avg"].values
        sigma[traj_i, :T, 2] = df["stress_xy_avg"].values
        eps_p[traj_i, :T, 0] = df["plastic_strain_xx_avg"].values
        eps_p[traj_i, :T, 1] = df["plastic_strain_yy_avg"].values
        eps_p[traj_i, :T, 2] = 2.0 * df["plastic_strain_xy_avg"].values

    np.savez(
        args.out,
        epsilon=epsilon,
        sigma=sigma,
        eps_p=eps_p,
        n_proportional=n_proportional,
        max_strain=args.max_strain,
    )
    print(f"\nWrote {len(all_trajectories)} trajectories with up to {max_T} time steps each")
    print(f"  Proportional: {n_proportional}")
    print(f"  Non-proportional (independent amplitudes; path still radial): {len(all_trajectories) - n_proportional}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
