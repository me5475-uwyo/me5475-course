"""
generate_data_analytic.py
=========================

Module 1 / L9-L11 demo support -- build the constitutive dataset WITHOUT MOOSE,
from the closed-form plane-strain stiffness.

Why this exists
---------------
`generate_data.py` builds `single_element.csv` by running MOOSE several hundred
times. That is the Lab 1 exercise and it stays the Lab 1 exercise. But the L9
learning-rate-finder demo, the L10 gallery and the L11 Optuna sweep all READ
that file, and it does not exist anywhere yet -- so none of those demos can be
rehearsed before class. This script produces the same six columns so they can.

This is NOT an approximation of the MOOSE answer
------------------------------------------------
`single_element_loadsweep.i` solves ONE linear QUAD4 element, plane strain,
`ComputeIsotropicElasticityTensor` with E = 1.0, nu = 0.3, and
`ComputeLinearElasticStress`. For a single element under prescribed uniform
strain that BVP has a closed form, and it is exactly

    sigma = C eps        (Voigt, gamma_xy = 2 eps_xy)

with C the plane-strain isotropic stiffness. MOOSE's Newton solve converges to
this to solver tolerance. So the rows below are what MOOSE would print, to
roughly machine precision -- not a stand-in with different statistics.

What it deliberately does NOT do
--------------------------------
It refuses to write to a file named `single_element.csv`. That name belongs to
the Lab 1 deliverable, which students produce from MOOSE; a file with that name
appearing in `data/` would silently become everyone's training set and nobody
would notice the MOOSE step had been skipped. Use --force if you mean it.

Usage
-----
    python generate_data_analytic.py
    python generate_data_analytic.py --n 200 --out data/single_element_analytic.csv
    python generate_data_analytic.py --noise 1e-4          # optional measurement noise
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

COLUMNS = ["eps_xx", "eps_yy", "gamma_xy", "sigma_xx", "sigma_yy", "sigma_xy"]

# Defaults mirror single_element_loadsweep.i and generate_data.py exactly, so a
# dataset from this script and one from MOOSE are drawn from the same design.
E_DEFAULT = 1.0        # single_element_loadsweep.i: youngs_modulus
NU_DEFAULT = 0.3       # single_element_loadsweep.i: poissons_ratio
RESERVED_NAME = "single_element.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="Number of samples (default: 200)")
    p.add_argument("--out", type=Path, default=Path("data/single_element_analytic.csv"),
                   help="Output CSV (default: data/single_element_analytic.csv)")
    p.add_argument("--max-strain", type=float, default=0.005,
                   help="Sample each strain component in [-max, +max] (default: 0.005)")
    p.add_argument("--youngs-modulus", type=float, default=E_DEFAULT,
                   help=f"E, matching the MOOSE input (default: {E_DEFAULT})")
    p.add_argument("--poissons-ratio", type=float, default=NU_DEFAULT,
                   help=f"nu, matching the MOOSE input (default: {NU_DEFAULT})")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Std-dev of Gaussian noise added to each stress component. "
                        "Default 0.0, which is what the MOOSE run gives.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--force", action="store_true",
                   help=f"Permit writing to a file named {RESERVED_NAME}")
    return p.parse_args()


def plane_strain_stiffness(E: float, nu: float) -> np.ndarray:
    """Isotropic plane-strain C in Voigt form, ordered (xx, yy, xy).

    The third row multiplies the ENGINEERING shear gamma_xy = 2 eps_xy, so its
    diagonal entry is the shear modulus G = E / (2(1+nu)) and NOT twice it.
    Dropping that factor of two is the standard Lab 1 error; it is written out
    here so the script can be read as the answer key.
    """
    f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return np.array([[f * (1.0 - nu), f * nu,           0.0],
                     [f * nu,         f * (1.0 - nu),   0.0],
                     [0.0,            0.0,              G  ]])


def main() -> None:
    args = parse_args()

    if args.out.name == RESERVED_NAME and not args.force:
        sys.exit(
            f"Refusing to write {args.out}.\n"
            f"'{RESERVED_NAME}' is the Lab 1 deliverable students generate from MOOSE.\n"
            "Writing analytic data under that name would replace it silently.\n"
            "Pick another name, or pass --force if you really mean to."
        )

    rng = np.random.default_rng(args.seed)
    C = plane_strain_stiffness(args.youngs_modulus, args.poissons_ratio)

    # Sampled one row at a time, with the same call shape as generate_data.py's
    # sample_strain(), so the same --seed gives the same strains as a MOOSE run.
    eps = np.array([rng.uniform(-args.max_strain, args.max_strain, size=3)
                    for _ in range(args.n)])
    sigma = eps @ C.T

    if args.noise > 0.0:
        sigma = sigma + rng.normal(0.0, args.noise, size=sigma.shape)

    rows = np.hstack([eps, sigma])

    # csv rather than pandas: the demo environment does not always have pandas,
    # and a six-column table does not need a DataFrame to be written correctly.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)

    print(f"Plane-strain stiffness used (E = {args.youngs_modulus}, nu = {args.poissons_ratio}):")
    for row in C:
        print("   [" + "  ".join(f"{v: .6f}" for v in row) + "]")
    print(f"\nWrote {len(rows)} rows to {args.out}")
    if args.noise > 0.0:
        print(f"Gaussian noise, std {args.noise:g}, added to each stress component.")
    else:
        print("No noise added -- these are the exact stresses, as MOOSE would return them.")
    print()
    print("  ".join(f"{c:>12s}" for c in COLUMNS))
    for r in rows[:5]:
        print("  ".join(f"{v: 12.6e}" for v in r))


if __name__ == "__main__":
    main()
