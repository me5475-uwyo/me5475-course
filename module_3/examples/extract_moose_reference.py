"""
extract_moose_reference.py
==========================

Module 3 / Lab 3 -- helper to sample the MOOSE reference solution of the
plate-with-hole problem at arbitrary points inside the PINN domain.

The reference is plate_square_hole_reference.i (this folder): the SAME problem
the PINN solves -- unit square [0,1]^2 minus the quarter disk r < 0.1, plane
strain, E = 1, nu = 0.3, u_1 = 0 on x = 0, u_2 = 0 on y = 0, u_1 = 1 on x = 1,
top and hole traction-free.  (Lab 0's plate_with_hole.i is a quarter annulus
with a traction load -- a different problem; do not compare the PINN to it.)

A converged copy is shared on ARCC:
    /project/me5475/examples/lab3_reference/plate_square_hole_reference_out.e

Two use cases:
  1. PINN-vs-MOOSE comparison dataset (Lab 3 Task 2): sample u_1, u_2 and the
     stresses on a regular grid, keeping only grid points inside the domain.
  2. Synthetic "DIC measurements" for the inverse problem (Lab 3 Task 5):
     sample the same fields at random points inside the domain.

Output CSV columns: x_1, x_2, u_1, u_2, and -- when the Exodus file carries
them as nodal fields (the reference does) -- sigma_11, sigma_22, sigma_12,
sigma_vm.

Requires numpy, pandas, scipy and netCDF4 (all in /project/me5475/envs/ml4sm).

Usage
-----
    REF=/project/me5475/examples/lab3_reference/plate_square_hole_reference_out.e

    # Full-field comparison grid (200x200 grid, in-domain points only):
    python extract_moose_reference.py --exodus $REF \
        --output reference_grid.csv --mode grid --nx 200 --ny 200

    # 200 random in-domain points for the inverse problem:
    python extract_moose_reference.py --exodus $REF \
        --output measurements.csv --mode random --n-points 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HOLE_RADIUS = 0.1
SHARED_REFERENCE = "/project/me5475/examples/lab3_reference/plate_square_hole_reference_out.e"

# CSV column -> MOOSE nodal variable name.  u_1/u_2 are required; the stress
# columns are written when present (the shared reference has all of them).
REQUIRED_FIELDS = {"u_1": "disp_x", "u_2": "disp_y"}
OPTIONAL_FIELDS = {
    "sigma_11": "stress_xx",
    "sigma_22": "stress_yy",
    "sigma_12": "stress_xy",
    "sigma_vm": "vonmises_stress",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exodus", type=Path, default=Path(SHARED_REFERENCE),
                   help=f"MOOSE Exodus output (.e file). Default: {SHARED_REFERENCE}")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path")
    p.add_argument("--mode", choices=["grid", "random"], required=True)
    p.add_argument("--nx", type=int, default=200, help="Grid mode: x resolution")
    p.add_argument("--ny", type=int, default=200, help="Grid mode: y resolution")
    p.add_argument("--n-points", type=int, default=200, help="Random mode: number of points")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def in_domain(x, y, hole_r=HOLE_RADIUS, plate_max=1.0):
    """Point is inside the PINN domain: [0,1]^2 with r >= hole_r."""
    return (
        (x >= 0)
        & (x <= plate_max)
        & (y >= 0)
        & (y <= plate_max)
        & (x ** 2 + y ** 2 >= hole_r ** 2)
    )


def read_moose_exodus(exodus_path: Path) -> tuple[np.ndarray, dict]:
    """
    Read node coordinates and last-timestep nodal fields from a MOOSE Exodus file.

    Uses netCDF4 directly to fetch the LAST time step, which is the
    steady-state MOOSE solution (step 0 is the initial condition).

    Returns
    -------
    coords : (N, 2) float array of (x, y) node positions
    fields : dict mapping MOOSE nodal variable name -> (N,) float array
    """
    import netCDF4 as nc4

    ds = nc4.Dataset(str(exodus_path))
    x = np.asarray(ds.variables["coordx"][:], dtype=float)
    y = np.asarray(ds.variables["coordy"][:], dtype=float)
    coords = np.stack([x, y], axis=1)

    names = [str(s).strip() for s in nc4.chartostring(ds.variables["name_nod_var"][:])]
    fields = {}
    for idx, name in enumerate(names, start=1):
        var_key = f"vals_nod_var{idx}"
        if var_key in ds.variables:
            fields[name] = np.asarray(ds.variables[var_key][-1], dtype=float)
    ds.close()
    return coords, fields


def interpolate_fields(coords: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear interpolation of nodal fields to query points.

    One Delaunay triangulation of the mesh nodes is shared by all fields.
    Every query point inside the domain lies inside the node cloud's convex
    hull, so no NaN is expected; NaN would mean a point outside the mesh.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    tri = Delaunay(coords)
    # Nudge points that sit exactly on the straight edges 1e-12 inward so that
    # round-off in the mesh coordinates cannot put them outside the hull.
    q = np.clip(query, 1e-12, 1.0 - 1e-12)
    return LinearNDInterpolator(tri, values, fill_value=np.nan)(q)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Reading {args.exodus} ...")
    coords, fields = read_moose_exodus(args.exodus)
    print(f"  {len(coords)} nodes")
    print(f"  Available nodal fields: {list(fields.keys())}")

    missing = [m for m in REQUIRED_FIELDS.values() if m not in fields]
    if missing:
        sys.exit(f"ERROR: required nodal field(s) {missing} not in {args.exodus}")

    # Build query points -- both modes keep only points inside the PINN domain.
    if args.mode == "grid":
        xs = np.linspace(0, 1, args.nx)
        ys = np.linspace(0, 1, args.ny)
        XX, YY = np.meshgrid(xs, ys)
        mask = in_domain(XX, YY)
        query = np.stack([XX[mask], YY[mask]], axis=1)
        print(f"  Grid mode: {args.nx}x{args.ny} -> {len(query)} in-domain points")
    else:
        # Rejection sampling, uniform over the domain.
        query = np.empty((0, 2))
        while len(query) < args.n_points:
            cand = rng.uniform(0, 1, (2 * args.n_points, 2))
            query = np.vstack([query, cand[in_domain(cand[:, 0], cand[:, 1])]])
        query = query[: args.n_points]
        print(f"  Random mode: {len(query)} in-domain points")

    col_map = dict(REQUIRED_FIELDS)
    col_map.update({c: m for c, m in OPTIONAL_FIELDS.items() if m in fields})
    values = np.stack([fields[m] for m in col_map.values()], axis=1)
    sampled = interpolate_fields(coords, values, query)

    cols = {"x_1": query[:, 0], "x_2": query[:, 1]}
    for j, (csv_col, moose_name) in enumerate(col_map.items()):
        cols[csv_col] = sampled[:, j]
        print(f"  Interpolated {moose_name} -> {csv_col}")
    df = pd.DataFrame(cols)

    n_nan = int(df.isna().any(axis=1).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} points fell outside the mesh (NaN) and were dropped -- "
              "is this Exodus file the square-with-hole reference?")
        df = df.dropna()

    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows ({n_nan} NaN rows dropped) to {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()
