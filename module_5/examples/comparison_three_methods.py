"""
comparison_three_methods.py
===========================

Module 5 / Lab 5 -- compare a network-as-material MOOSE run with the elastic MOOSE run, and,
on Lab 3's plate, with the Lab 3 PINN as well.

Why two modes. A comparison is only fair on one problem. Lab 0's plate (quarter annulus,
traction load) and Lab 3's plate (unit square with a quarter hole, prescribed displacement u_1 = 1)
are DIFFERENT problems, and the Lab 3 PINN solves only the second. So:

  plate mode (Lab 5 Task 3) -- Lab 0's plate, two methods, the SAME mesh:
      MOOSE + elastic material   vs   MOOSE + your network (ComputeTorchScriptStress)
    Both runs come from run_nn_simulation.sbatch. Because the mesh is identical, the fields are
    compared node by node (displacements) and element by element (stresses), with no
    interpolation: the only difference left is the material.

      python comparison_three_methods.py --elastic plate_elastic_out.e --nn plate_nn_out.e \
          --train-data single_element.csv
    --train-data (your Lab 1 CSV) adds a COVERAGE PROXY: how many elements' final element-average
    strains lie inside the convex hull of ALL the strains in that CSV (the "full dataset hull").
    It is a proxy, not a guarantee, for three reasons: the CSV also holds Lab 1's validation and
    test rows, so its hull is larger than the training rows' hull; the material evaluates the
    network at quadrature points, whose strains differ from the element average; and Newton's
    intermediate iterates are not checked at all.

  grid mode (Lab 5 Task 5, optional) -- Lab 3's plate, three methods, on Lab 3's 200 x 200 grid:
      the shared MOOSE reference   vs   MOOSE + your network   vs   your Lab 3 PINN
    All three are CSV files in the format of extract_moose_reference.py (x_1, x_2, u_1, u_2,
    sigma_11, sigma_22, sigma_12). The network run uses a displacement 1000x smaller (u_1 = 1e-3)
    to reduce extrapolation; this does not guarantee coverage of every material evaluation.
    --nn-scale 1000 rescales its output for comparison with the linear reference.

      python comparison_three_methods.py --grid-reference reference_grid.csv \
          --grid-nn nn_grid.csv --nn-scale 1000 [--grid-pinn pinn_grid.csv]

Errors are RELATIVE L2 norms over the domain, ||field - reference|| / ||reference||: in plate
mode area-weighted (stresses: element areas; displacements: lumped nodal areas, a quarter of each
neighboring element's area per node -- an approximation of the finite-element L2 integral), and
over the grid points in grid mode (as in Lab 3 Task 2). Grid mode refuses duplicate coordinates
and files with no usable common points, and reports the fraction of points it used.
Writes a figure (--out) and the numbers (--metrics).

Needs numpy, pandas, scipy, matplotlib and netCDF4 (all in /project/me5475/envs/ml4sm). Seconds to run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

# ----------------------------------------------------------------------------------------------
# Exodus reading (netCDF4, as extract_moose_reference.py)
# ----------------------------------------------------------------------------------------------
def read_exodus(path: Path) -> dict:
    import netCDF4 as nc4
    ds = nc4.Dataset(str(path))
    out = {"xy": np.stack([np.asarray(ds.variables["coordx"][:], float),
                           np.asarray(ds.variables["coordy"][:], float)], axis=1)}
    blocks = sorted(k for k in ds.variables if k.startswith("connect"))
    out["conn"] = [np.asarray(ds.variables[b][:], int) - 1 for b in blocks]
    nod = [str(s).strip() for s in nc4.chartostring(ds.variables["name_nod_var"][:])]
    out["nodal"] = {n: np.asarray(ds.variables[f"vals_nod_var{i}"][-1], float)
                    for i, n in enumerate(nod, 1) if f"vals_nod_var{i}" in ds.variables}
    out["elem"] = {}
    if "name_elem_var" in ds.variables:
        ele = [str(s).strip() for s in nc4.chartostring(ds.variables["name_elem_var"][:])]
        for i, n in enumerate(ele, 1):
            parts = [np.asarray(ds.variables[f"vals_elem_var{i}eb{b}"][-1], float)
                     for b in range(1, len(blocks) + 1) if f"vals_elem_var{i}eb{b}" in ds.variables]
            if parts:
                out["elem"][n] = np.concatenate(parts)
    ds.close()
    out["conn"] = np.concatenate(out["conn"])
    return out


def element_areas(xy: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Shoelace area of each element, using its corner nodes (first 4 for quadrilaterals)."""
    c = conn[:, :4] if conn.shape[1] >= 4 else conn
    x, y = xy[c, 0], xy[c, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


def rel_l2(a: np.ndarray, ref: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(ref) if w is None else w
    return float(np.sqrt(np.sum(w * (a - ref) ** 2) / np.sum(w * ref ** 2)))


def last_row(csv: Path) -> pd.Series | None:
    return pd.read_csv(csv).iloc[-1] if csv.exists() else None


# ----------------------------------------------------------------------------------------------
# plate mode
# ----------------------------------------------------------------------------------------------
def plate_mode(args) -> list[str]:
    E, N = read_exodus(args.elastic), read_exodus(args.nn)
    if E["xy"].shape != N["xy"].shape or np.abs(E["xy"] - N["xy"]).max() > 1e-12 \
            or not np.array_equal(E["conn"], N["conn"]):
        raise SystemExit("The two Exodus files are not on the same mesh -- run both with the same "
                         "input file and the same refine / overrides.")
    xy, conn = E["xy"], E["conn"]
    area = element_areas(xy, conn)
    w_node = np.zeros(len(xy))
    np.add.at(w_node, conn[:, :4], np.repeat(area[:, None] / 4, 4, axis=1))   # lumped nodal areas

    lines = [f"plate mode: elastic = {args.elastic}, network = {args.nn}",
             f"mesh: {len(conn)} elements, {len(xy)} nodes; total area {area.sum():.6f} "
             f"(Lab 0's quarter annulus exactly: pi/4 (1 - 0.1^2) = {np.pi / 4 * 0.99:.6f})", ""]

    # Fields: relative L2 (area-weighted) and the largest pointwise difference.
    ue = np.hypot(E["nodal"]["disp_x"], E["nodal"]["disp_y"])
    lines.append("(u: lumped nodal-area weights; stresses: element areas)")
    lines.append(f"{'field':<20}{'rel. L2 (NN vs elastic)':>26}{'max |diff| / max |elastic|':>30}")
    du = np.hypot(N["nodal"]["disp_x"] - E["nodal"]["disp_x"], N["nodal"]["disp_y"] - E["nodal"]["disp_y"])
    lines.append(f"{'u = (u_x, u_y)':<20}{np.sqrt(np.sum(w_node * du**2) / np.sum(w_node * ue**2)):>26.3e}"
                 f"{du.max() / ue.max():>30.3e}")
    for name in ("disp_x", "disp_y"):
        a, r = N["nodal"][name], E["nodal"][name]
        lines.append(f"{name:<20}{rel_l2(a, r, w_node):>26.3e}{np.abs(a - r).max() / np.abs(r).max():>30.3e}")
    for name in ("stress_xx", "stress_yy", "stress_xy", "vonmises_pe", "vonmises_stress"):
        if name in E["elem"] and name in N["elem"]:
            a, r = N["elem"][name], E["elem"][name]
            note = "   <- MOOSE's von Mises: the network's sigma_zz is 0" if name == "vonmises_stress" else ""
            lines.append(f"{name:<20}{rel_l2(a, r, area):>26.3e}{np.abs(a - r).max() / np.abs(r).max():>30.3e}{note}")

    # Postprocessors, side by side (the CSV next to each .e file).
    ce, cn = last_row(args.elastic.with_suffix(".csv")), last_row(args.nn.with_suffix(".csv"))
    if ce is not None and cn is not None:
        lines += ["", f"{'postprocessor':<22}{'elastic':>16}{'network':>16}{'(NN - el) / |el|':>20}"]
        for k in ce.index:
            if k == "time" or k not in cn.index:
                continue
            rel = (cn[k] - ce[k]) / abs(ce[k]) if ce[k] != 0 else float("nan")
            lines.append(f"{k:<22}{ce[k]:>16.6g}{cn[k]:>16.6g}{rel:>20.3e}")

    # Coverage proxy: is each element's FINAL element-average strain (eps_xx, eps_yy, gamma_xy =
    # 2 strain_xy) inside the convex hull of all the strains in the Lab 1 CSV? (Not quadrature-point
    # strains, not Newton's intermediate iterates, not the training split alone -- see the docstring.)
    if all(k in N["elem"] for k in ("strain_xx", "strain_yy", "strain_xy")):
        v = np.stack([N["elem"]["strain_xx"], N["elem"]["strain_yy"], 2 * N["elem"]["strain_xy"]], axis=1)
        lines += ["", "final element-average strains of the network run (gamma_xy = 2 strain_xy):"]
        for j, c in enumerate(("eps_xx", "eps_yy", "gamma_xy")):
            lines.append(f"  {c:<9} [{v[:, j].min(): .3e}, {v[:, j].max(): .3e}]")
        if args.train_data is not None:
            from scipy.spatial import Delaunay
            tr = pd.read_csv(args.train_data)[["eps_xx", "eps_yy", "gamma_xy"]].to_numpy()
            inside = Delaunay(tr).find_simplex(v) >= 0
            lines.append(f"  strains in {args.train_data} (all {len(tr)} rows: Lab 1's training, validation and test rows):")
            for j, c in enumerate(("eps_xx", "eps_yy", "gamma_xy")):
                lines.append(f"  {c:<9} [{tr[:, j].min(): .3e}, {tr[:, j].max(): .3e}]")
            lines.append(f"  coverage proxy -- elements whose final element-average strain lies inside the full dataset "
                         f"hull: {inside.sum()} of {len(v)} ({area[inside].sum() / area.sum():.2%} of the area)"
                         + ("" if inside.all() else "   <- the rest lies outside all the data"))
            if not inside.all():
                k = np.argmax(np.linalg.norm(v, axis=1) * ~inside)
                c = xy[conn[k, :4]].mean(axis=0)
                lines.append(f"  largest outside state: {np.array2string(v[k], precision=3)} "
                             f"in the element at ({c[0]:.3f}, {c[1]:.3f})")

    # Figure: rows u_x, u_y, sigma_xx, von Mises (pe); columns elastic | network | network - elastic.
    fig, axes = plt.subplots(4, 3, figsize=(13, 15.5))
    spec = [("disp_x", "nodal", r"$u_x$", None), ("disp_y", "nodal", r"$u_y$", None),
            ("stress_xx", "elem", r"$\sigma_{xx}$", 0.4), ("vonmises_pe", "elem",
             r"von Mises, $\sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy})$", 0.4)]
    for row, (name, kind, label, zoom) in enumerate(spec):
        if name not in E[kind]:
            continue
        fe, fn = E[kind][name], N[kind][name]
        lim = (min(fe.min(), fn.min()), max(fe.max(), fn.max()))
        d = fn - fe
        dl = np.abs(d).max() or 1.0
        for col, (vals, title, cmap, vlim) in enumerate(
                [(fe, f"elastic: {label}", "viridis", lim), (fn, f"network: {label}", "viridis", lim),
                 (d, f"network - elastic: {label}", "RdBu_r", (-dl, dl))]):
            ax = axes[row, col]
            if kind == "elem":
                pc = PolyCollection(xy[conn[:, :4]], array=vals, cmap=cmap, edgecolors="face", linewidths=0.3)
                pc.set_clim(*vlim)
                ax.add_collection(pc)
                m = pc
            else:
                tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
                m = ax.tripcolor(xy[:, 0], xy[:, 1], tri, vals, shading="gouraud", cmap=cmap,
                                 vmin=vlim[0], vmax=vlim[1])
            ax.set_xlim(0, zoom or 1.0); ax.set_ylim(0, zoom or 1.0)
            ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04, format="%.2e")
    fig.suptitle("Lab 0's plate: MOOSE elastic vs MOOSE + network (stress rows zoomed to the hole)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    lines += ["", f"figure: {args.out}"]
    return lines


# ----------------------------------------------------------------------------------------------
# grid mode
# ----------------------------------------------------------------------------------------------
COLS = ["u_1", "u_2", "sigma_11", "sigma_22", "sigma_12"]


def grid_mode(args) -> list[str]:
    ref = pd.read_csv(args.grid_reference)
    runs = {"network": pd.read_csv(args.grid_nn)}
    if args.grid_pinn:
        runs["PINN"] = pd.read_csv(args.grid_pinn)
    for name, df in [("reference", ref)] + list(runs.items()):
        missing = [c for c in ["x_1", "x_2"] + COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{name}: columns {missing} missing (expected x_1, x_2, {', '.join(COLS)})")
        if df[["x_1", "x_2"]].round(9).duplicated().any():
            raise SystemExit(f"{name}: duplicate (x_1, x_2) points -- the files cannot be aligned point by point")
    key = lambda df: list(zip(df["x_1"].round(9), df["x_2"].round(9)))
    ref_keys = key(ref)
    for name, df in runs.items():
        if key(df) != ref_keys:   # align on the reference's points if a file dropped or reordered some
            m = ref[["x_1", "x_2"]].round(9).merge(df.assign(x_1=df.x_1.round(9), x_2=df.x_2.round(9)),
                                                   on=["x_1", "x_2"], how="left")
            runs[name] = m
    scale = {"network": args.nn_scale, "PINN": 1.0}
    lines = [f"grid mode: reference = {args.grid_reference} ({len(ref)} points)",
             f"network = {args.grid_nn} (scaled by {args.nn_scale:g})"
             + (f", PINN = {args.grid_pinn}" if args.grid_pinn else ""), ""]
    ok = np.isfinite(ref[COLS].to_numpy(float)).all(axis=1)
    for df in runs.values():
        ok &= np.isfinite(df[COLS].to_numpy(float)).all(axis=1)   # NaN (missing after alignment) or +-inf
    if not ok.any():
        raise SystemExit("no point has finite values in every file -- nothing to compare")
    lines.append(f"points used: {ok.sum()} of {len(ref)} ({ok.mean():.2%}; points missing or non-finite in "
                 f"any file are left out)" + ("" if ok.all() else "   <- NOT all points"))
    lines.append(f"{'field':<14}" + "".join(f"{'rel. L2, ' + n:>22}" for n in runs))
    uref = ref.loc[ok, ["u_1", "u_2"]].to_numpy()
    lines.append(f"{'u (both)':<14}" + "".join(
        f"{np.linalg.norm(df.loc[ok, ['u_1', 'u_2']].to_numpy() * scale[n] - uref) / np.linalg.norm(uref):>22.3e}"
        for n, df in runs.items()))
    for c in COLS[2:]:
        r = ref.loc[ok, c].to_numpy()
        lines.append(f"{c:<14}" + "".join(f"{rel_l2(df.loc[ok, c].to_numpy() * scale[n], r):>22.3e}"
                                           for n, df in runs.items()))

    # Figure on the regular grid: rows u_1, u_2, sigma_11; columns reference, runs, differences.
    xs, ys = np.unique(ref["x_1"].round(9)), np.unique(ref["x_2"].round(9))
    ix, iy = np.searchsorted(xs, ref["x_1"].round(9)), np.searchsorted(ys, ref["x_2"].round(9))

    def to_img(v):
        img = np.full((len(ys), len(xs)), np.nan)
        img[iy, ix] = np.where(ok, v, np.nan)
        return img

    names = list(runs)
    ncol = 1 + 2 * len(names)
    fig, axes = plt.subplots(3, ncol, figsize=(3.6 * ncol, 9.5), squeeze=False)
    for row, c in enumerate(["u_1", "u_2", "sigma_11"]):
        r = ref[c].to_numpy()
        panels = [(r, f"MOOSE reference: {c}", "viridis", (np.nanmin(r), np.nanmax(r)))]
        for n in names:
            panels.append((runs[n][c].to_numpy() * scale[n], f"{n}: {c}", "viridis", (np.nanmin(r), np.nanmax(r))))
        for n in names:
            d = runs[n][c].to_numpy() * scale[n] - r
            dl = np.nanmax(np.abs(np.where(ok, d, np.nan))) or 1.0
            panels.append((d, f"{n} - reference: {c}", "RdBu_r", (-dl, dl)))
        for col, (v, title, cmap, (lo, hi)) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(to_img(v), origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9); ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2e")
    fig.suptitle("Lab 3's plate: MOOSE reference vs MOOSE + network" + (" vs PINN" if args.grid_pinn else ""))
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    lines += ["", f"figure: {args.out}"]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--elastic", type=Path, help="plate mode: Exodus file of the elastic run")
    p.add_argument("--nn", type=Path, help="plate mode: Exodus file of the network run (same mesh)")
    p.add_argument("--train-data", type=Path, default=None,
                   help="plate mode (optional): your Lab 1 CSV; checks which element strains it covers")
    p.add_argument("--grid-reference", type=Path, help="grid mode: reference_grid.csv (Lab 3 Task 2a)")
    p.add_argument("--grid-nn", type=Path, help="grid mode: the network run sampled on the same grid")
    p.add_argument("--grid-pinn", type=Path, default=None, help="grid mode (optional): your PINN on the same grid")
    p.add_argument("--nn-scale", type=float, default=1.0, help="grid mode: multiply the network run by this (1000 for u_1 = 1e-3)")
    p.add_argument("--out", type=Path, default=Path("comparison.png"))
    p.add_argument("--metrics", type=Path, default=Path("comparison_metrics.txt"))
    args = p.parse_args()

    if args.elastic and args.nn:
        lines = plate_mode(args)
    elif args.grid_reference and args.grid_nn:
        lines = grid_mode(args)
    else:
        p.error("give --elastic and --nn (plate mode) or --grid-reference and --grid-nn (grid mode)")
    text = "\n".join(lines)
    print(text)
    args.metrics.write_text(text + "\n")
    print(f"metrics: {args.metrics}")


if __name__ == "__main__":
    main()
