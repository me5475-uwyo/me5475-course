"""
generate_microstructures.py
============================

Module 6 / Lab 6 -- generate random 2-D RVE microstructures and compute each one's
apparent modulus E_app with MOOSE (mixed BCs, not periodic homogenization).

Each microstructure: a 64x64 grid with 2-5 random ellipsoidal inclusions
(stiffer phase) embedded in a softer matrix.

Each MOOSE simulation (rve_homogenization.i.template): plane strain, u_x prescribed on the left and right
edges (eps_xx_macro = 0.001), u_y = 0 along the bottom, top free. E_app = <sigma_xx> / 0.001, the cell's
apparent axial response under these BCs: E/(1 - nu^2) = 1.0989 E for a homogeneous cell, so 1.0989 (all
matrix) and 5.4945 (all inclusion) bound every composite.

Dataset output: rve_dataset.npz with:
    images : (N, 64, 64) binary microstructures (1 = inclusion, 0 = matrix)
    moduli : (N,) E_app values
    stress_fields : (N, 3, 64, 64) optional stress fields (for U-Net stretch task)

Usage
-----
    # Never on an ARCC login node. One MOOSE run took about 1.5 s on MedicineBow (Nov 5, 2026).
    python generate_microstructures.py --moose-exe "$MOOSE_EXE" --n 50 --out part.npz

    # On ARCC (faster via SLURM array, see data_generation.sbatch):
    sbatch data_generation.sbatch
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


GRID = 64
LX = LY = 1.0
E_MATRIX = 1.0
E_INCLUSION = 5.0
NU_BOTH = 0.3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("rve_dataset.npz"))
    p.add_argument("--moose-exe", default="moose-opt")
    p.add_argument("--template", type=Path,
                   default=Path(__file__).parent / "rve_homogenization.i.template")
    p.add_argument("--save-stress-fields", action="store_true",
                   help="Also save 64x64x3 stress fields (for U-Net stretch task)")
    p.add_argument("--min-inclusions", type=int, default=2)
    p.add_argument("--max-inclusions", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_random_ellipses(rng, n_inclusions: int):
    """Sample n random non-overlapping ellipse parameters in [0, 1]^2."""
    ellipses = []
    for _ in range(n_inclusions):
        for attempt in range(50):
            cx, cy = rng.uniform(0.15, 0.85, 2)
            a = rng.uniform(0.04, 0.12)
            b = rng.uniform(0.04, 0.12)
            theta = rng.uniform(0, np.pi)
            # Check non-overlap with previously placed ellipses
            ok = True
            for prev in ellipses:
                d = np.hypot(cx - prev[0], cy - prev[1])
                if d < (max(a, b) + max(prev[2], prev[3])) * 1.1:
                    ok = False
                    break
            if ok:
                ellipses.append((cx, cy, a, b, theta))
                break
    return ellipses


def rasterize_ellipses(ellipses, grid: int) -> np.ndarray:
    """Render the ellipse set as a binary (grid x grid) image."""
    xs = (np.arange(grid) + 0.5) / grid     # cell centers in [0, 1]
    ys = (np.arange(grid) + 0.5) / grid
    XX, YY = np.meshgrid(xs, ys)
    img = np.zeros((grid, grid), dtype=np.uint8)
    for (cx, cy, a, b, theta) in ellipses:
        xc = (XX - cx) * np.cos(theta) + (YY - cy) * np.sin(theta)
        yc = -(XX - cx) * np.sin(theta) + (YY - cy) * np.cos(theta)
        img |= ((xc / a) ** 2 + (yc / b) ** 2 < 1).astype(np.uint8)
    return img


def parsed_function_for_ellipses(ellipses) -> str:
    """
    Build a MOOSE ParsedFunction expression that returns E(x, y):
        E_matrix + (E_inclusion - E_matrix) * indicator(x, y inside any ellipse)

    We use a sum-of-sigmoids approximation for differentiability, but for a
    purely Material assignment the indicator function via tanh is fine.
    """
    terms = []
    for (cx, cy, a, b, theta) in ellipses:
        # transformed coords
        xc = f"((x - {cx}) * cos({theta}) + (y - {cy}) * sin({theta}))"
        yc = f"(-(x - {cx}) * sin({theta}) + (y - {cy}) * cos({theta}))"
        # smoothed indicator: 0.5 * (1 - tanh(scale * (rho^2 - 1)))
        # large scale = sharp transition; for material assignment, scale ~ 50 is enough
        rho_sq = f"(({xc})^2 / ({a})^2 + ({yc})^2 / ({b})^2)"
        terms.append(f"0.5 * (1 - tanh(50 * ({rho_sq} - 1)))")
    if not terms:
        return f"{E_MATRIX}"
    indicator = " + ".join(terms)
    indicator = f"min({indicator}, 1)"   # clip in case ellipses overlap
    return f"{E_MATRIX} + ({E_INCLUSION} - {E_MATRIX}) * ({indicator})"


# The [VectorPostprocessors] block the driver injects (in place of the template's
# __FIELD_SAMPLER__ marker) when --save-stress-fields is requested. ElementValueSampler
# writes one CSV row per element with the element centroid (x, y, z) and the sampled
# stress components -- trivially parsed with pandas, no Exodus/meshio dependency.
FIELD_SAMPLER_BLOCK = """[VectorPostprocessors]
  [stress_field]
    type     = ElementValueSampler
    variable = 'stress_xx stress_yy stress_xy'
    sort_by  = id
  []
[]"""


def render_moose_input(template: Path, E_expression: str, file_base: str,
                       save_fields: bool = False) -> str:
    """Substitute the E(x,y) expression into the MOOSE input file template.

    `save_fields=True` injects the per-element stress sampler so the full field
    is written to CSV (needed for the U-Net stretch task)."""
    import re
    text = template.read_text()
    text = text.replace("__E_EXPRESSION__", E_expression)
    text = text.replace("__FILE_BASE__", file_base)
    # Anchored replacement: only substitute the standalone __FIELD_SAMPLER__ line,
    # not the occurrence inside the template's comment block.
    text = re.sub(r"^__FIELD_SAMPLER__$",
                  FIELD_SAMPLER_BLOCK if save_fields else "",
                  text, flags=re.MULTILINE)
    return text


def csv_to_stress_field(workdir: Path, file_base: str, grid: int = GRID) -> np.ndarray:
    """Parse the ElementValueSampler CSV and bin it onto a (3, grid, grid) array:
    channels (sigma_xx, sigma_yy, sigma_xy), [row=y, col=x] layout to match
    `rasterize_ellipses`.

    Placement uses each element's centroid (the x/y columns), so it is robust to
    MOOSE's element numbering. The exact CSV filename (timestep suffix) is matched
    by glob; registration is confirmed on the first real MOOSE run during ARCC
    validation."""
    matches = sorted(workdir.glob(f"{file_base}_out_stress_field*.csv"))
    if not matches:
        raise RuntimeError(
            f"No stress-field CSV found for {file_base} "
            f"(expected {file_base}_out_stress_field*.csv in {workdir}; "
            f"MOOSE file_base = {file_base}_out → VPP CSV = {file_base}_out_stress_field_NNNN.csv)"
        )
    df = pd.read_csv(matches[-1])               # last timestep
    ix = np.clip((df["x"].to_numpy() * grid).astype(int), 0, grid - 1)
    iy = np.clip((df["y"].to_numpy() * grid).astype(int), 0, grid - 1)
    field = np.zeros((3, grid, grid), dtype=np.float32)
    for k, name in enumerate(["stress_xx", "stress_yy", "stress_xy"]):
        field[k, iy, ix] = df[name].to_numpy()
    return field


def run_moose_once(moose_exe: str, input_text: str, workdir: Path, file_base: str,
                   save_field: bool = False, grid: int = GRID):
    """Run MOOSE for one RVE. Returns (E_eff, stress_field), where stress_field
    is a (3, grid, grid) array if save_field else None."""
    input_path = workdir / f"{file_base}.i"
    input_path.write_text(input_text)
    result = subprocess.run(
        [moose_exe, "-i", str(input_path)],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"MOOSE failed for {file_base}:", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("MOOSE returned non-zero exit code")
    csv_path = workdir / f"{file_base}_out.csv"
    df = pd.read_csv(csv_path)
    sigma_xx_avg = float(df["stress_xx_avg"].iloc[-1])
    eps_imposed = 0.001    # set in the input template
    E_eff = sigma_xx_avg / eps_imposed

    field = None
    if save_field:
        field = csv_to_stress_field(workdir, file_base, grid)
    return E_eff, field


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not shutil.which(args.moose_exe):
        sys.exit(f"MOOSE executable not found: {args.moose_exe}")

    images = []
    moduli = []
    ellipses_all = []
    stress_fields = []

    with tempfile.TemporaryDirectory(prefix="moose_m6_") as tmpdir:
        tmppath = Path(tmpdir)
        for i in range(args.n):
            n_inc = rng.integers(args.min_inclusions, args.max_inclusions + 1)
            ellipses = sample_random_ellipses(rng, n_inc)
            img = rasterize_ellipses(ellipses, GRID)
            E_expr = parsed_function_for_ellipses(ellipses)
            input_text = render_moose_input(
                args.template, E_expr, f"run_{i:04d}", save_fields=args.save_stress_fields
            )

            try:
                E_eff, field = run_moose_once(
                    args.moose_exe, input_text, tmppath, f"run_{i:04d}",
                    save_field=args.save_stress_fields,
                )
            except RuntimeError:
                print(f"RVE {i} failed; skipping.")
                continue

            images.append(img)
            moduli.append(E_eff)
            ellipses_all.append(ellipses)
            if args.save_stress_fields:
                stress_fields.append(field)
            if (i + 1) % 20 == 0:
                print(f"  done {i+1}/{args.n}; current modulus stats: "
                      f"mean={np.mean(moduli):.3f}, std={np.std(moduli):.3f}")

    if not images:
        sys.exit("No microstructures succeeded.")

    images_arr = np.stack(images, axis=0)
    moduli_arr = np.array(moduli, dtype=np.float32)

    savez_kwargs = dict(
        images=images_arr,
        moduli=moduli_arr,
        ellipses=np.array(ellipses_all, dtype=object),
        E_matrix=E_MATRIX,
        E_inclusion=E_INCLUSION,
        nu=NU_BOTH,
        grid_size=GRID,
    )
    if args.save_stress_fields:
        savez_kwargs["stress_fields"] = np.stack(stress_fields, axis=0).astype(np.float32)

    np.savez(args.out, **savez_kwargs)
    print(f"\nWrote {len(images)} microstructures to {args.out}")
    if args.save_stress_fields:
        print(f"  + stress_fields shape {savez_kwargs['stress_fields'].shape}")
    print(f"Modulus range: {moduli_arr.min():.3f} -- {moduli_arr.max():.3f}")
    print(f"Modulus mean: {moduli_arr.mean():.3f}, std: {moduli_arr.std():.3f}")


if __name__ == "__main__":
    main()
