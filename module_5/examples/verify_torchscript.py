"""
verify_torchscript.py
=====================

Module 5 / Lab 5 Task 1 -- check an exported TorchScript constitutive network BEFORE MOOSE uses it.
It runs the network the way the course material ComputeTorchScriptStress does (converted to
float64, one (1, 3) strain at a time is the same as a batch here) and prints four checks.

What decides RESULT: PASS or FAIL
  1. Stress at 10 test strains, against plane-strain Hooke's law (E = 1, nu = 0.3 unless
     --E / --nu say otherwise). A judged test passes if the relative error of the stress vector
     is below --tolerance (default 5 %). With --data (your Lab 1 CSV), a test strain outside the
     convex hull of ALL the strains in that file is marked "outside" and is not judged. The CSV
     holds Lab 1's training, validation and test rows together, so this "full dataset hull" is a
     coverage proxy: it is larger than the hull of the rows the network was actually trained on.
     At least --min-judged tests (default 5) must be judged; with fewer, the file is NOT
     validated, and the result is FAIL.
  4. (with --checkpoint) The TorchScript file against the eager PyTorch model rebuilt from the
     Lab 1 checkpoint, on the 10 test strains and 1000 random strains in [-0.005, 0.005]^3:
     the float32 file against the float32 model (FAIL above --export-tol, default 1e-5, times the
     largest stress), and the float64 copy MOOSE runs against the model in float64 (FAIL above
     1e-12 times the largest stress; a dtype cast inside forward() fails here). An export bug
     shows up in check 4, a training error does not. Without --checkpoint this check is not run,
     and the result says so.
  Any non-finite stress or tangent anywhere is a FAIL.

Diagnostics (printed, not part of PASS/FAIL)
  2. The stress at ZERO strain. A trained network rarely returns exactly 0 there. Inside MOOSE
     this offset acts at every quadrature point, like a built-in residual stress.
  3. The tangent d sigma / d (eps_xx, eps_yy, gamma_xy), by autograd, exactly as the MOOSE
     material computes it for Newton's method, at zero strain and at one test strain, against
     the plane-strain stiffness C.

Exit status 1 on FAIL, so a job script can stop before MOOSE runs.

Usage (seconds; fine on a login node)
-----
    python verify_torchscript.py --model mlp_constitutive_ts.pt --data single_element.csv \
        --checkpoint mlp_constitutive.pt | tee verify_output.txt
"""

from __future__ import annotations

import argparse
import copy
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

F64_TOL = 1e-12   # check 4: float64 copy vs the eager model in float64, as a fraction of the largest stress

# Test strains (eps_xx, eps_yy, gamma_xy) inside the cube [-0.005, 0.005]^3 that Lab 1's driver
# requests. The single element of Lab 1 carries a mix of the requested strains, so some of these can
# lie outside the strains actually in the CSV; --data checks that.
TESTS = [
    ([0.001, 0.0, 0.0], "uniaxial strain, x"),
    ([0.0, 0.001, 0.0], "uniaxial strain, y"),
    ([0.0, 0.0, 0.001], "pure shear (gamma)"),
    ([0.002, 0.0, 0.0], "uniaxial strain, x, doubled"),
    ([0.001, -0.001, 0.0], "pure shear (principal axes)"),
    ([0.001, 0.001, 0.0], "equibiaxial"),
    ([0.0015, -0.0008, 0.0005], "general"),
    ([-0.001, 0.0, 0.0], "uniaxial compression, x"),
    ([0.0, 0.0, -0.002], "negative shear"),
    ([0.002, 0.002, 0.002], "combined, larger"),
]


def plane_strain_C(E: float, nu: float) -> np.ndarray:
    """Plane-strain Voigt stiffness for (eps_xx, eps_yy, gamma_xy); C33 = mu (engineering shear)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    return f * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, (1 - 2 * nu) / 2]])


def tangent(model, eps: list[float]) -> np.ndarray:
    """J[a, b] = d sigma_a / d v_b by autograd, one row per output, as ComputeTorchScriptStress does."""
    x = torch.tensor([eps], dtype=torch.float64, requires_grad=True)
    y = model(x).reshape(3)
    rows = [torch.autograd.grad(y[a], x, retain_graph=a < 2)[0][0] for a in range(3)]
    return torch.stack(rows).detach().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="TorchScript file from export_torchscript.py")
    p.add_argument("--tolerance", type=float, default=0.05, help="max relative stress error per test (default 0.05)")
    p.add_argument("--E", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Lab 1 checkpoint: compare the file with the eager model (check 4)")
    p.add_argument("--export-tol", type=float, default=1e-5,
                   help="check 4, float32: max |file - eager| as a fraction of the largest stress (default 1e-5)")
    p.add_argument("--data", type=Path, default=None,
                   help="Lab 1 CSV: tests outside the hull of all its strains are not judged")
    p.add_argument("--min-judged", type=int, default=5,
                   help="fewest judged tests for a PASS (default 5 of 10)")
    args = p.parse_args()
    failures: list[str] = []

    print(f"Loading {args.model} and converting it to float64 (as the MOOSE material does)")
    model = torch.jit.load(str(args.model)).to(torch.float64).eval()
    C = plane_strain_C(args.E, args.nu)

    # --- 1. stress against Hooke ---
    eps = np.array([t[0] for t in TESTS])
    with torch.no_grad():
        pred = model(torch.tensor(eps, dtype=torch.float64)).numpy()
    if pred.shape != (len(TESTS), 3):
        sys.exit(f"The network returned shape {pred.shape}; MOOSE needs (N, 3).")
    finite = np.isfinite(pred).all(axis=1)
    if not finite.all():
        failures.append(f"non-finite stress at test(s) {np.flatnonzero(~finite).tolist()}")
    truth = eps @ C.T
    inside = np.ones(len(TESTS), bool)
    if args.data is not None:
        import pandas as pd
        from scipy.spatial import Delaunay
        tr = pd.read_csv(args.data)[["eps_xx", "eps_yy", "gamma_xy"]].to_numpy()
        inside = Delaunay(tr).find_simplex(eps) >= 0
        print(f"Strains in {args.data}: {len(tr)} rows -- ALL rows (Lab 1 trained on its training split only); "
              "ranges " + ", ".join(f"[{tr[:, j].min():.2e}, {tr[:, j].max():.2e}]" for j in range(3)))
    print(f"\n1. Stress vs plane-strain Hooke (E = {args.E}, nu = {args.nu}); tolerance {args.tolerance:.0%}")
    print(f"{'#':<3}{'eps_xx':>9}{'eps_yy':>9}{'gamma':>9}  {'sigma network':<40}{'sigma Hooke':<40}{'rel err':>8}")
    n_pass = n_judged = 0
    for i, ((e, label), sp, st, ins) in enumerate(zip(TESTS, pred, truth, inside)):
        rel = np.linalg.norm(sp - st) / np.linalg.norm(st)
        ok = bool(np.isfinite(rel) and rel < args.tolerance)
        n_judged += ins
        n_pass += ok and ins
        verdict = ("PASS" if ok else "FAIL") if ins else "outside the full dataset hull (not judged)"
        print(f"{i:<3}{e[0]:>9.4f}{e[1]:>9.4f}{e[2]:>9.4f}  "
              f"{' '.join(f'{v: .4e}' for v in sp):<40}{' '.join(f'{v: .4e}' for v in st):<40}"
              f"{rel:>8.2%}  {verdict}  {label}")
    print(f"   passed {n_pass}/{n_judged} judged" + ("" if n_judged == len(TESTS) else
          f"; {len(TESTS) - n_judged} outside the hull of all the strains in {args.data}"))
    if n_pass < n_judged:
        failures.append(f"{n_judged - n_pass} judged test(s) above {args.tolerance:.0%}")
    if n_judged < args.min_judged:
        failures.append(f"NOT VALIDATED: only {n_judged} of {len(TESTS)} tests lie inside the data hull "
                        f"(at least {args.min_judged} needed)")

    # --- 2. stress at zero strain (diagnostic) ---
    with torch.no_grad():
        s0 = model(torch.zeros(1, 3, dtype=torch.float64)).numpy()[0]
    print("\n2. Stress at zero strain (exact material: 0) -- diagnostic, not part of PASS/FAIL")
    print(f"   sigma(0) = [{', '.join(f'{v: .3e}' for v in s0)}]"
          f"   = {np.abs(s0).max() / 1e-3:.2%} of a stress of 1e-3")
    if not np.isfinite(s0).all():
        failures.append("non-finite stress at zero strain")

    # --- 3. tangent (diagnostic) ---
    print("\n3. Tangent d sigma / d (eps_xx, eps_yy, gamma_xy) by autograd vs C -- diagnostic, not part of PASS/FAIL")
    print("   C =\n" + textwrap.indent(np.array2string(C, precision=4), "   "))
    for e in ([0.0, 0.0, 0.0], [0.0015, -0.0008, 0.0005]):
        J = tangent(model, e)
        if not np.isfinite(J).all():
            failures.append(f"non-finite tangent at eps = {e}")
        rel = np.linalg.norm(J - C) / np.linalg.norm(C)
        print(f"   at eps = {e}: ||J - C|| / ||C|| = {rel:.2%}; J =\n"
              + textwrap.indent(np.array2string(J, precision=4), "   "))

    # --- 4. file vs eager model ---
    if args.checkpoint is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_torchscript import MLP, StandardizedMLP
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        a = ck.get("args", {}) or {}
        m = MLP(hidden=a.get("hidden", 32), layers=a.get("layers", 4))
        m.load_state_dict(ck["model_state"])
        s = ck["scalers"]
        eager = StandardizedMLP(m.eval(), s["mu_x"], s["sigma_x"], s["mu_y"], s["sigma_y"]).eval()
        # Strains drawn in float64, as MOOSE passes them: a float32 cast inside forward() changes
        # these (it would not change strains that were float32 to begin with).
        g = torch.Generator().manual_seed(0)
        x64 = torch.cat([torch.tensor(eps, dtype=torch.float64),
                         (torch.rand(1000, 3, generator=g, dtype=torch.float64) * 2 - 1) * 5e-3])
        x32 = x64.float()
        file32 = torch.jit.load(str(args.model)).eval()
        # float64 reference built from the MLP and the stored standardization directly, NOT through
        # StandardizedMLP.forward, so a cast added to that forward() cannot hide on both sides
        mu_x, sd_x, mu_y, sd_y = (torch.tensor(s[k], dtype=torch.float32).flatten().double()
                                  for k in ("mu_x", "sigma_x", "mu_y", "sigma_y"))
        m64 = copy.deepcopy(m).double()
        with torch.no_grad():
            y_eager = eager(x32)
            d32 = (file32(x32) - y_eager).abs().max().item()
            d64 = (model(x64) - (m64((x64 - mu_x) / sd_x) * sd_y + mu_y)).abs().max().item()
        scale = y_eager.abs().max().item()
        tol32, tol64 = args.export_tol * scale, F64_TOL * scale
        print(f"\n4. File vs eager model from {args.checkpoint}, {len(x32)} strains "
              f"(the 10 tests + 1000 in [-0.005, 0.005]^3); largest stress {scale:.3e}:")
        print(f"   float32 file vs float32 model:            max |diff| = {d32:.3e}   (gate {tol32:.3e} = {args.export_tol:g} x largest)")
        print(f"   float64 copy (as MOOSE runs it) vs model in float64: max |diff| = {d64:.3e}   (gate {tol64:.3e} = {F64_TOL:g} x largest)")
        if not (np.isfinite([d32, d64, scale]).all() and d32 <= tol32 and d64 <= tol64):
            failures.append(f"the file does not reproduce the checkpoint's model (float32 {d32:.3e}, float64 "
                            f"{d64:.3e}): an export problem (or a dtype cast in forward), not a training error")
    else:
        print("\n4. File vs eager model: NOT RUN (no --checkpoint), so an export bug that stays within the")
        print("   Hooke tolerance would not be caught. Give --checkpoint for the Lab 5 record.")

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        if n_pass < n_judged:
            print("  Hooke tests off by ~1000x: the standardization is missing (check the StandardizedMLP buffers).")
            print("  Off by a constant factor on gamma: a Voigt factor-of-2 problem in the training data.")
            print("  Off by a few to tens of percent: the Lab 1 network is under-trained, or the strain is")
            print("  near the edge of its range. Check 4 tells an export bug from a training error.")
        sys.exit(1)
    print("\nRESULT: PASS" + ("" if args.checkpoint is not None else " (check 4 not run: no --checkpoint)"))


if __name__ == "__main__":
    main()
