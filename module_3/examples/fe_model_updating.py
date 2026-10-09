"""
fe_model_updating.py
====================

Module 3 -- optional companion to L18 (module_3/readings/fe_model_updating_walkthrough.md).

L18's inverse problem, solved with finite elements instead of a PINN: recover E and nu of
the Lab 3 plate from 20 noisy displacement points plus one noisy reaction force.  This is
finite element model updating (FEMU): an optimizer wrapped around ordinary MOOSE solves.

Same data as plate_with_hole_inverse.py.  --seed picks the same 20 of the 200 points in
measurements.csv and draws the same noise, with the same code in the same order (lines
marked "as plate_with_hole_inverse.py"), so seed 0 here is seed 0 there.

Forward model: one MOOSE solve of plate_square_hole_reference.i with command-line overrides
(E, nu and the mesh).  No input file is edited.  The displacement is sampled at the 20
points with extract_moose_reference.py's interpolator; the force is the solve's
reaction_right_x.  By default the inversion uses the coarse 24/48 mesh: the data came from
the finest mesh (192/384), and inverting with the very discretization that made the data
(the "inverse crime") removes one source of model discrepancy, an optimistic test.  The
wrapper's 2 GB / 10 min covers 24/48 and 48/96 (measured peaks 0.56 and 1.4 GB); 192/384 needs about 48 GB.

The optimization pattern is general FEMU, with no problem-specific shortcut: joint (E, nu)
weighted least squares by Levenberg-Marquardt (scipy.optimize.least_squares, method="lm",
relative finite-difference step 1e-4), one extra solve per parameter per Jacobian.  It starts
where the PINN starts, (0.5, 0.25).  This file is an adapter for the Lab 3 plate: another
problem needs its own forward model (input, parameters) and observation operator (what is
measured, and how it is read from the FE output).  LM here is unconstrained: nothing keeps
E > 0 or -1 < nu < 0.5 (positive shear and bulk moduli) in general; a result outside that
range is flagged.  (Lab 3 Task 5c flags nu outside (0, 0.5): a narrower, non-auxetic course choice.)

Weights: 1 % of the largest measured |u| and of the measured force -- plug-in estimates of
the noise levels, held fixed during the fit.  The reported standard errors come from
(J^T J)^-1 at the solution: a local covariance approximation, conditional on those weights
and on the measured locations.  They leave out noise-scale estimation, point selection,
model discrepancy and optimization variability.

Before optimizing, a LOCAL SENSITIVITY SCREEN (L18 slide 7: ask before you train): the
weighted sensitivities at the start point (0.5, 0.25), by forward differences with an
absolute step 1e-4 in the present units of E and nu (two extra solves), and their singular
values.  If the smaller is below 1e-6 times the larger, the misfit is insensitive to first
order along some direction there; the script names it and stops (a diagnostic policy, not a
general proof of non-identifiability; elsewhere, scale the parameters sensibly and check
other points and steps).  For this plate without the force, the screen agrees with the
physics of L18 slide 5: the displacements do not depend on E.

Each run works in its own new folder under --workdir and removes only that folder (a job
killed mid-run, e.g. by scancel, leaves its own folder behind).  Results
go to femu_<seed>_<data>_<mesh>_<run id>.json (run id = the SLURM job id, or a time stamp
and process id); an existing file of that name is never overwritten.  An explicit --save
path is overwritten.  The exit status is nonzero if the optimizer does not report success.

--no-force drops the force from the misfit (displacements only): L18 slide 5's question,
asked of the FE route.

Run on ARCC (MOOSE lives there) with fe_model_updating.sbatch, or inside any job/allocation
that has the course MOOSE runtime loaded (see run_plate_reference.sbatch).

    python fe_model_updating.py --measurements measurements.csv --seed 0
    python fe_model_updating.py --measurements measurements.csv --seed 0 --no-force
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_moose_reference import interpolate_fields, read_moose_exodus  # noqa: E402

SHARED_FORCE_CSV = "/project/me5475/examples/lab3_reference/plate_square_hole_reference_out.csv"
MOOSE_EXE = "/project/me5475/software/rom_opt_arcc/rom_opt-opt"
INPUT = Path(__file__).resolve().parent / "plate_square_hole_reference.i"


def draw_measurements(meas_csv, force_csv, seed, n_meas, noise_level):
    """The measurement model, as plate_with_hole_inverse.py lines 145-157 (same draws, same order)."""
    df = pd.read_csv(meas_csv)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_meas, replace=False)
    meas = df.iloc[idx].reset_index(drop=True)
    sigma_noise = noise_level * np.abs(meas[["u_1", "u_2"]].values).max()
    meas["u_1_noisy"] = meas["u_1"] + rng.normal(0, sigma_noise, n_meas)
    meas["u_2_noisy"] = meas["u_2"] + rng.normal(0, sigma_noise, n_meas)
    F_true = float(pd.read_csv(force_csv)["reaction_right_x"].iloc[-1])
    F_meas = F_true * (1.0 + noise_level * rng.normal())
    return meas, F_meas, F_true


class PlateModel:
    """One MOOSE solve per call: (E, nu) -> displacements at the measured points, and the reaction."""

    def __init__(self, points, workdir, nt, nr, ranks):
        self.points = points
        self.workdir = Path(workdir)                      # a folder this run owns (see main)
        self.mesh = [f"nt={nt}", f"nr={nr}"]
        self.ranks = ranks
        self.n_solves = 0
        self.solve_seconds = 0.0
        self.cache = {}

    def __call__(self, E, nu):
        key = (round(float(E), 12), round(float(nu), 12))
        if key in self.cache:
            return self.cache[key]
        self.n_solves += 1
        base = self.workdir / f"solve_{self.n_solves:04d}"
        cmd = [MOOSE_EXE, "-i", str(INPUT), "--color", "off", *self.mesh,
               f"Materials/elasticity_tensor/youngs_modulus={float(E):.17g}",
               f"Materials/elasticity_tensor/poissons_ratio={float(nu):.17g}",
               f"Outputs/file_base={base}"]
        if os.environ.get("SLURM_JOB_ID"):
            cmd = ["srun", f"--ntasks={self.ranks}", "--quiet", *cmd]
        t0 = time.time()
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.solve_seconds += time.time() - t0
        if run.returncode != 0:
            sys.exit(f"MOOSE failed at E={E}, nu={nu}:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
        coords, fields = read_moose_exodus(Path(f"{base}.e"))
        u = np.column_stack([interpolate_fields(coords, fields[v], self.points) for v in ("disp_x", "disp_y")])
        F = float(pd.read_csv(f"{base}.csv")["reaction_right_x"].iloc[-1])
        for ext in (".e", ".csv"):                       # keep the work folder small
            Path(f"{base}{ext}").unlink(missing_ok=True)
        self.cache[key] = (u, F)
        return u, F


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--force-csv", type=Path, default=Path(SHARED_FORCE_CSV))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-measurements", type=int, default=20)
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--no-force", action="store_true", help="displacements only")
    p.add_argument("--nt", type=int, default=24)
    p.add_argument("--nr", type=int, default=48)
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--workdir", type=Path, default=Path("femu_runs"),
                   help="parent folder; each run makes and removes its own subfolder")
    p.add_argument("--save", type=Path, default=None,
                   help="result file (overwritten if it exists); default: a new, unique name")
    args = p.parse_args()

    meas, F_meas, F_true = draw_measurements(args.measurements, args.force_csv, args.seed,
                                             args.n_measurements, args.noise_level)
    print(f"Force measurement: {F_meas:.5f} (reference {F_true:.5f}, {args.noise_level*100:.1f}% noise)")
    # Weights: plug-in noise levels, computed from the measured (noisy) values and held fixed.
    sig_u = args.noise_level * np.abs(meas[["u_1_noisy", "u_2_noisy"]].values).max()
    sig_F = args.noise_level * abs(F_meas)

    tag = f"seed{args.seed}_{'u_only' if args.no_force else 'u_and_F'}_{args.nt}x{args.nr}"
    run_id = os.environ.get("SLURM_JOB_ID") or f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    save = args.save or Path(f"femu_{tag}_{run_id}.json")
    if args.save is None and save.exists():
        sys.exit(f"{save} already exists; not overwriting it (pass --save to choose a name)")
    args.workdir.mkdir(parents=True, exist_ok=True)
    rundir = Path(tempfile.mkdtemp(prefix=f"{tag}_{run_id}_", dir=args.workdir))   # new, owned by this run
    try:
        code = fit(args, meas, F_meas, sig_u, sig_F, tag, rundir, save)
    finally:
        shutil.rmtree(rundir, ignore_errors=True)        # only the folder this run created
    sys.exit(code)


def fit(args, meas, F_meas, sig_u, sig_F, tag, rundir, save):
    """Screen, then fit; write the result file.  Returns the exit status."""
    pts = meas[["x_1", "x_2"]].values
    u_meas = meas[["u_1_noisy", "u_2_noisy"]].values
    model = PlateModel(pts, rundir, args.nt, args.nr, args.ranks)
    t_start = time.time()
    out = {"seed": args.seed, "data": "u only" if args.no_force else "u and F", "mesh": f"{args.nt}/{args.nr}",
           "F_meas": F_meas, "sigma_u_plugin": sig_u, "sigma_F_plugin": sig_F}

    def finish(code):
        out.update(n_solves=model.n_solves, solve_seconds=round(model.solve_seconds, 1),
                   wall_seconds=round(time.time() - t_start, 1))
        print(f"{model.n_solves} MOOSE solves, {model.solve_seconds:.0f} s in MOOSE runs, "
              f"{out['wall_seconds']:.0f} s in all")
        save.write_text(json.dumps(out, indent=1))
        print(f"wrote {save}")
        return code

    from scipy.optimize import least_squares

    def residuals(p):
        """Weighted misfit: each entry is (model - measured) / its noise level."""
        E, nu = p
        u, F = model(E, nu)
        r = ((u - u_meas) / sig_u).ravel()
        return r if args.no_force else np.concatenate([r, [(F - F_meas) / sig_F]])

    # --- local sensitivity screen at the start point, before any optimizing ---
    x0, h, rel_tol = np.array([0.5, 0.25]), 1e-4, 1e-6
    r0 = residuals(x0)
    J0 = np.column_stack([(residuals(x0 + h * np.eye(2)[j]) - r0) / h for j in range(2)])
    _, sv0, Vt0 = np.linalg.svd(J0, full_matrices=False)
    print(f"local sensitivity screen at (E, nu) = (0.5, 0.25), step {h:g}: "
          f"singular values {sv0[0]:.3e}, {sv0[1]:.3e}")
    out.update(screen_point=x0.tolist(), screen_step=h, screen_rel_threshold=rel_tol,
               screen_singular_values=sv0.tolist())
    if sv0[1] < rel_tol * sv0[0]:
        v = Vt0[1]
        msg = (f"SCREEN: at the start point the misfit is insensitive, to first order, along "
               f"(dE, dnu) = ({v[0]:+.4f}, {v[1]:+.4f}).  Stopping before optimizing (diagnostic policy).")
        print(msg)
        out.update(screen_rank_deficient=True, insensitive_direction=v.tolist(), message=msg)
        return finish(0)
    out["screen_rank_deficient"] = False

    res = least_squares(residuals, x0=x0, method="lm", diff_step=h, x_scale=[1.0, 0.3])
    E, nu = float(res.x[0]), float(res.x[1])
    admissible = E > 0 and -1 < nu < 0.5               # positive shear and bulk moduli
    out.update(E=E, nu=nu, cost=float(res.cost), nfev=int(res.nfev), success=bool(res.success),
               status=int(res.status), message=res.message, admissible=admissible)
    print(f"optimizer: success={res.success}, status={res.status}: {res.message}")
    if not res.success:
        print(f"FAILED: the optimizer stopped without success at E = {E:.4f}, nu = {nu:.4f}; "
              f"no standard errors reported.")
        return finish(3)
    J = res.jac                                          # weighted sensitivities, (n_data, 2)
    sv = np.linalg.svd(J, compute_uv=False)
    out["jac_singular_values"] = sv.tolist()
    if sv[-1] > 1e-8 * sv[0]:
        cov = np.linalg.inv(J.T @ J)                     # local; residuals already divided by sigma
        out.update(E_se_local=float(np.sqrt(cov[0, 0])), nu_se_local=float(np.sqrt(cov[1, 1])),
                   corr_local=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))
    print(f"seed={args.seed} data={out['data']} mesh={out['mesh']}: E = {E:.4f}, nu = {nu:.4f}"
          + (f"  (local s.e. E {out['E_se_local']:.4f}, nu {out['nu_se_local']:.4f})" if "E_se_local" in out
             else "  (weighted Jacobian is rank-deficient: no standard errors)"))
    print(f"singular values of the weighted Jacobian at the solution: {', '.join(f'{v:.3e}' for v in sv)}")
    if not admissible:
        print("WARNING: outside the isotropic elastic range (need E > 0 and -1 < nu < 0.5); "
              "report the run, do not drop it.")
    return finish(0)


if __name__ == "__main__":
    main()
