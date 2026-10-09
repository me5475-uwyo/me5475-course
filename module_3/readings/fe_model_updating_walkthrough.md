# Finding E and ν with Finite Elements: Model Updating for the L18 Plate — A Walkthrough

**Optional — not required.** Nothing in this reading is needed for Lab 3, any other lab, or any exam. It is for students with some FE and optimization background who want to see L18's inverse problem solved without a PINN.

*Course reading · ME 5475 · Module 3 · **[Optional]**, after L18 (inverse problems). Numbers cite `module_3/readings/measured_results.md` §10 (FE runs) and §4 (PINN runs).*

**After reading this, you should be able to:**

- pose L18's inverse problem as an optimization problem around an FE model: finite element model updating (FEMU);
- run it on ARCC for the Lab 3 plate, with the course MOOSE input and no edits to it;
- run a local sensitivity screen before optimizing, and say what it does and does not show;
- compare its cost with the PINN's on this plate, and project that cost to other problems.

---

## 1 · Quick start: files, running it, results

### Where the files are

The first four are in your course repo under `module_3/examples/`, and on ARCC in `/project/me5475/examples/`:

- `fe_model_updating.py`: the driver (data, misfit, optimizer, MOOSE calls);
- `fe_model_updating.sbatch`: the ARCC job script (CPU);
- `plate_square_hole_reference.i`: the Lab 3 plate in MOOSE (from L14), used unchanged;
- `extract_moose_reference.py`: its sampler (from L14), reused to read MOOSE's output;
- `measurements.csv`: yours, from Lab 3 Task 5a, the same file `plate_with_hole_inverse.py` reads.

### How to run it (on ARCC, where MOOSE is)

```bash
mkdir femu && cd femu
cp /project/me5475/examples/{fe_model_updating.py,fe_model_updating.sbatch,extract_moose_reference.py,plate_square_hole_reference.i} .
cp ../path/to/your/measurements.csv .                # the Task 5a file
sbatch fe_model_updating.sbatch                      # seed 0: displacements + force
sbatch fe_model_updating.sbatch --seed 1             # another measurement draw
sbatch fe_model_updating.sbatch --seed 0 --no-force  # displacements only
sbatch fe_model_updating.sbatch --nt 48 --nr 96      # a finer inversion mesh
```

- **Resources:** partition `mb` (CPU), 1 core, 2 GB, at most 10 minutes; no GPU. That is sized to what the runs used (peak memory about 0.56 GB on the 24/48 mesh and 1.4 GB on 48/96; about 1–2 and 3.5 minutes per job), so a small job can start sooner in a busy queue. It covers those two meshes only.
- **Output:** the log `femu-<jobid>.out`, and `femu_seed<k>_<data>_<mesh>_<jobid>.json` with the recovered values, local standard errors, singular values, the number of solves and the time. Each run works in its own new folder under `femu_runs/` and removes only that folder. A job canceled or killed mid-run leaves its own folder behind; it holds only that run's files and can be deleted.
- **Exit status:** nonzero if the optimizer does not report success; the result file then says so and carries no standard errors.
- **The runtime** is the same MOOSE build and module stack as `run_plate_reference.sbatch`.

### What you should see

The end of seed 0's log (ledger §10, job 24429484):

```text
Force measurement: 1.06876 (reference 1.07345, 1.0% noise)
local sensitivity screen at (E, nu) = (0.5, 0.25), step 0.0001: singular values 4.999e+02, 9.735e+01
optimizer: success=True, status=4: Both `ftol` and `xtol` termination conditions are satisfied.
seed=0 data=u and F mesh=24/48: E = 0.9988, nu = 0.2953  (local s.e. E 0.0101, nu 0.0018)
singular values of the weighted Jacobian at the solution: 5.691e+02, 9.945e+01
17 MOOSE solves, 100 s in MOOSE runs, 106 s in all
wrote femu_seed0_u_and_F_24x48_24429484.json
```

### The results in one look

| seed | FE model updating (E, ν) | PINN, same data (E, ν), §4 |
|---|---|---|
| 0 | 0.9988, 0.2953 | 0.9992, 0.2949 |
| 1 | 1.0002, 0.3006 | 1.0005, 0.3005 |
| 2 | 1.0089, 0.2964 | 1.0084, 0.2968 |

- **Close agreement:** the two methods agree to within about 0.001 in E and ν on these three data sets. The truth is (1.0, 0.3).
- **Cost on this plate:** FE took 17 MOOSE solves and one to two minutes per seed on one CPU core, depending on the node. The PINN took 8 min 40 s to 9 min 18 s per seed on a GPU (§4).
- **Without the force,** the FE route cannot find E either. Its screen stops before optimizing, in agreement with L18 slide 5's physics.

The rest of this reading explains how it works (sections 2–7), gives the full results (8), the cost and a projection to other problems (9), and the FE-or-PINN question (10).

---

## 2 · The idea: an optimizer around the FE model

L18 recovered E and ν with a PINN: E and ν became trainable scalars in the network's loss. FEMU keeps the FE model exactly as it is and wraps an optimizer around it:

1. guess (E, ν);
2. solve the plate with FE;
3. read the predicted displacements at the 20 measured points, and the predicted reaction force;
4. compare them with the measurements;
5. update (E, ν), and repeat until the misfit stops falling.

The PINN does steps 2–5 in one training run. FEMU does them as a loop of ordinary solves. Two ingredients make the loop work: a **misfit** to minimize (section 4), and the **gradient** of that misfit with respect to (E, ν), which tells the optimizer which way to step (section 6). **Everything L18 said about identifiability applies unchanged:** it is a property of the experiment and the equations, not of the solver.

The *optimization pattern* is general and does not use L18 slide 5's shortcut for this plate (u depends on ν only, and F = E·F̂(ν)). The script itself is an adapter for the Lab 3 plate. Another problem needs its own **forward model** (the input and its parameters) and **observation operator** (what is measured, and how it is read from the FE output).

## 3 · The data: exactly Task 5's

`fe_model_updating.py` draws its data with the same code, in the same order, as `plate_with_hole_inverse.py` (L18 slide 8):

- `--seed` picks the same 20 of the 200 points in `measurements.csv`;
- it adds the same displacement noise and the same 1 % force noise.

So seed 0 here is seed 0 there. As a check, both scripts print the same force measurement for each seed: 1.06876, 1.07403 and 1.08040 for seeds 0–2 (reference 1.07345; §4, §10).

## 4 · The misfit

Stack every data mismatch, each **divided by its noise level**, into one residual vector:

r(E, ν) = [ (u_FE(xᵢ; E, ν) − uᵢ,meas) / σ_u ,  …  ,  (F_FE(E, ν) − F_meas) / σ_F ]

- **Size:** 20 points × 2 components give 40 displacement entries, plus 1 force entry: 41 in all.
- **Objective:** minimize ½‖r‖². Dividing by the noise levels makes every entry dimensionless and comparable.
- **The ideal case:** if the noise is independent and Gaussian with *known, fixed* standard deviations, this minimizer is the maximum-likelihood estimate.
- **What the script does:** it does not know the true noise levels. It uses plug-in estimates computed from the measured values (1 % of the largest measured |u|, and 1 % of the measured force) and holds them fixed during the fit. So this is weighted least squares that approximates the ideal case.

In the code, this is the whole objective:

```python
def residuals(p):
    """Weighted misfit: each entry is (model - measured) / its noise level."""
    E, nu = p
    u, F = model(E, nu)
    r = ((u - u_meas) / sig_u).ravel()
    return r if args.no_force else np.concatenate([r, [(F - F_meas) / sig_F]])
```

Compare the PINN's loss: there, the same three kinds of mismatch were 15 *unweighted* mean squares, side by side with the physics residuals (L18 slides 6 and 11). Here the physics is not in the misfit at all. The FE solve satisfies it.

## 5 · The forward model: one MOOSE solve per guess

`model(E, nu)` runs the course input with **command-line overrides** (the L13 reading introduced them), so the input file is never edited:

```python
cmd = [MOOSE_EXE, "-i", str(INPUT), "--color", "off", *self.mesh,
       f"Materials/elasticity_tensor/youngs_modulus={float(E):.17g}",
       f"Materials/elasticity_tensor/poissons_ratio={float(nu):.17g}",
       f"Outputs/file_base={base}"]
```

Then:

- **Displacements at the 20 points:** read from the Exodus output, and interpolated with `extract_moose_reference.py`'s function, the same one that built `measurements.csv`.
- **The force:** the reaction on the loaded edge, `reaction_right_x`, from the run's CSV.

**Which mesh.** The measurements came from the finest mesh, 192/384 (§1). The inversion uses the coarse 24/48 mesh by default (`--nt 24 --nr 48`), which solves in a few seconds on one core. Using a *different* mesh is deliberate. Inverting synthetic data with the very discretization that generated it is called the **inverse crime**: it removes one source of model discrepancy, so the test is optimistic. It does not remove the measurement noise. The coarse mesh differs from the finest by only 9.4e-6 in relative L2 of u (§1), far below the 1 % noise, and one finer inversion mesh (48/96) changes E by 6e-5 and ν by 5e-7 (section 8).

## 6 · The optimizer, and where the gradients come from

`scipy.optimize.least_squares(..., method="lm")`, the **Levenberg–Marquardt** method. Each iteration:

- linearizes the residuals around the current guess, r(p + δ) ≈ r(p) + J δ, where **J** is the 41 × 2 matrix of sensitivities ∂rᵢ/∂(E, ν);
- solves for the step δ that minimizes the linearized misfit, damped toward a small gradient step when the linearization is poor. This is Gauss–Newton with a safety net.

**Where J comes from: finite differences.** Perturb E and solve; perturb ν and solve (a relative step of 1e-4). That is one extra FE solve per parameter for each Jacobian, which is cheap with two parameters. With many parameters it is not (section 9). There, **adjoint methods** make the gradient of the scalar misfit affordable: about one extra linear solve, however many parameters there are. An adjoint gives that gradient, not the whole Jacobian, so it pairs with gradient-based optimizers. MOOSE has an optimization module built for this; it is not set up in the course build, so it is named here only.

**Start:** (0.5, 0.25), where the PINN starts. **No constraints:** Levenberg–Marquardt here is unconstrained, so nothing keeps E > 0 or −1 < ν < 0.5 in general, the range in which the shear and bulk moduli are positive. The script flags a result outside that range, and exits with a nonzero status if the optimizer does not report success. (Lab 3 Task 5c flags ν outside (0, 0.5), a narrower course choice that excludes auxetic materials.)

## 7 · A local sensitivity screen before optimizing

L18 slide 7: ask what each measurement can see, *before* you train. FEMU lets you compute a first answer. At the start point (0.5, 0.25), the script computes J by forward differences (absolute step 1e-4 in the present units of E and ν; two extra solves) and its **singular values**:

- **Both clearly nonzero:** near the start point, the misfit responds to every direction in (E, ν). Go ahead.
- **The smaller below 10⁻⁶ times the larger:** near the start point, the misfit is insensitive, to first order, along some direction. The script names that direction and stops. That is a diagnostic policy, not a proof.

**What the screen does not show.** It is local and first order. A zero derivative at one point does not prove that the misfit never changes in that direction: the derivative of p² is zero at p = 0, yet p² changes. And two nonzero singular values at one point do not prove a unique or precise answer. On another problem, scale the parameters sensibly, and check other points and step sizes.

**On this plate it agrees with the physics.** Without the force (`--no-force`), the screen stops, and L18 slide 5 says why: the displacements do not depend on E at all.

## 8 · Full results (ledger §10)

ARCC, partition `mb`, 1 core, 2026-10-09. 24/48 inversion mesh unless stated.

| run | E | ν | MOOSE solves | job time |
|---|---|---|---|---|
| seed 0 | 0.9988 | 0.2953 | 17 | 1 min 05 s |
| seed 1 | 1.0002 | 0.3006 | 17 | 56 s |
| seed 2 | 1.0089 | 0.2964 | 17 | 56 s |
| seed 0, 48/96 mesh | 0.9987 | 0.2953 | 17 | 3 min 25 s |
| seed 0, no force | stops | stops | 3 | 15 s |

**Stops:** the screen found the misfit insensitive to E at the start point, so nothing was optimized. Job time is SLURM's elapsed time for the whole job, start-up included. Each seed's solve count includes the two of the screen.

What the table shows:

- **The estimates agree closely** on these three data sets, to within about 0.001 in E and ν. That is consistent with measurement noise contributing to the departures from the truth (1.0, 0.3). It does not separate data, optimization and modeling errors.
- **One finer mesh barely matters:** the 48/96 inversion mesh moves E by 6e-5 and ν by 5e-7, at about three times the cost.
- **Without the force:** at the start point the two singular values are 499 and 2.9e-10, and the insensitive direction is (dE, dν) = (1, 0): pure E. The script stops after three solves. Slide 5's physics says the same.
- **Local standard errors** (seeds 0–2): 0.010 for E and 0.0017–0.0022 for ν, from (JᵀJ)⁻¹ at the solution. This is a local covariance approximation, conditional on the plug-in weights and the measured locations. It leaves out noise-scale estimation, the choice of points, model discrepancy and optimization variability, so it is not the same object as Task 5's seed spread.
- **Seed 0's ν**, 0.2953, sits 2.7 of its local standard errors from 0.3. With three seeds, and with the qualifications above, you cannot tell whether that is chance or the local estimate being optimistic.

## 9 · Cost: this plate, and a projection to other problems

**Measured on this plate** (§4, §10):

| | FE model updating | PINN |
|---|---|---|
| hardware | 1 CPU core | 1 GPU |
| one seed | 17 solves, 1–2 min per job | 8 min 40 s – 9 min 18 s per job |
| time per FE solve | 2.9–3.5 s on `mb`, 5.9 s on another node (24/48 mesh); 11.3 s (48/96) | — |

The time per FE solve is the MOOSE run time divided by the number of solves. It includes the job-step launch and MOOSE's start-up, and excludes the Python interpolation and bookkeeping.

The FEMU cost has a simple, approximate structure: cost ≈ (number of solves) × (time per solve). With finite differences, each Jacobian costs one solve per parameter beyond the evaluation at the current point. The measured total here is 17 solves: 2 for the screen, and 15 inside the optimizer (its function evaluations and finite-difference Jacobian columns together).

**A projection, not a measurement.** The cases below apply that formula with this plate's 3.5 s per solve (or the stated solve time). They are arithmetic, not runs, and real problems may need more iterations.

- **10 material constants instead of 2.** About 11 solves per iteration (the current point plus 10 for the Jacobian): about 40 s per iteration. Still minutes; FEMU stays the easy route.
- **A nonlinear 3-D model at 10 minutes per solve.** 17 solves take about 3 hours. FEMU then costs hours, not seconds. A 3-D nonlinear PINN is also much harder to train (L17); not measured.
- **A stiffness value in every element: E(x) on the 24/48 mesh, 2 304 unknowns.** A finite-difference Jacobian needs 2 305 solves, about 2.2 hours per iteration. Adjoint gradients (section 6) make this practical, with a gradient-based optimizer.
- **Many specimens, each with its own data set.** FEMU repeats its whole loop for each specimen. Two network routes could shorten this, and they are different. (a) A *forward* surrogate, like L16's parametric network, maps (E, ν) and x to the fields; it can replace the FE solves *inside* each specimen's inverse search, but that search remains, and the network must first be trained. (b) An *inverse* network maps a specimen's measurements directly to (E, ν); that needs a different training setup, costs training up front, and still faces any non-uniqueness in the data. Neither is measured here.

Two points apply to both methods:

- **More unknowns need more information.** Recovering 2 304 unrestricted values from 41 observations is underdetermined. It needs more independent observations (full-field DIC is one option) and/or restrictions or priors (regularization), for FEMU and for a PINN alike.
- **What scales.** FEMU's finite-difference cost grows with the number of parameters, one solve each per Jacobian. A PINN has no such per-parameter multiplier, but how its training cost changes with the parameterization and the physics is not measured in this course.

## 10 · FE or PINN?

On **this** problem, the evidence favors FEMU: two scalar unknowns, a trusted FE model and seconds per solve. That is one plate, one loading and three seeds, measured here and in §4; it is not a general verdict.

The usual case made for a PINN is a problem unlike this one:

- **The unknown is a function**, such as a stiffness map E(x) or a constitutive law. FEMU then works with many parameters, where adjoint gradients make it practical; a PINN adds the field as another network output (L18 slide 15). Either way, regularization or extra data is needed.
- **No solver loop:** one optimization fits the field and the parameters together, with sparse, noisy data in the same loss.
- **No convenient FE implementation**, or a mesh that is hard to build, even though the governing equations are known. A PINN still needs those equations and the boundary and measurement assumptions.

Those are arguments made in the literature; this course has not measured them.

What carries over either way is L18's main point: **check what the data can see before you trust any fit.** In FEMU, a first check is the local sensitivity screen; on this plate, without the force, it agrees with slide 5's physics.

## Questions to think about

1. Why does each Levenberg–Marquardt Jacobian cost two extra FE solves here? How would that scale with 1 000 parameters, and what would you use instead?
2. Run `--no-force`. Which direction in (E, ν) does the screen report as insensitive, and what does slide 5's physics say about it?
3. Why is inverting on the 192/384 mesh that generated the data an optimistic validation, even though the measurement noise remains? (Do not run it with this wrapper: that mesh needs about 48 GB.)
4. The weights σ_u and σ_F are plug-in estimates from the measured values. What happens to the answer if you set σ_F ten times too large? Too small?
5. Pick a problem from your own research. Where would it sit in section 9's projection, and which route would you try first?
