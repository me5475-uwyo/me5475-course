# The MOOSE Material Interface, and the Network Material (Without Writing C++)

A student-facing reference for Lab 5: what MOOSE asks a material to compute, and what the course material
`ComputeTorchScriptStress` does with your network. Read it once; come back to it when a run fails. Numbers are in
`module_5/readings/measured_results.md` (cited as "ledger §n").

---

## What a material does in MOOSE

A MOOSE *material* computes properties at the quadrature points of each element, every time the residual or the
Jacobian is assembled. For small-strain solid mechanics the chain is:

```
displacements u (the unknowns)
   │
   ▼  strain calculator (ComputeSmallStrain, added by the QuasiStatic action)
mechanical strain ε at each quadrature point
   │
   ▼  stress calculator  ← the [Materials] block chooses this one
stress σ  and  the tangent dσ/dε (MOOSE calls it the Jacobian multiplier)
   │
   ▼  StressDivergenceTensors kernels (added by the action)
residual  R = ∫ σ : ∇v dV − ∫ t · v dA      and Newton's Jacobian, built from dσ/dε
```

Newton's method needs both pieces: σ for the residual, dσ/dε for the Jacobian. A wrong tangent does not change the
converged answer, but it can turn Newton's quadratic convergence into slow, linear convergence, or stop it.

For Lab 0's elastic plate the stress calculator is `ComputeLinearElasticStress` (σ = C : ε, with C from
`ComputeIsotropicElasticityTensor`), and the tangent is the constant C. The plate is linear, so Newton converges in
one iteration (ledger §3).

---

## The network material: `ComputeTorchScriptStress`

It replaces the pair of elastic materials with one block:

```
[Materials]
  [nn_stress]
    type     = ComputeTorchScriptStress
    filename = mlp_constitutive_ts.pt     # the TorchScript file from export_torchscript.py
  []
[]
```

`filename` is the only parameter you set. Everything else in the input file (mesh, action, boundary conditions,
solver) stays as it was. `plate_with_hole_nn_material.i` keeps both versions in one file and chooses with one line:

```
[Materials]
  active = 'nn_stress'                     # or: active = 'elasticity_tensor elastic_stress'
  ...
[]
```

**Where it comes from.** It is a course material, compiled into the course executable
`/project/me5475/software/moose_torch/me5475/me5475-opt` (MOOSE built with libtorch 2.5.1, the version of the course
Python environment). The older course executable `rom_opt-opt` does not have it; with that executable the run stops
at once with "A 'ComputeTorchScriptStress' is not a registered object" (ledger §5). MOOSE's own `TorchScriptMaterial` is not a substitute: it feeds a network
only from postprocessors (global numbers), not from the strain at each point.

### What it does at each quadrature point

From its source (C++, `ComputeTorchScriptStress::computeQpStress`), in words:

1. **Load, once.** At start-up it loads the file, converts the whole module to float64, and sets libtorch to one
   thread per MPI rank (run it with MPI ranks, as `run_nn_simulation.sbatch` does, not with MOOSE threads).
2. **Input.** It takes MOOSE's mechanical strain tensor and forms x = (ε_xx, ε_yy, γ_xy) with γ_xy = ε_xy + ε_yx =
   2 ε_xy: plane Voigt notation with engineering shear, exactly Lab 1's columns. x is a (1, 3) float64 tensor.
3. **Stress.** y = network(x). The material stops the run if y does not hold exactly 3 values, or if any of them is
   not finite (the message names the element, the quadrature point and the strain). Otherwise y gives σ_xx, σ_yy,
   σ_xy (σ_xy is written to both off-diagonal entries). **σ_zz is set to 0.** All other components are 0.
4. **Tangent.** Three backward passes through the network give J[a, b] = ∂σ_a/∂x_b by automatic differentiation:
   the network's exact derivative. It is mapped to MOOSE's fourth-order tangent; because x_3 = ε_xy + ε_yx, the
   shear column fills both tensor positions, so an isotropic network gives C_xyxy = μ, as Hooke's law does.
5. **Elastic strain.** The whole strain is reported as elastic strain (there is no inelastic part).

Your exported module must therefore take a (1, 3) tensor of raw strain and return 3 raw stresses, with the
standardization inside it. `module_5/readings/torchscript_export_guide.md` covers the export.

### Checks it passed before Lab 5 (course build tests, 2026-10-04)

- A *linear* network σ = C x reproduces `ComputeLinearElasticStress` on Lab 0's plate: every postprocessor
  identical except MOOSE's von Mises (see below). Re-checked with the Lab 5 input file (ledger §3.4).
- **The tangent, for a nonlinear tanh network:**
  - PETSc's finite-difference Jacobian test agrees with the autograd tangent at every Newton step of a small
    single-element test: 13 checks, largest relative difference 9.0e-7, two converged solves (ledger §9).
  - The network's autograd derivative matches central finite differences at 200 strains to 1.6e-9 (ledger §9).
  - Both checks are at a deliberately small load. They check the derivative, not that Newton converges at larger
    loads.

The Lab 5 runs themselves are in ledger §3–§5.

---

## σ_zz, plane strain, and von Mises

In plane strain ε_zz = 0, and for an isotropic elastic material σ_zz = ν(σ_xx + σ_yy). It does not enter the
in-plane equilibrium equations, so the displacements and in-plane stresses do not depend on it. But MOOSE's
`vonmises_stress` output uses all components, σ_zz included.

The network was trained on in-plane data only and does not predict σ_zz, so `ComputeTorchScriptStress` sets it to 0.
Consequences on Lab 0's plate (ledger §3.4):

- MOOSE's `max_vonmises_stress` comes out *higher* in the network run than in the elastic run, for a reason that has
  nothing to do with the network's accuracy. At the top of the hole the stress is nearly uniaxial, and a uniaxial σ_xx
  has von Mises stress σ_xx when σ_zz = 0, but σ_xx √(1 − ν + ν²) ≈ 0.889 σ_xx when σ_zz = ν σ_xx (ν = 0.3).
- `plate_with_hole_nn_material.i` therefore adds `vonmises_pe`: von Mises rebuilt from the in-plane element stresses with
  σ_zz = ν(σ_xx + σ_yy), ν = 0.3, for *both* runs. Compare `max_vonmises_pe` with `max_vonmises_pe`.
  (For the network this assumes the ν it should have learned; it is a post-processing choice. And because von Mises
  is nonlinear, the von Mises of element-average stresses differs slightly from MOOSE's element average of
  quadrature-point von Mises; compare like with like.)

---

## When the run is slow, or does not converge

**Cost.** Every residual and every Jacobian evaluation runs the network (forward and, for the tangent, three backward
passes) at every quadrature point. The network run takes several Newton iterations where the elastic run takes one,
and each iteration is far more expensive. On Lab 0's plate at the Lab 5 mesh the network run took about 43 s with 3
Newton iterations, the elastic run about 3 s with 1 (ledger §3.3).

**Newton stalls or diverges.** An exact tangent gives Newton its fast local convergence only near a solution, with a
smooth residual, a nonsingular Jacobian there, a good enough starting point and a line search that does its job. An
exact derivative guarantees none of these, nor that the network material is stable or that a solution exists
nearby. On Lab 0's plate the converged runs took 2 to 5 Newton iterations (ledger §3.3, §4.1); on Lab 3's plate,
started from zero, 15 (§6.1).

When a run does not converge, a common cause is the network being evaluated at strains far from its data. There a
tanh network flattens out, its stiffness drops toward zero, and along such directions the stress it returns levels
off. Newton then stalls, fails its line search, or (with a near-zero tangent) cannot factor the Jacobian. Lab 5
Task 4 does this on purpose (ledger §4.1). A run that fails to converge shows that Newton failed with those
settings, not that no solution exists.

Check the `min_/max_strain_*` postprocessors against the strains in your Lab 1 CSV, or run
`comparison_three_methods.py --train-data`. A poor *initial guess* can slow or stop a run whose final strains are
fine: Lab 5 Task 5 shows it (ledger §6.1).

**A load error at start-up.** Wrong file name, the Lab 1 checkpoint instead of the TorchScript file, an output that
is not 3 values, or a dtype cast inside `forward`. The messages for each are in ledger §5.

**No error at all, wrong answer.** A file without its standardization runs and converges, to a wrong plate (ledger §5).
MOOSE cannot know what the right stress is; `verify_torchscript.py` can.

---

## Comparison with the elastic material

| | `ComputeLinearElasticStress` | `ComputeTorchScriptStress` |
|---|---|---|
| stress | C : ε, exact Hooke | the network's prediction (Lab 1 training error, plus whatever it does outside its data) |
| tangent | constant C | autograd through the network, at every point |
| Newton iterations on Lab 0's plate | 1 | 3 at refine 3, 4 at refine 4 (ledger §3.3) |
| run time on Lab 0's plate (refine 3, 4 MPI ranks) | about 3 s | about 43 s (ledger §3.3) |
| σ_zz | ν(σ_xx + σ_yy) in plane strain | 0 (not predicted) |
| history / internal variables | none needed | **none possible**: the material keeps no state (below) |
| what it can represent | isotropic linear elasticity | any stateless map from 3 strains to 3 stresses |

For linear elasticity the network is slower and less accurate. Its value is that the same input block deploys *any*
stateless strain-to-stress model trained on data, including ones with no closed form.

---

## What this material cannot do

- **Path dependence.** It evaluates σ(ε) from the current strain only and stores nothing between steps. Lab 4's LSTM
  (hidden state) and the MLP that is handed ε^p (an internal variable whose evolution it does not learn) both need a
  material that stores and updates state at each quadrature point. That is not part of the course build.
- **Finite strain.** It uses the small (mechanical) strain and fills a small-strain stress.
- **3-D or out-of-plane stress.** It is a plane model: three strains in, three stresses out, σ_zz = 0.

---

## What you do not need for Lab 5

- C++. The material is compiled into the course executable.
- libtorch's C++ API. It loads the file, calls `forward`, and differentiates the result.
- MOOSE's Action system beyond the `[Physics/SolidMechanics/QuasiStatic]` block you already used in Lab 0 (there
  under its older name `[Modules/TensorMechanics/Master]`).

What you do need: the export (`torchscript_export_guide.md`), the one-block input change above, and the habit of
checking which strains the network was evaluated at (the strain postprocessors, and the coverage proxy of
`comparison_three_methods.py`).
