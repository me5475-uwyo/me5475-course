# MOOSE Input File Anatomy

A section-by-section reference for `module_0/examples/plate_with_hole.i`. Read this alongside the file itself.

The MOOSE input language is hierarchical, brace-delimited (well, square-bracket delimited), and **declarative**: you describe *what* the problem is, not *how* to solve it. Each top-level block answers one question about the simulation.

| Block | Question it answers |
|-------|--------------------|
| `[Mesh]` | What geometry is the simulation on? |
| `[Variables]` | What unknowns are being solved for? |
| `[Kernels]` / `[Modules/...]` | What PDEs govern the unknowns? |
| `[BCs]` | What boundary conditions apply? |
| `[Materials]` | What material properties / constitutive relations apply? |
| `[AuxVariables]` / `[AuxKernels]` | What derived quantities should be computed? (Not solved for.) |
| `[Postprocessors]` | What scalar summaries should be reported? |
| `[Executioner]` | How is the problem stepped/solved? (Steady, Transient, etc.) |
| `[Outputs]` | Where and in what format do results go? |

There are more blocks (`[GlobalParams]`, `[Functions]`, `[Adaptivity]`, `[Constraints]`, etc.) but the above nine cover ~95% of inputs you will write in this course.

---

## `[Mesh]`

MOOSE accepts:

- **Built-in generators** (the `*MeshGenerator` family) — generate geometry parametrically in the input file. No external mesh tool needed.
- **External files** — read `.e`, `.msh`, `.xda`, etc.
- **Chained generators** — combine and modify generated meshes (delete blocks, rename boundaries, stitch).

In `plate_with_hole.i` we use the `AnnularMeshGenerator`, which produces a *quarter-annulus* — the FE discretization of just the upper-right quadrant of the full plate-with-hole, exploiting the problem's symmetry:

```
[Mesh]
  [annulus]
    type = AnnularMeshGenerator
    nr   = 8                            # radial elements
    nt   = 24                           # azimuthal elements (over 90 degrees)
    rmin = 0.1                          # hole radius
    rmax = 1.0                          # outer boundary radius
    dmin = 0                            # starting angle (degrees)
    dmax = 90                           # ending angle (degrees)
  []
  uniform_refine = 0                    # global isotropic mesh refinement
[]
```

**Why a quarter-annulus.** The full plate-with-hole has two planes of symmetry (x-axis and y-axis). By modeling only the first quadrant and applying symmetry BCs on the radial edges, we get the full solution from 1/4 the mesh — a 4× cost reduction without losing accuracy.

**Key parameter — `uniform_refine`.** Each unit increment splits every element into 4 (in 2D) or 8 (in 3D). So `uniform_refine = 4` gives you 256× more 2-D elements than `uniform_refine = 0`. This is the lever Lab 0 sweeps.

**Generator-emitted sidesets** (named, not numeric):

| Sideset | What it is |
|---------|------------|
| `rmin` | The hole curve (inner arc at r = 0.1) |
| `rmax` | The outer arc at r = 1.0 |
| `dmin` | The bottom radial edge (y = 0, θ = 0) |
| `dmax` | The left radial edge (x = 0, θ = 90°) |

To constrain the bottom edge (symmetry: u_y = 0): `boundary = 'dmin'`. To apply a traction on the outer arc: `boundary = 'rmax'`. The hole is `rmin` and stays traction-free by default. Confirm visually on first run with `--mesh-only` and ParaView.

---

## `[Variables]`

Declares the unknowns the FE solver actually solves for. For linear elasticity in 2D plane strain:

```
[Variables]
  [disp_x]
    family = LAGRANGE
    order  = FIRST                      # linear (P1) elements
  []
  [disp_y]
    family = LAGRANGE
    order  = FIRST
  []
[]
```

`FIRST` order Lagrange = linear elements. For quadratic, use `SECOND`. Higher order means better convergence rate (O(h⁴) instead of O(h²)) but more degrees of freedom.

---

## `[Modules/TensorMechanics/Master]`

This is a **MOOSE action** — a macro that expands into many kernels and material properties at parse time. The `TensorMechanics/Master` action is the standard way to set up a solid mechanics problem without writing each `StressDivergenceTensors` kernel by hand.

```
[Modules/TensorMechanics/Master]
  displacements = 'disp_x disp_y'         # REQUIRED at the parent level in modern MOOSE
  [all]
    displacements = 'disp_x disp_y'        # AND inside [all]
    add_variables = false                  # we declared disp_x, disp_y manually
    strain = SMALL                         # small-strain kinematics
    incremental = false                    # total-form solve (not rate)
    generate_output = 'stress_xx ... vonmises_stress'   # auxvars to compute
    planar_formulation = PLANE_STRAIN      # or PLANE_STRESS, or NONE (3D)
  []
[]
```

**Required parameter — `displacements`.** Modern MOOSE requires `displacements = 'disp_x disp_y'` (2-D) or `'disp_x disp_y disp_z'` (3-D) at both the parent block level AND inside `[all]`. Omitting it produces:

```
*** ERROR ***
The following error occurred in the object "all", of type "EmptyAction".
Missing required parameter 'Modules/TensorMechanics/Master/displacements'
```

This was a frequent stumble during initial validation.

**Key parameter — `planar_formulation`.** For a thin plate under uniform in-plane tension, `PLANE_STRESS` is physically correct. We use `PLANE_STRAIN` to match Min Lin's PINN setup so the Module 0 (MOOSE) and Module 3 (PINN) solutions can be directly compared. The infinite-plate Kirsch hoop-stress reference is ≈3 under either assumption; for ν = 0.3, the corresponding plane-strain von Mises reference is ≈2.67. In the assigned refine-4 run, expect ≈2.88 (hoop) and ≈2.48 (von Mises).

---

## `[BCs]`

Boundary conditions live here. Each named sub-block applies one BC:

```
[BCs]
  [fix_left_x]
    type     = DirichletBC                # u_x = const
    variable = disp_x
    boundary = '10002 15002'              # left edge (two half-sidesets)
    value    = 0.0
  []
  [tension_right]
    type          = Pressure              # Neumann / traction
    variable      = disp_x
    boundary      = '10004 15004'         # right edge
    displacements = 'disp_x disp_y'        # required for Pressure in modern MOOSE
    factor        = -1.0                   # negative = tensile pull along +x outward normal
  []
[]
```

**Boundary IDs.** Assigned by the mesh generator, NOT by us. `PolygonConcentricCircleMeshGenerator` with `generate_side_specific_boundaries = true` splits each polygon side into two halves and emits sidesets `10001/15001` (top), `10002/15002` (left), `10003/15003` (bottom), `10004/15004` (right). Always pass *both* halves to a BC that should cover the whole edge.

**`Pressure` BC sign convention.** Negative `factor` = tensile (outward-pointing in the surface-normal direction). We verified by inspection: `factor = -1.0` on the right edge produces `disp_x ≈ +1.0` on the right (tension).

**`Pressure` requires `displacements`.** In modern MOOSE the `Pressure` BC needs an explicit `displacements = 'disp_x disp_y'` parameter — the same as the parent-level requirement for `[Modules/TensorMechanics/Master]`.

---

## `[Materials]`

Constitutive relations and material parameters:

```
[Materials]
  [elasticity_tensor]
    type            = ComputeIsotropicElasticityTensor
    youngs_modulus  = 1.0
    poissons_ratio  = 0.3
  []
  [stress]
    type = ComputeLinearElasticStress     # sigma = C : epsilon
  []
[]
```

In Module 5 we will replace `ComputeLinearElasticStress` with a learned material that calls a TorchScript-exported neural network. This is the cleanest extension point for ML-augmented constitutive modeling — Materials block in, ML model in, output is still a `RankTwoTensor` named `stress`.

---

## `[Postprocessors]`

Compute scalar summaries of the solution at each timestep, written to CSV. For a Steady executioner, you get one row of values.

```
[Postprocessors]
  [max_vonmises_stress]
    type       = ElementExtremeValue
    variable   = vonmises_stress
    value_type = max
  []
  [max_stress_xx]
    type       = ElementExtremeValue
    variable   = stress_xx
    value_type = max
  []
  [num_elements]
    type = NumElements
  []
[]
```

These three postprocessors produce the two peak-stress series and the mesh-size x-axis for Lab 0. The refine-4 expectations are ≈2.48 for `max_vonmises_stress` and ≈2.88 for `max_stress_xx`; compare them with the ≈2.67 / ≈3 infinite-plate references.

Other commonly useful postprocessors: `NodalExtremeValue` (max nodal value), `ElementAverageValue` (volume-weighted average), `PointValue` (value at a coordinate), `SideAverageValue` (average over a boundary), `NumNonlinearIterations` (solver diagnostic).

---

## `[Executioner]`

How the problem is stepped through. For statics:

```
[Executioner]
  type       = Steady
  solve_type = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_rel_tol = 1e-10
[]
```

`solve_type = NEWTON` plus a direct LU preconditioner is the most robust choice for small linear problems. For larger problems you switch to `solve_type = PJFNK` (preconditioned Jacobian-free Newton-Krylov) plus an iterative preconditioner like `hypre`.

For transients (not in Module 0, but coming in Module 4): `type = Transient` plus `dt` and `end_time`.

---

## `[Outputs]`

Where simulation results go:

```
[Outputs]
  exodus = true                           # mesh + nodal/elemental fields, for ParaView
  csv    = true                           # postprocessor values, for plotting
[]
```

`exodus` is the standard scientific computing format for FE output — netCDF-based, supported by ParaView, VisIt, MATLAB, Python (`netCDF4`, `meshio`).

`csv` writes one row per timestep with all postprocessor values.

**File naming.** Output files are named `<file_base>.<ext>`, where `file_base` defaults to the input file name minus `.i`. To override (as Lab 0 does):

```
[Outputs]
  file_base = my_custom_name
  exodus    = true
  csv       = true
```

Or via the command line: `moose-opt -i input.i Outputs/file_base=my_custom_name`.

---

## Five idioms you will reuse all semester

**1. Override any input parameter from the command line.**

```bash
moose-opt -i input.i Materials/elasticity_tensor/youngs_modulus=2.5 \
                     Mesh/uniform_refine=2 \
                     Outputs/file_base=run_E2.5_refine2
```

**2. Run mesh inspection only (no solve).**

```bash
moose-opt -i input.i --mesh-only inspect.e
paraview inspect.e
```

**3. Dump the parsed parameter tree without solving.**

```bash
moose-opt -i input.i --show-input
```

Useful for debugging when MOOSE is silently ignoring a typo'd parameter.

**4. Echo every kernel/material being applied.**

```bash
moose-opt -i input.i --show-applied
```

**5. Parallel run.**

```bash
mpiexec -n 8 moose-opt -i input.i
```

Or via SLURM `srun --ntasks=8 moose-opt -i input.i`.

---

## Where the rest of the docs live

- *Syntax index.* mooseframework.inl.gov/syntax — every block, every type, every parameter.
- *Source code.* github.com/idaholab/moose — the C++ implementation, useful when input docs are ambiguous.
- *Examples.* mooseframework.inl.gov/getting_started/examples_and_tutorials — the official tutorial library.
- *Discussion.* moose-users Google group, MOOSE Discussions on GitHub — active responses from the INL team.
