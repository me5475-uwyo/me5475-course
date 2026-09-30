# Lab 3's Plate in MOOSE — A Walkthrough

*Module 3. Read it after Lecture 14 (Fri Oct 2), before Lab 3 Task 2. Runnable companions: `module_3/examples/plate_square_hole_reference.i`, `run_plate_reference.sbatch` and `extract_moose_reference.py`. It builds on the L13 reading, `moose_diffusion_walkthrough.md`, which introduced kernels, command-line overrides and the run folder.*

Lab 3 Task 2 compares your PINN with a MOOSE solution of the same plate. That solution is already computed, converged and shared: `/project/me5475/examples/lab3_reference/`. **You do not need to run MOOSE for the lab.** This reading shows how that reference is built, and how to run it yourself:
- the problem, and why the input file needs only three boundary conditions;
- the input file, block by block;
- running it as a job, and reading the checks;
- extracting the fields to a CSV.

It stops there. Comparing the PINN with MOOSE is Task 2. Numbers cite `module_3/readings/measured_results.md` §1.

---

## 1 · The problem

The same plate as L14 and the course's PyTorch port:
- **Domain:** the unit square [0, 1]² minus the quarter disk r < 0.1 centred at the origin.
- **Material:** linear isotropic elasticity with E = 1 and ν = 0.3, small strain, **plane strain**. The primer, `plane_stress_and_plane_strain_primer.md`, explains why plane strain and not plane stress.
- **Boundary conditions:**

| edge | displacement (written in `[BCs]`) | traction (holds naturally, §2) |
|---|---|---|
| left, x = 0 | u₁ = 0 | zero shear traction |
| bottom, y = 0 | u₂ = 0 | zero shear traction |
| right, x = 1 | u₁ = 1 | zero shear traction |
| top, y = 1 | — | traction-free: both components zero |
| hole, r = 0.1 | — | traction-free: σ·n = 0 |

These are the same ten conditions L14 counts: one displacement and one shear condition on each of the three straight edges, and two traction conditions on the top and two on the hole.

---

## 2 · Why `[BCs]` has only three entries

Multiply the equilibrium equation ∇·σ = 0 by a vector test function ψ and integrate by parts over the plate Ω:

> ∫_Ω σ(u) : ∇ψ dA − ∫_∂Ω (σn)·ψ ds = 0  for every admissible ψ.

**The first integral** is the solid-mechanics kernel: the term the `[Physics/SolidMechanics/QuasiStatic]` block adds for each displacement component. It plays the role `Diffusion` played in L13.

**The boundary integral** contains σn, the traction. On every part of the boundary, either of two things removes it:
- **a displacement component is prescribed there.** The test function's matching component is then zero, as in L13, where ψ vanished at the Dirichlet ends. These are the three `DirichletBC`s.
- **the traction component is zero.** It then contributes nothing, so nothing needs to be written. MOOSE satisfies these conditions *naturally*.

So of the ten conditions, only the three displacement conditions go into the file. The other seven are built into the weak form: the shear on the three straight edges, and the two traction components on the top and on the hole. L14's script is a strong-form PINN, so it imposes all ten conditions explicitly as terms of its loss.

Natural conditions hold in the weak, integrated sense, not exactly at each point of a finite mesh. The input file has a consistency check on them. `traction_top_*` and `traction_hole_*` are *integrated traction resultants*: each traction component integrated along the free top edge, or around the hole. They are small, and they shrink as the mesh is refined (§4). Because they are integrals, positive and negative parts can cancel, so they show consistency, not that the traction is small at every point.

---

## 3 · The input file, block by block

`module_3/examples/plate_square_hole_reference.i`. **Read its header comment first.** It states the problem and explains the mesh construction and the resolution parameters.

**Top-level variables.** `hole_radius = 0.1`, `nt = 192` elements along each 45° arc of the hole, and `nr = 384` elements along each ray from the hole to the outer edge. As with L13's `nx`, the mesh block reads them with `${nt}` and `${nr}`, so they can be changed from the command line.

**`[Mesh]`: two patches, bent into place.** The mesh is built in five steps:
1. **A square parameter mesh per patch.** Each patch starts as a `GeneratedMeshGenerator` on the unit square of parameters (ξ, η), with `QUAD9` elements.
2. **Bent into place.** A `ParsedNodeTransformGenerator` moves every node. η sets the angle: θ = η·45° for patch A, which runs from the hole to the right edge, and θ = 45° + η·45° for patch B, which runs to the top edge. ξ sets the radius, graded geometrically from the hole outward, so the elements are smallest where the stress concentrates.
3. **Named edges.** `RenameBoundaryGenerator` gives the edges the names the rest of the file uses: `left`, `bottom`, `right`, `top`, `hole`.
4. **Stitched.** `StitchMeshGenerator` joins the two patches along the 45° diagonal.
5. **Snapped.** A last transform snaps round-off onto the exact straight edges.

Every node of a `QUAD9` element on the hole, the mid-edge nodes included, lies on the exact arc r = 0.1. The hole is *body-fitted*, not a staircase. You never need to edit this block: `nt` and `nr` set the resolution.

**`[Variables]`.** `disp_x` and `disp_y`, second-order Lagrange, matching the `QUAD9` elements.

**`[Physics/SolidMechanics/QuasiStatic]`.** One block sets up the solid mechanics:

```
[Physics/SolidMechanics/QuasiStatic]
  displacements = 'disp_x disp_y'
  [all]
    displacements          = 'disp_x disp_y'
    add_variables          = false
    strain                 = SMALL
    incremental            = false
    planar_formulation     = PLANE_STRAIN
    generate_output        = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy vonmises_stress'
    material_output_family = LAGRANGE
    material_output_order  = FIRST
  []
[]
```

- **It adds the kernels:** the ∫ σ : ∇ψ term of §2, one kernel per displacement component, and the small-strain calculation.
- **`planar_formulation = PLANE_STRAIN`** is where the plane-strain choice lives.
- **The `generate_output` and `material_output_*` lines** write the stresses as nodal fields into the Exodus file, which is what `extract_moose_reference.py` reads.

Lab 0's `[Modules/TensorMechanics/Master]` is the older name of this same block.

**`[BCs]`: the three displacement conditions, and nothing else.**

```
[BCs]
  [left_ux]
    type     = DirichletBC
    variable = disp_x
    boundary = left
    value    = 0.0
  []
  [bottom_uy]
    type     = DirichletBC
    variable = disp_y
    boundary = bottom
    value    = 0.0
  []
  [right_ux]
    type     = DirichletBC
    variable = disp_x
    boundary = right
    value    = 1.0
  []
[]
```

**`[Materials]`: Hooke's law.**

```
[Materials]
  [elasticity_tensor]
    type           = ComputeIsotropicElasticityTensor
    youngs_modulus = 1.0
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]
```

The first material builds the elasticity tensor from E and ν. The second computes σ from the strain. In the PINN, Hooke's law is a residual in the loss; here it is evaluated directly at every quadrature point.

**`[Postprocessors]`: checks, not results.** Each one is something you can predict or bound before you run:

| postprocessor | what it should be |
|---|---|
| `area` | 1 − π/400, the plate's exact area: tests the geometry |
| `hole_length` | π/20, the quarter circle: tests the body-fitted hole |
| `ux_right_min`, `ux_right_max` | both 1: the Dirichlet condition |
| `reaction_right_x`, `reaction_left_x` | equal and opposite: global equilibrium |
| `traction_top_x`, `_y`, `traction_hole_x`, `_y` | integrated traction resultants on the free edges: small, and → 0 with refinement. A consistency check on the natural conditions of §2, not a pointwise one. |
| `stress_xx_hole_top` | σ₁₁ at (0, 0.1), the stress concentration: Lab 3 Task 2 quotes it |

The others (`max_stress_xx`, `max_vonmises_stress`, `avg_stress_xx_right`, `num_elements`, `num_dofs`) describe the solution and the mesh.

**`[Preconditioning]`, `[Executioner]`, `[Outputs]`.**
- `Steady`, Newton, and a direct LU solve (MUMPS), as in L13's files.
- `SMP` with `full = true` keeps the coupling between `disp_x` and `disp_y` in the preconditioner, so Newton converges in one step on this linear problem. With the nodal stress output the log shows two, as the file's comment explains.
- Exodus and CSV output.

---

## 4 · Run it on ARCC

Use the same pattern as L13's reading: a folder of its own, outside the repository, and a job.

```bash
cd ~/me5475/me5475-<your-github-username>
git pull
mkdir -p ~/me5475/moose_plate_24x48
cp module_3/examples/plate_square_hole_reference.i module_3/examples/run_plate_reference.sbatch \
   ~/me5475/moose_plate_24x48/
cd ~/me5475/moose_plate_24x48

sbatch run_plate_reference.sbatch        # the coarse mesh, nt=24 nr=48: seconds
squeue -u $USER
```

**The job script's default is the coarse level.** With no arguments, `run_plate_reference.sbatch` runs `nt=24 nr=48`, not the input file's own default of 192/384. Arguments replace that default and are passed to MOOSE as overrides. The job uses 8 MPI ranks; the coarse run took under 20 s from start to finish, most of it job start-up (§1).

| file | what it is |
|---|---|
| `moose-plate-<jobid>.out` | MOOSE's log, then the checks CSV |
| `plate_square_hole_reference_out.csv` | the postprocessors, one column each; the answer is the `time = 1` row |
| `plate_square_hole_reference_out.e` | displacements and nodal stresses (Exodus), for ParaView or the extractor (§5) |

**Reading the checks.** The CSV has 17 columns, too wide to read by eye. Load the course environment, then print the last row, one check per line:

```bash
source /project/me5475/setup5475.sh
python -c "import pandas as pd; print(pd.read_csv('plate_square_hole_reference_out.csv').iloc[-1])"
```

**What you should see** at 24/48 (§1):

| check | value |
|---|---|
| `area` | 0.99214602 |
| `hole_length` | 0.15707963 |
| `reaction_right_x` / `reaction_left_x` | 1.073361 / −1.073342 |
| `traction_top_x`, `_y` | 2.7e-5, 6.0e-5 |
| `traction_hole_x`, `_y` | 5.0e-5, −4.9e-5 |
| `ux_right_min`, `_max` | 1, 1 |
| `stress_xx_hole_top` | 3.23799 |

**Refine the mesh.** Use a new folder: a second run in the same folder overwrites the first, and two jobs running there at once write over each other.

```bash
mkdir -p ~/me5475/moose_plate_48x96
cp ~/me5475/moose_plate_24x48/{plate_square_hole_reference.i,run_plate_reference.sbatch} ~/me5475/moose_plate_48x96/
cd ~/me5475/moose_plate_48x96
sbatch run_plate_reference.sbatch nt=48 nr=96
```

`stress_xx_hole_top` becomes 3.24086, and the integrated traction resultants fall about 4× (§1). The shared reference is the 192/384 level, with 3.24179 (§1). You do not need to run it. If you do, it needs more memory: `sbatch --mem=48G run_plate_reference.sbatch nt=192 nr=384`. It took 1 min 40 s (§1).

**Change the material from the command line,** with the same syntax. Give the mesh too: any argument replaces the job script's coarse default, so `youngs_modulus=2` alone would run the file's 192/384 level. In a folder of its own:

```bash
mkdir -p ~/me5475/moose_plate_E2
cp ~/me5475/moose_plate_24x48/{plate_square_hole_reference.i,run_plate_reference.sbatch} ~/me5475/moose_plate_E2/
cd ~/me5475/moose_plate_E2
sbatch run_plate_reference.sbatch nt=24 nr=48 Materials/elasticity_tensor/youngs_modulus=2
```

The plate is loaded by a prescribed displacement, so the displacements do not change, and every stress and reaction doubles exactly (§1). Lab 3 Task 5 relies on this fact ("Why a force is needed").

---

## 5 · Extract the fields to a CSV

`extract_moose_reference.py` samples the Exodus file at points inside the plate. Run it on the login node, in the coarse run's folder:

```bash
cd ~/me5475/moose_plate_24x48
source /project/me5475/setup5475.sh
python /project/me5475/examples/extract_moose_reference.py \
    --exodus plate_square_hole_reference_out.e --mode grid --nx 50 --ny 50 --output coarse_grid.csv
python /project/me5475/examples/extract_moose_reference.py \
    --exodus plate_square_hole_reference_out.e --mode random --n-points 200 --output coarse_random.csv
```

- **The grid mode** keeps only the grid points inside the plate: 2 478 of the 50 × 50 grid, in about 5 s (§1).
- **The random mode** samples 200 random points inside the plate.
- **Both files have the same columns:** `x_1, x_2, u_1, u_2, sigma_11, sigma_22, sigma_12, sigma_vm`.

**Without `--exodus`, the extractor reads the shared reference.** That is what Lab 3 Tasks 2 and 5 use. Follow their commands exactly: your own coarse run is for learning how the reference is made, not a substitute for it.

---

## 6 · Next

- **Lab 3 Task 2:** the PINN against this reference.
- **Module 5:** the same plate in MOOSE again, with a trained network as the material law (Lab 3's closing notes).
- **Plane stress:** the primer shows how the course ran Min Lin's plane-stress problem with this same input, without editing it (§8 of the results file).
