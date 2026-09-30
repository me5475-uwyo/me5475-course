# The L13 Diffusion Problems in MOOSE — A Walkthrough

*Module 3. Read it after Lecture 13 (Wed Sep 30). Runnable companions: `module_3/examples/diffusion_1d.i`, `diffusion_2d.i` and `run_moose_diffusion.sbatch`. It uses the same MOOSE build and module stack as Lab 0; for MOOSE syntax in general, see `module_0/readings/moose_input_file_anatomy.md`.*

L13 solved two problems with a PINN: −u″ = π² sin(πx) on (0, 1), and its 2-D version on the unit square. This reading solves **the same two problems with finite elements in MOOSE**. That is the code you ran in Lab 0, and the code that produced Lab 3's plate reference. It is about *how*:
- turning the equation into the blocks of an input file;
- running the input file as a job on ARCC;
- reading what comes back.

Numbers cite `module_3/readings/measured_results.md` §7.6. The L14 reading, `moose_plate_walkthrough.md`, does the same for the plate.

---

## 1 · From the equation to what MOOSE solves

**The problem (the same as `pinn_diffusion_1d.py`).**

> −u″(x) = f(x) = π² sin(πx) on (0, 1),  u(0) = u(1) = 0,  exact solution u(x) = sin(πx).

**The weak form.** Finite elements do not work with −u″ directly. Take a *test function* ψ that is zero wherever u is prescribed, here ψ(0) = ψ(1) = 0. Multiply the equation by it and integrate over the domain:

> −∫₀¹ u″ ψ dx = ∫₀¹ f ψ dx.

Integrate the left side by parts: −∫ u″ψ dx = ∫ u′ψ′ dx − [u′ψ]₀¹. The bracket is zero because ψ vanishes at both ends. What remains is the **weak form**:

> ∫₀¹ u′ ψ′ dx − ∫₀¹ f ψ dx = 0  for every such ψ.

Two things changed:
- only *first* derivatives of u appear;
- the equation must hold when averaged against every test function, not at each point.

**Discretisation: the linear system K U = F.** MOOSE looks for u_h(x) = Σ_j U_j φ_j(x), where φ_j is the *hat function* of node j: 1 at x_j, 0 at every other node, and linear in between. So U_j = u_h(x_j). Take ψ = φ_i in the weak form at each *interior* node i. The two end hats are not allowed, because the weak form needs ψ(0) = ψ(1) = 0. Each interior node gives one equation:

> Σ_j K_ij U_j = F_i,  K_ij = ∫₀¹ φ_i′ φ_j′ dx,  F_i = ∫₀¹ f φ_i dx.

Two hats overlap only when their nodes are neighbours, so K is tridiagonal. On a uniform mesh of n_x elements, h = 1/n_x, each interior row of K is (1/h)(−1, 2, −1). The rows of the two end nodes are replaced by U₀ = 0 and U_nx = 0. MOOSE computes every integral element by element, at a few quadrature points per element, and adds up the pieces.

**Kernels: one kernel per integral.** MOOSE assembles the residual (the left-hand side above) as a sum of *kernels*, each contributing one integral:

| in the weak form | MOOSE kernel | what it adds to the residual |
|---|---|---|
| ∫ u′ψ′ dx | `Diffusion` | +∫ ∇u·∇ψ (in 1-D, ∫ u′ψ′) |
| −∫ f ψ dx | `BodyForce`, with f given as a function | −∫ f ψ |

`BodyForce` carries the minus sign itself, so a positive f is a source, as in −u″ = f. MOOSE then drives the sum of the kernels to zero.

**What a kernel is, in MOOSE's C++.** You never write C++ for this problem, but three short pieces of MOOSE's own source show what the input file asks for. They are quoted from the course's MOOSE build (git 437fbe5082), which you can read on ARCC in `/project/me5475/software/moose/framework/src/kernels/`.

The loop that every kernel runs, `Kernel.C`, lines 100–102:

```
  for (_i = 0; _i < _test.size(); _i++)
    for (_qp = 0; _qp < _qrule->n_points(); _qp++)
      _local_re(_i) += _JxW[_qp] * _coord[_qp] * computeQpResidual();
```

The integrand of `Diffusion`, `Diffusion.C`, lines 28 and 34. Line 28 returns the residual integrand u_h′φ_i′. Line 34 returns its derivative φ_j′φ_i′, the Jacobian integrand. Integrating and assembling this product gives K_ij:

```
  return _grad_u[_qp] * _grad_test[_i][_qp];
  return _grad_phi[_j][_qp] * _grad_test[_i][_qp];
```

The integrand of `BodyForce`, `BodyForce.C`, line 54, which is −f φ_i:

```
    return -_test[_i][_qp] * _scale * _postprocessor * _function.value(_t, _q_point[_qp]);
```

How to read them, at quadrature point `_qp` of the current element:
- `_grad_test[_i][_qp]` is φ_i′, and `_test[_i][_qp]` is φ_i;
- `_grad_u[_qp]` is u_h′, and `_grad_phi[_j][_qp]` is φ_j′;
- `_JxW[_qp]` is the quadrature weight times the element's Jacobian, the "dx";
- `_coord[_qp]` is 1 in Cartesian coordinates;
- `_scale` and `_postprocessor` are 1 unless you set `value` or `postprocessor` (their defaults, `BodyForce.C` lines 27 and 29–30);
- `_function.value(…)` is the `ParsedFunction` named `forcing`.

So for an interior node i the residual is R_i = Σ over elements Σ over quadrature points of w_q × (u_h′φ_i′ − f φ_i), and ∂R_i/∂U_j = K_ij. Here w_q is `_JxW[_qp]`, the quadrature weight including the element's Jacobian. The two end nodes have rows of their own: the `DirichletBC` sets R₀ = U₀ − 0 and R_nx = U_nx − 0. The problem is linear, so Newton's method solves R(U) = 0 in one step (§4).

**Boundary conditions.** The two conditions u(0) = u(1) = 0 are imposed directly on the end nodes, by a `DirichletBC`. Conditions of this kind, on u itself, are called *essential*. L14's reading meets the other kind, *natural* conditions, which need no entry at all.

**Every piece of the problem, and its place in the input file:**

| the math | the input file |
|---|---|
| the domain (0, 1), cut into nx elements | `[Mesh]`: `GeneratedMeshGenerator`, `dim = 1` |
| the unknown u_h, piecewise linear | `[Variables]`: `[u]` |
| f = π² sin(πx), and the exact solution for checking | `[Functions]`: two `ParsedFunction`s |
| ∫ u′ψ′ dx and −∫ fψ dx | `[Kernels]`: `Diffusion` and `BodyForce` |
| u(0) = u(1) = 0 | `[BCs]`: `DirichletBC` on `'left right'` |
| "solve residual = 0 once; nothing depends on time" | `[Executioner]`: `Steady` |
| checks: the error against sin(πx), and u at x = 0.5 | `[Postprocessors]` |
| u along the line, for plotting | `[VectorPostprocessors]`: `LineValueSampler` |
| what to write | `[Outputs]` |

---

## 2 · The input file, block by block

`module_3/examples/diffusion_1d.i`, below its header comment.

```
nx = 50            # number of elements; override on the command line: nx=100

[Mesh]
  [line]
    type = GeneratedMeshGenerator
    dim  = 1
    nx   = ${nx}
    xmin = 0.0
    xmax = 1.0
  []
[]
```

A **top-level variable**, `nx`, and a mesh generator that reads it with `${nx}`. This is the same idiom as the plate reference's `nt` and `nr`, and it lets you change the mesh without editing the file (§5). A 1-D `GeneratedMeshGenerator` names its two ends `left` and `right`.

```
[Variables]
  [u]
  []
[]
```

One unknown field, named `u`. An empty block takes the defaults: first-order Lagrange, one value per node.

```
[Functions]
  [forcing]
    type       = ParsedFunction
    expression = 'pi*pi*sin(pi*x)'
  []
  [exact]
    type       = ParsedFunction
    expression = 'sin(pi*x)'
  []
[]
```

Functions of (x, y, z, t), written as text. `pi` is built in. The name `forcing` is what the kernel refers to. `exact` is used **only to check the answer** and never enters the solve: the same rule as L13's PINN scripts, where the exact solution scores the network but is not in the loss. In the course's MOOSE build the parameter is `expression`; older MOOSE examples online write `value`, the deprecated name.

```
[Kernels]
  [diffusion]   # + integral u' psi' dx
    type     = Diffusion
    variable = u
  []
  [source]      # - integral f psi dx
    type     = BodyForce
    variable = u
    function = forcing
  []
[]
```

The two terms of the weak form, one sub-block each. The names in brackets (`diffusion`, `source`) are yours to choose; the `type` is the MOOSE class.

```
[BCs]
  [ends]
    type     = DirichletBC
    variable = u
    boundary = 'left right'
    value    = 0.0
  []
[]
```

One entry covers both ends, since they share the same value.

```
[Executioner]
  type                = Steady
  solve_type          = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'lu       mumps'
[]
```

`Steady` means solve once for the equilibrium state. The problem is linear, so Newton's method finishes in one step. The two PETSc lines ask for a direct LU factorisation (MUMPS), the same choice as the plate reference.

```
[Postprocessors]
  [l2_error]
    type     = ElementL2Error
    variable = u
    function = exact
  []
  [u_mid]
    type     = PointValue
    variable = u
    point    = '0.5 0 0'
  []
  [num_dofs]
    type = NumDOFs
  []
[]
```

A postprocessor reduces the solution to one number. Here there are three **checks**:
- `l2_error` is ‖u_h − sin(πx)‖, the *absolute* L2 norm of the error over the domain;
- `u_mid` is u_h at x = 0.5, where the exact value is 1;
- `num_dofs` is the number of unknowns.

The point is written with three coordinates, even in 1-D.

```
[VectorPostprocessors]
  [u_line]
    type        = LineValueSampler
    variable    = u
    start_point = '0 0 0'
    end_point   = '1 0 0'
    num_points  = 101
    sort_by     = x
  []
[]

[Outputs]
  exodus = true                # diffusion_1d_out.e: the full solution (ParaView)
  csv    = true                # diffusion_1d_out.csv + the u_line CSV
[]
```

A *vector* postprocessor writes a whole table: here u at 101 evenly spaced points, ready to plot (§6). `exodus = true` writes the full solution. `csv = true` writes the postprocessors, plus one CSV per vector postprocessor. The output names are the input name with `_out` added.

---

## 3 · One dimension up: `diffusion_2d.i`

The 2-D problem is −(∂²u/∂x₁² + ∂²u/∂x₂²) = 2π² sin(πx₁) sin(πx₂) on (0, 1)², with u = 0 on all four sides and exact solution sin(πx₁) sin(πx₂). MOOSE calls the coordinates x and y. Run `diff diffusion_1d.i diffusion_2d.i`: apart from comments, four things change.
1. **The mesh:** `dim = 2`, with `ny = ${nx}` and `ymin`/`ymax` added. `nx` is now the number of elements per side.
2. **The functions** gain a factor `sin(pi*y)`: `'2*pi*pi*sin(pi*x)*sin(pi*y)'` and `'sin(pi*x)*sin(pi*y)'`.
3. **The boundary condition** covers four sides: `boundary = 'left right bottom top'`.
4. **The checks:** `u_centre` at (0.5, 0.5), and the line sampler runs along y = 0.5, where the exact solution is again sin(πx).

**The kernels do not change.** `Diffusion` is ∫ ∇u·∇ψ in any dimension, and `BodyForce` is −∫ fψ.

---

## 4 · Run it on ARCC

MOOSE runs as a **job**, as in Lab 0, never on the login node. The job script loads the course's module stack and calls the MOOSE build by its full path, so you need no set-up of your own. The first commands give the files a folder of their own, outside your repository, because the runs write several files.

```bash
cd ~/me5475/me5475-<your-github-username>
git pull
mkdir -p ~/me5475/moose_diffusion
cp module_3/examples/diffusion_1d.i module_3/examples/diffusion_2d.i \
   module_3/examples/run_moose_diffusion.sbatch ~/me5475/moose_diffusion/
cd ~/me5475/moose_diffusion

sbatch run_moose_diffusion.sbatch        # the 1-D problem; prints "Submitted batch job <jobid>"
sbatch run_moose_diffusion.sbatch 2d     # the 2-D problem
squeue -u $USER                          # seconds; when your jobs are gone from the list, they are done
ls
```

The two cases write different files, so they can share a folder. Each job took 1–8 s from start to finish (§7.6). After the 1-D run the folder holds:

| file | what it is |
|---|---|
| `moose-diffusion-<jobid>.out` | MOOSE's log, then the checks |
| `diffusion_1d_out.csv` | the postprocessors: `time, l2_error, num_dofs, u_mid` |
| `diffusion_1d_out_u_line_0001.csv` | u at the 101 line points: `id, u, x, y, z` |
| `diffusion_1d_out_u_line_0000.csv` | a header only, written before the solve; ignore it |
| `diffusion_1d_out.e` | the full solution (Exodus), for ParaView |

**Reading the log.** Open it with `less moose-diffusion-<jobid>.out` and look for these, in order:
- **A note about modules "not unloaded"**, from the script's `module purge`. It is harmless.
- **`Num DOFs: 51`**: 50 elements, 51 nodes, one unknown per node.
- **Two residual lines, then `Solve Converged!`**: `0 Nonlinear |R| = …` and `1 Nonlinear |R| = …`. One Newton step solves a linear problem; in the 1-D run the residual fell from 9.87e-1 to 6.9e-14 (§7.6).
- **The postprocessor table**, and after it the same numbers printed from the CSV.

**The CSV has two rows.** The `time = 0` row is all zeros, written before the solve. The `time = 1` row is the answer; a `Steady` solve is labelled time 1.

**What you should see** (§7.6):

| | `l2_error` | `u_mid` / `u_centre` | unknowns |
|---|---|---|---|
| 1-D, nx = 50 | 2.33e-4 | 1.0000000 | 51 |
| 2-D, nx = 50 | 1.64e-4 | 1.00033 | 2 601 |

If your numbers match, your run worked.

---

## 5 · Refine the mesh from the command line

Anything after `1d` or `2d` is passed to MOOSE as a **command-line override**. `nx=100` replaces the top-level `nx`, and `${nx}` carries it into the mesh:

```bash
mkdir -p ~/me5475/moose_diffusion_nx100
cd ~/me5475/moose_diffusion
cp diffusion_1d.i run_moose_diffusion.sbatch ~/me5475/moose_diffusion_nx100/
cd ~/me5475/moose_diffusion_nx100
sbatch run_moose_diffusion.sbatch 1d nx=100
```

**Use one folder per run of the same case.** A second 1-D run in the same folder overwrites the first run's files. Two 1-D runs going *at the same time* in one folder write over each other: in one of the course's test jobs, the numbers printed at the end of the log were the other job's (§7.6). Instead of a new folder, you can add `Outputs/file_base=run_nx100`, which renames the outputs. That is one of Lab 0's override idioms (`moose_input_file_anatomy.md`, "Five idioms").

**What refining does** (§7.6):

| nx | 1-D `l2_error` | 2-D `l2_error` |
|---|---|---|
| 25 | 9.30e-4 | 6.58e-4 |
| 50 | 2.33e-4 | 1.64e-4 |
| 100 | 5.82e-5 | 4.11e-5 |
| 200 | 1.45e-5 | 1.03e-5 |

Each doubling of nx divides the error by 4.00. The error goes as h², the expected L2 rate for linear elements. Try nx = 25 as well: `u_mid` becomes 0.99803, because x = 0.5 is then the middle of an element rather than a node, and the value there is interpolated between the two nodes on either side.

---

## 6 · Look at the solution

The line CSV plots with pandas. On the login node, a short analysis like this is fine. Paste the whole block into the terminal: it goes back to the 50-element run's folder, loads the course environment, and runs the Python lines up to `EOF`.

```bash
cd ~/me5475/moose_diffusion
source /project/me5475/setup5475.sh
python - <<'EOF'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # write a file; no window on ARCC
import matplotlib.pyplot as plt

line = pd.read_csv("diffusion_1d_out_u_line_0001.csv")   # columns: id, u, x, y, z
x = np.linspace(0, 1, 201)
plt.plot(line["x"], line["u"], label="MOOSE, 50 elements")
plt.plot(x, np.sin(np.pi * x), "--", label="exact: sin(pi x)")
plt.xlabel("x"); plt.ylabel("u"); plt.legend()
plt.savefig("moose_diffusion_1d.png", dpi=120)
print(len(line), "points; max |u - exact| on them:",
      np.abs(line["u"] - np.sin(np.pi * line["x"])).max())
EOF
```

It prints `101 points; max |u - exact| on them:` followed by about 4.9e-4 (§7.6). Half of the 101 points fall between nodes, where the piecewise-linear u_h differs from the curve the most. For the 2-D run, read `diffusion_2d_out_u_line_0001.csv`: the same columns, along the line y = 0.5.

**To see the whole 2-D field,** copy `diffusion_2d_out.e` to your laptop (`scp`, as in Lab 0) and open it in ParaView.

---

## 7 · Next

- **L14's reading, `moose_plate_walkthrough.md`,** runs Lab 3's plate in MOOSE the same way: an input file, a job script, overrides from the command line, and checks in a CSV.
- **The MOOSE types used here** (`Diffusion`, `BodyForce`, `DirichletBC`, `ParsedFunction`, `ElementL2Error`, `PointValue`, `LineValueSampler`) are documented in the syntax index, mooseframework.inl.gov/syntax. Search a type's name there for its full list of parameters.
