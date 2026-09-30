# DeepXDE in Two Pages

*Module 3. Skim it after Lecture 12 (Mon Sep 28), before the live demo in Lecture 13 (Wed Sep 30); keep it open through Lab 3 (due Mon Oct 19). Runnable companions: `module_3/examples/plate_with_hole_fixed.py` and `plate_with_hole_hard_bc.py`, and L13's `pinn_diffusion_1d.py` and `pinn_diffusion_2d.py`.*

A practitioner's quickstart. Keep this open while writing PINN scripts. The course environment is DeepXDE 1.15.0 with the PyTorch 2.5.1 backend (`/project/me5475/envs/ml4sm`); every call below was checked against that version. Numbers from course runs cite `module_3/readings/measured_results.md` by section.

**On ARCC, anything that trains — even the 1-D template at the end — runs as a submitted job**, never on the login node: `sbatch pinn_diffusion.sbatch` for the 1-D and 2-D diffusion examples (a CPU job — they need no GPU; `measured_results.md` §7), `sbatch pinn_train.sbatch` for the plate PINNs, or a copy of either pointed at your script. Course rule: at most two of your GPU jobs at a time. The last subsection, *Running the L13 demo on ARCC, step by step*, walks through the diffusion jobs command by command.

---

## What DeepXDE is

A Python library that handles the boilerplate of PINN training: geometry, collocation point sampling, BC enforcement, autograd-managed PDE residuals, and the Adam + L-BFGS training loop. The user-facing API is backend-agnostic; the backend is chosen once, when `deepxde` is imported.

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"     # MUST be set BEFORE importing dde
import deepxde as dde                     # prints "Using backend: pytorch"
```

Without `DDE_BACKEND`, DeepXDE uses the backend named in `~/.deepxde/config.json`; if that file does not exist, it picks an installed backend and writes the file. In the course environment (no TensorFlow) that search finds PyTorch — but a config file left behind by a personal TensorFlow environment (for example one you made to run Min's notebook) would silently switch every later script to TensorFlow, and PyTorch-specific code such as `torch.cat` in an `output_transform` would then fail. Setting the variable in the script makes the choice explicit. Check the first line DeepXDE prints.

---

## The five concepts

**Geometry.** Where collocation points live.

```python
geom = dde.geometry.Interval(0, 1)                              # 1-D
rect = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])         # 2-D box
hole = dde.geometry.Disk(center=[0, 0], radius=0.1)             # 2-D disk
geom = rect - (rect & hole)                                     # set ops: the Lab 3 plate
```

DeepXDE 1.15 provides `Interval`, `Rectangle`, `Disk`, `Ellipse`, `Triangle`, `Polygon`, `Cuboid`, `Sphere`, `Hypercube`, `Hypersphere` and more, and the set operations `|` (union), `-` (difference) and `&` (intersection). There is no `+`.

**PDE residual.** A function `pde(x, u)` that takes a batch of collocation points and the network outputs, and returns the residual — or a list of residuals — to drive to zero.

```python
def pde(x, u):
    du_xx = dde.grad.hessian(u, x)               # d2u/dx2 (output 0, input 0)
    return -du_xx - source_term(x)               # PDE residual R(x)
```

`dde.grad.jacobian(u, x, i=I, j=J)` returns ∂u[:, I] / ∂x[:, J] — **I is the output, J is the input**. Get the indices right; this is a common bug source. `dde.grad.hessian(u, x, component=C, i=I, j=J)` returns ∂²u[:, C] / ∂x[:, I] ∂x[:, J]; `component` defaults to 0.

In the plate problem the network has five outputs, (u₁, u₂, σ₁₁, σ₂₂, σ₁₂) — the **mixed (u, σ) formulation** — so the residual uses only first derivatives: two momentum-balance residuals and three constitutive residuals (`pde` in `plate_with_hole_fixed.py`).

**Boundary conditions.** Each BC says (a) where it applies, (b) what the value is, and (c) which network output component it constrains.

```python
bc_left_u1 = dde.icbc.DirichletBC(
    geom,
    func=lambda x: 0.0,                                        # boundary value
    on_boundary=lambda x, on_bnd: on_bnd and np.isclose(x[0], 0),
    component=0,                                               # which network output
)
bc_neumann = dde.icbc.NeumannBC(geom, func, on_boundary, component=0)   # du/dn = func
bc_hole_tx = dde.icbc.OperatorBC(
    geom,
    lambda x, u, _: u[:, 2:3] * x[:, 0:1] / 0.1 + u[:, 4:5] * x[:, 1:2] / 0.1,  # t_1 = σ₁₁n₁ + σ₁₂n₂
    is_hole,
)
```

`OperatorBC` is the flexible workhorse — it constrains *any expression* of the inputs and outputs to be zero on a boundary. Because the stresses are network **outputs** in the mixed formulation, every traction condition is **algebraic in the outputs**: σ₂₂ = 0 on the top edge is simply a `DirichletBC` on output 3, and zero traction on the hole is an `OperatorBC` with no derivatives. The course scripts use no `NeumannBC`.

The plate has **ten** boundary conditions, two per edge and two on the hole: u₁ = 0 and σ₁₂ = 0 on the left, u₂ = 0 and σ₁₂ = 0 on the bottom, u₁ = 1 and σ₁₂ = 0 on the right, σ₂₂ = σ₁₂ = 0 on the top, two traction components on the hole. Leave out the three σ₁₂ = 0 conditions and the PINN still reaches a tiny training loss while solving a different, under-constrained problem (`measured_results.md` §2).

**Data.** Combines geometry, PDE and BCs into a single trainable object.

```python
data = dde.data.PDE(
    geom,
    pde,
    [bc_left_u1, bc_right_u1, ...],   # all ten, in a fixed order
    num_domain=4000,                  # interior collocation points
    num_boundary=600,                 # boundary collocation points
    num_test=2000,                    # PDE test points
)
```

4 000 + 600 is the course default in all four plate scripts. Min Lin's notebook used 1 500 + 300; with that budget the hard-BC PINN fitted its training points and oscillated between them (`measured_results.md` §3).

Two things the test loss is and is not. Its PDE part is evaluated on `num_test` points sampled separately from the training points, so a test loss far above the training loss is the signal of that overfitting. Its BC part **reuses the training boundary points** (DeepXDE 1.15 `PDE.test_points`), so a small test loss is not an independent check of the boundary conditions — check them yourself on fresh points, as Lab 3 Task 3c asks.

**Network.** The neural network architecture.

```python
net = dde.nn.FNN(                     # dde.maps.FNN, used in the course scripts, is the same class
    [n_inputs] + [hidden] * depth + [n_outputs],
    "tanh",
    "Glorot uniform",
)
```

Optionally apply an output transform — the hard-BC ansatz of L15 and `plate_with_hole_hard_bc.py`:

```python
def output_transform(x, y):                # x: (n, 2) points, y: (n, 5) raw network outputs
    x1, x2 = x[:, 0:1], x[:, 1:2]
    u1 = x1 + x1 * (1 - x1) * y[:, 0:1]    # u1 = 0 at x1 = 0 and u1 = 1 at x1 = 1, for any y
    u2 = x2 * y[:, 1:2]                    # u2 = 0 at x2 = 0, for any y
    return torch.cat([u1, u2, y[:, 2:5]], dim=1)   # stresses unchanged
net.apply_output_transform(output_transform)
```

What the transform guarantees is exactly the three displacement conditions, for any network. It does not touch the stresses, so the **seven traction conditions** (including σ₁₂ = 0 on the left, bottom and right) stay in the loss: ten soft terms become seven. It does not guarantee faster training or a more accurate solution — under the same schedule the course's hard-BC runs were not faster (`measured_results.md` §6).

---

## The training recipe

```python
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)                                    # lr is η, the learning rate
losshistory, train_state = model.train(iterations=50000, display_every=1000)
model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=100)
```

Adam first, then L-BFGS to refine — the schedule of Min's notebook and of every course plate script (*rule of thumb*: Adam copes with the early, rough landscape; L-BFGS makes fast progress once it is near a minimum). `train` after L-BFGS takes no `iterations`: L-BFGS stops at its own tolerances or at DeepXDE's default cap of 15 000 iterations (`dde.optimizers.set_LBFGS_options`).

**Loss weights.** The loss is a weighted sum ℒ = Σ λ_k ℒ_k. Here λ is a **loss weight** (λ_BC, λ_data in L12's notation) — not L5's Hessian eigenvalue, not L10's weight-decay coefficient.

```python
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=[1] * 5 + [100] * 10,
    # one weight per loss term: the 5 PDE residuals first, then the 10 BCs in list order
)
```

Each ℒ_k is the **mean** of the squared residuals over that term's own points (DeepXDE's `"MSE"`), so a term's pull on the optimizer is set by its weight and its residual size, not by how many points it has: the 4 000 interior points do not "outvote" the far smaller set of points on one edge. And no finite weight *guarantees* that a boundary condition holds exactly — it only changes the trade-off. A transform like the one above does guarantee it, for the conditions it builds in.

**Saving and loading.** `model.save("plate.pt")` writes `plate.pt-<step>.pt`, where `<step>` is the final iteration; the file holds the network weights and the optimizer state. To predict with a trained network, rebuild the problem and load only the weights: `net.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"])` (Lab 3 Task 2b shows the full pattern).

---

## Seven gotchas

1. **Set `DDE_BACKEND` before importing dde.** Setting it after has no effect.
2. **`dde.grad.jacobian(u, x, i=I, j=J)` is ∂u[I] / ∂x[J]** — output first, input second. Other PINN libraries order these differently.
3. **The hard-BC ansatz uses `torch.cat` (PyTorch backend), not `tf.concat`.** If you copy code from a TensorFlow example, this is the line to change. (Min's notebook has no output transform; the ansatz is the course's addition.)
4. **`num_domain=0, num_boundary=0, anchors=X`** lets you supply collocation points as a numpy array directly. `plate_with_hole_parametric.py` (arrives with L16) uses it, because its points live in (x₁, x₂, E, ν).
5. **`model.predict(x)` returns numpy, not torch.** Don't try to `.backward()` on it.
6. **`loss_weights` needs one entry per loss term, in order** — 15 for the soft plate (5 + 10), 12 for the hard plate (5 + 7).
7. **Pass `external_trainable_variables` to *every* `compile`.** Each `compile` resets the list, so an L-BFGS `compile` without it freezes E and ν for the rest of training. Each `compile` also builds a fresh optimizer.

---

## Seven functions you'll use most often

| Function | Use |
|----------|-----|
| `dde.grad.jacobian(u, x, i=I, j=J)` | ∂u[I]/∂x[J] |
| `dde.grad.hessian(u, x, component=C, i=I, j=J)` | ∂²u[C]/∂x[I]∂x[J] |
| `dde.icbc.DirichletBC(geom, func, on_boundary, component)` | u[:, component] = func(x) on the boundary part |
| `dde.icbc.OperatorBC(geom, func, on_boundary)` | func(x, u, X) = 0 on the boundary part |
| `dde.icbc.PointSetBC(points, values, component)` | observation constraint at given points (the inverse problem's measurements) |
| `dde.Variable(initial_value)` | learnable scalar for inverse problems |
| `dde.callbacks.VariableValue(vars, period, filename)` | log learnable scalars during training |

For the inverse problem (L18 / Lab 3 Task 5), the key call is `model.compile(..., external_trainable_variables=[E_param, nu_param])`, which adds the learnable scalars to the optimizer's parameters. In DeepXDE 1.15 with PyTorch, `dde.Variable` returns a plain `torch.Tensor` with `requires_grad=True` — use it directly in the residual and read it with `float(E_param.detach().cpu())`; it has no `.value`. See `plate_with_hole_inverse.py`, which also explains why the plate needs one force measurement besides the displacements: under a prescribed displacement the displacement field does not depend on E, so displacement data alone cannot identify it.

---

## A complete 1-D template

The same problem as `module_3/examples/pinn_diffusion_1d.py` (L13's demo): −u″ = π² sin(πx) on (0, 1), u(0) = u(1) = 0, exact solution u = sin(πx).

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch

dde.config.set_random_seed(42)       # training is non-convex: fix the seed (L10's checklist)

# 1. Geometry
geom = dde.geometry.Interval(0, 1)

# 2. PDE
def pde(x, u):
    du_xx = dde.grad.hessian(u, x)
    return -du_xx - np.pi**2 * torch.sin(np.pi * x)

# 3. BC
bc = dde.icbc.DirichletBC(geom, lambda x: 0.0,
                          lambda _, on_b: on_b)

# 4. Data
data = dde.data.PDE(geom, pde, [bc],
                    num_domain=50, num_boundary=2, num_test=100)

# 5. Network
net = dde.nn.FNN([1] + [32] * 4 + [1], "tanh", "Glorot uniform")

# 6. Model + train
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(iterations=5000)
model.compile("L-BFGS")
model.train()

# 7. Predict + compare
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
u_pred = model.predict(x_test)
print("max |error|:", np.abs(u_pred - np.sin(np.pi * x_test)).max())
```

Substituting the geometry, residual, BCs and network shape takes you to the 2-D warm-up (`pinn_diffusion_2d.py`) and to the forward plate problems of Lab 3 Tasks 1 and 3. The sweep and the inverse problem add a few more pieces; read `pinn_optuna_sweep.py` and `plate_with_hole_inverse.py` for those.

### Running the L13 demo on ARCC, step by step

These are the steps L13 runs live, for the two scripts `pinn_diffusion_1d.py` and `pinn_diffusion_2d.py` and their job script `pinn_diffusion.sbatch`. Follow along in class, or use them afterwards to catch up. Off campus you need the UW VPN.

**1 · Log in, make a folder, copy the three files.**

```bash
ssh <netid>@medicinebow.arcc.uwyo.edu
mkdir -p ~/me5475/L13 && cd ~/me5475/L13
cp /project/me5475/examples/pinn_diffusion.sbatch \
   /project/me5475/examples/pinn_diffusion_1d.py \
   /project/me5475/examples/pinn_diffusion_2d.py .
```

- **Any folder works.** The job writes its output next to the files.
- **The same three files** are in your course repository, under `module_3/examples/`.
- **No `setup5475.sh` needed:** the job script activates the course environment itself.

**2 · Submit both jobs, then watch the 1-D one.**

```bash
J1=$(sbatch --parsable pinn_diffusion.sbatch 1d)
J2=$(sbatch --parsable pinn_diffusion.sbatch 2d)
echo $J1 $J2
squeue -u $USER
tail -F pinn-diffusion-$J1.out        # Ctrl-C stops watching, not the job
```

- **`--parsable`** makes `sbatch` print only the job number, which the two variables keep.
- **Both are CPU jobs,** four cores and no GPU, so they do not count against the two-GPU-jobs rule.
- **`squeue`** lists them as `PD` (waiting) or `R` (running).
- **If the 1-D job has not started yet,** `tail` first says *cannot open … No such file or directory*. That is harmless: it keeps waiting, and starts following as soon as the job writes.

**3 · What the 1-D job prints, in order.**
1. **A note that two modules "were not unloaded".** It is harmless.
2. **`----- Job start: …`**, then **`Script: pinn_diffusion_1d.py`**.
3. **`Using backend: pytorch`.** If it names another backend, see the table below. The lines suggesting other backends are DeepXDE's advertising; ignore them.
4. **`Compiling model...`, then `Training model...`, then the Adam loss table,** one row every 1 000 steps. `Train loss` is [ℒ_PDE, ℒ_BC] on the training points. `Test loss` is the same two terms, with ℒ_PDE on separately sampled test points; ℒ_BC reuses the training boundary points (see *Two things the test loss is and is not*, above). `Test metric` is the relative L2 error against the exact solution.
5. **A second `Compiling model...` and the L-BFGS table,** rows 5000 and 5060: L-BFGS stopped by itself after 60 steps.
6. **The summary and the plot.** The 1-D job takes about half a minute from start to finish (§7.1).

**Watch the last two rows of the Adam table** (`measured_results.md` §7.1). From step 4000 to step 5000, ℒ_PDE falls from 6.80e-05 to 5.78e-05, but the test error *rises* threefold, from 3.57e-04 to 1.09e-03, and ℒ_BC rises too. A falling loss is not a falling error. The job ends with:

```
=== L13 summary, 1-D ===
                               L_PDE      L_BC  test rel. L2 error
end of Adam (5000 its)      5.78e-05  8.40e-07            1.09e-03
end of L-BFGS (+60 its)     1.21e-05  6.82e-10            2.77e-05
max |u_NN - u| on 1001 evenly spaced points: 3.54e-05
Wrote pinn_diffusion_1d.png
```

**4 · Check that both jobs finished, then read the 2-D result.** The 2-D job takes about two minutes, and more when the node is busy (§7.1–§7.2).
- **An empty `squeue` only means the jobs have left the queue, not that they succeeded.** Check with `sacct -j $J1,$J2 --format=JobID,State,Elapsed`, which should say `COMPLETED`, or look at the end of the output:

```bash
tail -8 pinn-diffusion-$J2.out
```

```
=== L13 summary, 2-D ===
                               L_PDE      L_BC  test rel. L2 error
end of Adam (5000 its)      4.74e-04  5.07e-03            9.61e-02
end of L-BFGS (+1501 its)   1.94e-06  9.03e-07            1.05e-03
max |u_NN - u| on a 101 x 101 grid: 2.54e-03
Wrote pinn_diffusion_2d.png
```

- **In a new terminal, `$J1` and `$J2` are gone.** Run `ls pinn-diffusion-*.out`: the larger number is the 2-D job.
- **These are the numbers the course's ARCC runs printed, to every digit (§7.1).** Another machine, or a GPU, can differ in the last digits (§7.3).

**5 · Look at the plots.** In VS Code connected to ARCC (as in L8's guide, `module_0/readings/running_the_l8_demo_on_arcc_guide.md`), click the `.png` in the file tree. Or copy it to your laptop, from a terminal *on the laptop*:

```bash
scp <netid>@medicinebow.arcc.uwyo.edu:me5475/L13/pinn_diffusion_1d.png .
```

- **1-D, four panels:** the solution, with the 52 collocation points as ticks along the bottom; the error on a log axis; the two loss terms against step; and the test error against step.
- **2-D:** the top row shows the exact solution, the PINN, and the error, on its own colour scale. The bottom row shows the loss terms and the test error, where Adam's plateau is followed by the L-BFGS drop.

**If something goes wrong**

| you see | what it means / what to do |
|---|---|
| `cp: cannot stat '/project/me5475/…': Permission denied` | Your account is not in the course group yet. Tell the instructor; meanwhile, copy the files from your repository's `module_3/examples/`. |
| `sbatch: error: … account` or `… partition` | Your ARCC account is not on the course allocation `me5475` yet. Tell the instructor. |
| still `PD` in `squeue` after a few minutes | The queue is busy. The job starts by itself; nothing to do. `squeue -u $USER` shows the reason in the last column. |
| `tail: cannot open …` | The job has not started writing yet. It is harmless; `tail -F` keeps waiting. |
| a backend other than `pytorch` on the first lines | The script sets the backend before importing DeepXDE, so this means the copy was edited. Copy the three files again. |
| `squeue` empty, but no summary at the end of the `.out` | The job failed. `grep -i error pinn-diffusion-*.out` shows why; check that the script is unedited and that you submitted it with `sbatch`. |
| `ModuleNotFoundError: No module named 'deepxde'` | You ran `python pinn_diffusion_1d.py` directly, outside the course environment and on the login node. Submit it with `sbatch` instead. |
| different numbers from the ones above | The recorded course CPU runs matched these displayed digits. Different hardware or runtime versions can change the last digits; a laptop or a GPU may. First check the script, the options and the environment. |
