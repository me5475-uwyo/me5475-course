# TorchScript Export Guide

A practitioner's reference for Lab 5 Task 1. Keep it open while you export. Every number here is in
`module_5/readings/measured_results.md` (cited as "ledger §n").

---

## What TorchScript is

PyTorch's serialization format for *deployable* models. A TorchScript file (`.pt`, a zip archive) holds the network's
computation and its weights. It can be loaded

- in Python, with `torch.jit.load`, without the Python class that defined the network, and
- in C++, through libtorch, with no Python at all. That is how MOOSE loads it.

There are two ways to convert a model:

```python
traced   = torch.jit.trace(model, example_input)   # runs the model once and records the operations
scripted = torch.jit.script(model)                 # compiles the model's Python source
```

**A TorchScript file is not a checkpoint.** Lab 1's `checkpoints/mlp_constitutive.pt` is a `torch.save` dictionary
(weights, scalers, history, arguments); it needs the Python class to be useful. The TorchScript file
`mlp_constitutive_ts.pt` is the network itself. Both end in `.pt`, which is why `export_torchscript.py` gives the
second a different name and refuses to overwrite the first. Handing MOOSE the checkpoint fails at load time
(ledger §5).

---

## What the MOOSE material expects

`ComputeTorchScriptStress` (`module_5/readings/moose_material_interface_reference.md`) calls your module like this, once per
quadrature point:

| | what MOOSE passes / expects |
|---|---|
| input | a float64 tensor of shape **(1, 3)**: (ε_xx, ε_yy, γ_xy), raw units, γ_xy = 2 ε_xy |
| output | **3 numbers**: (σ_xx, σ_yy, σ_xy), raw units (it reshapes the output to 3 values) |
| dtype | the module is converted to **float64** when MOOSE loads it (`module.to(torch.float64)`) |
| gradient | MOOSE differentiates the output with respect to the input (autograd), so the forward pass must be differentiable |

`export_torchscript.py` produces exactly this from a Lab 1 checkpoint.

---

## Trace or script?

**`torch.jit.trace`** records what the model did on one example input. Use it when `forward` has no branch or loop
that depends on the *values* of the input. Lab 1's MLP qualifies, and `export_torchscript.py` traces it with
`torch.zeros(1, 3)`. The recorded operations work for any batch size.

**`torch.jit.script`** compiles the source, so `if` and `for` survive. Use it when the computation depends on the
values (a return-mapping iteration, a branch on a yield function). It accepts only a subset of Python and reports the
offending line when it refuses something.

What a trace misses: in

```python
def forward(self, x):
    if x.norm() > 0.01:
        return self.layer1(x)
    return self.layer2(x)
```

the trace keeps only the branch the example input took. PyTorch prints a `TracerWarning` when it sees such a
conversion of a tensor to a Python bool; do not ignore it.

---

## The translation issues

### 1. Standardization — the most common bug

Lab 1 trained on standardized strain and stress. MOOSE passes raw strain (≈ 1e-3) and uses whatever comes out as raw
stress. The fix is to put the standardization inside the exported module:

```python
class StandardizedMLP(nn.Module):
    def __init__(self, model, mu_in, sd_in, mu_out, sd_out):
        super().__init__()
        self.model = model
        self.register_buffer("mu_in",  torch.tensor(mu_in,  dtype=torch.float32).flatten())
        self.register_buffer("sd_in",  torch.tensor(sd_in,  dtype=torch.float32).flatten())
        self.register_buffer("mu_out", torch.tensor(mu_out, dtype=torch.float32).flatten())
        self.register_buffer("sd_out", torch.tensor(sd_out, dtype=torch.float32).flatten())

    def forward(self, eps_raw):
        return self.model((eps_raw - self.mu_in) / self.sd_in) * self.sd_out + self.mu_out
```

`register_buffer` matters twice: buffers are saved in the file (plain Python attributes or numbers captured during a
trace are frozen as constants instead), and buffers are converted to float64 together with the weights.

*What a missing wrapper looks like* (ledger §5): the raw network, traced without the wrapper, returns
σ = (−2.2e-4, 1.96e-3, 2.5e-4) for ε = (1e-3, 0, 0), where Hooke gives (1.35e-3, 5.8e-4, 0).
`verify_torchscript.py` fails it on all 10 tests. **MOOSE does not notice:** the plate solved with it converged in 2
Newton iterations, to strains more than twice the correct ones. Only a check against known answers catches this.

### 2. Dtype — keep `forward` dtype-neutral

Your file holds float32 weights; MOOSE converts the whole module to float64 and passes float64 strains. That works
as long as `forward` does not force a dtype. Measured on the course reference network (ledger §2):
- the float64 copy is identical to the same network evaluated in float64 (difference 0);
- it differs from the float32 original by at most 1.2e-9, 1.5e-7 of the largest stress. That is float32 round-off
  in the original, not an error in the copy.

*The trap:* an `eps.float()` (or `.to(torch.float32)`) inside `forward` is recorded by the trace and survives the
conversion. What it then does depends on what follows it (ledger §5):

- **Directly into a layer** (no float64 buffer in between): the float32 input meets float64 weights and every call
  fails with `RuntimeError: mat1 and mat2 must have the same dtype`.
- **Into the standardization wrapper:** the float64 buffers promote it back to float64, so the call works, but the
  strain has been rounded to float32 (about 7 digits). In MOOSE this showed as Newton stalling: the residual stopped
  falling near 1e-7 of its starting value, short of the input file's tolerance (1e-10 relative), and the run failed
  after 15 iterations. The same network without the cast converged in 3.

Do not cast inside `forward`. Both course scripts catch the second case:
- `export_torchscript.py` stops;
- `verify_torchscript.py --checkpoint` fails check 4.

They compare the float64 copy with the network evaluated in float64 on strains drawn in float64, as MOOSE passes
them. The cast shows as a difference of 3.9e-10, against a gate of 1e-12 of the largest stress (ledger §2). Strains
that started as float32 would hide it: rounding them to float32 changes nothing.

### 3. Shape — a batch of one

MOOSE passes one quadrature point at a time as a (1, 3) tensor and needs 3 numbers back. A model that calls
`x.squeeze()` or `x.flatten()` can lose the batch dimension and break the layers after it; a model that returns
fewer than 3 outputs is refused (ledger §5). Trace with `torch.zeros(1, 3)` and check a (1, 3) call after loading.

### 4. The PyTorch version

The course executable links libtorch **2.5.1**, the version in the course environment
(`/project/me5475/envs/ml4sm`). Export on ARCC with that environment and the file loads. A file exported on a laptop
with torch 2.8.0 also loaded, and the plate run with it gave identical postprocessors (ledger §5); other versions are
untested. If MOOSE refuses a laptop-exported file, re-export on ARCC.

---

## The export workflow (what `export_torchscript.py` does)

```python
import copy
import torch            # MLP and StandardizedMLP are the classes in export_torchscript.py

ckpt = torch.load("mlp_constitutive.pt", map_location="cpu", weights_only=False)   # the Lab 1 checkpoint
model = MLP(hidden=ckpt["args"]["hidden"], layers=ckpt["args"]["layers"])
model.load_state_dict(ckpt["model_state"]); model.eval()

s = ckpt["scalers"]
wrapped = StandardizedMLP(model, s["mu_x"], s["sigma_x"], s["mu_y"], s["sigma_y"]).eval()

traced = torch.jit.trace(wrapped, torch.zeros(1, 3))
traced.save("mlp_constitutive_ts.pt")

# check: reload, and compare with the network -- in float32 (expect identical) and in float64 (as MOOSE runs it).
# Draw the strains in float64, as MOOSE passes them: a float32 cast inside forward() only shows on such strains.
loaded = torch.jit.load("mlp_constitutive_ts.pt").eval()
x64 = (torch.rand(1000, 3, dtype=torch.float64) * 2 - 1) * 5e-3
assert torch.equal(loaded(x64.float()), wrapped(x64.float()))
loaded64 = torch.jit.load("mlp_constitutive_ts.pt").to(torch.float64).eval()
m64 = copy.deepcopy(model).double()           # the network in float64, standardization applied by hand
b = {k: getattr(wrapped, k).double() for k in ("mu_in", "sd_in", "mu_out", "sd_out")}
ref64 = m64((x64 - b["mu_in"]) / b["sd_in"]) * b["sd_out"] + b["mu_out"]
print((loaded64(x64) - ref64).abs().max())    # 0 for a correct export; the script stops above 1e-12 of the largest stress
```

`weights_only=False` is needed because the checkpoint also stores Python objects (its arguments). With torch 2.5.1 it
is the default; with torch 2.8.0, a plain `torch.load(...)` of a Lab 1 checkpoint fails with
`UnpicklingError: Weights only load failed` (ledger §5). Only load checkpoints you made or got from the course.

Then run `verify_torchscript.py`. It checks the stress against Hooke's law at 10 strains, the stress at zero strain,
the autograd tangent that MOOSE will use, and (with `--checkpoint`) the file against the eager model.
- **With `--data`:** test strains outside the convex hull of all the strains in your Lab 1 CSV are reported but not
  judged. That hull includes the validation and test rows, so it is a proxy for the training data. With fewer than 5
  judged tests the file is not validated.
- **What decides the result:** the judged Hooke tests, the coverage minimum, check 4 and any non-finite value.
  σ(0) and the tangent are diagnostics.
- **Exit status:** 1 on FAIL.

---

## Common symptoms

| symptom | likely cause | where to look |
|---|---|---|
| `verify_torchscript.py` fails every test by a large factor | standardization not in the file | §1; compare with `--checkpoint` |
| fails by a few percent at some strains only | the Lab 1 network's own error, or a strain outside its data | check 4 of the verify script (an export bug shows there, a training error does not); `--data` |
| `RuntimeError: mat1 and mat2 must have the same dtype` when called in float64; or MOOSE's Newton stalls near 1e-7 of the initial residual | a dtype cast inside `forward` | §2 |
| MOOSE: "returned 2 values; it must return 3" | the module returns other than 3 values | §3 |
| MOOSE: "Error while loading torchscript file … No such file or directory" | wrong path or name | ledger §5 |
| MOOSE: "Error while loading torchscript file … failed locating file constants.pkl" | the Lab 1 checkpoint given instead of the TorchScript file | ledger §5 |
| MOOSE converges, but the strains are far too large | standardization missing: MOOSE does **not** stop for this | run the verify script first |
| right at the training points, wrong in the plate | the plate asks about strains the network never saw | Lab 5 Task 4; `comparison_three_methods.py --train-data` |

---

## File size

The course reference network (2 339 parameters) exports to a 25.5–25.7 KB file (ledger §2). Most of it is the weights in
float32 plus the recorded code. Size is not a concern for networks of this kind.
