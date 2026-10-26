"""
export_torchscript.py
=====================

Module 5 / Lab 5 Task 1 -- export a trained Lab 1 constitutive MLP to a TorchScript file that
the course MOOSE material ComputeTorchScriptStress can load.

The exported module
  - takes RAW plane Voigt strain (eps_xx, eps_yy, gamma_xy), gamma_xy = 2 eps_xy, shape (N, 3);
  - returns RAW plane Voigt stress (sigma_xx, sigma_yy, sigma_xy), shape (N, 3);
  - carries Lab 1's standardization inside it as buffers (StandardizedMLP below), so MOOSE
    needs to know nothing about how the network was trained.
MOOSE calls it with one quadrature point at a time, a (1, 3) tensor, and converts the module
to float64 when it loads it.

The output file is a NEW file (default mlp_constitutive_ts.pt). It is not the Lab 1 checkpoint,
and the script refuses to overwrite the checkpoint.

After writing the file, it reloads it and, on 1000 random strains in [-0.005, 0.005]^3, stops
with an error (exit 1) unless
  - the float32 file reproduces the eager model exactly, and
  - the float64 copy (what MOOSE runs) is finite and reproduces, to within 1e-12 of the largest
    stress, the network evaluated in float64 directly from the MLP and the standardization (not
    through the wrapper's forward). A dtype cast inside forward() fails this: the copy then rounds
    its input to float32 (module_5/readings/measured_results.md section 5).
It also prints, for information, the float64 copy against the eager float32 model (float32
round-off).

Usage (seconds; fine on a login node or a laptop with torch)
-----
    python export_torchscript.py --checkpoint ../../lab_1/<you>/training/checkpoints/mlp_constitutive.pt
    python export_torchscript.py --checkpoint mlp_constitutive.pt --out mlp_constitutive_ts.pt

The network's width and depth are read from the checkpoint (mlp_constitutive.py saves its
arguments); --hidden / --layers override them for a checkpoint that does not carry them.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

F64_TOL = 1e-12   # float64 copy vs the eager model in float64, as a fraction of the largest stress


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Lab 1 checkpoint from mlp_constitutive.py (holds model_state and scalers)")
    p.add_argument("--out", type=Path, default=Path("mlp_constitutive_ts.pt"),
                   help="TorchScript file to write (default: mlp_constitutive_ts.pt)")
    p.add_argument("--hidden", type=int, default=None, help="hidden width (default: from the checkpoint, else 32)")
    p.add_argument("--layers", type=int, default=None, help="number of Linear layers (default: from the checkpoint, else 4)")
    return p.parse_args()


class MLP(nn.Module):
    """The architecture of module_1/examples/mlp_constitutive.py (Linear + tanh, no tanh on the output)."""

    def __init__(self, hidden: int = 32, layers: int = 4):
        super().__init__()
        widths = [3] + [hidden] * (layers - 1) + [3]
        modules: list[nn.Module] = []
        for i in range(len(widths) - 1):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class StandardizedMLP(nn.Module):
    """Raw strain in, raw stress out. The training standardization is stored as buffers:
    buffers are saved with the module (and converted to float64 with it), but are not trained."""

    def __init__(self, model: nn.Module, mu_in, sd_in, mu_out, sd_out):
        super().__init__()
        self.model = model
        self.register_buffer("mu_in", torch.tensor(mu_in, dtype=torch.float32).flatten())
        self.register_buffer("sd_in", torch.tensor(sd_in, dtype=torch.float32).flatten())
        self.register_buffer("mu_out", torch.tensor(mu_out, dtype=torch.float32).flatten())
        self.register_buffer("sd_out", torch.tensor(sd_out, dtype=torch.float32).flatten())

    def forward(self, eps_raw):
        return self.model((eps_raw - self.mu_in) / self.sd_in) * self.sd_out + self.mu_out


def main():
    args = parse_args()
    if args.out.resolve() == args.checkpoint.resolve():
        raise SystemExit(f"--out {args.out} is the checkpoint itself; pick another name "
                         "(the default is mlp_constitutive_ts.pt).")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt or "scalers" not in ckpt:
        raise SystemExit("Checkpoint lacks 'model_state' or 'scalers'. It must come from "
                         "module_1/examples/mlp_constitutive.py, which saves both.")

    saved = ckpt.get("args", {}) or {}
    hidden = args.hidden or saved.get("hidden", 32)
    layers = args.layers or saved.get("layers", 4)
    print(f"Architecture: {layers} Linear layers, width {hidden} (tanh)")

    # 1. Rebuild the network and load the trained weights.
    model = MLP(hidden=hidden, layers=layers)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Wrap it with the standardization from training.
    s = ckpt["scalers"]
    wrapped = StandardizedMLP(model, s["mu_x"], s["sigma_x"], s["mu_y"], s["sigma_y"])
    wrapped.eval()

    # 3. Sanity check on one strain (plane-strain Hooke, E = 1, nu = 0.3: sigma = (1.3462, 0.5769, 0) * 1e-3).
    eps = torch.tensor([[1e-3, 0.0, 0.0]])
    with torch.no_grad():
        sig = wrapped(eps)
    print(f"\nSanity check: eps = {eps.tolist()[0]}")
    print(f"  network sigma = [{', '.join(f'{v:.6e}' for v in sig[0].tolist())}]")
    print("  Hooke   sigma = [1.346154e-03, 5.769231e-04, 0.000000e+00]  (E = 1, nu = 0.3, plane strain)")

    # 4. Export by tracing. The network has no data-dependent branches, so a trace records all of it.
    traced = torch.jit.trace(wrapped, torch.zeros(1, 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.out))
    print(f"\nExported {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")

    # 5. Reload and compare with the eager model on strains inside Lab 1's range [-0.005, 0.005].
    #    The strains are drawn in float64, as MOOSE passes them: a float32 cast inside forward()
    #    changes these (it would not change strains that were float32 to begin with).
    g = torch.Generator().manual_seed(0)
    x64 = (torch.rand(1000, 3, generator=g, dtype=torch.float64) * 2 - 1) * 5e-3
    x = x64.float()
    loaded = torch.jit.load(str(args.out)).eval()
    with torch.no_grad():
        y_eager = wrapped(x)
        d32 = (loaded(x) - y_eager).abs().max().item()
        loaded64 = torch.jit.load(str(args.out)).to(torch.float64).eval()   # what MOOSE does on load
        y64 = loaded64(x64)
        d64 = (y64 - y_eager.double()).abs().max().item()
        # float64 reference from the MLP and the standardization buffers directly, NOT through
        # StandardizedMLP.forward, so a cast added to that forward() cannot hide on both sides
        m64 = copy.deepcopy(model).double()
        b = {k: getattr(wrapped, k).double() for k in ("mu_in", "sd_in", "mu_out", "sd_out")}
        ref64 = m64((x64 - b["mu_in"]) / b["sd_in"]) * b["sd_out"] + b["mu_out"]
        d64e = (y64 - ref64).abs().max().item()
        one = loaded64(torch.tensor([[1e-3, 0.0, 0.0]], dtype=torch.float64))
        finite = bool(torch.isfinite(y64).all() and torch.isfinite(y_eager).all())
    scale = y_eager.abs().max().item()
    print("Reload check on 1000 strains in [-0.005, 0.005]^3:")
    print(f"  float32 file vs eager model: max |diff| = {d32:.3e}   (must be 0)")
    print(f"  float64 copy (as MOOSE runs it) vs the network in float64: max |diff| = {d64e:.3e} "
          f"= {d64e / scale:.1e} of the largest stress   (must be finite and <= {F64_TOL:g})")
    print(f"  float64 copy vs eager float32 (information): max |diff| = {d64:.3e} "
          f"= {d64 / scale:.1e} of the largest stress   (float32 round-off)")
    print(f"  float64 copy at eps = (1e-3, 0, 0): sigma = [{', '.join(f'{v:.6e}' for v in one[0].tolist())}]")
    if d32 != 0.0:
        raise SystemExit("The reloaded file does not reproduce the eager model -- do not use it.")
    if not (finite and d64e <= F64_TOL * scale):
        raise SystemExit("The float64 copy MOOSE will run is non-finite or does not match the model -- "
                         "look for a dtype cast inside forward() (module_5/readings/torchscript_export_guide.md).")
    print("\nOK. Next: python verify_torchscript.py --model", args.out)


if __name__ == "__main__":
    main()
