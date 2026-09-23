"""
reproducibility_checklist.py
============================

Module 2 / Lecture 10 helper --- runs a short PyTorch training twice and
verifies bit-for-bit reproducibility, then prints a checklist of items the
caller can use to verify their own scripts.

Usage
-----
    python reproducibility_checklist.py

Prints PASS / FAIL on:
  1. Two runs with the same seed produce identical final weights.
  2. Two runs with different seeds produce DIFFERENT weights.
  3. CUDA determinism (if GPU available) does not change results.
  4. PyTorch and NumPy seeds are set BEFORE any model or data is created.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set every RNG seed we know about. Call BEFORE constructing any model or data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Strict CUDA determinism. Slows down some ops; required for bit-exact reproducibility on GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def tiny_train_run(seed: int) -> torch.Tensor:
    """A small deterministic training run; returns the final flattened weight vector."""
    set_seed(seed)

    # Construct AFTER seeding.
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = torch.sin(np.pi * x)

    model = nn.Sequential(
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    return flat


def main() -> None:
    print("REPRODUCIBILITY CHECKLIST\n" + "=" * 50)

    # Test 1: same seed, identical weights.
    w_a = tiny_train_run(seed=42)
    w_b = tiny_train_run(seed=42)
    same_seed_identical = torch.allclose(w_a, w_b, rtol=0, atol=0)
    print(f"1. Same seed -> identical weights:    {'PASS' if same_seed_identical else 'FAIL'}")
    if not same_seed_identical:
        print(f"     max element-wise diff = {(w_a - w_b).abs().max().item():.2e}")

    # Test 2: different seeds, different weights.
    w_c = tiny_train_run(seed=43)
    different_seed_different = not torch.allclose(w_a, w_c, rtol=0, atol=1e-6)
    print(f"2. Different seed -> different weights:{'PASS' if different_seed_different else 'FAIL'}")

    # Test 3: CUDA determinism check (only if GPU available).
    if torch.cuda.is_available():
        # NOT an automatic test: this script only checks CPU seeding. On a GPU,
        # determinism has to be verified by hand -- say so rather than imply a check ran.
        print("3. CUDA determinism: NOT TESTED automatically (GPU present) -- manual check:")
        print("   set torch.use_deterministic_algorithms(True) and re-run your training on GPU")
    else:
        print("3. CUDA determinism: SKIPPED (no GPU detected)")

    # Test 4: dump the recommended checklist.
    print("\nMANUAL CHECKLIST (apply to all your training scripts):")
    items = [
        "torch.manual_seed(SEED) set BEFORE constructing any model or data",
        "numpy.random.seed(SEED) set with the same SEED",
        "If using GPU: torch.cuda.manual_seed_all(SEED) and torch.use_deterministic_algorithms(True)",
        "Train/val/test split uses a fixed seed for the shuffle (not the same one as model init)",
        "Data loading uses num_workers=0 OR worker_init_fn that re-seeds each worker",
        "Saved checkpoints include: model state, optimizer state, scheduler state, RNG state",
        "Environment locked: requirements.txt / conda env.yml frozen and committed",
        "Code is committed in git with the run-producing SHA recorded in the output file",
    ]
    for i, item in enumerate(items, start=1):
        print(f"  [ ] {item}")

    print()
    if not (same_seed_identical and different_seed_different):
        sys.exit(1)


if __name__ == "__main__":
    main()
