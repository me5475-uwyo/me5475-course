"""
aggregate_parts.py
==================

Module 6 / Lab 6 Task 1b -- combine the array workers' partial datasets into one file.

Each worker of data_generation.sbatch writes rve_dataset_part_<k>.npz. This script stacks every
per-sample field (images, moduli, the ellipse lists, and stress_fields when the parts have them)
and keeps the run constants (E_matrix, E_inclusion, nu, grid_size) once, after checking that all
parts agree on them.

Usage (in the folder holding the parts):
    python aggregate_parts.py                                  # -> rve_dataset.npz
    python aggregate_parts.py --pattern "rve_dataset_part_*.npz" --out rve_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PER_SAMPLE = ("images", "moduli", "ellipses", "stress_fields")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="rve_dataset_part_*.npz")
    p.add_argument("--out", type=Path, default=Path("rve_dataset.npz"))
    args = p.parse_args()

    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r} in {Path('.').resolve()}")
    parts = [np.load(f, allow_pickle=True) for f in paths]

    keys = set(parts[0].files)
    for f, part in zip(paths, parts):
        if set(part.files) != keys:
            raise SystemExit(f"{f} has fields {sorted(part.files)}, unlike {paths[0]} ({sorted(keys)})")

    out = {}
    for k in sorted(keys):
        if k in PER_SAMPLE:
            out[k] = np.concatenate([part[k] for part in parts])
        else:                                   # run constants: must agree across parts
            first = parts[0][k]
            for f, part in zip(paths[1:], parts[1:]):
                if not np.array_equal(part[k], first):
                    raise SystemExit(f"{k} differs between {paths[0]} and {f}")
            out[k] = first

    np.savez(args.out, **out)
    m = out["moduli"]
    print(f"Combined {len(paths)} parts: {len(m)} RVEs -> {args.out}")
    print(f"Fields: {', '.join(sorted(out))}")
    print(f"Modulus: mean {m.mean():.4f}, std {m.std():.4f}, range {m.min():.4f} -- {m.max():.4f}")


if __name__ == "__main__":
    main()
