"""
icnn_rank_one_check.py
======================

Module 4 / Lecture 21 companion -- "check, don't trust the name."

An ICNN is convex in its INPUTS by construction. That is not the same as a
polyconvex energy psi(F). For the inputs of icnn_hyperelastic.py,
y = (I_1 - 3, I_2 - 3, J - 1, -ln J), the standard SUFFICIENT construction
(Klein et al. 2022) also makes psi non-decreasing in I_1, I_2 and -ln J. The skip
weights are unconstrained, so nothing enforces that. This script checks a
trained model in two ways:

1. Monotonicity, the sufficient condition: the sign of dpsi/dy for I_1 - 3,
   I_2 - 3 and -ln J at sample F (J - 1 may have either sign). Failing it does
   NOT disprove polyconvexity: on the course window 0.5 < J < 1.8 the neo-Hookean
   truth is polyconvex, yet it decreases in -ln J where J > exp(mu / lam) ~ 1.65.
   (It is not polyconvex for all J > 0: its volumetric term stops being convex
   in J at large J.) Failing the test shows only that the
   sufficient construction is not what is holding the model up.
2. Rank-one convexity, which polyconvexity implies (a necessary condition): the curvature
   d^2/dt^2 psi(F + t a (x) b) at t = 0 must be >= 0 for every F and every pair
   of directions a, b. Checked at random (F, a, b), for the model and for the
   neo-Hookean truth.

A clearly negative curvature PROVES the model is not polyconvex. No negative
curvature at the samples does NOT prove it is: the check only looks where it
samples. Curvatures within --tol of zero are reported separately, because the
sign of a value that close to zero can differ between machines.

Usage
-----
    python icnn_hyperelastic.py                  # trains and writes icnn_hyperelastic.pt
    python icnn_rank_one_check.py --model icnn_hyperelastic.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import icnn_hyperelastic as ih


def sample_F(n, generator):
    """F's from the training window of icnn_hyperelastic.py: I + symmetric perturbation, 0.5 < J < 1.8."""
    eta = 0.3 * torch.randn(n, 3, 3, generator=generator)
    F = torch.eye(3).unsqueeze(0) + 0.5 * (eta + eta.transpose(-1, -2))
    J = torch.linalg.det(F)
    return F[(J > 0.5) & (J < 1.8)]


def neo_hookean_psi(F, mu=0.5, lam=1.0):
    """Energy only, differentiable twice (same constants as ih.neo_hookean)."""
    C = torch.einsum("nij,nik->njk", F, F)
    I_1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    J = torch.linalg.det(F)
    return (mu / 2) * (I_1 - 3 - 2 * torch.log(J)) + (lam / 2) * torch.log(J) ** 2


def rank_one_curvature(psi_fn, F, M):
    """d^2/dt^2 psi(F + t M) at t = 0, for each sample, by autograd twice."""
    t = torch.zeros(F.shape[0], requires_grad=True)
    value = psi_fn(F + t[:, None, None] * M)
    d1 = torch.autograd.grad(value.sum(), t, create_graph=True)[0]
    return torch.autograd.grad(d1.sum(), t)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("icnn_hyperelastic.pt"))
    parser.add_argument("--n-F", type=int, default=200, help="sample F's drawn (some are rejected by the J window)")
    parser.add_argument("--n-dirs", type=int, default=20, help="random (a, b) pairs per F")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6, help="curvatures in [-tol, 0) are reported as near zero")
    args = parser.parse_args()

    saved = torch.load(args.model, weights_only=True)   # restricted loader: tensors and primitives only
    model = ih.HyperelasticICNN(hidden=saved["hidden"], depth=saved["depth"])
    model.load_state_dict(saved["state_dict"])
    model.eval()

    g = torch.Generator().manual_seed(args.seed)
    F = sample_F(args.n_F, g)

    # 1. monotonicity in the ICNN inputs
    y = ih.invariants_of_F(F).detach().requires_grad_(True)
    dpsi_dy = torch.autograd.grad(model.icnn(y).sum(), y)[0]
    print(f"{args.model}: {len(F)} sample F's, {args.n_dirs} direction pairs each")
    print("1. Monotonicity (the sufficient construction wants dpsi/dy >= 0 for I_1 - 3, I_2 - 3, -ln J):")
    for k, name in [(0, "I_1 - 3"), (1, "I_2 - 3"), (3, "-ln J")]:
        neg = (dpsi_dy[:, k] < 0).float().mean().item()
        print(f"   dpsi/d({name:7s}) < 0 at {neg:6.1%} of the sample F's")
    # the truth, psi = mu/2 (I_1 - 3) + mu (-ln J) + lam/2 (-ln J)^2: dpsi/d(-ln J) = mu - lam ln J
    J = torch.linalg.det(F)
    print(f"   (neo-Hookean truth: dpsi/d(-ln J) = mu - lam ln J < 0 at {(0.5 - 1.0 * torch.log(J) < 0).float().mean().item():.1%}"
          " of the sample F's -- and it is polyconvex on this window)")

    # 2. rank-one convexity, model vs the neo-Hookean truth
    total = neg_model = neg_truth = near_zero = 0
    worst = float("inf")
    for _ in range(args.n_dirs):
        a = torch.randn(len(F), 3, generator=g)
        b = torch.randn(len(F), 3, generator=g)
        M = (a / a.norm(dim=1, keepdim=True))[:, :, None] * (b / b.norm(dim=1, keepdim=True))[:, None, :]
        c_model = rank_one_curvature(model, F, M)
        c_truth = rank_one_curvature(neo_hookean_psi, F, M)
        total += len(F)
        neg_model += int((c_model < -args.tol).sum())
        near_zero += int(((c_model >= -args.tol) & (c_model < 0)).sum())
        neg_truth += int((c_truth < 0).sum())
        worst = min(worst, float(c_model.min()))
    print("2. Rank-one curvature d2/dt2 psi(F + t a(x)b) (polyconvexity needs >= 0 everywhere):")
    print(f"   model:       below -{args.tol:g} at {neg_model} / {total} ({neg_model / total:.1%}), minimum {worst:+.3e};"
          f" in [-{args.tol:g}, 0): {near_zero}")
    print(f"   neo-Hookean: negative at {neg_truth} / {total}")
    if neg_model:
        print("Verdict: NOT polyconvex -- a clearly negative rank-one curvature was found.")
    else:
        print("Verdict: no violation at these samples. That does not prove polyconvexity;"
              " it only means this check found none.")


if __name__ == "__main__":
    main()
