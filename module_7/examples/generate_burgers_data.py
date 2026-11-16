"""
generate_burgers_data.py
=========================

Module 7 / Lab 7 -- generate 1-D viscous Burgers' equation training data
via spectral method.

The PDE:
    u_t + u u_x = nu u_xx,  x in [0, 1] (periodic), t in [0, 1],  u(x, 0) = u_0(x)

For each random initial condition u_0(x) drawn from a Gaussian random field,
solve to T=1 using a Fourier-spectral method with 4th-order Runge-Kutta time
stepping. Save (u_0, u_T) pairs.

Usage
-----
    python generate_burgers_data.py --n 1000 --N 128 --out burgers_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000, help="Number of (u_0, u_T) pairs")
    p.add_argument("--N", type=int, default=128, help="Spatial grid resolution")
    p.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    p.add_argument("--T", type=float, default=1.0, help="Final time")
    p.add_argument("--dt", type=float, default=1e-3, help="Time step")
    p.add_argument("--length-scale", type=float, default=0.3, help="GRF length scale")
    p.add_argument("--out", type=Path, default=Path("burgers_data.npz"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_grf(N, length_scale, rng):
    """Sample a Gaussian random field with given correlation length scale on [0,1] periodic."""
    k = np.fft.fftfreq(N, d=1.0 / N)
    # Power spectrum: Gaussian decay with k
    spectrum = np.exp(-(length_scale * k * np.pi) ** 2)
    # Sample complex Gaussian, multiply by spectrum, inverse FFT
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    z = z * spectrum
    u = np.fft.ifft(z).real * N
    u = u / np.abs(u).max() * 1.0     # normalize to ~|u| < 1
    return u


def solve_burgers_spectral(u0, nu, T, dt, N):
    """Solve 1-D periodic Burgers' equation with 4th-order RK time stepping."""
    k = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
    k_sq = k * k
    # A simplified aliasing filter: zero the top third of the modes of u_x (the product u * u_x
    # itself is not filtered, so this is not full 2/3-rule dealiasing). fftfreq order puts the
    # highest |k| in the middle of the array, so the middle is zeroed directly. (Do not fftshift
    # this mask: that would zero the LOW modes and delete the nonlinear term, leaving the heat
    # equation.) Validated for the course's smooth GRF inputs, nu = 0.01, T = 1 and N = 64-512
    # against Cole-Hopf and finer-grid references (module_7/readings/measured_results.md); it is
    # not a certificate for rough inputs, other viscosities or other grids.
    cutoff = N // 3
    dealias = np.ones(N)
    dealias[cutoff : N - cutoff] = 0

    def rhs(u):
        """du/dt = -u * du/dx + nu * d^2 u / dx^2"""
        u_hat = np.fft.fft(u)
        # First derivative (with dealias)
        ux_hat = 1j * k * u_hat * dealias
        ux = np.fft.ifft(ux_hat).real
        # Second derivative
        uxx_hat = -k_sq * u_hat
        uxx = np.fft.ifft(uxx_hat).real
        return -u * ux + nu * uxx

    # Explicit RK4 is stable for the diffusion term only while nu * k_max^2 * dt stays below ~2.8,
    # with k_max = pi * N. The default dt = 1e-3 gives about 1.6 at N = 128 and is kept unchanged
    # there; on finer grids (N = 256, 512, ...) dt is reduced so the number stays near 1.6.
    # Without this, every sample diverges at N >= 256 (checked Nov 13, 2026).
    stab = nu * (np.pi * N) ** 2 * dt
    if stab > 2.0:
        dt = dt * 1.6 / stab
    u = u0.copy()
    n_steps = int(np.ceil(T / dt - 1e-9))
    dt = T / n_steps
    for _ in range(n_steps):
        # RK4
        k1 = dt * rhs(u)
        k2 = dt * rhs(u + 0.5 * k1)
        k3 = dt * rhs(u + 0.5 * k2)
        k4 = dt * rhs(u + k3)
        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return u


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    u0_all = []
    uT_all = []

    for i in range(args.n):
        u0 = sample_grf(args.N, args.length_scale, rng)
        try:
            uT = solve_burgers_spectral(u0, args.nu, args.T, args.dt, args.N)
            if not np.all(np.isfinite(uT)):
                print(f"Sample {i}: solution diverged, skipping.")
                continue
            u0_all.append(u0)
            uT_all.append(uT)
        except Exception as e:
            print(f"Sample {i} failed: {e}; skipping.")
        if (i + 1) % 100 == 0:
            print(f"  done {i+1}/{args.n}")

    u0_arr = np.stack(u0_all, axis=0).astype(np.float32)
    uT_arr = np.stack(uT_all, axis=0).astype(np.float32)
    x_grid = np.linspace(0, 1, args.N, endpoint=False, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        u0=u0_arr,
        uT=uT_arr,
        x_grid=x_grid,
        nu=args.nu,
        T=args.T,
        N=args.N,
    )
    print(f"\nWrote {len(u0_all)} pairs to {args.out}")
    print(f"u0 range: [{u0_arr.min():.3f}, {u0_arr.max():.3f}]")
    print(f"uT range: [{uT_arr.min():.3f}, {uT_arr.max():.3f}]")


if __name__ == "__main__":
    main()
