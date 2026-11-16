#!/usr/bin/env python3
"""heat_fourier_example.py -- Module 7 / L32: Fourier basics on a problem you can solve by hand.

The periodic 1-D heat equation (Burgers without its u u_x term, same domain and viscosity as the Lab 7 data):

    u_t = nu u_xx,  x in [0, 1) periodic,  u(x, 0) = sin(2 pi x) + 0.5 sin(6 pi x),  nu = 0.01

By hand: each Fourier mode decays on its own, a_k' = -nu kappa_k^2 a_k with wavenumber kappa_k = 2 pi k / L (L = 1),
so  u(x, t) = exp(-4 pi^2 nu t) sin(2 pi x) + 0.5 exp(-36 pi^2 nu t) sin(6 pi x).

With the FFT: rfft the samples, multiply coefficient k by exp(-nu kappa_k^2 t), irfft back. The transform only
changes the representation; the PDE supplies the multiplier. This script does both and compares them. It is a teaching
calculation, not a trained model. Numbers it prints are recorded in module_7/readings/measured_results.md (section 3).

Usage:  python heat_fourier_example.py                 # N = 64, nu = 0.01, t = 1
        python heat_fourier_example.py --N 256
        python heat_fourier_example.py --figdir figs   # also write the three L32 figures (PDF)
"""
import argparse
from pathlib import Path

import numpy as np


def initial_condition(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)


def exact_solution(x, nu, t):
    """The hand solution: each mode decays at its own rate nu * kappa_k^2."""
    return (np.exp(-4 * np.pi**2 * nu * t) * np.sin(2 * np.pi * x)
            + 0.5 * np.exp(-36 * np.pi**2 * nu * t) * np.sin(6 * np.pi * x))


def fft_solution(u0, nu, t):
    """Sampled u0 -> rfft -> multiply mode k by exp(-nu kappa_k^2 t) -> irfft."""
    N = u0.size
    k = np.fft.rfftfreq(N, d=1.0 / N)          # integer mode index 0, 1, ..., N/2
    kappa = 2 * np.pi * k                      # physical wavenumber on a domain of length L = 1
    u_hat = np.fft.rfft(u0)                    # numpy's default normalization: forward unscaled
    return np.fft.irfft(u_hat * np.exp(-nu * kappa**2 * t), n=N)   # inverse divides by N


def amplitudes(u):
    """Real amplitude of each mode read from rfft: 2 |u_hat_k| / N for interior modes 0 < k < N/2.

    The two end bins are not doubled: k = 0 (the mean, signed value u_hat_0.real / N) and, for even N, the Nyquist
    bin k = N/2 (amplitude |u_hat_k| / N)."""
    N = u.size
    a = 2 * np.abs(np.fft.rfft(u)) / N
    a[0] /= 2
    if N % 2 == 0:
        a[-1] /= 2
    return a


def make_figures(figdir, nu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BROWN, GOLD, INK, GRAY = "#492F24", "#8A5E0D", "#4A423A", "#9A8F84"
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
    figdir.mkdir(parents=True, exist_ok=True)
    xf = np.linspace(0, 1, 400)
    N = 64
    x = np.arange(N) / N

    # 1. a function as a sum of two waves
    fig, ax = plt.subplots(figsize=(3.2, 1.75), dpi=200)
    ax.plot(xf, np.sin(2 * np.pi * xf), "--", color=GOLD, lw=1.1, label=r"$\sin 2\pi x$  (mode 1)")
    ax.plot(xf, 0.5 * np.sin(6 * np.pi * xf), ":", color=GRAY, lw=1.3, label=r"$\frac{1}{2}\sin 6\pi x$  (mode 3)")
    ax.plot(xf, initial_condition(xf), "-", color=BROWN, lw=1.8, label=r"$u_0(x)$, their sum")
    ax.set_xlabel("$x$"); ax.set_xlim(0, 1); ax.axhline(0, color=GRAY, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6,
              columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "l32_heat_waves.pdf", metadata={"CreationDate": None}); plt.close(fig)

    # 2. the same function as coefficients, at t = 0 and t = 1
    a0, a1 = amplitudes(initial_condition(x)), amplitudes(fft_solution(initial_condition(x), nu, 1.0))
    ks = np.arange(0, 7)
    fig, ax = plt.subplots(figsize=(2.3, 1.75), dpi=200)
    ax.bar(ks - 0.18, a0[:7], width=0.34, color=GOLD, label="$t = 0$")
    ax.bar(ks + 0.18, a1[:7], width=0.34, color=BROWN, label="$t = 1$")
    ax.set_xticks(ks); ax.set_xlabel("mode $k$"); ax.set_ylabel(r"amplitude $2|\hat u_k|/N$")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "l32_heat_spectrum.pdf", metadata={"CreationDate": None}); plt.close(fig)

    # 3. hand solution (lines) vs FFT solution (dots) at three times
    fig, ax = plt.subplots(figsize=(3.2, 1.75), dpi=200)
    for t, c, ls in [(0.0, GOLD, "--"), (0.25, GRAY, "-."), (1.0, BROWN, "-")]:
        ax.plot(xf, exact_solution(xf, nu, t), ls, color=c, lw=1.4, label=f"$t = {t:g}$")
        ax.plot(x[::2], fft_solution(initial_condition(x), nu, t)[::2], "o", ms=2.0, color=c, mew=0)
    ax.set_xlabel("$x$"); ax.set_xlim(0, 1); ax.axhline(0, color=GRAY, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc="lower left", title="lines: by hand   dots: FFT", title_fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "l32_heat_solution.pdf", metadata={"CreationDate": None}); plt.close(fig)
    print(f"Saved 3 figures to {figdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=64, help="grid points x_j = j / N (no duplicate endpoint)")
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--figdir", type=Path, default=None)
    args = p.parse_args()
    if args.N < 7:
        p.error("--N must be at least 7: mode 3 must lie below the Nyquist mode N/2, or the grid aliases it "
                "and the FFT solution no longer matches the hand solution")

    x = np.arange(args.N) / args.N
    u0 = initial_condition(x)
    u_hat = np.fft.rfft(u0)
    shown = np.round(u_hat, 9) + 0.0           # hide roundoff-level real parts (about 1e-14) in the printout
    print(f"Grid N = {args.N}, nu = {args.nu}, t = {args.t}")
    print(f"rfft returns N//2 + 1 = {u_hat.size} complex coefficients (modes 0 .. {args.N // 2})")
    print(f"  u_hat_1 = {shown[1].real:+.3g} {shown[1].imag:+.3g}i,  u_hat_3 = {shown[3].real:+.3g} "
          f"{shown[3].imag:+.3g}i   (a sine of amplitude A at mode k gives -i A N / 2)")
    a = amplitudes(u0)
    print("  amplitudes (2|u_hat_k|/N inside, not doubled at k = 0 or Nyquist), k = 0..5:", " ".join(f"{v:.3f}" for v in a[:6]))
    for k in (1, 3):
        print(f"  mode {k}: decay factor exp(-nu (2 pi k)^2 t) = {np.exp(-args.nu * (2 * np.pi * k)**2 * args.t):.4f}")

    u_fft, u_exact = fft_solution(u0, args.nu, args.t), exact_solution(x, args.nu, args.t)
    err = u_fft - u_exact
    print(f"FFT vs hand solution at t = {args.t}: max |error| = {np.abs(err).max():.1e}, "
          f"relative L2 = {np.linalg.norm(err) / np.linalg.norm(u_exact):.1e}")
    print("(Roundoff level: u0 contains only modes 1 and 3, so the grid represents it exactly.)")
    if args.figdir is not None:
        make_figures(args.figdir, args.nu)


if __name__ == "__main__":
    main()
