#!/usr/bin/env python3
"""fft_basics_examples.py -- Module 7 / L32: five small FFT examples for students who have never used one.

Convention (NumPy's default, as on the L32 slides): u_hat_k = sum_j u_j exp(-2 pi i j k / N), the inverse divides by
N; grid x_j = j / N on [0, 1) with no duplicate endpoint. A cosine of amplitude A at mode k gives A N / 2 (real), a
sine gives -i A N / 2, so the amplitude of an interior mode is 2 |u_hat_k| / N (the bin k = 0, and for even N the bin
k = N/2, are not doubled).

  1  Find the waves: rfft a sum of a sine and a cosine and read which modes are present, how big, and which kind.
     The position of a nonzero coefficient is the mode, its size the amplitude, its angle sine vs cosine.
  2  Natural frequency from a vibration record: a damped spring-mass free-vibration displacement record sampled at
     100 Hz for 20 s. In this run the largest rfft bin falls within half a bin (bin spacing 1/T) of the known damped
     frequency; bin spacing is not a general accuracy bound.
  3  Smoothing a noisy measurement: zero every mode above k_c = 10 and transform back (a low-pass filter).
     Noise spreads over all modes; the signal lives in a few low ones, so most of the noise is removed.
  4  Derivative with the FFT: multiply mode k by i kappa_k and transform back, against a central finite difference.
     For this resolved two-mode function the spectral derivative matches to roundoff; for a general smooth periodic
     function the accuracy depends on resolving its Fourier content.
  5  Convolution theorem: a periodic 5-point moving average done directly and as a product of transforms.
     Both give the same result to roundoff; for long kernels the FFT route is the cheap one (N log N vs N^2).

A teaching calculation (NumPy only; matplotlib only for --figdir), deterministic (fixed seeds), runs in about a
second. The numbers it prints are recorded in module_7/readings/measured_results.md (section 4).

Usage:  python fft_basics_examples.py                    # all five examples
        python fft_basics_examples.py --example 2
        python fft_basics_examples.py --figdir figs       # also write one small PDF per example
"""
import argparse
from pathlib import Path

import numpy as np

BROWN, GOLD, INK, GRAY = "#492F24", "#8A5E0D", "#4A423A", "#9A8F84"


def grid(N):
    """x_j = j / N, j = 0..N-1: periodic grid on [0, 1), no duplicate endpoint."""
    return np.arange(N) / N


def amplitudes(u_hat, N):
    """Real amplitude per rfft bin: 2 |u_hat_k| / N inside, |u_hat_k| / N at k = 0 and (even N) k = N/2."""
    a = 2 * np.abs(u_hat) / N
    a[0] /= 2
    if N % 2 == 0:
        a[-1] /= 2
    return a


def kind_from_angle(c):
    """u = a cos + b sin at an interior mode gives u_hat = (a - i b) N / 2; name the pure cases by the angle."""
    deg = np.degrees(np.angle(c))
    for target, name in [(0, "cosine"), (180, "-cosine"), (-180, "-cosine"), (-90, "sine"), (90, "-sine")]:
        if abs(deg - target) < 1e-6:
            return deg, name
    return deg, "mixed (cosine and sine)"


# ---------------------------------------------------------------- 1. find the waves
def example_1():
    N = 64
    x = grid(N)
    u = 2 * np.sin(2 * np.pi * x) + 0.5 * np.cos(10 * np.pi * x)
    u_hat = np.fft.rfft(u)
    a = amplitudes(u_hat, N)
    present = np.flatnonzero(np.abs(u_hat) > 1e-9 * N)
    print("Example 1 -- find the waves: u = 2 sin 2 pi x + 0.5 cos 10 pi x, N = 64")
    print(f"  rfft returns {u_hat.size} coefficients (k = 0 .. {N // 2}); nonzero at k = {present.tolist()}")
    for k in present:
        c = np.round(u_hat[k], 9) + 0.0
        deg, name = kind_from_angle(c)
        print(f"  k = {k}: u_hat = {c.real:+.4g} {c.imag:+.4g}i, amplitude 2|u_hat|/N = {a[k]:.4f}, "
              f"angle {deg:+.0f} deg -> {name}")
    return dict(x=x, u=u, a=a)


# ---------------------------------------------------------------- 2. natural frequency from a vibration record
def example_2():
    m, k_spring, zeta = 1.0, 400.0, 0.02
    wn = np.sqrt(k_spring / m)                       # 20 rad/s
    fn = wn / (2 * np.pi)
    fd = fn * np.sqrt(1 - zeta**2)
    fs, T = 100.0, 20.0
    N = int(round(fs * T))
    t = np.arange(N) / fs                            # no duplicate endpoint, as on the periodic grid
    x = np.exp(-zeta * wn * t) * np.cos(wn * np.sqrt(1 - zeta**2) * t)
    X = np.fft.rfft(x - x.mean())                    # remove the mean so bin 0 cannot dominate
    f = np.fft.rfftfreq(N, d=1 / fs)                 # 0, 1/T, 2/T, ..., fs/2  (Hz)
    i = np.argmax(np.abs(X))
    df = f[1] - f[0]
    print("Example 2 -- natural frequency from a vibration record: m = 1 kg, k = 400 N/m, zeta = 0.02")
    print(f"  omega_n = {wn:.1f} rad/s, f_n = {fn:.4f} Hz, exact damped f_d = f_n sqrt(1 - zeta^2) = {fd:.4f} Hz")
    print(f"  sampled at fs = {fs:g} Hz for T = {T:g} s: N = {N} samples, rfft returns {X.size} bins, "
          f"Nyquist fs/2 = {f[-1]:g} Hz")
    print(f"  bin spacing 1/T = {df:.3f} Hz; largest bin at {f[i]:.3f} Hz (bin {i}); "
          f"peak - f_d = {f[i] - fd:+.4f} Hz (this run: within half a bin, {df / 2:.3f} Hz)")
    print(f"  envelope factor at t = T: exp(-zeta omega_n T) = {np.exp(-zeta * wn * T):.1e}. The record starts at 1 "
          "and ends near 0, so it is not periodic across the seam,\n  and a decaying transient has a broadened "
          "spectrum: the peak locates f_d here, but bin spacing is not an error bound")
    return dict(f=f, A=np.abs(X), fd=fd, fpk=f[i], t=t, x=x)


# ---------------------------------------------------------------- 3. smoothing a noisy measurement
def example_3():
    N, sigma, kc = 256, 0.05, 10
    x = grid(N)
    clean = np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)       # modes 1 and 3
    noisy = clean + sigma * np.random.default_rng(0).standard_normal(N)
    u_hat = np.fft.rfft(noisy)
    u_hat[kc + 1:] = 0.0                                                 # keep k = 0 .. k_c
    smooth = np.fft.irfft(u_hat, n=N)
    rms = lambda e: np.sqrt(np.mean(e**2))
    print(f"Example 3 -- smoothing: u = sin 2 pi x + 0.3 sin 6 pi x plus Gaussian noise (sigma = {sigma}, seed 0), "
          f"N = {N}")
    print(f"  low-pass: keep modes k = 0 .. {kc} ({kc + 1} of {N // 2 + 1}), zero the rest, irfft")
    print(f"  RMS error vs the clean signal: before {rms(noisy - clean):.4f}, after {rms(smooth - clean):.4f}")
    return dict(x=x, clean=clean, noisy=noisy, smooth=smooth)


# ---------------------------------------------------------------- 4. derivative with the FFT
def example_4():
    N = 32
    x = grid(N)
    h = 1 / N
    u = np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    du = 2 * np.pi * np.cos(2 * np.pi * x) + 3 * np.pi * np.cos(6 * np.pi * x)
    kappa = 2 * np.pi * np.fft.rfftfreq(N, d=1 / N)                     # kappa_k = 2 pi k / L, L = 1
    mult = 1j * kappa
    mult[-1] = 0.0                                                       # Nyquist bin: no odd derivative
    du_spec = np.fft.irfft(mult * np.fft.rfft(u), n=N)
    du_fd = (np.roll(u, -1) - np.roll(u, 1)) / (2 * h)                   # periodic central difference
    print(f"Example 4 -- derivative: u = sin 2 pi x + 0.5 sin 6 pi x, N = {N}")
    print(f"  max |error| vs the exact u': spectral (i kappa_k, Nyquist zeroed) {np.abs(du_spec - du).max():.1e}, "
          f"central finite difference {np.abs(du_fd - du).max():.3f}  (max |u'| = {np.abs(du).max():.2f})")
    return dict(x=x, du=du, du_spec=du_spec, du_fd=du_fd)


# ---------------------------------------------------------------- 5. convolution theorem
def example_5():
    N = 256
    x = grid(N)
    u = (np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)
         + 0.05 * np.random.default_rng(0).standard_normal(N))         # the noisy signal of example 3
    direct = sum(np.roll(u, s) for s in (-2, -1, 0, 1, 2)) / 5          # y_j = (u_{j-2} + ... + u_{j+2}) / 5
    kernel = np.zeros(N)
    kernel[[0, 1, 2, -1, -2]] = 1 / 5                                    # the same average as a periodic kernel
    via_fft = np.fft.irfft(np.fft.rfft(u) * np.fft.rfft(kernel), n=N)
    print(f"Example 5 -- convolution theorem: periodic 5-point moving average of the example-3 signal, N = {N}")
    print(f"  max |direct - via FFT| = {np.abs(direct - via_fft).max():.1e}  (roundoff)")
    print(f"  cost for a kernel as long as the signal: direct N^2 = {N**2}, FFT about N log2 N = "
          f"{int(N * np.log2(N))} per transform")
    return dict(x=x, u=u, direct=direct, via_fft=via_fft)


# ---------------------------------------------------------------- figures
def make_figures(figdir, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
    figdir.mkdir(parents=True, exist_ok=True)
    meta = {"CreationDate": None}
    saved = []

    def save(fig, name):
        fig.tight_layout(pad=0.3)
        fig.savefig(figdir / name, metadata=meta)
        plt.close(fig)
        saved.append(name)

    if 1 in res:
        r = res[1]
        fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=200)
        ks = np.arange(0, 9)
        ax.bar(ks, r["a"][:9], width=0.6, color=GOLD)
        ax.set_xticks(ks); ax.set_xlabel("mode $k$"); ax.set_ylabel(r"amplitude $2|\hat u_k|/N$")
        ax.set_title("sine at $k = 1$, cosine at $k = 5$", fontsize=7, color=INK)
        save(fig, "fft_ex1_spectrum.pdf")
    if 2 in res:
        r = res[2]
        fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=200)
        sel = r["f"] <= 8
        ax.plot(r["f"][sel], r["A"][sel], "-", color=BROWN, lw=1.0)
        ax.axvline(r["fd"], color=GOLD, ls="--", lw=0.9, label=f"exact $f_d$ = {r['fd']:.3f} Hz")
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel(r"$|\hat x_k|$")
        ax.legend(frameon=False, fontsize=6)
        save(fig, "fft_ex2_vibration.pdf")
    if 3 in res:
        r = res[3]
        fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=200)
        ax.plot(r["x"], r["noisy"], ".", ms=1.6, color=GRAY, label="noisy")
        ax.plot(r["x"], r["clean"], "--", color=GOLD, lw=1.2, label="clean")
        ax.plot(r["x"], r["smooth"], "-", color=BROWN, lw=1.0, label="low-pass ($k \\leq 10$)")
        ax.set_xlabel("$x$"); ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        save(fig, "fft_ex3_smoothing.pdf")
    if 4 in res:
        r = res[4]
        fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=200)
        xf = np.linspace(0, 1, 400)
        ax.plot(xf, 2 * np.pi * np.cos(2 * np.pi * xf) + 3 * np.pi * np.cos(6 * np.pi * xf), "-", color=INK,
                lw=0.8, label="exact $u'$")
        ax.plot(r["x"], r["du_spec"], "o", ms=2.2, color=BROWN, mew=0, label="spectral")
        ax.plot(r["x"], r["du_fd"], "s", ms=2.0, color=GOLD, mew=0, label="central difference")
        ax.set_xlabel("$x$"); ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        save(fig, "fft_ex4_derivative.pdf")
    if 5 in res:
        r = res[5]
        fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=200)
        ax.plot(r["x"], r["u"], ".", ms=1.6, color=GRAY, label="signal")
        ax.plot(r["x"], r["direct"], "-", color=BROWN, lw=1.2, label="direct average")
        ax.plot(r["x"][::8], r["via_fft"][::8], "o", ms=2.0, color=GOLD, mew=0, label="via FFT")
        ax.set_xlabel("$x$"); ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        save(fig, "fft_ex5_convolution.pdf")
    print(f"Saved {len(saved)} figures to {figdir}: {', '.join(saved)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example", choices=["1", "2", "3", "4", "5", "all"], default="all")
    p.add_argument("--figdir", type=Path, default=None, help="also write one PDF per example here")
    args = p.parse_args()
    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}
    chosen = list(examples) if args.example == "all" else [int(args.example)]
    res = {}
    for n in chosen:
        res[n] = examples[n]()
        print()
    if args.figdir is not None:
        make_figures(args.figdir, res)


if __name__ == "__main__":
    main()
