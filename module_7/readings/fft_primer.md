# FFT Primer

A short introduction for students who have never used a fast Fourier transform. Read it before or right after L32
(slides 4–10); the FNO's spectral layer is built on everything here. Runnable companion:
`module_7/examples/fft_basics_examples.py`. Every computed number below is recorded in `measured_results.md` §4.

---

## 1. A function as a sum of waves

On a periodic domain [0, L) almost any function you meet in mechanics can be written as a sum of sines and cosines:

u(x) = a_0 + Σ_k [ a_k cos(κ_k x) + b_k sin(κ_k x) ],   κ_k = 2πk / L.

- **Mode index k** (an integer) counts oscillations across the domain: mode 1 is one full wave, mode 5 is five.
  Mode 0 is the mean.
- **Wavenumber κ_k = 2πk / L** is the physical frequency in space (radians per unit length). With L = 1, as in the
  course examples, κ_k = 2πk. A derivative d/dx multiplies mode k by iκ_k; a second derivative by −κ_k².
- **Two representations of the same thing.** On a computer you hold N **samples** u_j = u(x_j). The FFT turns them
  into N **coefficients** û_k, "how much of each wave". Nothing is lost: the inverse gives the samples back exactly.

## 2. The DFT and what "fast" means

The discrete Fourier transform (DFT) of N samples, and its inverse (the convention on the L32 slides and in NumPy):

û_k = Σ_{j=0}^{N−1} u_j e^{−2πi jk/N},   u_j = (1/N) Σ_{k=0}^{N−1} û_k e^{+2πi jk/N}.

In words: **û_k compares the samples with a wave that oscillates k times across the domain**; a large |û_k| means a
lot of that wave. The inverse adds the waves back up, and it is the inverse that carries the 1/N.

Computing every û_k directly from the sum costs about N² multiplications. The **fast Fourier transform** (FFT;
Cooley & Tukey, 1965) splits the sum into halves recursively and costs about **N log₂ N**. The DFT is the transform;
the FFT is the algorithm (the halving argument is the radix-2 case, N a power of two; N log N is an asymptotic cost,
not an exact count). Here we use NumPy's FFT routines.

## 3. NumPy in practice

| Call | What it returns |
|---|---|
| `np.fft.fft(u)` | all N coefficients, k = 0, …, N−1 (complex) |
| `np.fft.rfft(u)` | for real u: only k = 0, …, floor(N/2), i.e. **N//2 + 1** numbers (the rest are their complex conjugates) |
| `np.fft.irfft(u_hat, n=N)` | back to N real samples (pass `n`, or an odd N is lost) |
| `np.fft.fftfreq(N, d)`, `np.fft.rfftfreq(N, d)` | the frequency of each bin, with sample spacing d |

- **Normalization.** NumPy's default: the forward transform is unscaled, the inverse divides by N. Other libraries
  (and some textbooks) put 1/N or 1/√N elsewhere; check before you compare numbers.
- **Mode index from `rfftfreq`.** With `d = 1/N` it returns the integer modes 0, 1, …, N/2; with `d = 1/fs` (a time
  signal sampled at fs Hz) it returns frequencies in Hz.
- **Reading amplitude.** A cosine of amplitude A at an interior mode k (0 < k < N/2) gives û_k = AN/2, a real number;
  a sine gives −iAN/2. So **amplitude = 2|û_k|/N**.
- **Reading phase.** The angle of û_k (`np.angle`) is the phase: 0° is a cosine, −90° a sine, anything in between
  a shifted wave (u = a cos + b sin gives û_k = (a − ib)N/2).
- **The end-bin rule.** Do not double the mean, û_0/N. For **even** N, the Nyquist bin k = N/2 is also not doubled
  (amplitude |û_k|/N), and a sine exactly at Nyquist vanishes at the samples, so its phase cannot be recovered. For odd
  N the last `rfft` bin is an ordinary bin and is doubled like the others.

## 4. Sampling: what the grid can and cannot see

- **Periodic grid, no duplicate endpoint.** Use x_j = j/N, j = 0, …, N−1 (`np.arange(N) / N`). The point x = 1 is
  x = 0 again; `np.linspace(0, 1, N)` includes it twice and quietly breaks periodicity.
- **Nyquist.** N samples can only distinguish modes |k| ≤ N/2. In time: frequencies up to fs/2.
- **Aliasing.** A mode above that limit is not lost; it is disguised. On the grid x_j = j/N,
  e^{2πi(N−1)j/N} = e^{−2πij/N}, so **mode N−1 looks exactly like mode −1**, and cos(2π(N−1)x) is indistinguishable
  from cos(2πx) at the samples. Resolve the highest mode present, or it folds onto a low one.
- **Bin spacing.** The frequencies the DFT reports are spaced **1/T** apart for a record of length T (the spacing of
  `rfftfreq`). More samples over the same T do not change that spacing. Bin spacing is not an accuracy guarantee:
  damping, noise, the window and neighboring modes all affect how well a peak locates a frequency.
- **Leakage.** The FFT assumes the signal repeats with the window's period. If it does not (it ends at a different
  value than it starts, or contains a wave that does not fit a whole number of times), the jump at the seam spreads
  energy into many modes. This matters in the course: the plate of Modules 3–6 is **not periodic**, which is why
  FNOs on non-periodic domains pad the input (see `fno_primer.md`, Weaknesses).

## 5. Five examples

All five run in about a second: `python module_7/examples/fft_basics_examples.py` (or `--example 1` … `5`; add
`--figdir figs` for one small PDF per example). Results: `measured_results.md` §4.

**Example 1 · Find the waves.** u(x) = 2 sin 2πx + 0.5 cos 10πx on N = 64 points.
`--example 1` → `rfft` gives 33 coefficients, nonzero only at k = 1 (û_1 = −64i: amplitude 2.0000, angle −90°,
a sine) and k = 5 (û_5 = 16: amplitude 0.5000, angle 0°, a cosine).
*Takeaway:* position = which wave, size = how much, angle = which kind.

**Example 2 · Natural frequency from a vibration record.** The spring–mass oscillator you know: m = 1 kg,
k = 400 N/m, so ω_n = 20 rad/s, f_n = 3.1831 Hz; damping ζ = 0.02 gives f_d = f_n√(1−ζ²) = 3.1825 Hz. A free
vibration displacement record x(t) = e^{−ζω_n t} cos(ω_d t), sampled at fs = 100 Hz for T = 20 s (N = 2000), the kind
of record a displacement sensor gives. `--example 2` subtracts the mean, takes the `rfft` and finds the largest bin
with `rfftfreq`: **3.200 Hz**, 0.0175 Hz from f_d. In this run the peak falls within half a bin of the known damped
frequency (bin spacing 1/T = **0.050 Hz**). Two cautions: the record starts at 1 and its envelope has decayed to
3.4e-4 by t = T, so it is *not* periodic across the seam (repeating it creates a jump of about 1); and a decaying
transient has a broadened spectrum, not a single sharp line. The peak still locates f_d well here, but the bin spacing
is not a guaranteed error bound.
*Takeaway:* the largest FFT bin of a vibration record points at its dominant frequency; how precisely depends on the
record (length, damping, noise, window), not on the bin spacing alone.

**Example 3 · Smoothing a noisy measurement.** A smooth profile sin 2πx + 0.3 sin 6πx plus Gaussian noise
(σ = 0.05, seed 0), N = 256. `--example 3` keeps modes k ≤ 10 (11 of 129), zeros the rest and `irfft`s: RMS error
against the clean signal falls from **0.0506** to **0.0155**.
*Takeaway:* noise spreads over all modes, the signal lives in a few low ones; cutting the high modes is a low-pass
filter. (It would also cut real fine detail; choose k_c with care.)

**Example 4 · Derivative with the FFT.** u = sin 2πx + 0.5 sin 6πx on N = 32. `--example 4` multiplies each
coefficient by iκ_k (κ_k = 2πk, Nyquist bin set to zero for a first derivative) and transforms back. Max error
against the exact u′ (whose largest value is 15.71): spectral **4.6e-14**, central finite difference **0.576**.
*Takeaway:* for this resolved two-mode function the spectral derivative agrees with the exact one to roundoff. For a
general smooth periodic function the accuracy depends on resolving its Fourier content: it converges very fast as N
grows, but is not exact at every N. This is how spectral PDE solvers compute derivatives.

**Example 5 · The convolution theorem.** A periodic 5-point moving average of the Example-3 signal, done directly
(sum of five shifted copies) and as `irfft(rfft(u) * rfft(kernel))`. `--example 5` → max difference
**6.7e-16** (roundoff).
*Takeaway:* convolution in x is multiplication in k. A 5-point kernel is cheap either way, but for a kernel as long
as the signal the direct route costs N² (65,536 at N = 256) against about N log₂ N per transform (2,048). The
*spectral branch* of an FNO layer is such a global kernel, applied as a multiplication (with a learned channel-mixing
matrix per retained mode); the full layer also has a pointwise path and an activation.

## 6. Where the course uses it

- **Lab 7's Burgers data** (`generate_burgers_data.py`): a spectral solver that takes u_x and u_xx with the FFT, as
  in Example 4.
- **The L32 heat example** (`heat_fourier_example.py`, `fno_primer.md` "one problem solved twice"): `rfft`, multiply
  mode k by e^{−νκ_k²t}, `irfft`.
- **The FNO's spectral layer** (`fno_burgers.py`, `fno_microstructure.py`): `rfft`, keep the lowest `modes`
  coefficients, multiply them by learned complex weights, `irfft`. That is Example 3's truncation and Example 5's
  convolution theorem, with the multiplier learned instead of chosen.

## 7. Further reading

- J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series,"
  *Mathematics of Computation* 19 (1965), 297–301. The paper that made the FFT standard.
- NumPy documentation, "Discrete Fourier Transform (numpy.fft)": the conventions, normalization options and the
  `rfft`/`fftfreq` helpers used above.
