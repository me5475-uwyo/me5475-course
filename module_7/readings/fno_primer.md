# Fourier Neural Operator Primer

A two-page reference for L32 and Lab 7.

---

## Before the FNO: one problem solved twice (L32, slides 7–10)

The periodic heat equation u_t = ν u_xx on [0, 1), with ν = 0.01 (Burgers without its u u_x term) and
u(x, 0) = sin 2πx + ½ sin 6πx.

- **Mode index and wavenumber.** Mode k (an integer) has k oscillations across the domain; its wavenumber is
  κ_k = 2πk / L (here L = 1). A second derivative multiplies mode k by −κ_k². Mode 0 is the mean.
- **By hand.** Each mode decays on its own: a_k' = −ν κ_k² a_k, so
  u(x, t) = e^(−4π²νt) sin 2πx + ½ e^(−36π²νt) sin 6πx. Mode 3 decays 9× faster than mode 1.
- **With the FFT.** Sample at x_j = j/N (no duplicate endpoint), `rfft`, multiply coefficient k by e^(−νκ_k²t),
  `irfft`. NumPy's `rfft` returns N//2 + 1 complex numbers; a sine of amplitude A at an interior mode k (0 < k < N/2)
  appears as −iAN/2, so 2|û_k|/N is its amplitude. The end bins are not doubled: the mean is û_0/N, and for even N the
  Nyquist bin k = N/2 has amplitude |û_k|/N. The result matches the hand solution to roundoff, because u(x, 0) contains only modes
  1 and 3 (`module_7/examples/heat_fourier_example.py`; numbers in `measured_results.md` §3).
- **What the FNO changes.** The heat equation's multiplier is a known scalar per mode. An FNO layer learns a complex
  matrix R_k per retained mode that mixes latent feature channels (ẑ_{o,k} = Σ_i R_{io,k} v̂_{i,k}), and adds a
  pointwise path and an activation, because a nonlinear operator such as Burgers couples the modes. Its four blocks
  are network depth, not four time steps.

---

## The big idea

Replace a learned spatial convolution (its kernel weights are tied to grid cells) with a learned **multiplication in Fourier space** (its weights are tied to Fourier modes, so the same layer can be applied on a grid of any size N).

For a function v(x) on a uniform grid:

```
FFT
v(x)  -----------> v_hat(k)              spatial -> spectral
                    |
                    | multiply by learned R(k)
                    v
                  out_hat(k)
                    |
                    | IFFT
                    v
                  out(x)                  spectral -> spatial
```

The R(k) values are learned. They act as a "multiplicative filter" in Fourier space.

**Why it is designed to run on any grid.** The FFT is defined at any grid size N, and the learned R(k) are indexed by Fourier mode k = 0, 1, ..., modes − 1, which is the same frequency on any grid. So a model trained at N = 128 can be *applied* at another N without changing a weight, as long as the grid has enough modes (N//2 + 1 ≥ modes). Whether its *accuracy* carries over to other N is a separate question; Lab 7 Task 4 measures it.

---

## The FNO architecture

A full FNO consists of:

```
Input v(x) on N grid points
   |
   v
[Lift]                          point-wise linear: R^in -> R^channels
   |
   v
[Repeated n_layers times:
   (Spectral Conv + Spatial Linear) + Activation]
   |
   v
[Project]                       point-wise linear: R^channels -> R^out
   |
   v
Output s(x) on N grid points
```

Each spectral-conv layer:

```
v(x) in (B, C, N)
   |
   FFT
   v
v_hat in (B, C, N//2 + 1)  -- complex values
   |
   keep first `modes` modes
   |
   matrix-multiply by R (complex, learned)
   |
   pad rest with zeros
   |
   IFFT
   v
out(x) in (B, C, N)
```

---

## PyTorch implementation (1-D)

```python
import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        scale = 1.0 / (in_c * out_c)
        self.weights = nn.Parameter(
            scale * torch.randn(in_c, out_c, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, in_c, N)
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                      # (B, in_c, N//2 + 1)
        # Truncate: an N-point grid has only N//2 + 1 modes, so use at most that many
        out_ft = torch.zeros(B, self.out_c, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = torch.einsum(
            "bik,iok->bok", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, modes=16, width=64, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(in_dim + 1, width)  # +1 for position
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.spatial = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, N) -- input function on N grid points
        B, N = x.shape
        # grid positions x_j = j / N on [0, 1), as in the data's x_grid
        pos = (torch.arange(N, device=x.device, dtype=x.dtype) / N).unsqueeze(0).expand(B, -1)
        x = torch.stack([x, pos], dim=-1)
        x = self.lift(x).permute(0, 2, 1)
        for sp, sl in zip(self.spectral, self.spatial):
            x = self.activation(sp(x) + sl(x))
        x = x.permute(0, 2, 1)
        x = self.activation(self.project1(x))
        return self.project2(x).squeeze(-1)
```

This is the model in `module_7/examples/fno_burgers.py`: course defaults modes = 16, width = 64, n_layers = 4, projection 64→128→1, GELU; 549,569 parameters counted as real numbers (287,425 tensor entries, since each complex spectral weight is one entry but two real numbers; the script prints both). The position channel uses `torch.arange(N) / N`, the same points x_j = j/N as the data; `torch.linspace(0, 1, N)` would include x = 1, which is the point x = 0 again on a periodic domain. The `min(self.modes, ...)` guard keeps the layer from indexing modes a coarse grid does not have; on such a grid the layer silently uses fewer modes.

For 2-D FNO, replace `rfft` with `rfft2` and the spectral conv becomes a 2-D Fourier truncation. See `module_7/examples/fno_microstructure.py`.

---

## Key hyperparameters

| HP | Typical | What it controls |
|----|---------|------------------|
| modes | 8–32 at N = 128 (course: 16) | how many low Fourier modes each spectral layer acts on; keep modes ≤ N//2 + 1 at the coarsest grid you evaluate |
| width | 32–128 (course: 64) | channel count throughout the network |
| n_layers | 4–8 (course: 4) | depth of spectral processing |
| learning rate | 1e-3 to 1e-4 (course: 1e-3) | as usual |

**Modes is the key FNO-specific choice.** Too few: each layer's global (spectral) mixing acts only on the lowest frequencies; the pointwise path can still pass local, high-frequency information, and the nonlinearities can create higher modes, but accurate fine detail has to be learned. Too many: parameter count grows (width² × modes complex weights per layer); spurious frequencies may appear.

For 1-D at N = 128: modes 8–32 (course: 16); keep modes ≤ N//2 + 1 at the coarsest grid you evaluate (33 at N = 64).
For 2-D with N=64x64: modes=8x8 to modes=16x16 typical (the course's 2-D example uses 12x12).

---

## Strengths

- **Designed to run on any grid.** The weights are indexed by Fourier mode, so a model trained at one N can be applied at another N (it needs N//2 + 1 ≥ modes). Whether accuracy carries over is what Lab 7 Task 4 measures.
- **Computationally efficient.** FFT is O(N log N). Spectral truncation is cheap.
- **Native handling of periodicity.** Burgers' equation on a periodic domain: FNO is natural.

---

## Weaknesses

- **Implicit periodicity assumption.** The FFT treats the input as periodic. For non-periodic domains, a generic remedy is to pad (e.g., zero-pad) before the Fourier layers and crop afterward; `fno_microstructure.py` does this. Irregular geometries are a separate problem: Geo-FNO (Li, Huang, Liu & Anandkumar, JMLR 24, 2023) learns a deformation that maps the physical domain to a regular latent grid, where the FFT is applied.
- **No exact symmetry guarantees.** The spectral and pointwise layers commute with periodic shifts, but the position channel x_j = j/N tells the network where each point is, so it breaks exact translation equivariance: a shifted input need not give a shifted output. In 2-D there is no built-in rotation equivariance either: a rotated input need not give a rotated prediction.
- **Grid-aligned.** For irregular meshes, see M8 (GNN).
- **Memory grows with grid size and width.** Activations are stored at every grid point for every channel and layer (about B × width × N^d values per layer, kept for backpropagation), and each spectral layer holds about width² × modes^d complex weights (a 3-D FNO with modes=12 has 12^3 = 1728 modes per channel pair). Large 3-D grids can be GPU-heavy.

---

## When to use FNO over DeepONet

- You want to apply one model on grids of different resolution: FNO is designed for this, and Lab 7 Task 4 tests whether its accuracy carries over on Burgers.
- Input function is naturally on a uniform grid.
- Periodic or near-periodic problems.
- 2-D or 3-D problems on rectangular domains.

---

## When NOT to use FNO

- Irregular geometries → GNN (M8), or Geo-FNO.
- Sparse query points (you only want s at a few y values, not the whole field) → DeepONet.
- Strong non-periodic boundaries → padding (see Weaknesses), DeepONet, or PI-FNO with boundary handling.
- Very large 3-D grids (e.g., 256³ and up), where activations (grid size × width) and the modes³ spectral weights strain GPU memory.

---

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Predictions show ringing artifacts | Spectral truncation too aggressive | Increase modes |
| Strange artifacts at edges of non-periodic domain | FFT periodicity violated | Pad (zero or reflective) before the Fourier layers and crop afterward |
| Memory error during training | Batch, grid, width or modes^d too large | Reduce batch size; then width or modes |
| Loss decreases but predictions look wrong | Output normalization missing | Standardize outputs per-channel before training |

---

## A note on `torch.fft`

PyTorch's `torch.fft.rfft` returns the *positive-frequency* half of the FFT (since the input is real and the negative-frequency half is the conjugate of the positive). Output shape is `N // 2 + 1`.

For 2-D: `torch.fft.rfft2` returns `(H, W // 2 + 1)`.

The "modes" parameter in FNO refers to how many of these complex coefficients are kept (multiplied by R; the rest are zeroed). Course default: modes = 16 at N = 128 (typical range 8–32); keep modes ≤ N//2 + 1 at the coarsest grid you evaluate.
