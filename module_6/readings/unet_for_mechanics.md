# U-Net for Mechanics — A Practitioner's Reference

The U-Net architecture (Ronneberger 2015) is the standard image-to-image deep learning model in mechanics. This reference covers the architecture, the skip-connection rationale, and the practical patterns for solid mechanics field prediction.

---

## The architecture in 60 seconds

```
Input image (B, 1, 64, 64)
       |
       v
[Encoder]
    enc1: Conv-Conv         (B, 32, 64, 64) -----------------------+
    Pool                     -> (B, 32, 32, 32)                     |
    enc2: Conv-Conv          (B, 64, 32, 32) ----------------+      |
    Pool                     -> (B, 64, 16, 16)               |      |
    enc3: Conv-Conv          (B, 128, 16, 16) --------+       |      |
    Pool                     -> (B, 128, 8, 8)         |       |      |
[Bottleneck]
    bottleneck: Conv-Conv    (B, 256, 8, 8)            |       |      |
[Decoder]
    Upsample                 -> (B, 128, 16, 16)        |       |      |
    Concat with enc3         -> (B, 256, 16, 16)  <-----+       |      |
    dec3: Conv-Conv          (B, 128, 16, 16)                   |      |
    Upsample                 -> (B, 64, 32, 32)                 |      |
    Concat with enc2         -> (B, 128, 32, 32)  <-------------+      |
    dec2: Conv-Conv          (B, 64, 32, 32)                           |
    Upsample                 -> (B, 32, 64, 64)                        |
    Concat with enc1         -> (B, 64, 64, 64)  <---------------------+
    dec1: Conv-Conv          (B, 32, 64, 64)
       |
       v
[Output]
    1x1 Conv                 -> (B, 3, 64, 64)
```

Parameter count: 1,927,907 with base channels = 32 and 3 levels (`module_6/readings/measured_results.md` §2). Training
time depends on the dataset size, epochs and node; the ledger records our one 100-RVE test run.

---

## Why skip connections matter

Without skip connections, the decoder must reconstruct spatial detail from the bottleneck (~8×8 features). Reconstruction from such a small feature map is hard — the decoder essentially has to *generate* detail from very limited information.

Skip connections route encoder features *directly* to decoder layers at the same resolution. At each decoder level the model sees:

- Global context (via the bottleneck path)
- Local detail (via the skip)

It can combine both: "this region is in the stress-concentration zone (global) AND this particular pixel is exactly here (local)."

For mechanics, this matters because stress concentration around an inclusion boundary is *very local* but the overall stress distribution is *global*. U-Net handles both.

---

## The PyTorch idiom

```python
import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base=32):
        super().__init__()
        self.enc1 = double_conv(in_channels, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.bottleneck = double_conv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)
        self.out_conv = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
```

The complete reference is in `module_6/examples/unet_microstructure_to_stress.py`.

---

## Upsampling: ConvTranspose vs Bilinear+Conv

Two ways to double spatial size:

**ConvTranspose2d.** Learnable upsampling. With the same channel counts, kernel size and groups it has as many weights
as a regular convolution. It can produce "checkerboard" artifacts (alternating bright/dark pixels) if the kernel size
and stride are not chosen with care.

```python
self.up3 = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
```

**Bilinear + Conv.** Fixed bilinear upsample (no parameters) + a regular conv. No checkerboarding; slightly less expressive.

```python
nn.Sequential(
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
    nn.Conv2d(in_c, out_c, 1),
)
```

For mechanics with continuous fields, bilinear+conv is often preferred because the checkerboard artifacts of ConvTranspose are particularly visible in smooth stress fields. For Lab 6 either works.

---

## When to use U-Net for mechanics

**Use it for:**

- Microstructure → full stress / strain field.
- Microstructure evolution under load (image-to-image-over-time, in a sequence).
- Image segmentation of mechanical phenomena (crack detection from images, phase identification).
- Topology optimization: input is the loading + constraints as images; output is the optimal density distribution.

**Don't use it for:**

- Scalar outputs (use regression CNN with global pooling).
- Mesh-based simulations on irregular geometries (use GNN in M8).
- Family of solutions over a continuous parameter (use operator learning in M7).
- Very large images (>256×256) — memory grows quadratically; consider patch-based or progressive approaches.

---

## Limitations of U-Net (worth knowing for the final project)

**Resolution.** A fully convolutional U-Net accepts other image sizes (multiples of its downsampling factor), but it
learned features at the training resolution, so accuracy at another resolution has to be tested. Options: train at
several resolutions, or use a model designed for discretization transfer (FNO, M7), which is a motivation for that
design, not a guarantee of accuracy ([Li et al., 2020](https://arxiv.org/abs/2010.08895)).

**Translation, not rotation.** A U-Net is roughly translation-**equivariant** (shift the input, the output field
shifts with it), up to the effects of padding at the edges and of downsampling. It is not rotation-equivariant. And in
mechanics, rotating the geometry while keeping the load along x is a different problem with a different answer: an
augmentation must rotate the load and the stress components consistently, or use equivariant networks (Cohen and
Welling style).

**Boundary conditions encoded only implicitly.** If the BCs change (different loading direction, different geometry boundaries), the U-Net trained on one BC pattern won't generalize. Fix: include BC as additional input channels.

**Conservation-law violations.** U-Net's output isn't constrained to satisfy the equilibrium equation div(σ) = 0. How
close it comes has to be measured. PINN-style residual losses (M3) penalize the violation but do not remove it; only
constraints built into the architecture or the output parametrization enforce it exactly.

---

## Comparison to other field-prediction methods

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| U-Net | Image input/output; well-understood; fast inference | Mesh-dependent; no physics constraints |
| FNO (M7) | Designed for transfer across discretizations; works in Fourier space | Accuracy across resolutions still has to be tested |
| DeepONet (M7) | Function-space input; not tied to grid | Architecture more complex; slower per-call |
| GNN (M8) | Irregular meshes; natural for mesh-based mechanics | Slower training; more bookkeeping |
| PINN (M3) | Physics in the loss; mesh-free | Constraints only penalized (exact only if hard-coded); slow to train; struggles with sharp features |

For Lab 6 we use U-Net because the inputs are image-shaped microstructures. For an FE-mesh deployment in production we'd use a GNN.
