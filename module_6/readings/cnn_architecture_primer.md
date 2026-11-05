# CNN Architecture Primer

A two-page reference for the conv/pool/receptive-field concepts.

---

## The 2-D convolutional layer

**Operation.** Slide a learnable kernel K over an input image I:

```
Output[c_out, i, j] = sum over c_in, m, n of Kernel[c_out, c_in, m, n] * Input[c_in, i+m, j+n] + bias[c_out]
```

For an image with C_in channels and a kernel that produces C_out channels, the parameter count is `C_out * C_in * kH * kW + C_out`.

**Standard kernel sizes:** 3×3 (workhorse), 1×1 (channel mixing only), 5×5, 7×7. Rarely larger.

**Output size formula:**

```
output_size = floor( (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride ) + 1
```

With dilation 1 (the default) this is `floor((input_size - kernel_size + 2 * padding) / stride) + 1`. For an **odd**
kernel with stride 1, `padding = kernel_size // 2` keeps the size (`output_size = input_size`); PyTorch's
`padding="same"` does this for you and is allowed only with stride 1.
([PyTorch `Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).)

---

## Pooling

**Max pooling.** Take the maximum in each k×k window. Reduces spatial size.

```
input 64x64 -> MaxPool2x2 -> output 32x32
```

**Average pooling.** Mean instead of max.

**Global pooling.** Pool over the entire spatial extent. `nn.AdaptiveAvgPool2d(1)` produces a `(C, 1, 1)` tensor from any input size — useful for image-to-scalar regression.

**When to use which.** Max pooling is common inside classification networks (it keeps "is this feature present?").
Global average pooling is the common choice before a regression head (it keeps an average magnitude); Lab 6's network
uses max pooling inside its blocks and global average pooling at the end. Neither is better in general: compare them on
your validation set.

---

## Receptive field

The receptive field of a network output is the set of input pixels that influence it. Computed by stacking layers:

Track two numbers through the layers: the receptive field `r` (in input pixels) and the **jump** `j`, the distance
in input pixels between neighboring outputs of a layer. Start with `r_0 = 1`, `j_0 = 1`. A layer with kernel `k`,
stride `s` and dilation `d` (pooling counts as a layer, with its window as `k`) gives

```
r_l = r_(l-1) + (k_l - 1) * d_l * j_(l-1)
j_l = s_l * j_(l-1)
```

Example, Lab 6's blocks (3x3 conv, stride 1, then 2x2 max pool, stride 2): after one block r = 4, j = 2; after two,
r = 10, j = 4; after three, r = 22, j = 8; after four, r = 46, j = 16. Without pooling, stride-1 3x3 convolutions
add only 2 pixels each, so pooling and strides are what make the receptive field large.
([Araujo, Norris & Sim, *Computing receptive fields of convolutional neural networks*, Distill 2019](https://distill.pub/2019/computing-receptive-fields/).)

**For mechanics:**

- Microstructure-to-modulus: the modulus is a property of the whole cell, so the prediction should depend on the whole
  image. Lab 6 gets this from global average pooling at the end.
- Stress at a point: geometry near the point matters most, but equilibrium and the loading make the stress depend on
  the whole cell too. A stress-field network needs local detail and global context, which is the U-Net's design (L30).

---

## Padding and stride

**Padding.** Adds zero rows/columns to image edges so the kernel can be centered at the edge.

- `padding = 0` (`"valid"`): output smaller than input.
- `padding = "same"`: output the same size as the input; stride 1 only. For an odd kernel this equals
  `padding = kernel // 2`.
- Larger outputs come from transposed convolutions (`nn.ConvTranspose2d`) or upsampling, not from a `Conv2d` padding
  mode; PyTorch's `Conv2d` accepts `"valid"`, `"same"` or numbers.

**Stride.** How many pixels the kernel moves per step.

- stride 1: dense; default.
- stride 2: skip every other pixel; downsamples by 2. Cheaper than pool+conv but loses some information.
- stride > 2: rare; usually used in the first layer of a deep network for fast downsampling.

---

## Batch normalization

**The pattern.** Apply normalization after every conv:

```python
nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, padding=1),
    nn.BatchNorm2d(out_c),                # normalize over batch
    nn.ReLU(inplace=True),
)
```

**What it does.** In training, normalizes each channel to zero mean and unit variance using statistics over the
batch **and** the spatial positions (N, H, W), then applies a learnable scale and shift per channel. It also keeps
running averages of those statistics, which it uses instead in evaluation (`model.eval()`), so the two modes behave
differently. ([PyTorch `BatchNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).)

**Why it helps.** Stabilizes training; reduces sensitivity to initialization; acts as a mild regularizer. Default in almost every modern CNN.

**The gotcha.** BatchNorm is sensitive to batch size — with very small batches (<8), it can be unstable. For Lab 6 with batch size 32+, no issue.

---

## Common CNN architectures (by name)

**LeNet (1998).** 7 layers. Used for MNIST. ~60k parameters. Historical.

**AlexNet (2012).** 8 layers, 60M parameters. ImageNet 2012 winner. Ushered in deep learning.

**VGG (2014).** Deep stacks of 3x3 convs. Simple uniform architecture. 138M parameters for VGG-16.

**Inception / GoogLeNet (2014).** Multi-scale convolutional modules. Computational efficiency.

**ResNet (2015).** Skip connections, very deep. 152-layer networks. The current default backbone.

**DenseNet (2017).** Connectivity between every layer, not just adjacent. Memory-hungry but excellent.

**EfficientNet (2019).** Systematic scaling of depth, width, resolution. SOTA on ImageNet.

For Lab 6: a 4-block custom architecture (97,761 parameters at the defaults; `module_6/readings/measured_results.md` §2)
is appropriate. Residual (ResNet-style) skips are not in the Lab 6 sweep; adding them would be your own extension.

---

## The image-to-scalar template

For a CNN that takes an image and outputs a scalar (e.g., effective modulus):

```python
nn.Sequential(
    # Spatial downsampling via stacked conv+pool
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
    # Collapse spatial to a single value per channel
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),                # (B, 128, 1, 1) -> (B, 128)
    nn.Linear(128, 1),           # final scalar
)
```

Input: (B, 1, 64, 64). Output: (B, 1). Parameter count: 97,761 at the defaults (§2).

---

## The image-to-image template (U-Net)

See `module_6/readings/unet_for_mechanics.md` and `module_6/examples/unet_microstructure_to_stress.py` for the full implementation. The pattern:

```
encoder: conv -> conv -> pool   (repeat, halving spatial size)
bottleneck: conv -> conv at the smallest size
decoder: upsample -> concat with encoder skip -> conv -> conv   (repeat, doubling spatial size)
```

Output: same spatial size as input, with `out_channels` features per pixel.

---

## When CNNs fail in mechanics

Three failure modes worth knowing:

1. **Out-of-distribution topologies.** Train on inclusion-based RVEs; test on stripes. The network still returns a
   number, and nothing guarantees it is close; how far off it is has to be measured (Task 4).
2. **Resolution.** Lab 6's network ends in global average pooling, so it accepts a 128×128 image and returns a
   number, but it was trained on features at the 64×64 scale; whether that number is accurate has to be tested.
   Operator learning (M7) is designed for transfer across discretizations, which is a motivation, not a guarantee of
   accuracy ([Li et al., *Fourier Neural Operator*, 2020](https://arxiv.org/abs/2010.08895)).
3. **Rotation.** Convolutions are translation-equivariant, not rotation-equivariant. Rotating the microstructure
   while keeping the load along x can legitimately change E_app, so a rotated image with the same label is not a valid
   augmentation here: the load and any tensor labels would have to rotate too.

For Lab 6: limit testing to the training topology + same resolution. The OOD task (Task 4) explores the topology issue deliberately.
