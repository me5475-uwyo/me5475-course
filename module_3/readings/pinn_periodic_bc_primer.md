# Periodic boundary conditions in a PINN: a feature transform layer

*Course reading · ME 5475 · Module 3 · **[Optional]**, after L15 (hard boundary conditions).*

**After reading this, you should be able to:**
- explain why replacing a coordinate input by Fourier features makes a network field periodic in that coordinate, for
  every θ;
- state what that guarantees for strains, and what *more* stresses and tractions need;
- say what periodicity alone leaves undetermined.

---

## Why periodic conditions come up

A composite with a repeating microstructure is often modelled with one **periodic cell** (a representative volume
element, RVE) of side Q. The **total** displacement splits into a macroscopic part and a **fluctuation**:

u(x) = ε̄·x + ũ(x)

- **The fluctuation ũ is periodic:** ũ(x + Q eⱼ) = ũ(x) on opposite faces.
- **The total displacement is not.** It jumps by an affine amount: u(x + Q eⱼ) − u(x) = ε̄·(Q eⱼ).

Periodicity ties *pairs* of points on opposite faces together, so it is neither a Dirichlet nor a Neumann condition.
Equilibrium between neighbouring cells also needs the tractions on opposite faces to be equal and opposite.

## Soft or hard? The L15 question again

**Soft:** sample pairs of points on opposite faces and add penalty terms for what the problem requires. That is the
fluctuation jump ũ(x + Q eⱼ) − ũ(x), and the traction mismatch t(x + Q eⱼ) + t(x). As in L15, each pair is a
*request*. No finite weight guarantees it holds, and the terms trade against the PDE residuals.

**Hard:** build the periodicity of ũ into the network, so it holds for every θ, before training.

## The construction: a feature transform layer

A function of cos(2πx/Q) and sin(2πx/Q) is periodic in x with period Q, whatever that function is. So replace each
coordinate input xⱼ by Fourier features:

xⱼ  →  [cos(2πxⱼ/Q), sin(2πxⱼ/Q), cos(4πxⱼ/Q), sin(4πxⱼ/Q), …]

Call the network's output field v. It is used as the fluctuation: ũ = v, up to the translation fix below.

**What this guarantees:**
- **ũ(x + Q eⱼ) = ũ(x) for every θ.** No training and no penalty are needed for the fluctuation's periodicity.
- **Its derivatives are periodic too,** for a smooth activation such as tanh or Swish. The output is then a smooth
  function of smooth periodic features, so every spatial derivative inherits the period. In particular the strain
  sym ∇ũ is periodic.

**What stresses and tractions additionally need.** The stress is

σ = C(x) : [ε̄ + sym ∇ũ − ε∗(x)], with ε∗ any eigenstrain.

It is periodic only if the **constitutive data are periodic**:
- C(x) and ε∗(x) take matching values at paired points on opposite faces;
- any non-spatial inputs (load mode, moduli) are held the same.

Then σ matches on opposite faces, and because the normals are opposite, the tractions are equal and opposite. A
periodic strain times a modulus that differs between the paired faces would not give matching stresses.

**This is a boundary compatibility identity, not a solution.** Interior equilibrium, and traction continuity at
material interfaces inside the cell, still come from the PDE residuals and the interface treatment.

**How many terms:**
- Lu et al. (2021) argue that the first pair, cos and sin of 2πx/Q, is enough *in principle*. The higher harmonics
  are nonlinear functions of the first (for example, cos 2θ = 2cos²θ − 1), which the network can produce itself.
  That is not a promise that a finite trained network reproduces them exactly.
- Min Lin's thesis uses two pairs (four terms) per coordinate, and reports that this worked better than one pair in
  her tests.

![A network diagram. On the left, the inputs x₁ and x₂ each pass through a box labelled ‘Feature Transform Layer’ holding cos(2πxⱼ), sin(2πxⱼ), cos(4πxⱼ) and sin(4πxⱼ). These eight features feed fully connected hidden layers whose two outputs, labelled u₁ and u₂, are the periodic network field (ũ in this primer). On the right, a ‘PDE’ region shows the operators ∂/∂x₁ and ∂/∂x₂, the stresses σ₁₁, σ₂₂, σ₁₂, and two residual boxes, f₁ = ∂σ₁₁/∂x₁ + ∂σ₁₂/∂x₂ and f₂ = ∂σ₁₂/∂x₁ + ∂σ₂₂/∂x₂, which feed the ‘PDE Loss’ and then the ‘Loss’. The derivative part is schematic: it does not draw the cross derivatives or the constitutive law that produce the stresses.](figures/lin2023_pinn_periodic_feature_layer_contrast.png)

*The feature transform layer, for a unit cell (Q = 1). Reproduced from M. Lin, PhD dissertation, University of
Wyoming, 2023, Chapter 5 (the PINN-with-periodic-boundary-conditions network diagram). Colors adjusted for
accessibility, and the background made white. The scientific content is unchanged.*

**Reading the figure:**
- **Notation.** The figure's u₁ and u₂ are the periodic network field: in our decomposition that is ũ, *not* the
  total displacement.
- **The right-hand side is schematic.** It draws ∂/∂x₁ after u₁ and ∂/∂x₂ after u₂ only. The stresses actually come
  from the **full symmetric gradient** and the **constitutive law**:
  - ε₁₁ = ε̄₁₁ + ∂₁ũ₁;
  - ε₂₂ = ε̄₂₂ + ∂₂ũ₂;
  - ε₁₂ = ε̄₁₂ + (∂₂ũ₁ + ∂₁ũ₂)/2;
  - then σ = C : (ε − ε∗).

  The cross derivatives and the constitutive step are not all drawn.
- **The two equilibrium residuals,** in text: f₁ = ∂σ₁₁/∂x₁ + ∂σ₁₂/∂x₂ and f₂ = ∂σ₁₂/∂x₁ + ∂σ₂₂/∂x₂.
- **Outputs.** This network outputs displacements only. Lab 3's mixed network also outputs the three stresses.

## Compare with L15, and the DeepXDE hook

L15 built *Dirichlet* conditions into the *output*: u = g + B·N, installed with `net.apply_output_transform`.
Periodicity is built into the *input*: the network sees only features that are already periodic. DeepXDE has the
matching hook, which stores a function applied to the inputs before the first layer:

```python
import deepxde as dde
import numpy as np
import torch

Q = 1.0  # cell size

def periodic_features(x):            # x: (N, 2) points; periodic in x1 and x2
    feats = []
    for j in range(2):
        for k in (1, 2):             # two pairs per coordinate, as in Min Lin's thesis
            feats += [torch.cos(2 * np.pi * k * x[:, j:j+1] / Q),
                      torch.sin(2 * np.pi * k * x[:, j:j+1] / Q)]
    return torch.cat(feats, dim=1)   # (N, 8) features instead of (N, 2) coordinates

net = dde.maps.FNN([8] + [50] * 5 + [2], "tanh", "Glorot uniform")  # input width = number of features
net.apply_feature_transform(periodic_features)
```

*An illustration of the API, **not run in this course**.* The network's first layer must take as many inputs as there
are features.

**Combining hooks needs care.** An output transform receives the *original* coordinates, so it can destroy the
periodicity that the features built in. L15's factor x(1 − x) applied to a periodic field is an example. Take
q(x) = x(1 − x)·cos(2πx):
- its end values match, q(0) = q(1) = 0;
- its derivatives do not: q′(0) = 1 but q′(1) = −1.

Any output transform you add must itself preserve periodicity, and its derivatives' periodicity.

## What periodicity does not fix, and how the thesis handles it

1. **The macroscopic loading.** A periodic field can represent only ũ; ε̄·x is not periodic. Two equivalent ways to
   write the purely mechanical problem:
   - **Directly:** σ = C(x) : [ε̄ + sym ∇ũ].
   - **As an eigenstrain:** σ = C(x) : [sym ∇ũ − ε_T], which is the same problem only if ε_T = −ε̄. Mind the sign.

   Min Lin's chapter loads its influence-function problems through a **thermal/eigenstrain** term in the
   constitutive law, with a mode-dependent and spatially varying field. When you compare that with a macrostrain ε̄,
   check the sign convention.
2. **A rigid translation.** Adding a constant to ũ keeps it periodic, so ũ is unique only up to a translation.
   - The chapter describes zero corner displacements for 2-D EHM (§5.3.2, in the energy-method discussion). For a
     periodic fluctuation the corners are equivalent points, so fixing one corner value fixes the translation.
   - In a periodic network, one clean way to express this is a gauge shift, ũ(x) = v(x) − v(x₀). It subtracts a
     constant, so every derivative and the periodicity are kept. (This is a mathematical statement of the fix, not a
     recipe for evaluating the network inside its own output hook.)
3. **Many load cases and materials in one network.** The chapter adds inputs for the load mode, the thermal/eigenstrain
   loading and the two phases' Young's moduli. One trained network then serves many influence-function calculations,
   within the ranges it was trained on. That is L16's parametric idea (E and ν as inputs) in a multiscale setting.

## What to read

**M. Lin, *Multiscale Modeling of Nonlinear Composite Materials using Interface-Enriched Generalized Finite Element
Method and Eigendeformation-based Reduced Order Homogenization Models*, PhD dissertation, Mechanical Engineering,
University of Wyoming, August 2023.** Read Chapter 5, *Physics-Informed Neural Networks for Enhanced Reduced Order
Modeling of Composites*:
- **§5.2:** the governing equations, and the energy (variational) form used by the Deep Energy Method.
- **§5.3.2:** collocation PINNs (our strong form) vs energy-based PINNs. Note how each handles Dirichlet and Neumann
  conditions.
- **§5.3.3, "PINN enhanced EHM": the core of this reading.** Periodic features, the thermal/eigenstrain loading, and
  the extra inputs (the three network diagrams captioned "… with periodic boundary conditions …").
- **§5.4:** skim it, for how the PINN-computed influence functions are checked against an IGFEM-based reduced-order
  model.

**Lu et al. 2021,** *Physics-informed neural networks with hard constraints for inverse design*, SIAM J. Sci.
Comput. 43(6), B1105–B1132, doi:10.1137/21M1397908. You already have it as L15's primary reading (read before L16).
Its periodic construction (its §2.3) is the one the thesis uses.

**Notation differences from the course:**
- **Activation:** the thesis uses Swish, where L14–L15 use tanh. Both are smooth.
- **Loss:** the chapter's displayed collocation loss writes the residuals without squares. Compute each term as the
  mean of the squares of the individual residual components, as DeepXDE does, not as the square of a summed residual.
- **Outputs:** its collocation PINN is described with displacement outputs (§5.3.2); Lab 3's is mixed (u, σ).

## Questions to think about

1. Why does every spatial derivative of ũ inherit the period when only the inputs were transformed? What must the
   activation function provide?
2. The periodicity of ũ holds for every θ. What must also hold for the stresses, and so the tractions, on opposite
   faces to match? What still has to be learned, and which loss terms remain?
3. Why can't ε̄·x be part of the periodic network output? If you apply the loading as an eigenstrain instead, what
   sign must it have?
4. You want to fix the translation on top of the feature transform. Check two candidates:
   - Does ũ(x) = x₁(1 − x₁)·v(x) keep ũ, and its derivatives, periodic in x₁?
   - Does ũ(x) = v(x) − v(x₀)?
5. In L15, hard BCs were not faster and not uniformly more accurate (slide 13). What would you measure to compare
   soft and hard periodicity on a periodic cell?
