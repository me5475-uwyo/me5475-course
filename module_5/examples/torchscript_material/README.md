# `ComputeTorchScriptStress`: the course's neural-network stress material

This is the C++ source of the material Lab 5 runs. It is compiled into the course MOOSE executable on ARCC; **you do
not build or edit it for Lab 5.** It is here so you can read what happens to your network at every quadrature point.
L25 compares it, side by side, with MOOSE's own `ComputeLinearElasticStress`.

| File | What it is |
|---|---|
| `ComputeTorchScriptStress.h` | the class declaration and its documentation comment |
| `ComputeTorchScriptStress.C` | `validParams` (one parameter, `filename`), the constructor (loads the TorchScript file in float64), and `computeQpStress` |

**What `computeQpStress` does at each quadrature point:**
1. It builds x = (ε_xx, ε_yy, γ_xy) from MOOSE's mechanical strain, with γ_xy = 2 ε_xy, as a (1, 3) float64 tensor.
2. It calls the network. It stops the run unless the network returns 3 finite values, then sets σ_xx, σ_yy and σ_xy, with σ_zz = 0.
3. It computes the tangent ∂σ/∂ε by automatic differentiation, with one backward pass per stress component.
4. It maps that tangent into MOOSE's rank-four `_Jacobian_mult`.

It stores nothing between steps, so it can deploy a stateless network such as Lab 1's MLP, but not Lab 4's LSTM.

**To compare with MOOSE:** MOOSE's `ComputeLinearElasticStress.C` is in `modules/solid_mechanics/src/materials/` of
the MOOSE source (github.com/idaholab/moose). Both classes derive from `ComputeStressBase` and set the same three
things per point: `_stress`, `_Jacobian_mult` and `_elastic_strain`.

**Background reading:** `module_5/readings/moose_material_interface_reference.md`.
