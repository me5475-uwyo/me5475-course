# Starter Prompts for Lab 1

You may adapt these prompts freely. Logging them in your `prompts_*.md` files is required. As in M0, **lead and review run in separate sessions with no shared context.**

These are starting points — feel free to make them more specific to your situation.

---

## Tasks 2–3 setup: verify the MOOSE input file before running 200 times

It is much cheaper to find a bug in `single_element_loadsweep.i` *before* generating 200 samples than after.

**Lead prompt (theory-leaning):**

```
Read the MOOSE input file single_element_loadsweep.i. Verify the following
by inspection (no need to run MOOSE):

1. The mesh is exactly ONE element (a single QUAD4 on a unit square).
2. The four-edge displacement BCs are consistent with a UNIFORM strain field
   on that element. Specifically:
   - u_x(x=0) = 0;  u_x(x=L) = eps_xx_val * L;   so eps_xx = eps_xx_val.
   - u_y(y=0) = 0;  u_y(y=L) = eps_yy_val * L;   so eps_yy = eps_yy_val.
   - The shear BC should make eps_xy = gamma_xy_val / 2, i.e. gamma_xy in
     Voigt convention equals gamma_xy_val.
3. The material model is linear isotropic plane-strain elasticity with
   E = 1, nu = 0.3.
4. The postprocessors report element-averaged stress AND strain components.

Reply with one paragraph per item: explain what part of the input file
implements that requirement and whether it does so correctly.
```

**Review prompt (independent session):**

```
Read the analysis below. Verify each claim against the actual file
single_element_loadsweep.i:

1. Mesh is one element: count nx * ny.
2. BCs are consistent with uniform strain: confirm by writing out the
   strain field implied by the four-edge displacement BCs and checking
   it has constant epsilon_xx, epsilon_yy, epsilon_xy across the element.
3. Material parameters: read [Materials] block.
4. Postprocessor output: read [Postprocessors] block.

Specifically check whether the shear BC implementation gives gamma_xy or
2 * gamma_xy. Voigt-vs-tensor factor-of-2 confusion is the #1 bug pattern
in this lab.

PASS / FAIL each item with one-sentence justification.

Analysis:
<paste lead analysis>
```

---

## Task 2 main: review the data-generation driver

**Lead prompt:**

```
Read generate_data.py in this directory. Examine specifically how it constructs
the gamma_xy column in the output CSV. Voigt convention says gamma_xy = 2 *
epsilon_xy. MOOSE's tensor_mechanics module reports strain_xy_avg as the
*tensor* component (i.e., epsilon_xy, not gamma_xy).

Confirm the script multiplies by 2 in the right place. If not, identify the
bug.
```

**Review prompt:**

```
Read the script and the lead's analysis below. Independently verify:

1. The lead correctly identified MOOSE's convention for strain_xy_avg
   (it reports the tensor component, not the engineering shear).
2. The script handles the factor-of-2 conversion correctly in BOTH
   directions: when reading strain_xy_avg back out, AND when imposing
   gamma_xy via the BCs (the right_y_fn and top_x_fn).
3. The final CSV column 'gamma_xy' contains 2*epsilon_xy (engineering
   shear), not epsilon_xy (tensor shear).

PASS / FAIL each item.

Analysis:
<paste>
```

---

## Task 3: PyTorch training script review

**Lead prompt:**

```
Read mlp_constitutive.py. Without running it, predict and report:

1. The total parameter count of the default 4-layer, width-32 MLP.
2. What happens if I forget the optimizer.zero_grad() call -- describe in
   one sentence what the loss curve would look like after ~10 epochs.
3. What happens if I forget the .standardize() pre-processing -- describe
   the expected effect on the optimal learning rate.
4. The expected MSE on the test set after 5000 epochs for this problem
   (within an order of magnitude).
```

**Review prompt:**

```
Read the script and the lead's predictions below. Verify each prediction
against the actual code:

1. Parameter count: count Linear-layer weights and biases.
2. zero_grad: trace what happens to .grad attributes across epochs.
3. Standardization: examine the standardize() function and confirm what
   happens to the gradient magnitudes.
4. Expected MSE: this is a hard prediction -- justify it with reference
   to the noise floor of the dataset (which is essentially zero for
   linear elasticity with exact analytical labels).

PASS / FAIL each.

Lead's predictions:
<paste>
```

---

## Task 4: stiffness verification

**Lead prompt (run after training):**

```
The training script reported the following implied stiffness matrix:

[paste the printed C_learned from mlp_constitutive.py output]

The analytical plane-strain stiffness for E=1.0, nu=0.3 is:

[[1.3462, 0.5769, 0     ]
 [0.5769, 1.3462, 0     ]
 [0,      0,      0.3846]]

Compute the element-wise relative error. Identify any entry where the
relative error exceeds 5%. For each such entry, propose the most likely cause:

- (1,3), (2,3), (3,1), (3,2): these should be exactly zero by isotropy.
  Nonzero values suggest the MLP has not fully learned the isotropic
  symmetry from the data.
- (3,3): if off by a factor of 2, suspect a Voigt factor-of-2 bug
  somewhere in data generation or stiffness extraction.
- (1,1), (2,2): should be 1.3462. Errors > 1% suggest training has not
  converged.
- (1,2), (2,1): should be 0.5769. By symmetry they should be equal to each
  other (the trained MLP is not symmetry-constrained, so they may differ
  slightly).
```

**Review prompt:**

```
The student's analysis is below. Independently verify:

1. The analytical stiffness values are correctly computed for E=1.0, nu=0.3
   in plane strain.
2. The element-wise relative error is computed correctly (not, say, mixing
   absolute and relative).
3. The diagnosis of "Voigt factor-of-2 bug" is appropriate for off-by-2
   discrepancies in the (3,3) entry but NOT for off-by-2 errors in (1,1)
   or (2,2) (those would indicate a different bug).

PASS / FAIL each.

Analysis to review:
<paste>
```

---

## A meta-tip for the rest of the course

For mechanics-derivation tasks, the most reliable failure-finding *review* prompts include physical limit checks. For this lab specifically, three checks catch most bugs:

- **Set gamma_xy = 0 in the input. Does the model predict zero shear stress?** (Should: by isotropy.)
- **Set eps_yy = eps_xx, gamma_xy = 0. Does it predict equal normal stresses?** (Should: by symmetry of isotropic.)
- **Set eps_xx = eps_yy = 0, gamma_xy = some value. Does it predict zero normal stresses?** (Should: by decoupling of normal and shear in isotropic.)

These three checks together test the structural form of the isotropic stiffness matrix without needing the analytical answer. Add them to your review prompts.
