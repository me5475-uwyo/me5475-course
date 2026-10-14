# Starter Prompts for Lab 4

Lead and review in **separate sessions, no shared context**. Logged prompts mandatory.

---

## Task 1 — MOOSE J2 input file verification

**Lead prompt:**

```
Read j2_cyclic_loading.i in this directory. The file is supposed to apply 3
fully reversed loading cycles (0 -> +a -> -a -> 0, three times) to a single
2-D unit-quad element with J2 plasticity (E=100, nu=0.3, yield_stress=1.0,
hardening=5.0).

Verify by inspection:
1. The Functions block defines a triangle-wave cyclic_factor that goes
   0 -> 1 -> -1 -> 0, three times, across t in [0, 3].
2. The BCs apply this factor to all three controlled strain components
   (eps_xx, eps_yy, gamma_xy/2) via displacement BCs u = eps(t) . x on
   all four edges, so the single element carries a homogeneous strain.
3. The Materials block uses ComputeMultipleInelasticStress with
   IsotropicPlasticityStressUpdate as the inelastic model -- the correct
   stack for J2 + linear hardening in MOOSE.
4. The Executioner is Transient with 60 timesteps total (dt=0.05, t_max=3).
5. The Postprocessors capture both stress AND plastic strain components,
   averaged over the element.
```

**Review prompt:**

```
Read the lead's analysis. Independently verify each item:

1. Plot the cyclic_factor(t) function mentally. Confirm it is a triangle
   wave of period 1: +1 at t = 0.25, 1.25, 2.25; -1 at t = 0.75, 1.75,
   2.75; 0 at every half-integer and integer t.
2. On every edge u_x = (eps_xx_amp * x + gamma_xy_amp/2 * y) * cyclic_factor
   and u_y = (gamma_xy_amp/2 * x + eps_yy_amp * y) * cyclic_factor. At a
   peak (cyclic = 1) this gives eps_xx = eps_xx_amp, eps_yy = eps_yy_amp,
   gamma_xy = gamma_xy_amp everywhere in the element, and at a negative
   peak (cyclic = -1) the same strains with the opposite sign. Correct?
   Why would separate BCs on the right and top edges alone not do this?
3. ComputeMultipleInelasticStress invokes IsotropicPlasticityStressUpdate
   inside its inelastic_models list -- this is the standard return-mapping
   driver.
4. With dt=0.05 over t in [0, 3], we get 60 steps.

PASS / FAIL each.

Analysis:
<paste>
```

---

## Task 2 — Stateless MLP failure mode diagnosis

**Lead prompt:**

```
After running stateless_mlp_baseline.py on trajectories.npz, my stateless_baseline.png
shows the MLP prediction failing to track the cyclic stress-strain loops.
Diagnose why.

The truth shows, for a yielding trajectory (the strain goes 0 -> +a ->
-a -> 0, three times; a is the amplitude vector, so the plotted eps_xx
peak can be positive or negative):
  - Loading branch from (0, 0) toward (eps_max, sigma_max), yielding on
    the way.
  - Unloading branch from (eps_max, sigma_max): elastic at first, a
    straight line with the elastic slope (for most trajectories all the
    way back to eps = 0), which reaches eps = 0 at a nonzero (residual)
    stress.
  - Reverse-loading branch, yielding again in the opposite direction on
    the way to -eps_max (for 127 of the 134 yielding trajectories the
    sigma_xx-eps_xx curve visits both the first and the third quadrant).
  - Reloading from -eps_max, elastic at first, then yielding on the way
    back out to +eps_max. The STRAIN path is closed each cycle, but the
    stress-strain curve does not close: the plastic state has changed,
    so the stress at eps = 0 differs from cycle to cycle, and the stress
    magnitude |sigma| at the same peak strain grows each cycle
    (hardening). The branches drift rather than retracing one loop.

The MLP prediction shows a single-valued curve that cuts between the
branches and follows neither.

Explain why a stateless MLP cannot represent these loops. Frame the
explanation in terms of:
1. The mathematical structure of the constitutive map sigma = f(eps, q).
2. Why a function sigma_NN(eps) is single-valued and cannot have
   different sigma at the same eps.
3. The role of the internal variable q in disambiguating same-eps,
   different-sigma points along the loop.
```

**Review prompt:**

```
Read the lead's diagnosis. Check independently:

1. The claim that "a stateless function cannot return different outputs for
   the same input" -- is this mathematically correct? (Yes for any pure
   function; not for stochastic processes -- but the MLP is deterministic.)
2. The proposed cure (feed eps_p as additional input) -- does it preserve
   the function structure (yes, because (eps, eps_p) IS sufficient to
   determine sigma uniquely for rate-independent plasticity)?
3. The proposed cure relies on observed eps_p in the training data.
   What if eps_p is not directly observable (a real-experiment case)?
   The lead's analysis should mention that a history/state
   representation then becomes necessary (an LSTM is one option). Also
   note: given the current eps_p, the MLP can map (eps, eps_p) to the
   present stress, but it does not learn how eps_p evolves.

PASS / FAIL each.

Diagnosis:
<paste>
```

---

## Task 3 — LSTM architecture verification

**Lead prompt:**

```
Read lstm_plasticity.py. Verify the LSTM is set up correctly for
sequence-to-sequence stress prediction:

1. The input_size is 3 (Voigt strain components).
2. The hidden_size defaults to 32, num_layers defaults to 2.
3. batch_first=True is set, so input shape is (B, T, 3) -- B is the number
   of trajectories per batch, T is the time dimension.
4. The linear head maps (hidden_size,) -> 3 (stress components) at each
   time step, applied via nn.Linear on the LSTM's output sequence.
5. The training loss uses a *masked* MSE that ignores NaN-padded steps in
   trajectories.

For item 5, the mask is constructed from `~np.isnan(epsilon[..., 0])`. Each
trajectory has up to T_max steps, but some may have fewer if MOOSE didn't
write all the timesteps. The masked MSE divides by the number of valid
(B, T, 3) entries, not by B*T_max*3.
```

**Review prompt:**

```
Read the script and the lead's analysis. Verify:

1. The LSTM's input shape (B, T, 3) matches the dataset's
   (n_traj_in_batch, T, 3) shape.
2. The masked_mse function correctly normalizes by (mask.sum() * 3),
   NOT just by mask.sum() (which would undercount the 3 stress
   components per time step) and NOT by (B * T * 3) (which would
   include masked-out NaN entries in the denominator).
3. The standardization uses ONLY training-set statistics, not pooled
   train+test statistics (no leakage).
4. The torch.manual_seed and np.random.seed are set BEFORE the
   train/test split (so the split is reproducible).

PASS / FAIL each.

Script:
<paste>
```

---

## Task 4 (stretch) — Level-set yield surface SDF verification

**Lead prompt:**

```
After running levelset_yield_surface.py, the trained model should predict
phi(sigma) such that:

1. phi(sigma) = 0 on the analytical J2 yield surface (a circle of radius
   kappa in the simplified 2-D stress space we are using).
2. phi(sigma) < 0 inside the circle.
3. phi(sigma) > 0 outside the circle.
4. The Eikonal-like property |grad phi| ~ 1 should hold approximately
   (since we trained on signed-distance values, not just signed labels).

Verify each by computing predictions on test points:
- A point on the surface (1, 0): phi should be 0.
- Origin (0, 0): the exact SDF gives -kappa (i.e., -1 for kappa=1). This
  is the cone tip of |sigma| - kappa, which a smooth network rounds off,
  so expect a noticeably different value; treat it as a diagnostic.
- A far-away point (5, 0): the exact SDF gives 4 (i.e., 5 - kappa).
  Training points were sampled only in the box [-3, 3]^2, so this is
  extrapolation; treat it as a diagnostic too.
  Say what you expect at each before you look.
- Random unit-norm direction multiplied by some r: phi should be
  approximately r - kappa.

Compute these and report.
```

**Review prompt:**

```
Read the lead's verification. Independently confirm:

1. The four test points above span the relevant regions (interior, on,
   far-outside).
2. Accuracy is judged only where the network was trained and the SDF
   is smooth: in the ring 0.5 <= |sigma| <= 2 (kappa = 1), the ABSOLUTE
   error |phi_NN - (|sigma| - kappa)| should be small (our run: at most
   0.026 at the sampled grid points there, measured_results.md
   section 3). Use absolute, not
   relative, error: near the zero contour the exact value is 0. The
   origin and the far point are diagnostics with no pass threshold --
   report the values and explain them (our run: -0.791 at the origin,
   3.209 at (5, 0)).
3. The Eikonal property (|grad phi| ~ 1) -- this is NOT enforced explicitly
   by the H1 Sobolev loss (it matches only the gradient's direction);
   verify by sampling grad_phi at random points and computing the norm.
   Expect it to be near 1 in the well-trained regions but deviate near
   the origin (which is the cone of singularity for the SDF) and far away.

PASS / FAIL each.

Verification:
<paste>
```

---

## Meta-tip for Lab 4

The most-bug-causing thing in Lab 4 is the train/test split. **By trajectory, not by time step.** A time-step-wise split leaks future data from the training set into the test set, and your LSTM will look mysteriously perfect on the test set because it's already seen those points.

When you set up the test set, write the indices on paper and verify no trajectory appears in both train and test. Have the review agent verify this *specifically*.
