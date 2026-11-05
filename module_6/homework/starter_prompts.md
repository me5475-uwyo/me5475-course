# Starter Prompts for Lab 6

Lead and review in **separate sessions, no shared context**. Logged prompts mandatory. Replace every `<...>` with
your own values.

---

## Task 1 — Data generation verification

**Lead prompt (after the first workers finish):**

```
I'm generating 500 RVEs with generate_microstructures.py and MOOSE (rve_homogenization.i.template).
Read the template's header first: the boundary conditions are mixed, not periodic, and the driver
reports E_app = <sigma_xx> / 0.001 under those conditions.

My first outputs: <paste E_app values and the "Wrote N microstructures" lines from your m6_rve-*.out files>.

Check the physics:
1. Homogeneous limit. For a homogeneous cell (E constant, nu = 0.3) under these BCs, derive E_app.
   (Hint: sigma_yy = 0 and eps_zz = 0.) Evaluate it for E = 1 and E = 5.
2. Range. Explain why every composite of these two phases must have E_app between those two values
   under displacement-controlled loading, and check that all my values do.
3. Trend. With the area fraction f = img.mean() of each image, does E_app increase with f? Describe the
   scatter; don't assume a formula for it.
4. Completeness. Did every worker write all of its RVEs? If any were skipped, quote the error.
```

**Review prompt:**

```
Read the lead's analysis and the template's header. Verify:

1. The homogeneous-limit derivation and its two numbers.
2. The argument for the range, and that the reported values lie inside it.
3. The trend description matches the data (it should not be forced onto a bound or a formula).
4. Skipped RVEs, if any, are counted from the worker printouts, with their errors quoted.

PASS / FAIL each, with the reason.

Lead analysis:
<paste>
```

---

## Task 2 — CNN training diagnosis

**Lead prompt (after the first training run):**

```
Read the parity plot at checkpoints/cnn_modulus.png and the training printout m6_cnn-<jobid>.out.
Diagnose the training. Use the training and VALIDATION curves for any decision about what to change;
the test set is only for the final report.

1. Which device did it run on (the "Device:" line), and on which split seed?
2. How does the test MAE compare with the two Task 2d baselines (predict the training mean; a straight
   line in the area fraction) on the same test rows? A small MAE / mean(E_app) alone means little when
   E_app spans a narrow range.
3. Does the validation loss track the training loss, or is there a gap? When did early stopping trigger?
4. Are the predictions spread along the diagonal or compressed toward the mean?

For each observation, propose a change to try, and say how you would check it on the validation set.
```

**Review prompt:**

```
Read the lead's diagnosis. Verify:

1. The metric: MAE = mean |y_pred - y_true| over the test rows, divided by mean y_true over the same rows.
2. The baseline comparison uses the same split (same split seed) as the CNN.
3. The diagnosis matches the plot and the printout.
4. Proposed changes are judged on validation data, not by looking at the test parity plot again.

PASS / FAIL each.

Diagnosis:
<paste>
```

---

## Task 4 — OOD test setup

**Lead prompt:**

```
I want to test my trained CNN on 3 out-of-distribution subgroups, 10 microstructures each:

1. More inclusions: 6-8 (training had 2-5). generate_microstructures.py takes --min-inclusions and
   --max-inclusions.
2. One large circular inclusion (a different topology).
3. Stripes (a different geometry): I write the image and the E(x, y) expression myself.

Write ood_microstructures.py: make the images and E(x, y) expressions, run MOOSE on each as a job
(not on the login node), and save images and E_app. Then load my Task 2 checkpoint (the model predicts
standardized E_app; the scalers are in the checkpoint) and predict.

Before running, write down a hypothesis for each subgroup and say how uncertain it is. After running,
report the measured errors next to the hypotheses, and the area-fraction line's errors on the same RVEs.
A result that contradicts the hypothesis is a finding, not a failure.
```

**Review prompt:**

```
Read the lead's script, its job printout and the results table. Verify:

1. Each subgroup has the intended microstructures (look at a few images) and all MOOSE solves succeeded
   (count them from the printout; no silent skips).
2. Every E_app lies between the pure-phase limits 1.0989 and 5.4945.
3. Predictions were un-standardized with the checkpoint's scalers.
4. The conclusions follow from the measured numbers, whatever order the errors came out in.

PASS / FAIL each.

Results:
<paste>
```

---

## Meta-tip — if MOOSE solves fail

Start from the evidence before changing the data-generating model:

1. Read the failing run's MOOSE log and the error the driver printed.
2. Check the material field: is E(x, y) between 1 and 5 everywhere (the template's `E_avg` postprocessor and a plot
   of the expression on a grid help)?
3. Check resolution: very thin or touching inclusions may be poorly resolved by a 64 × 64 mesh with a smoothed
   interface.

Only then change something (inclusion sizes, spacing, the interface sharpness), one thing at a time, and record what
you changed and why in `prompts_data_gen.md`. Note that changing the generator changes the dataset you are learning.
