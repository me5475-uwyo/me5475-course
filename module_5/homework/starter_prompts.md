# Starter Prompts for Lab 5

Lead and review in **separate sessions, no shared context**. Logged prompts are mandatory.

Lab 5 is a pipeline, and every step prints a check: `export_output.txt`, `verify_output.txt`, the job's `.out`, and
`comparison_metrics.txt`. Paste the real output into the prompt; do not describe it from memory. Numbers measured by
the course are in `module_5/readings/measured_results.md` (cited as §n).

---

## Task 1 — Is the TorchScript file right?

**Lead prompt (after running the export and `verify_torchscript.py`):**

```
Below are the outputs of export_torchscript.py and verify_torchscript.py for my Lab 1 network
(plane-strain Voigt strain (eps_xx, eps_yy, gamma_xy) -> stress, E = 1, nu = 0.3).

1. Did the export reproduce my network exactly in float32 and in float64 (check 4)? The export also
   prints the float64 copy against the float32 model: is that size what float32 round-off should give?
2. For each judged test: is the relative error consistent with my network's Lab 1 test error and
   implied stiffness (pasted below)? Flag any test that is not.
3. Which tests are "outside" the hull of all the strains in my CSV, and is the error larger there?
   That hull includes my validation and test rows: what does it miss about my training data?
4. Is sigma(0) small compared with the stresses in the tests? What would a nonzero sigma(0) do at
   every quadrature point of a MOOSE run?
5. Does the autograd tangent at zero strain match the implied stiffness Lab 1 printed?
If anything FAILS, rank the likely causes: standardization missing from the file; a dtype cast inside
forward(); a Voigt factor of 2 in the data; an under-trained network; a strain outside the data.

<paste export_output.txt>
<paste verify_output.txt>
<paste the stiffness comparison from Lab 1 Task 4>
```

**Review prompt:**

```
Check the lead's analysis against the pasted outputs, independently:
1. Every number the lead quotes appears in the outputs, with the same value.
2. A failure off by roughly a factor of 1000 or with the wrong sign pattern on every test is a
   missing standardization, not a training error. Check 4 of verify_torchscript.py (file vs eager
   model) separates the two: an export bug shows there, a training error does not.
3. The lead does not call a test FAIL that the script marked "outside (not judged)", and does not
   ignore how many tests were outside.
PASS / FAIL each, with the line of output that decides it.

<paste the lead's analysis and the two outputs>
```

---

## Task 2 — The MOOSE job

**Lead prompt (if the job fails, or looks wrong):**

```
My ARCC job ran plate_with_hole_nn_material.i with ComputeTorchScriptStress (a course MOOSE material
that loads a TorchScript file and evaluates it at every quadrature point), then the same input with
the elastic material. The .out file and the end of the MOOSE log are below.

Classify the failure:
1. the executable lacks ComputeTorchScriptStress (wrong executable);
2. the TorchScript file is missing, or is the Lab 1 checkpoint rather than the exported network;
3. the file loads but the call fails (output not 3 values; a dtype cast inside forward());
4. the solve ran but Newton did not converge, or converged slowly;
5. a SLURM problem (time, memory, partition).
Quote the line that decides it and propose one specific fix. If the job succeeded, instead check:
the Newton iterations of both runs, reaction_x_left in both, and the min/max strain postprocessors
against the strain ranges of my Lab 1 CSV (pasted).

<paste nn_moose-<jobid>.out>
<paste the last 40 lines of plate_nn_out.log>
<paste the CSV strain ranges printed by verify_torchscript.py>
```

**Review prompt:**

```
Read the lead's classification and the output. Verify:
1. The quoted line exists and supports the classification.
2. The fix is specific ("re-export with the course environment and point MODEL at the new file"),
   not "try again".
3. If the error is an unknown object type ComputeTorchScriptStress, the diagnosis is the executable,
   not the input file or the .pt file.
4. For a successful run: reaction_x_left should be close to -sigma_inf for BOTH materials. Why is it
   insensitive to the material? Does the lead say so correctly?
PASS / FAIL each.

<paste the lead's analysis>
```

---

## Task 3 — Reading the comparison

**Lead prompt:**

```
Below is comparison_metrics.txt: the same plate, same mesh, solved with the elastic material and with
my network as the material. Rows: relative L2 errors (area-weighted) of u, stress components and von
Mises; postprocessors side by side; a coverage proxy (how many elements' final element-average strains
lie inside the hull of all the strains in my Lab 1 CSV).

1. Rank the fields by relative error and explain the ranking. In particular: why can the displacement
   error be much larger than the sigma_xx error in a plate loaded only by tractions? Relate the
   displacement error to my network's implied stiffness error (pasted).
2. Why do sigma_yy and sigma_xy have larger relative errors than sigma_xx? Look at their sizes.
3. Explain the difference between max_vonmises_stress and max_vonmises_pe in the network run.
4. From the wall times, Newton iterations and residual evaluations, estimate how much more each
   residual evaluation costs with the network than with the elastic material.

<paste comparison_metrics.txt>
<paste the stiffness comparison from Lab 1 Task 4>
```

**Review prompt:**

```
Check the lead's analysis:
1. Every number quoted matches comparison_metrics.txt.
2. The explanation of the u vs sigma_xx gap does not claim the network is "more accurate for stress";
   it should rest on how traction-loaded elastic stress fields depend on the elastic constants.
3. The von Mises explanation names sigma_zz = 0 in the network material, and says vonmises_pe rebuilds
   it as nu (sigma_xx + sigma_yy) in both runs.
4. The cost estimate uses numbers from the output, not general claims about neural networks.
PASS / FAIL each.

<paste the lead's analysis>
```

---

## Task 4 — Extrapolation

**Lead prompt:**

```
I ran the same plate with my network as the material at three more loads.
- sigma_inf = 1 (strains about 1000x larger than at 1e-3): my run <failed / converged>. Below: the job
  output (Newton history) and the end of plate_nn_s1_out.log, and my network's stress at growing
  uniaxial strains (the Task 4a probe).
- sigma_inf = 2e-3 and 1e-4: below, comparison_metrics_s2e-3.txt and comparison_metrics_s1e-4.txt,
  each with its coverage line (elements whose final element-average strain lies inside the hull of
  all the strains in my Lab 1 CSV).

1. At sigma_inf = 1: what did Newton's method do, and which lines of the log show it?
2. What does the probe show a tanh MLP returning far outside its data along this strain direction,
   and what stiffness does that give the solver? What does a one-direction probe NOT tell me: about
   other strain states, and about whether an equilibrium exists at this load at all? Keep "Newton did
   not converge with these settings" apart from "no solution exists".
3. At 2e-3: which fields got worse than at 1e-3, by how much, and how does that relate to the
   coverage line? What can that proxy miss (quadrature points, Newton iterates, the training split)?
4. At 1e-4: did the errors grow or shrink compared with 1e-3? Relate it to sigma(0) from
   verify_output.txt.

<paste the job output and the end of plate_nn_s1_out.log for sigma_inf = 1>
<paste the probe output>
<paste comparison_metrics_s2e-3.txt>
<paste comparison_metrics_s1e-4.txt and sigma(0) from verify_output.txt>
```

**Review prompt:**

```
Check, PASS / FAIL each:
1. The explanation uses the strain ranges and coverage fractions actually printed, and calls the
   coverage a proxy.
2. It keeps "Newton did not converge with these settings" apart from "no solution exists".
3. It does not blame MOOSE or the mesh for what the network does outside its data.
4. The small-load argument compares sigma(0) with the stress levels of that load.

<paste the lead's analysis>
```

---

## Meta-tip — your Lab 1 network sets your Lab 5 result

The same plate, the same mesh, the same solver: the only difference between the two runs is the material. Whatever
you see in Task 3 is your Lab 1 network, measured in a structure instead of on a test set. Discuss in your reflection
how the Lab 1 numbers (test error, implied stiffness, the strains in your CSV) carried into Lab 5.
