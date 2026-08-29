# Starter Prompts for Lab 0

You may adapt these prompts freely. Logging them (paraphrased or verbatim) in your `prompts_*.md` files is required; reusing them verbatim is permitted.

The pattern throughout: **lead and review run in separate sessions with no shared context.** That independence is what makes review effective.

---

## Task 2 — Install MOOSE locally

**Lead prompt** (provided in `module_0/examples/install_moose_prompt.md`; copy it verbatim, fill in your OS).

**Review prompt** (run in a fresh session after install completes):

```
Read the install log below and independently verify the following items:

1. The conda/mamba environment 'moose' actually exists.
2. moose-opt is on PATH when the environment is active.
3. moose-opt --version prints a reasonable version string (not empty, not an error).
4. Python in the environment is 3.11.x as requested.
5. No error or warning messages were silently ignored by the install agent.

Reply PASS / FAIL per item with one sentence of justification.

Install log:
<paste log here>
```

---

## Task 3 — ARCC hello-world

**Lead prompt** (after activating Claude Code in your local copy of the course repo):

```
Modify the provided hello.sbatch in this directory for my ARCC account.
My account string is: <fill in>
My email for SLURM notifications: <fill in>@uwyo.edu

Confirm the partition name, time limit, and module loads against the current
UW ARCC user guide at <URL>. Then submit the job via sbatch and report the
SLURM job ID it returned. After the job completes, show me the contents of
the output file.
```

**Review prompt** (fresh session):

```
Read the SLURM script and the job output below. Verify independently:

1. The #SBATCH directives are well-formed (no typos, correct syntax).
2. The partition exists on UW ARCC (look it up in the user guide).
3. The module load lines reference modules that exist.
4. The script runs python3, not python2 (some clusters default to python2).
5. The output file shows the expected hello-world content plus a SLURM job ID.

PASS / FAIL per item.

SLURM script:
<paste>

Output:
<paste>
```

---

## Task 4a — Modify plate_with_hole.i for five refinement levels

**Lead prompt:**

```
Read the file plate_with_hole.i in this directory. I want to run a
mesh-refinement convergence study with uniform_refine in {0, 1, 2, 3, 4}.

Propose ONE of these two approaches and implement it:

Option A: Generate five separate input files
  plate_with_hole_refine_0.i ... plate_with_hole_refine_4.i
  each with the correct uniform_refine value and a distinct Outputs/file_base
  so the runs don't overwrite each other's outputs.

Option B: Keep a single plate_with_hole.i and arrange to override
  Mesh/uniform_refine and Outputs/file_base at the MOOSE command line.

Whichever you pick, explain your choice in one sentence so I can decide if I
agree. Then implement it.
```

**Review prompt** (fresh session):

```
The files below are intended for a mesh-refinement convergence study. Verify
independently:

1. uniform_refine takes the values {0, 1, 2, 3, 4} across the five
   configurations (or command-line invocations).
2. Each configuration writes to a distinct Outputs/file_base so outputs are
   not overwritten.
3. All other inputs (material properties, BCs, mesh generators) are identical
   across the five configurations.
4. The CSV output is enabled (csv = true under [Outputs]).
5. The Postprocessors block defines max_vonmises_stress, max_stress_xx, and
   num_elements.

PASS / FAIL per item.

<paste files or command-line invocations>
```

---

## Task 4b — Modify run_convergence_study.sbatch for ARCC

**Lead prompt:**

```
Read run_convergence_study.sbatch in this directory. Modify it for my ARCC
account.

My account string: <fill in>
My email: <fill in>@uwyo.edu
Current ARCC partition (per the user guide): <fill in>
MOOSE on ARCC is available via: <module load moose OR conda env at /project/...>

Verify the SLURM array runs all five refinement levels. Confirm the resource
requests are sensible -- for the finest level (refine=4) my mesh will have
~10^4 to 10^5 elements; a few minutes on 4 cores should be ample.

After modification, submit the job and confirm it shows up in squeue.
```

**Review prompt** (fresh session):

```
Read the SLURM array script below. Verify independently:

1. The #SBATCH --array directive covers exactly the integers 0,1,2,3,4.
2. SLURM_ARRAY_TASK_ID is used correctly to vary uniform_refine.
3. Outputs/file_base is set per-task so the 5 runs don't overwrite each other.
4. Resource requests (time, ntasks, mem) are reasonable for the workload.
5. The MOOSE activation (module load OR conda activate) is correct for ARCC.

PASS / FAIL per item.

<paste script>
```

---

## Task 4c — Modify extract_stress.py to read 5 CSVs and plot

**Lead prompt:**

```
Read extract_stress.py in this directory. Modify it to:

1. Glob for plate_with_hole_refine_*_out.csv files in the outputs/ subdirectory.
2. Parse each one for max_vonmises_stress, max_stress_xx, and num_elements
   (last row of the CSV).
3. Sort by refinement level (extracted from the filename).
4. Plot both peak-stress series vs. element count on log-log axes.
5. Overlay the infinite-plate reference lines: 2.67 for plane-strain von Mises
   and 3.0 for the Kirsch hoop stress.
6. Save the plot to outputs/convergence.png at 150 DPI.
7. Print a summary table and report whether the refine-4 peaks are within ±2%
   of the validated expectations: 2.48 for von Mises and 2.88 for hoop stress.

Make sure the script handles missing files gracefully (skip and warn) and
exits nonzero if no files are found.

Use only pandas, numpy, and matplotlib.
```

**Review prompt** (fresh session):

```
Read the Python script below. Verify independently:

1. The CSV parsing picks up the LAST row, not the header row.
2. The glob pattern matches the expected file naming.
3. The sort order is by refinement level (integer), not lexicographic.
4. The log-log axes are correctly set on both x and y.
5. Both reference lines are correct: 2.67 for plane-strain von Mises and 3.0
   for Kirsch hoop stress.
6. The script handles the case of no files matched (early exit, clear error).
7. There are no off-by-one or sign errors in the relative-error computation.
8. The refine-4 checks use 2.48 for von Mises and 2.88 for hoop stress, with a
   ±2% tolerance; they do not treat the reference lines as refine-4 endpoints.

PASS / FAIL per item.

<paste script>
```

---

## A meta-tip on prompting

Keep prompts *concrete and bounded*. "Modify the script to read the CSVs" is too vague — the agent has to guess. "Modify the script to glob for `plate_with_hole_refine_*_out.csv`, parse the last row for the two peak-stress columns plus `num_elements`, plot both series with the 2.67 / 3.0 reference lines, and check the refine-4 values against 2.48 / 2.88" is enough information for one round-trip to nail it.

The same applies to review. "Check this script" gets you a vibe. "Verify items 1–6 below, PASS/FAIL each with justification" gets you a checklist you can act on.
