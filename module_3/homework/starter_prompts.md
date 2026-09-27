# Starter Prompts for Lab 3

Lead and review run in **separate sessions, no shared context**. Logged prompts mandatory: every lead and review prompt, with the agent's answer, goes into that task's `prompts_taskN.md`.

These prompts assume the M0–M2 starter prompts have already taught you the *style* of effective prompt engineering. The Lab 3 prompts focus on the PINN-specific traps. They follow `lab_3.md` task by task; if a prompt and the handout ever disagree, the handout wins.

**Every Lab 3 agent needs these facts** — it cannot know them unless you give them:

- **The problem:** the square [0, 1] × [0, 1] minus a quarter hole of radius 0.1 at the origin; plane-strain linear elasticity, E = 1, ν = 0.3; **ten** boundary conditions — left u₁ = 0, σ₁₂ = 0; bottom u₂ = 0, σ₁₂ = 0; right u₁ = 1, σ₁₂ = 0; top σ₂₂ = 0, σ₁₂ = 0; the hole traction-free (two conditions).
- **The reference:** the MOOSE solution of this *same* problem, shared read-only in `/project/me5475/examples/lab3_reference/`. Never Lab 0's `plate_with_hole_refine_4_out.e`: that is a different problem (a quarter annulus loaded by a traction).
- **Training runs only as a submitted job** (`sbatch`), never on the login node — and never more than two of your GPU jobs at once.
- **Commands** are exactly the ones in `lab_3.md`; every block starts from the root of your course repository.

---

## Task 1 — Port Min's notebook and run it on ARCC

**Lead prompt:**

```
I am doing Lab 3 Task 1 of a PINN course. Two files:

- module_3/examples/min_lin_2D_hole_example.ipynb -- Min Lin's 2022 DeepXDE
  notebook (TensorFlow backend, outputs cleared). I only READ it: the course
  environment has no TensorFlow.
- module_3/examples/plate_with_hole_fixed.py -- the course's PyTorch-backend
  port of it, which I will train as a job.

The problem: the square [0,1]^2 minus a quarter hole of radius 0.1 at the
origin; plane-strain linear elasticity, E = 1, nu = 0.3; network outputs
(u_1, u_2, sigma_11, sigma_22, sigma_12). Ten boundary conditions:
  left   (x_1 = 0):  u_1 = 0,       sigma_12 = 0
  bottom (x_2 = 0):  u_2 = 0,       sigma_12 = 0
  right  (x_1 = 1):  u_1 = 1,       sigma_12 = 0
  top    (x_2 = 1):  sigma_22 = 0,  sigma_12 = 0
  hole   (r = 0.1):  both traction components = 0

1. Tabulate the ten conditions: for each, the notebook's BC object, the port's
   BC object, the output component it constrains and the boundary function it
   uses. Flag any condition missing from either file.
2. Explain the port line by line: the five PDE residuals (2 momentum +
   3 constitutive), the geometry, the BC list, build_problem(), train(),
   plot_solution() and main().
3. List every difference between the notebook and the port, and say whether it
   changes the problem being solved. Check at least: the backend; the
   constitutive constants (which 2-D idealisation does each file use?); the
   collocation counts; the sign of the hole normal; the order of the BC list;
   the seeding; what is saved.
4. Tell me which files the job writes. pinn_train.sbatch runs the port with
   --epochs 50000 --seed ${SEED:-42}.

Do not change the port's PDE or its ten BCs, and do not suggest running it on
the login node. I submit exactly as lab_3.md 1c says:

  cd "$(git rev-parse --show-toplevel)"
  mkdir -p labs/lab_3/<your-github-handle>/task1_port
  cd labs/lab_3/<your-github-handle>/task1_port
  cp ../../../../module_3/examples/plate_with_hole_fixed.py ../../../../module_3/examples/pinn_train.sbatch .
  sbatch pinn_train.sbatch            # Adam 50 000 + L-BFGS; about 10 minutes on an A30
  squeue -u $USER
```

**Review prompt (separate session):**

```
Compare Min Lin's notebook (module_3/examples/min_lin_2D_hole_example.ipynb,
TensorFlow backend) with the course's PyTorch port
(module_3/examples/plate_with_hole_fixed.py), and check the lead's notes.
Verify the following independently:

1. Both files impose the same TEN boundary conditions: u_1 = 0 and
   sigma_12 = 0 on the left; u_2 = 0 and sigma_12 = 0 on the bottom; u_1 = 1
   and sigma_12 = 0 on the right; sigma_22 = 0 and sigma_12 = 0 on the top;
   zero traction (two components) on the hole. The three sigma_12 = 0
   conditions on the left, bottom and right MUST be in the list the port
   passes to dde.data.PDE (an earlier port dropped them; see
   module_3/readings/measured_results.md, section 2, for what that did).
2. Each BC constrains the correct output (0 u_1, 1 u_2, 2 sigma_11,
   3 sigma_22, 4 sigma_12) on the correct boundary.
3. The hole normal is n = -x/r in the notebook and n = +x/r in the port. For a
   traction-FREE condition the sign cannot matter (t = 0 exactly when -t = 0).
4. The two momentum residuals and the shear relation (C33 = E/(1+nu) = 2*mu)
   are the same in both files. The normal-stress relations are NOT: the
   notebook's C11 = E/(1-nu^2), C12 = E*nu/(1-nu^2) are the PLANE-STRESS
   constants; the port's Lame form (lambda + 2*mu, lambda) is PLANE STRAIN.
   The lab's problem and the MOOSE reference (planar_formulation =
   PLANE_STRAIN in module_3/examples/plate_square_hole_reference.i) are plane
   strain, so the port must stay plane strain. FAIL if the lead "fixed" the
   port to match the notebook.
5. Same network ([2] + [50]*6 + [5], tanh, Glorot uniform) and optimizer
   sequence (Adam, lr = 1e-3, 50 000 iterations, then L-BFGS). The collocation
   budget differs on purpose: 4 000 domain + 600 boundary points in the port,
   1 500 + 300 in the notebook (measured_results.md, section 3).
6. dde.grad.jacobian(u, x, i=i, j=j) returns the partial derivative of
   u[:, i] with respect to x[:, j] -- NOT the other way round (a common bug).
   Check every jacobian call in the port.
7. DDE_BACKEND is set to "pytorch" BEFORE deepxde is imported; DeepXDE reads
   it only at import.
8. The seeds are set before build_problem() samples the points and builds the
   network, so the same --seed gives the same points and starting weights
   (Task 3 relies on this).
9. The port's BC order -- left u_1, right u_1, bottom u_2, left/bottom/right
   sigma_12, top sigma_22, top sigma_12, hole t_1, hole t_2 -- puts the loss
   columns in the order 0-4 PDE, 5-7 displacement BCs, 8-14 traction BCs.
10. The von Mises panel uses the plane-strain formula: sigma_33 =
    nu*(sigma_11 + sigma_22), not zero.

PASS / FAIL each, with a one-sentence justification.

Notebook:
<paste the notebook's code cells>

Port:
<paste plate_with_hole_fixed.py -- your copy, if you modified it>

Lead's notes:
<paste>
```

---

## Task 2 — Compare with the MOOSE reference

**Lead prompt:**

```
Lab 3 Task 2: compare my trained PINN from Task 1 with the MOOSE reference of
the SAME problem (the square [0,1]^2 minus a quarter hole r = 0.1, plane
strain, E = 1, nu = 0.3, ten BCs), shared read-only in
/project/me5475/examples/lab3_reference/.

I extracted the reference on a 200 x 200 grid exactly as lab_3.md 2a says (on
the login node; it takes seconds and is not training):

  cd "$(git rev-parse --show-toplevel)"
  mkdir -p labs/lab_3/<your-github-handle>/task2_moose_comparison
  cd labs/lab_3/<your-github-handle>/task2_moose_comparison
  python /project/me5475/examples/extract_moose_reference.py --mode grid --nx 200 --ny 200 --output reference_grid.csv

reference_grid.csv holds only points inside the plate, with columns
x_1, x_2, u_1, u_2, sigma_11, sigma_22, sigma_12, sigma_vm.

The skeleton below (lab_3.md 2b) rebuilds the problem, loads ONLY the network
weights from my Task 1 checkpoint (the checkpoint also holds the L-BFGS
optimizer state) and predicts u_1, u_2, sigma_11, sigma_22, sigma_12 at the
grid points. Extend it into comparison_script.py that:

1. Computes relative L2 errors as FIELD norms over all grid points,
   ||PINN - MOOSE|| / ||MOOSE||: one for the displacement (u_1, u_2)
   together, and one for each of sigma_11, sigma_22, sigma_12. No pointwise
   relative errors: they blow up where the reference stress is near zero.
2. Evaluates the PINN exactly at (x_1, x_2) = (0, 0.1), the top of the hole,
   and compares its sigma_11 with the reference value there, 3.2418
   (module_3/readings/measured_results.md, section 1). The grid has no point
   at (0, 0.1), so do not read it off the grid.
3. Writes the four errors and the hole-top value, saying where it was
   evaluated, to comparison_metrics.txt.
4. Plots comparison_plot.png: for each of the five fields,
   PINN | MOOSE | |PINN - MOOSE|, with the same colour range on the PINN and
   MOOSE panels.

It only predicts (no training), so it is a short analysis for the login node.

Skeleton:
<paste the code block from lab_3.md 2b>
```

**Review prompt:**

```
Read the comparison script and its outputs. Verify independently:

1. reference_grid.csv was extracted with extract_moose_reference.py's default
   --exodus: the shared reference of THIS problem in
   /project/me5475/examples/lab3_reference/. FAIL if anything uses Lab 0's
   plate_with_hole_refine_4_out.e -- a different problem (a quarter annulus
   loaded by a traction). The extractor reported "0 NaN rows dropped".
2. The PINN is evaluated at exactly the CSV's x_1, x_2 columns (no
   re-gridding or interpolation of its own), and the predicted columns
   (u_1, u_2, sigma_11, sigma_22, sigma_12) are matched to the CSV's columns
   by name, not by position (the CSV starts with x_1, x_2 and ends with
   sigma_vm).
3. Only the network weights are loaded ("model_state_dict" from the Task 1
   checkpoint); nothing is trained.
4. Each relative L2 error is a field norm, sqrt(sum (PINN - MOOSE)^2) /
   sqrt(sum MOOSE^2) over all grid points -- (u_1, u_2) together, then each
   stress component on its own. No maximum pointwise relative error anywhere.
5. The hole-top stress is read at exactly (0, 0.1) -- the top of the quarter
   hole, where sigma_11 peaks -- and compared with the MOOSE reference value
   there, sigma_11 = 3.2418 (module_3/readings/measured_results.md, section 1).
   It is not taken from the nearest grid point, (0, 0.1005).
6. The plot uses the SAME colorbar range for PINN and MOOSE side-by-side
   panels so visual comparison is meaningful.
7. comparison_metrics.txt says where the hole-top stress was evaluated.

PASS / FAIL each.

Extraction command and its output:
<paste>

Script:
<paste>

comparison_metrics.txt:
<paste>
```

---

## Task 3 — Hard boundary conditions: a controlled comparison

This is the highest-stakes prompt because a subtle off-by-one in the ansatz silently breaks the BCs — and forgetting that the ansatz fixes only the displacements silently drops traction conditions.

**Lead prompt (3a — what the ansatz guarantees):**

```
Lab 3 Task 3: the hard-BC plate PINN, module_3/examples/plate_with_hole_hard_bc.py.
The three displacement conditions
  u_1 = 0 on x_1 = 0,    u_1 = 1 on x_1 = 1,    u_2 = 0 on x_2 = 0
are built into the network output by output_transform(x, y), where y (shape
(N, 5)) is the raw network output:

  u_1_hard = x_1 + x_1 * (1 - x_1) * y[:, 0:1]
  u_2_hard = x_2 * y[:, 1:2]
  the stresses y[:, 2:5] pass through unchanged

1. Verify by direct algebra that, for ANY raw outputs:
   a. At x_1 = 0:  u_1_hard = 0 + 0 * (1 - 0) * y[:, 0:1] = 0
   b. At x_1 = 1:  u_1_hard = 1 + 1 * (1 - 1) * y[:, 0:1] = 1
   c. At x_2 = 0:  u_2_hard = 0 * y[:, 1:2] = 0
   d. At x_1 = 0.5 (interior): u_1_hard = 0.5 + 0.25 * y[:, 0:1] -- depends on
      the network output, as it must.
2. Write a short script that imports output_transform from
   plate_with_hole_hard_bc.py, feeds RANDOM raw outputs (torch.randn(N, 5))
   through it at points exactly on the three edges -- left x_1 = 0 with
   0.1 <= x_2 <= 1, right x_1 = 1 with 0 <= x_2 <= 1, bottom x_2 = 0 with
   0.1 <= x_1 <= 1 (the hole removes the rest) -- and writes the largest
   violation of each identity to ansatz_check.txt. Nothing is trained, so it
   runs on the login node.
3. From build_problem_hard_bc(), list the SEVEN conditions still in the loss.
   The transform fixes only u_1 and u_2, so every traction condition stays
   soft, including sigma_12 = 0 on the left, bottom and right: ten soft
   conditions become seven.

Do not tell me hard BCs train faster or end more accurate: Task 3 measures that.
```

**Lead prompt (3b–3c — the controlled comparison):**

```
I trained the soft and the hard PINN with the same seed, exactly as lab_3.md 3b
says:

  cd "$(git rev-parse --show-toplevel)"
  mkdir -p labs/lab_3/<your-github-handle>/task3_hard_bc
  cd labs/lab_3/<your-github-handle>/task3_hard_bc
  cp ../../../../module_3/examples/plate_with_hole_fixed.py ../../../../module_3/examples/plate_with_hole_hard_bc.py \
     ../../../../module_3/examples/pinn_train.sbatch .
  SEED=43 sbatch pinn_train.sbatch            # soft
  SEED=43 sbatch pinn_train.sbatch hard_bc    # hard

With the same seed both use the same 4 000 + 600 collocation points, the same
starting weights, the same 6 x 50 network and the same schedule (Adam 50 000 +
L-BFGS): the only difference is how the three displacement conditions are
enforced.

Adapt my Task 2 comparison_script.py into a script that, for BOTH runs, writes
soft_vs_hard_metrics.txt and soft_vs_hard.png:

1. Relative L2 errors (field norms) of (u_1, u_2) together and of each stress
   component against ../task2_moose_comparison/reference_grid.csv. Soft run:
   build_problem() + plate_with_hole_fixed.pt-*.pt; hard run:
   build_problem_hard_bc() + plate_with_hole_hard_bc.pt-*.pt (load only
   "model_state_dict", as in Task 2).
2. The largest error in u_1 = 0 (left), u_1 = 1 (right) and u_2 = 0 (bottom)
   on FRESHLY sampled edge points, not the training points.
3. sigma_11 exactly at (0, 0.1), against 3.2418
   (module_3/readings/measured_results.md, section 1).
4. Training cost: the wall time of each phase from each job log (it prints
   "'train' took ... s" twice: Adam, then L-BFGS), and the number of L-BFGS
   steps, steps[-1] - 50000, from "steps" in each .loss.npz (the step count
   carries on from the end of Adam).
5. soft_vs_hard.png: loss curves of the COMPARABLE components only. Each
   .loss.npz stores one column per term in loss_train. Columns 0-4 are the PDE
   residuals in both; the seven traction terms are columns 8-14 in the soft
   file and 5-11 in the hard file; the soft file's columns 5-7 are the three
   displacement conditions the hard run does not have. Never plot or compare
   the two TOTAL losses: they sum different terms.

It only predicts and reads files, so it runs on the login node.
```

**Review prompt:**

```
Read the lead's algebra, the ansatz check, the metrics script and its outputs.
Verify independently:

1. The four algebra checks (x_1 = 0, x_1 = 1, x_2 = 0, x_1 = 0.5) hold for
   arbitrary raw outputs.
2. The implementation slices x[:, 0:1] and x[:, 1:2] (keeping the trailing
   dimension), NOT x[:, 0] (which would broadcast wrong), and concatenates
   [u_1_hard, u_2_hard, y[:, 2:5]] with torch.cat along dim=1 -- the original
   output order, with the stresses untouched.
3. ansatz_check.txt comes from random raw outputs at points exactly on the
   three edges and reports a largest violation of 0 for each identity.
4. The seven conditions listed are exactly the BC list of
   build_problem_hard_bc(): sigma_12 = 0 on the left, bottom and right;
   sigma_22 = 0 and sigma_12 = 0 on the top; the two hole tractions. None of
   the three sigma_12 conditions is missing.
5. Both runs used the same SEED (so the same collocation points and starting
   weights), the same network and the same schedule.
6. The displacement-condition error is measured on freshly sampled edge
   points for both runs (hard: 0 by construction; soft: whatever it is).
7. Every error is a field-norm relative L2 error on reference_grid.csv, and
   sigma_11 is evaluated exactly at (0, 0.1) against 3.2418.
8. The loss plot compares only matching columns (PDE 0-4 in both; traction
   8-14 soft against 5-11 hard) and never the totals. The soft file has 15
   loss columns, the hard file 12.
9. The L-BFGS step counts are steps[-1] - 50000, and the phase times come
   from the two "'train' took ... s" lines of each job log.
10. The conclusions follow from these numbers. What the ansatz GUARANTEES is
    exact displacement conditions -- nothing more. FAIL any claim that hard
    BCs are faster or more accurate in general; either method can be the
    right answer if the evidence supports it.

PASS / FAIL each.

Algebra and ansatz_check.txt:
<paste>

Metrics script and soft_vs_hard_metrics.txt:
<paste>
```

---

## Task 4 — Optuna sweep on the GPUs

**Lead prompt (4a–4b — before you submit):**

```
Lab 3 Task 4: an Optuna sweep of the hard-BC plate PINN with the course scripts
pinn_optuna_sweep.py and pinn_optuna_sweep.sbatch. Before I submit, explain
both to me and confirm each point below against the code, citing the lines.

My commands, exactly as lab_3.md 4a-4b (create-study runs on the login node;
the sweep itself only as the submitted array):

  cd "$(git rev-parse --show-toplevel)"
  mkdir -p labs/lab_3/<your-github-handle>/task4_optuna_sweep
  cd labs/lab_3/<your-github-handle>/task4_optuna_sweep
  cp ../../../../module_3/examples/pinn_optuna_sweep.py ../../../../module_3/examples/pinn_optuna_sweep.sbatch \
     ../../../../module_3/examples/plate_with_hole_hard_bc.py ../../../../module_3/examples/plate_with_hole_fixed.py .
  cp ../task2_moose_comparison/reference_grid.csv .
  optuna create-study --study-name pinn_lab3_sweep --storage "sqlite:///$(pwd)/pinn_opt.db" --direction minimize
  sbatch pinn_optuna_sweep.sbatch
  squeue -u $USER
  grep -h "split fingerprint" pinn_sweep-*.out | sort -u     # must print exactly one line

Search space (in the script):
  lr               log-uniform in [1e-4, 1e-2]
  hidden_width     categorical {32, 50, 64}
  hidden_layers    integer in [4, 8]
  adam_epochs      categorical {10000, 25000, 50000}
  bc_loss_weight   log-uniform in [1, 100]
TPE sampler; MedianPruner(n_startup_trials=3, n_warmup_steps=2).

Confirm:
1. The array is --array=0-5%2: six workers x four trials = 24 trials, at
   most two GPUs at a time. I will not remove the %2 or submit a second array
   while one is running.
2. The reference grid is split ONCE, by --split-seed 42 (the same in every
   worker), into a validation half and a test half (split_reference). Every
   trial -- its pruning reports and its final value -- is scored only on the
   VALIDATION half: the relative L2 error of (u_1, u_2).
3. loss_weights has 12 entries: the 5 PDE residuals at weight 1, then the 7
   traction BCs (sigma_12 = 0 on left, bottom and right; sigma_22 = 0 and
   sigma_12 = 0 on top; the two hole tractions) at bc_loss_weight. The ansatz
   fixes u_1 and u_2, so no displacement BC is in the loss.
4. Each trial reports its validation error every 10 % of adam_epochs
   (trial.report); the pruner can stop it during Adam, and only surviving
   trials go on to L-BFGS.
5. Each worker has its own seed (--seed 42 + the array task ID) for the
   sampler and, plus the trial number, for the initialisation; the workers
   share one study asynchronously -- so re-running the sweep will NOT
   reproduce it. What every worker shares is the split.

Pruned trials are expected: how many complete is up to the pruner. What counts
is that the study holds all 24 attempted trials (complete, pruned and failed).
If a worker dies for a cluster reason (timeout, node failure -- its log ends
without "Worker done"), I resubmit just that worker once, after the first
array has left the queue, and then count:

  sbatch --array=<task-id> pinn_optuna_sweep.sbatch       # e.g. --array=3 for pinn_sweep-<jobid>_3.out
  python -c "import optuna; s=optuna.load_study(study_name='pinn_lab3_sweep', storage='sqlite:///pinn_opt.db'); print(sum(t.state.is_finished() for t in s.trials))"

If you suggest narrowing the lr range (lab_3.md Hints), justify it from my
Task 1 and Task 3 runs and L9-L10, and change nothing else.
```

**Lead prompt (4e — retrain the best configuration):**

```
Lab 3 Task 4e. Write retrain_best.py for my task4_optuna_sweep/ folder
(pinn_optuna_sweep.py, plate_with_hole_hard_bc.py, plate_with_hole_fixed.py and
reference_grid.csv are already there). It must:

1. Load the study (study_name "pinn_lab3_sweep", storage
   "sqlite:///pinn_opt.db") and read study.best_params: lr, hidden_width,
   hidden_layers, adam_epochs, bc_loss_weight.
2. Rebuild the hard-BC problem as the sweep's objective does: the data from
   build_problem_hard_bc() (4 000 + 600 points); a network
   [2] + [hidden_width]*hidden_layers + [5] (tanh, Glorot uniform) with
   net.apply_output_transform(output_transform); loss_weights =
   [1]*5 + [bc_loss_weight]*7. Train Adam for adam_epochs at lr, then L-BFGS
   with the same loss_weights. Set the seed first and print it.
3. Split reference_grid.csv with split_reference(ref, 42) from
   pinn_optuna_sweep.py, print the fingerprint (it must match the one in the
   sweep logs) and report the relative L2 error of (u_1, u_2) on the TEST half
   (l2_error_vs_reference).
4. Load my Task 3 hard-BC network into the net from build_problem_hard_bc()
   (../task3_hard_bc/plate_with_hole_hard_bc.pt-*.pt; "model_state_dict" only,
   as in lab_3.md 2b) and report its error on the same test half.
5. Write everything to retrain_log.txt.

Call the test half what it is: unused by the Optuna objective, not an
untouched end-to-end test set (Tasks 2 and 3 already looked at the whole grid).
retrain_best.py trains, so it runs as a job: I copy pinn_train.sbatch and point
SCRIPT/ARGS at retrain_best.py. Show me that edit.
```

**Review prompt:**

```
Read the lead's setup, the worker logs, study_summary.txt and retrain_log.txt.
study_summary.txt is the output of (login node):

  python /project/me5475/examples/analyze_study.py --storage sqlite:///pinn_opt.db --study-name pinn_lab3_sweep --out-dir . | tee study_summary.txt

Verify independently:

1. pinn_optuna_sweep.sbatch still says --array=0-5%2 (at most two GPUs at a
   time), and no second array ran while the first was running. A worker was
   resubmitted only if it died for a cluster reason, at most once, and only
   after the first array had left the queue.
2. Exactly one split fingerprint appears across all worker logs: the split
   seed (--split-seed 42) is the same for every worker.
3. The objective scores every trial -- its intermediate reports and its final
   value -- on the VALIDATION half only. The other half is described as
   "unused by the Optuna objective", not as an untouched test set.
4. The loss_weights passed to dde.Model.compile have 12 entries:
   5 PDE residuals at weight 1 + 7 traction BCs at bc_loss_weight (the three
   sigma_12 = 0 edges, the top's sigma_22 = 0 and sigma_12 = 0, the two hole
   tractions). FAIL any count of 9, or any "4 BCs".
5. MedianPruner(n_startup_trials=3, n_warmup_steps=2): no trial is pruned
   until 3 trials have completed in the study, and a trial's first two
   reports (steps 0 and 1) are never pruned. The reports are *intermediate*
   validation errors (trial.report inside the objective, every 10 % of
   adam_epochs); L-BFGS is never pruned.
6. The sampler seed differs per worker (42 + array task ID) and the workers
   run asynchronously, so the sweep is NOT reproducible run to run. FAIL any
   claim that a fixed sampler seed makes it reproducible.
7. The study holds at least 24 attempted trials -- complete, pruned and
   failed all count; "running" trials do not (complete + pruned + failed on
   the "Trials total" line of study_summary.txt, or the one-liner in
   lab_3.md 4b). Pruned trials are the pruner working, not a
   fault; FAIL any demand for a minimum number of *completed* trials. If
   trials still show as "running" after every job has ended, a worker died
   mid-trial -- that should be said in prompts_task4.md.
8. The hyperparameter-importance ranking is treated as exploratory.
9. retrain_best.py ran as a job; its test-half error comes from
   split_reference(ref, 42) with the same fingerprint as the sweep logs, and
   sits next to the Task 3 hard-BC run's error on the same test half.

PASS / FAIL each.

Setup, worker-log excerpts, study_summary.txt, retrain_log.txt:
<paste>
```

---

## Task 5 — Inverse problem (stretch)

**Lead prompt:**

```
Lab 3 Task 5 (stretch): recover E and nu with the inverse PINN
module_3/examples/plate_with_hole_inverse.py, run for seeds 0-9 by
pinn_inverse_ensemble.sbatch. Explain the script first, then help me aggregate.

Facts from lab_3.md Task 5:
- The plate is loaded by a PRESCRIBED DISPLACEMENT (u_1 = 1 on the right edge),
  so the displacement field does not depend on E at all -- only on nu. Stress,
  and so force, scales with E. Displacement data alone can identify nu but
  never E.
- So the script also uses ONE measured force: the reaction on the loaded edge,
  F = integral of sigma_11(1, x_2) dx_2 over 0 <= x_2 <= 1, read from the
  reference's reaction_right_x (--force-csv, default the shared
  plate_square_hole_reference_out.csv) and perturbed with the same 1 % noise
  as the displacements.
- Each seed picks 20 of the 200 points in measurements.csv, draws the noise on
  them and on the force, and sets the collocation points and the initial
  weights -- all at once.
- Each array task writes inverse_results_seed_<N>.txt (a header plus one CSV
  row: seed,E_recovered,nu_recovered,E_true,nu_true,n_measurements,noise_level)
  and inverse_variables_seed_<N>.dat (E and nu every 500 Adam steps).

My commands, exactly as lab_3.md 5a-5b (--array=0-9%2: ten seeds, at most two
GPUs at a time; nothing trains on the login node):

  cd "$(git rev-parse --show-toplevel)"
  mkdir -p labs/lab_3/<your-github-handle>/task5_inverse
  cd labs/lab_3/<your-github-handle>/task5_inverse
  cp ../../../../module_3/examples/plate_with_hole_inverse.py ../../../../module_3/examples/pinn_inverse_ensemble.sbatch .
  python /project/me5475/examples/extract_moose_reference.py --mode random --n-points 200 --output measurements.csv
  sbatch pinn_inverse_ensemble.sbatch

1. Explain how the force enters the loss (a sixth residual returned by the PDE
   function), how E and nu become trainable, and every term of the loss.
2. Write a short script that reads every inverse_results_seed_*.txt and writes
   ensemble_summary.txt with:
   - every run's seed, E and nu -- and any missing seed, with the reason from
     its job log (inverse-<jobid>_<N>.out / .err);
   - the mean, standard deviation (say which: ddof=0 or ddof=1) and range
     (min, max) of E and of nu;
   - a flag on every inadmissible run (E <= 0, or nu outside (0, 0.5)) and
     every failed run.
   Keep every run in the table and the statistics. Never drop a run because
   it lands far from the true values: that is a result.
3. Plot ensemble_plot.png: the ten (E, nu) estimates, with the truth
   (1.0, 0.3) and the ensemble mean marked.
4. Do not turn the spread into an interval with a probability attached (no
   "95 %", no mean +/- 2 std): ten runs are not enough to claim a calibrated
   95 % interval, and the spread mixes several sources of randomness.
5. Help me plan -- not run -- how I would separate measurement uncertainty
   from optimisation randomness.
```

**Review prompt:**

```
Read the lead's explanation, the aggregation script and its outputs. Verify
independently against plate_with_hole_inverse.py and
pinn_inverse_ensemble.sbatch:

1. The explanation says why a force is needed: under displacement loading the
   displacement field depends only on nu, so displacement data alone cannot
   identify E. FAIL any claim that displacements alone identify E.
2. The force comes from reaction_right_x in --force-csv (by default the
   shared reference CSV) and gets the same relative noise level as the
   displacements; each seed's log prints its "Force measurement" line.
3. The displacement noise standard deviation is 1 % of the *maximum absolute
   displacement* among the 20 chosen points (u_1 and u_2 together), the same
   for every measurement -- not 1 % of each individual measurement.
4. measurements.csv came from extract_moose_reference.py --mode random
   --n-points 200 --output measurements.csv (the shared reference), and each
   seed draws its own 20 points. Do not accept "measurements on the edges
   carry no information": only the prescribed components (u_1 on the left and
   right, u_2 on the bottom) are fixed there; the other components are free.
5. E_param and nu_param are passed via
   external_trainable_variables=[E_param, nu_param] to BOTH compile calls
   (Adam and L-BFGS), and the PDE closure uses the variables themselves
   (in DeepXDE 1.15 with PyTorch a dde.Variable is a tensor; there is no
   .value), so it never captures a stale copy.
6. The loss has 15 terms: 5 PDE residuals + the force mismatch (returned as a
   sixth PDE residual) + 7 traction BCs + 2 displacement-data terms. In the
   train-loss vectors the job log prints, the sixth entry (index 5) is
   (F_pred - F_meas)^2 and the last two are the u_1 and u_2 data misfits.
7. The aggregation reads every inverse_results_seed_<N>.txt, lists every run,
   reports mean, standard deviation (stating ddof) and range, flags
   inadmissible runs (E <= 0, or nu outside (0, 0.5)) and failed or missing
   seeds, and drops nothing.
8. For any run far from the truth, the lead checked whether its force and
   displacement fits were still good -- that separates an optimisation
   failure from a genuinely ambiguous fit.
9. The interpretation says each seed changes the measured points, the noise
   on them and on the force, the collocation points and the initial weights
   together, so the spread is their COMBINED effect, not the uncertainty of
   one experiment. FAIL any interval with a probability attached (e.g.
   mean +/- 2 std, "95 %"): ten runs are not enough for a calibrated 95 %
   interval.

PASS / FAIL each.

Explanation and aggregation script:
<paste>

ensemble_summary.txt:
<paste>
```

---

## Meta-tip — verifying PINN convergence

A useful third-agent role for Lab 3 is what you might call the "physics police":

**Physics-police prompt:**

```
I have a trained PINN model for the plate-with-hole problem. Apply the following
sanity checks independently:

1. At a point well in the interior, say (0.5, 0.5), the predicted u_1 should be
   between 0 and 1 (monotone interpolation between left and right edges). Run
   model.predict([[0.5, 0.5]]) and confirm.

2. At the top of the hole, (0, 0.1), the predicted sigma_11 should be close to
   the MOOSE reference value 3.2418 (module_3/readings/measured_results.md,
   section 1). Do NOT use the infinite-plate Kirsch factor of exactly 3: this is
   a finite plate pulled by a prescribed displacement. For scale, the average
   sigma_11 on the loaded (right) edge is 1.073 in the reference.

3. At the top edge (x_2 = 1), sigma_22 and sigma_12 should be near zero
   (traction-free BC).

For each check, report PASS/FAIL and the actual value. If any check fails, find
out why before training longer: first confirm the problem is posed right (all
ten BCs, plane-strain constants). A port missing three BCs reached a tiny
training loss with almost no stress concentration at the hole
(measured_results.md, section 2).
```

Use this prompt against your Task 1 / Task 3 outputs to confirm the PINN actually solved the physics, not just minimized loss.
