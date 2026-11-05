# Lab 6 — Microstructure to Modulus with a CNN

**Module:** 6 — CNNs for Field Prediction
**Released:** Thu Nov 5, 2026 (before Lecture 28)
**Due:** Mon Nov 16, 2026, 11:59 PM Mountain Time
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_6` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

> **Overlaps Lab 5 (due Wed Nov 11).** Lab 6's compute is short. In our test runs on ARCC (Nov 5), one MOOSE
> solve took about 1.5 s, so each of the ten data-generation workers finished its 50 RVEs in under a minute;
> a CNN training job on a GPU node took under a minute; the ten CPU sweep workers took a few minutes each. Queue time
> comes on top (`module_6/readings/measured_results.md` §1–§3). Start Task 1 any time; L28 (Fri Nov 6) covers convolutions, L29 (Mon Nov 9) this pipeline.

---

## What this lab does

You generate your own dataset with a finite-element code and train a network on it. Each sample is a 64 × 64 image
of a square cell, a **representative volume element** (RVE): a soft matrix (E = 1) with two to five stiff elliptical
inclusions (E = 5), both with ν = 0.3. For each image, MOOSE solves a plane-strain tension test of the cell and
averages σ_xx; we call the ratio ⟨σ_xx⟩ / 0.001 the cell's **apparent modulus E_app**.

**What E_app is, exactly.** The boundary conditions are mixed, not periodic: u_x = 0 on the left edge and
u_x = 0.001 on the right (both free in y), u_y = 0 along the bottom (free in x), and a traction-free top. So E_app is
the cell's apparent axial response under those conditions, not a periodic homogenization result, not the constrained
stiffness C₁₁ and not an unconstrained Young's modulus. For a homogeneous cell these conditions give σ_yy = 0 and
ε_zz = 0, so E_app = E / (1 − ν²) = 1.0989 E: **1.0989 for all matrix, 5.4945 for all inclusion**, and every
composite lies between the two (`rve_homogenization.i.template` explains why).

The question that matters is not only "is the error small?" but **"how does the network compare with simple
baselines?"** E_app of a dilute composite depends strongly on how much stiff phase there is. Task 2 makes you measure
that: compare the CNN, on the same test rows, with predicting the mean and with a straight line in the inclusion area
fraction. Whichever way it comes out, say what the comparison does and does not show.

---

## Before you start

- **Get the files.** New course files arrive on `main`: `git checkout main && git pull`
  (`module_0/readings/git_basics_and_course_repo_primer.md` explains why, and how to merge `main` into a lab branch).
- **Work on ARCC.** Data generation and training run as jobs on MedicineBow. Never run MOOSE or training on the
  login node.
- **Set your username once per terminal session.** Every command block below uses it:

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first: your GitHub username, no angle brackets
```

---

## Deliverables

```
labs/lab_6/<your-github-username>/
├── data_generation/
│   ├── generate_microstructures.py         your copy (possibly modified)
│   ├── rve_homogenization.i.template
│   ├── data_generation.sbatch
│   ├── aggregate_parts.py
│   ├── rve_dataset.npz                     500 RVEs, about 2 MB
│   ├── sample_microstructures.png          9 samples with their moduli (Task 1c)
│   ├── dataset_summary.txt                 Task 1d
│   └── prompts_data_gen.md
├── cnn_training/
│   ├── cnn_microstructure_to_modulus.py
│   ├── cnn_train.sbatch
│   ├── m6_cnn-<jobid>.out                  the training printout
│   ├── checkpoints/cnn_modulus.pt          trained model (about 0.4 MB)
│   ├── checkpoints/cnn_modulus.png         parity and loss plot
│   ├── baselines.py, baselines.txt         Task 2d
│   └── prompts_training.md
├── hp_sweep/
│   ├── cnn_optuna_sweep.py, cnn_optuna_sweep.sbatch, analyze_study.py
│   ├── cnn_opt.db                          the Optuna study
│   ├── study_summary.txt                   the analyze_study.py printout
│   ├── study_analysis/*.html               parallel coordinates, importances, history, slices
│   └── prompts_sweep.md
├── ood_test/
│   ├── ood_microstructures.py              your generator for the three subgroups
│   ├── ood_results.txt                     predicted vs true, per subgroup
│   ├── ood_comparison.png
│   └── prompts_ood.md
├── stretch_unet/                           (optional, +15 bonus; do not commit the field dataset)
│   ├── checkpoints/unet_stress.png
│   ├── m6_cnn-<jobid>.out
│   └── prompts_unet.md
└── reflection.md                           ≤ 500 words
```

---

## Tasks

### Task 1 — Generate the dataset on ARCC (20 points)

**1a.** Copy the scripts and submit the ten-worker array (50 RVEs each, 500 in all):

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first
cd ~/me5475-"$U" && git checkout main && git pull
git checkout -b lab_6
mkdir -p labs/lab_6/"$U"/{data_generation,cnn_training,hp_sweep,ood_test,stretch_unet}
cd labs/lab_6/"$U"/data_generation
cp ../../../../module_6/examples/{generate_microstructures.py,rve_homogenization.i.template,data_generation.sbatch,aggregate_parts.py} .
sbatch data_generation.sbatch
squeue -u $USER                    # ten array tasks
```

Each worker uses its own seed (42 + 1000 × worker number), so the ten parts are different RVEs. A failed MOOSE run is
skipped and the worker continues; `m6_rve-<jobid>_<k>.out` reports how many each worker wrote.

**1b.** When all ten tasks have finished (`sacct -j <jobid>`), combine the parts:

```bash
source /project/me5475/setup5475.sh
python aggregate_parts.py          # -> rve_dataset.npz, and prints the count and modulus statistics
```

**1c.** Plot nine random samples, each titled with its modulus. Commit `sample_microstructures.png`.

**1d.** In `dataset_summary.txt`, report the number of RVEs, the E_app mean, standard deviation and range, and
the range of the **inclusion area fraction** (the mean of each binary image; MOOSE sees a smoothed version of the same
inclusions). Check that every E_app lies between the pure-phase limits 1.0989 and 5.4945. Then answer: is the E_app
range wide or narrow compared with its mean, and what does that imply for an error target like "MAE below 5 % of the
mean"?

### Task 2 — Train the CNN and compare it with baselines (30 points)

**2a.** Train on a GPU:

```bash
cd ../cnn_training
cp ../../../../module_6/examples/{cnn_microstructure_to_modulus.py,cnn_train.sbatch} .
cp ../data_generation/rve_dataset.npz .
sbatch cnn_train.sbatch modulus     # -> m6_cnn-<jobid>.out and checkpoints/cnn_modulus.{pt,png}
```

The script splits the 500 RVEs 70/15/15 (350 train, 75 validation, 75 test) with `--split-seed` (default 42),
standardizes E_app on the training set, trains with early stopping on the validation loss, and prints the device it
used and the test MAE and MAE / mean(E_app). `--seed` changes only the initialization and the minibatch order; keep
`--split-seed` fixed for every run you compare, so all of them use the same test rows.

**2b.** Read the parity plot and the loss curves. Are the test points on the diagonal across the whole range, or
pulled toward the mean? Did training stop early, and why? (Look, but don't tune on the test set: Task 3 tunes on the
validation set.)

**2c.** Commit the checkpoint, the plot and the printout.

**2d. Baselines.** Write `baselines.py` that uses the **same split** as the training script
(`split_data(images, moduli, seed=42)` from `cnn_microstructure_to_modulus.py`) and reports the test MAE and
MAE / mean(E_app) of:

1. **predicting the training mean** for every test RVE;
2. **a straight line in the inclusion area fraction**, fitted by least squares on the training set.

Put the CNN beside them in `baselines.txt`. How does the CNN compare with the line on this metric and these 75
test rows? Keep the conclusion to what the comparison shows: doing better than a *straight* line does not by itself
prove the network uses information beyond the area fraction (a curve in the area fraction might do as well), and not
doing better does not say what the network learned. A nonlinear fraction-only baseline is an optional extra.

### Task 3 — Hyperparameter sweep with Optuna (15 points)

**3a.** Copy the sweep files and create the study from the login node, in `hp_sweep/`:

```bash
cd ../hp_sweep
cp ../../../../module_6/examples/{cnn_optuna_sweep.py,cnn_optuna_sweep.sbatch,cnn_microstructure_to_modulus.py} .
cp ../../../../module_2/examples/analyze_study.py .
cp ../data_generation/rve_dataset.npz .
source /project/me5475/setup5475.sh
optuna create-study --study-name lab6_cnn_sweep --storage sqlite:///cnn_opt.db --direction minimize
```

**3b.** Submit the sweep: ten workers, three trials each by default (30 trials; `N_TRIALS=5 sbatch …` for more).
The search space is in the header of `cnn_optuna_sweep.py`: learning rate, base channels, depth, weight decay,
batch size. Every worker uses the same split (`--split-seed 42`) and judges trials on the validation rows; the test
rows take no part. The workers run on CPU nodes. The median pruner stops unpromising trials early, so expect many
PRUNED trials.

```bash
sbatch cnn_optuna_sweep.sbatch
```

**3c.** When all ten workers have finished:

```bash
python analyze_study.py --study-name lab6_cnn_sweep --storage sqlite:///cnn_opt.db \
    --out-dir study_analysis | tee study_summary.txt
```

**3d.** Retrain with the best configuration as a job, in `cnn_training/`, with a new output name so the default
checkpoint is kept (example values; use yours from `study_summary.txt`):

```bash
cd ../cnn_training
sbatch cnn_train.sbatch modulus --lr 7e-4 --base-channels 16 --depth 3 --weight-decay 1e-5 --batch-size 32 \
    --out checkpoints/cnn_best.pt
sbatch cnn_train.sbatch modulus --seed 43 --out checkpoints/cnn_default_seed43.pt   # run-to-run variation
```

Compare the best configuration's test MAE with Task 2's default run and with the Task 2d line, all on the same test
rows. Is the difference larger than the change you see between the two default runs that differ only in `--seed`?

### Task 4 — Out of distribution (15 points)

**4a.** Write `ood_microstructures.py` that makes 30 microstructures from **different** distributions and computes
their true moduli with the same MOOSE template:

- 10 with **6–8 inclusions** (training had 2–5). `generate_microstructures.py` already takes
  `--min-inclusions` and `--max-inclusions`.
- 10 with **one large circular inclusion** (a different topology).
- 10 with **stripes** (a different geometry altogether).

The functions in `generate_microstructures.py` (`rasterize_ellipses`, `parsed_function_for_ellipses`,
`render_moose_input`, `run_moose_once`) can be imported; for stripes you write the image and the MOOSE expression
E(x, y) yourself. Run the MOOSE part as a job (copy the module and environment lines of `data_generation.sbatch`),
not on the login node.

**4b.** Load your Task 2 checkpoint, predict the 30 moduli (remember the model predicts **standardized** moduli: the
scalers are in the checkpoint), and report the MAE of each subgroup in `ood_results.txt`, beside the Task 2d line's
predictions for the same RVEs.

**4c.** `ood_comparison.png`: one example per subgroup with its true and predicted modulus.

### Task 5 (optional, +15 bonus) — U-Net for the full stress field

**5a.** Regenerate with fields (a separate folder; the field dataset is about 27 MB for 500 RVEs, so don't commit it):

```bash
cd ../stretch_unet
cp ../data_generation/{generate_microstructures.py,rve_homogenization.i.template,aggregate_parts.py} .
cp ../../../../module_6/examples/{unet_microstructure_to_stress.py,cnn_train.sbatch} .
sed 's/--seed \$SEED/--seed $SEED --save-stress-fields/' ../data_generation/data_generation.sbatch > data_generation.sbatch
sbatch data_generation.sbatch
# when it finishes:
python aggregate_parts.py          # rve_dataset.npz now also holds stress_fields (N, 3, 64, 64)
sbatch cnn_train.sbatch unet
```

**5b.** Commit `checkpoints/unet_stress.png` and the printout. Which stress component does the U-Net predict worst,
and why might that be?

### Task 6 — Reflection (15 points, ≤ 500 words)

1. **Cost.** How long did Task 1 take? Report elapsed time per worker and in total
   (`sacct -j <jobid> -o JobID,Elapsed,TotalCPU,AllocCPUS`), and say which you are quoting: elapsed (wall) time, the
   CPU time the processes actually used (TotalCPU), or allocated core-time (Elapsed × AllocCPUS). Time the trained
   CNN predicting 500 E_app values, and say on what (CPU or GPU, batch size). With these numbers, when is a
   surrogate worth building for this problem, and when would you just run MOOSE?
2. **The baselines.** From Task 2d and Task 4: how did the CNN compare with the area-fraction line, and what can
   and can't you conclude from that about what it learned?
3. **The sweep.** Which hyperparameter mattered most? Was the best configuration's test improvement real, given the
   seed-to-seed variation you measured?
4. **Out of distribution.** Which subgroup failed worst, and why? Did the area-fraction line do better or worse
   there?
5. **A better dataset.** If you could change one thing in `generate_microstructures.py` to make this a problem where
   a CNN clearly earns its keep, what would it be?

---

## Grading rubric (out of 100, +15 bonus)

| Component | Points |
|---|---|
| Task 1: dataset generated on ARCC, samples plotted, summary (1d) | 20 |
| Task 2: CNN trained, parity plot read, baselines compared (2d) | 30 |
| Task 3: Optuna sweep, analysis, retrain and comparison | 15 |
| Task 4: OOD test on three subgroups | 15 |
| Task 6: reflection, all five questions answered concretely | 15 |
| Prompts logged in each sub-folder | 5 |
| Task 5 (optional): U-Net stress field | +15 bonus |

---

## Hints

- **If a job ends within seconds**, read both `.err` and `.out`; the job scripts print their own errors to `.out`.
- **Sourcing `setup5475.sh` before `sbatch` is fine.** The job scripts were fixed on Nov 5 so that they work whether
  or not you did.
- **The area fraction** of an image is `img.mean()`. It is a property of the image, not of the network, so it is a
  fair baseline. (MOOSE sees a smoothed version of the inclusions, so the FE stiff-phase fraction is close to, not
  equal to, the image's.)
- **Run-to-run variation.** With 75 test RVEs, small differences in test MAE can be noise. Compare against a rerun
  with a different seed before calling one configuration better. On a GPU, even two runs with the same seeds can
  differ a little (some GPU kernels are not deterministic): that is part of the variation too.
- **Loading a checkpoint.** The checkpoints store CPU copies of the weights, so `torch.load(path, weights_only=False)`
  works on a login node or a CPU job as well as on a GPU.
- **Out of distribution, the CNN still returns a number.** Nothing in the network says "I have not seen this". That
  is the point of Task 4.

---

## What this lab leads into

- **L30 (Wed Nov 11)** turns the image-to-number network into an image-to-image one (the U-Net of Task 5).
- **Module 7** (operator learning) maps functions to functions; the same RVEs can be used to compare approaches.
- **Final project.** Microstructure surrogates are a common final-project starting point.

---

## Academic integrity

Agents are encouraged. Logged prompts in each sub-folder are mandatory. Your dataset, networks and numbers must be
your own runs; copying a classmate's files is plagiarism. Discussing strategy with classmates is fine and encouraged.

---

## How to submit

A pull request from branch `lab_6`, with your files under `labs/lab_6/<your-github-username>/`
(`module_0/readings/git_submission_guide.md` walks through it).

```
U=YOUR_GITHUB_USERNAME            # EDIT first, as above
cd ~/me5475-"$U"                  # the repository root: the paths below start here
git checkout lab_6                # the branch you made in Task 1a
printf 'm6_rve-*.err\n__pycache__/\n*/rve_dataset_part_*.npz\nstretch_unet/rve_dataset*.npz\nstretch_unet/checkpoints/*.pt\n' > labs/lab_6/"$U"/.gitignore
git add labs/lab_6/"$U"
git commit -m "Lab 6: what you did"
git push -u origin lab_6
# then open the pull request on github.com -- that PR is the submission
```
