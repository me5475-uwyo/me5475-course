# Lab 7 — DeepONet and FNO for 1-D Burgers

**Module:** 7 — Operator Learning
**Released:** Mon Nov 16, 2026 (before Lecture 32)
**Due:** Mon Nov 23, 2026, 11:59 PM Mountain Time
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_7` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

> **Overlaps Lab 6 (due tonight, Mon Nov 16) and Lab 8 (released Wed Nov 18).** Lab 7's compute is small and needs
> no MOOSE: the Burgers data are made by a short NumPy solver. In our test runs on ARCC CPU nodes (Nov 12–13), making
> the 1000 training pairs took 1.5–2 minutes and training the DeepONet 2–3.5 minutes; each training job fits well
> inside its time limit. Queue time comes on top (`module_7/readings/measured_results.md`). L31 (Fri Nov 13)
> covered the DeepONet and its demo; L32 (Mon Nov 16) covers the FNO.

---

## What this lab does

Train a DeepONet and a Fourier Neural Operator (FNO) on the same operator, the 1-D viscous Burgers equation

u_t + u u_x = ν u_xx,  x in [0, 1) periodic,  t in [0, 1],  ν = 0.01,

learning the map from the initial condition u₀(x) to the solution u(x, 1) (L31 slides 5–6). Compare the two models:
accuracy, cost, and what happens when you **evaluate them on grids they were not trained on**. The FNO is *designed*
to run on any grid, because its weights are indexed by Fourier mode rather than by grid point (L32); the DeepONet's
branch reads a fixed set of 128 sensors, so on another grid its input must be interpolated onto those sensors.
Whether, and how well, each model keeps its accuracy across grids is what Task 4 measures: there is no answer given
here. The optional stretch task applies a 2-D FNO to your Lab 6 stress fields and compares it with your U-Net.

---

## Before you start

- **Get the files.** New course files arrive on `main`: `git checkout main && git pull`. You need, in
  `module_7/examples/`: `generate_burgers_data.py`, `burgers_data.sbatch`, `deeponet_burgers.py`, `fno_burgers.py`,
  `operator_train.sbatch`, and for the stretch task `fno_microstructure.py`. Readings are in `module_7/readings/`
  (`deeponet_primer.md`, `fno_primer.md`, `reading_list.md`); `module_7/homework/starter_prompts.md` has agent prompts.
- **Work on ARCC.** Data generation and training run as jobs on MedicineBow (CPU partition `mb`: the Burgers scripts
  do not use a GPU). Never run them on the login node.
- **Set your username once per terminal session:**

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first: your GitHub username, no angle brackets
```

---

## Deliverables

```
labs/lab_7/<your-github-username>/
├── data/
│   ├── generate_burgers_data.py, burgers_data.sbatch
│   ├── burgers_data.npz                       1000 (u0, u(x,1)) pairs at N = 128, about 1 MB
│   ├── data_samples.png                       four pairs
│   └── prompts_data.md
├── deeponet/
│   ├── deeponet_burgers.py, operator_train.sbatch
│   ├── deeponet_burgers.pt, deeponet_burgers.png
│   ├── m7_op-<jobid>.out                       the training printout
│   └── prompts_deeponet.md
├── fno/
│   ├── fno_burgers.py, operator_train.sbatch
│   ├── fno_burgers.pt, fno_burgers.png
│   ├── m7_op-<jobid>.out
│   ├── comparison.md                          Task 3c table
│   └── prompts_fno.md
├── resolution_test/
│   ├── test_resolutions.py                    your script (Task 4b)
│   ├── resolution_results.txt                 relative L2 of each model at N = 64, 128, 256, 512
│   ├── resolution_plot.png
│   └── prompts_resolution.md
├── stretch_fno_micro/                          (optional, +15)
│   ├── fno_microstructure.py, operator_train.sbatch
│   ├── fno_micro.png, m7_op-<jobid>.out       (fno_micro.pt and the Lab 6 dataset are not committed)
│   └── prompts_stretch.md
└── reflection.md                               ≤ 500 words
```

---

## Tasks

### Task 1 — Generate the Burgers data (15 points)

**1a.** Copy the scripts and submit the data job (1000 pairs at N = 128, seed 42; about 1.5–2 minutes):

```bash
U=YOUR_GITHUB_USERNAME            # EDIT first
cd ~/me5475-"$U" && git checkout main && git pull
git checkout -b lab_7
mkdir -p labs/lab_7/"$U"/{data,deeponet,fno,resolution_test,stretch_fno_micro}
cd labs/lab_7/"$U"/data
cp ../../../../module_7/examples/{generate_burgers_data.py,burgers_data.sbatch} .
sbatch burgers_data.sbatch
squeue -u $USER
```

When it has finished, `m7_data-<jobid>.out` should end with `Wrote 1000 pairs to burgers_data.npz`. The file holds
`u0` and `uT` (1000 × 128 each), the grid `x_grid` (x_j = j/128) and the settings.

**1b.** Plot four random pairs (u₀ dashed, u(x, 1) solid, one panel each) and commit `data_samples.png`. Check that
each u₀ is a smooth random function with |u₀| ≤ 1, and describe where u(x, 1) has steepened into a **front** (a narrow
zone where u drops sharply, L31 slide 5). Does a front form where u₀ was decreasing or increasing, and why?

### Task 2 — Train the DeepONet (25 points)

**2a.** Train it (500 epochs; one epoch is one pass over the 800 training functions):

```bash
cd ../deeponet
cp ../../../../module_7/examples/{deeponet_burgers.py,operator_train.sbatch} .
cp ../data/burgers_data.npz .
sbatch operator_train.sbatch deeponet
```

**2b.** From `m7_op-<jobid>.out` report the parameter count, the final test relative L² error, and the training
time (the `training:` line). Count parameters as **real numbers** (the DeepONet's are all real; the FNO's printout
gives both conventions, see 3c). The relative L² error is the scripts' pooled definition: ‖pred − true‖ / ‖true‖ over all
200 × 128 test values. With the default data and seed your run reproduces the L31 demo (about 8.7 %), up to the last
digits.

**2c.** Read `deeponet_burgers.png`: where on each test curve is the prediction good, and where is it not?

### Task 3 — Train the FNO (25 points)

**3a.** Train it on the same data (500 epochs, 20 functions per batch):

```bash
cd ../fno
cp ../../../../module_7/examples/{fno_burgers.py,operator_train.sbatch} .
cp ../data/burgers_data.npz .
sbatch operator_train.sbatch fno
```

**3b.** Report the same three numbers as 2b, and read `fno_burgers.png` the same way as 2c.

**3c.** In `comparison.md`, put the two models side by side: test relative L² error, parameter count in real
numbers (each complex FNO weight is two: its real and imaginary parts; the FNO printout also gives the tensor-entry
count, which counts it once), training time
(both were CPU jobs on partition `mb`; say which nodes ran them, from `sacct -j <jobid> -o JobID,Elapsed,NodeList`,
since nodes differ in speed). Then say what the comparison does and does not show: one run each, one dataset, one
set of hyperparameters.

### Task 4 — Cross-resolution test (25 points)

Both models were trained at N = 128. Now evaluate them, **without retraining**, at N = 64, 128, 256 and 512.

**4a. Test sets.** Make 100 fresh pairs at each N, with a seed that is **not 42** (seed 42 at N = 128 would reproduce
your training pairs). Keep ν, T and the length scale at their defaults so the test functions come from the same
family. For example:

```bash
cd ../resolution_test
cp ../data/{generate_burgers_data.py,burgers_data.sbatch} .
cp ../deeponet/{deeponet_burgers.py,deeponet_burgers.pt} ../fno/{fno_burgers.py,fno_burgers.pt} .
for N in 64 128 256 512; do
  sbatch burgers_data.sbatch --n 100 --N $N --seed $((1000 + N)) --out test_N$N.npz
done
```

The solver takes smaller time steps on finer grids to stay stable, so N = 512 takes a few minutes. Its accuracy was
checked for exactly this family (smooth random inputs, ν = 0.01, T = 1, N = 64 to 512); it is not certified beyond it.
Each N gets its own random test functions, so differences across N mix the change of grid with a change of test
samples: keep that in mind in 4c.

**4b. Evaluate.** Write `test_resolutions.py` (agents welcome; log the prompts) that, for each N:

1. loads `test_N$N.npz` and both trained checkpoints, rebuilding each model from the `args` saved in its checkpoint
   (import `DeepONet` and `FNO1d` from the two training scripts);
2. **DeepONet:** interpolates each u₀ from its N points onto the 128 training sensors x_j = j/128 (periodic linear
   interpolation, e.g. `numpy.interp` with the first point repeated at x = 1), and asks the trunk for the answer at
   the N test points x_j = j/N;
3. **FNO:** applies the model to the N-point input directly;
4. computes the pooled relative L² error of each model against the test set's `uT`.

Run it as a job, not on the login node: copy `burgers_data.sbatch` to `resolution.sbatch` and change its `srun` line
to `srun python test_resolutions.py`. Commit `resolution_results.txt` (one row per N, both models) and
`resolution_plot.png` (error against N, log–log axes).

**4c. Explain.** Describe what you see for each model, and explain it from how each model reads its input and
produces its output. Is any change with N larger than the differences you would expect from sampling alone? One way
to judge: make a second test set at N = 128 with another seed and see how much the error moves. (Retraining with
`--seed 7` changes the train/test split as well as the initialization, so it measures something broader.)

### Task 5 (optional, +15 bonus) — FNO on your Lab 6 stress fields

This needs your Lab 6 Task 5 field dataset (`labs/lab_6/<you>/stretch_unet/rve_dataset.npz`, not committed) and your
U-Net checkpoint (`stretch_unet/checkpoints/unet_stress.pt`).

```bash
cd ../stretch_fno_micro
cp ../../../../module_7/examples/{fno_microstructure.py,operator_train.sbatch} .
cp ../../../../module_6/examples/unet_microstructure_to_stress.py .      # for the U-Net comparison
cp ../../../lab_6/"$U"/stretch_unet/rve_dataset.npz .
cp ../../../lab_6/"$U"/stretch_unet/checkpoints/unet_stress.pt .
sbatch operator_train.sbatch fno-micro --data rve_dataset.npz --unet-ckpt unet_stress.pt
```

`fno_microstructure.py` uses the **same split, standardization, loss, epochs and test metric** as the U-Net script,
so the two test errors it prints refer to the same test cells, in physical units. It pads the cell before the Fourier
layers because the Lab 6 cells are not periodic. This is the longest job in the lab: on a 40-cell test set it took
3 minutes on a CPU node, and the time grows with the number of training cells (your 350), so the job allows two
hours. Commit `fno_micro.png` (microstructure | MOOSE σ_xx | FNO | U-Net for
three test cells) and the printout. Which model is better on which stress component, and is the difference larger than
a rerun would change?

### Task 6 — Reflection (5 points, ≤ 500 words)

1. **Accuracy.** Which model had the lower test error on Burgers, and by how much? Is one run each enough to say so?
2. **Cost.** How did training times and parameter counts compare (Task 3c)? Was the more accurate model also the
   more expensive one?
3. **Resolution.** What did Task 4 show for each model, and how does it follow from the architectures (L32's "the
   weights are indexed by mode"; L31's fixed sensors)? How far did interpolation get the DeepONet?
4. **When to use which.** From Tasks 2–4, state a rule for choosing a DeepONet or an FNO for a new mechanics problem
   (think about grids, geometry and boundary conditions).
5. **Connection to M6.** If you did the stretch: did the FNO beat your U-Net on the microstructure problem? Why might
   it, or might it not? If you did not: what would you expect, and why?

---

## Grading rubric (out of 100, +15 bonus)

| Component | Points |
|---|---|
| Task 1: data generated on ARCC, samples plotted, fronts described | 15 |
| Task 2: DeepONet trained and read | 25 |
| Task 3: FNO trained, comparison table and its limits | 25 |
| Task 4: cross-resolution test, results, plot and explanation | 25 |
| Task 6: reflection, all five questions answered concretely | 5 |
| Prompts logged in each sub-folder | 5 |
| Task 5 (optional): FNO on the Lab 6 stress fields | +15 bonus |

---

## Hints

- **If a job ends within seconds**, read both `.err` and `.out`. The job scripts `source /project/me5475/setup5475.sh`
  themselves, so it does not matter whether you sourced it before `sbatch`.
- **Arguments after the variant go to the script:** `sbatch operator_train.sbatch fno --modes 8 --out fno_m8.pt`.
  Use a new `--out` name for each variant so you do not overwrite your main result.
- **`train=` in the DeepONet and FNO printouts** is the loss of the last mini-batch of that epoch, not an epoch
  average; `test=` and `rel-L2=` use all 200 test functions.
- **FNO `modes`.** The spectral layer keeps at most N//2 + 1 modes (the script guards this). With the default 16 modes
  every grid in Task 4 has enough.
- **Loading a checkpoint.** `torch.load(path, map_location="cpu", weights_only=False)` returns a dictionary with the
  weights (`model_state`) and the training arguments (`args`); rebuild the model from `args` before
  `load_state_dict`.
- **Interpolating periodic data.** For `numpy.interp`, append the first value at x = 1 so the interpolation wraps
  around: `np.interp(x128, np.r_[xN, 1.0], np.r_[u, u[0]])`.

---

## What this lab leads into

- **L33 (Wed Nov 18)** compares PINNs, DeepONets, FNOs and plain FE, and how to choose among them.
- **Module 8** (graph neural networks, from Fri Nov 20) handles irregular meshes, where an FFT grid does not exist.
- **Final project.** Many projects use an FNO or a DeepONet; Lab 7 is the foundation.

---

## Academic integrity

Agents are encouraged. Logged prompts in each sub-folder are mandatory. Your data, networks and numbers must be your
own runs; copying a classmate's files is plagiarism. Discussing strategy with classmates is fine and encouraged.

---

## How to submit

A pull request from branch `lab_7`, with your files under `labs/lab_7/<your-github-username>/`
(`module_0/readings/git_submission_guide.md` walks through it).

```
U=YOUR_GITHUB_USERNAME            # EDIT first, as above
cd ~/me5475-"$U"                  # the repository root: the paths below start here
git checkout lab_7                # the branch you made in Task 1a
printf 'm7_*.err\n__pycache__/\nstretch_fno_micro/rve_dataset.npz\nstretch_fno_micro/*.pt\n' > labs/lab_7/"$U"/.gitignore
git add labs/lab_7/"$U"
git commit -m "Lab 7: what you did"
git push -u origin lab_7
# then open the pull request on github.com -- that PR is the submission
```
