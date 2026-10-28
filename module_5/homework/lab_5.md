# Lab 5 — Your Lab 1 Network as a Material Inside MOOSE

**Module:** 5 — NN-FEA Integration with MOOSE
**Released:** Tue Oct 27, 2026 (the day before Lecture 24)
**Due:** Wed Nov 11, 2026, 11:59 PM Mountain Time (the day Lecture 30 is taught)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request from branch `lab_5` in your instructor-created private repository in the `me5475-uwyo` course organization (`me5475-uwyo/me5475-<your-github-username>`).

> **Overlaps Lab 4 (due Fri Nov 6) and the midterm (due Sun Nov 8, extended from Nov 1).** Lab 5 is deliberately small, and its compute
> is short: the export and checks take seconds, and each core MOOSE job took about a minute in our reference runs at
> the default mesh, not counting the queue; the optional Task 5 takes longer (`module_5/readings/measured_results.md`,
> cited below as §n).
> Start with Task 1 any time; Tasks 2–4 follow L25–L27.

---

## What this lab does

In Lab 1 you trained an MLP that maps plane-strain Voigt strain (ε_xx, ε_yy, γ_xy) to stress (σ_xx, σ_yy, σ_xy). In
Lab 5 that network becomes the **material of a finite-element simulation**. The course MOOSE executable has a material,
`ComputeTorchScriptStress`, that loads a TorchScript file and evaluates your network at every quadrature point;
Newton's method uses the network's exact derivative, computed by automatic differentiation.

You will run Lab 0's plate with a hole twice, on the same mesh: once with your network and once with Lab 0's elastic
material. Then you compare the two fields, look at what the network costs, and load the plate outside the strains
the network was trained on to see what a learned material does there.

Two things to know before you start:

- **The load is scaled.** Lab 0 used σ∞ = 1 with E = 1, which strains the plate up to about 2.4 at the hole on the
  Lab 5 mesh. Your network was trained on strains of a few times 1e-3. The Lab 5 input applies σ∞ = 1e-3. With the
  course reference network, the final element-average strains of all but a few elements at the hole then lie inside
  the convex hull of the strains in the Lab 1 CSV (§3.1). That is a coverage proxy, not a guarantee; Task 1d and
  Task 3a say why. The elastic problem is linear: its answer just scales by 1e-3.
- **The comparison is between two MOOSE runs.** Lab 3's PINN solved a different plate (a unit square with a
  prescribed displacement), so it is compared only on that plate, in the optional Task 5.

---

## Before you start

- **Your Lab 1 files:** `labs/lab_1/<you>/training/checkpoints/mlp_constitutive.pt` (the checkpoint) and
  `labs/lab_1/<you>/data_generation/data/single_element.csv` (the data it was trained on).
- **On ARCC:** `git pull` in your repository for the `module_5/` files, and `source /project/me5475/setup5475.sh`
  in every session (it gives you the course Python with torch 2.5.1).
- **The executable:** `/project/me5475/software/moose_torch/me5475/me5475-opt`. It is the course MOOSE build that
  has the network material; `run_nn_simulation.sbatch` already names it. The older `rom_opt-opt` does not have the
  material.

**Course reference network** (read-only; copy from, never write into): `/project/me5475/examples/lab5_reference/`
holds a Lab 1 network the course trained with the unchanged Lab 1 scripts and the Lab 1 MOOSE data
(`mlp_constitutive.pt`), its training data (`single_element.csv`), its TorchScript export, `README.txt` and
`SHA256SUMS`. Use your own network. If your Lab 1 network cannot be used (no checkpoint, or it fails Task 1c for
reasons you cannot fix), you may copy the reference checkpoint and CSV and do every task with them. If you do, say
so in `export/README.md`: the path you copied from, the SHA-256 of each file you copied (`sha256sum <file>`), whether
you modified anything, and why you used the fallback.

---

## Deliverables

```
labs/lab_5/<your-github-handle>/
├── export/
│   ├── export_torchscript.py               your copy (possibly modified)
│   ├── verify_torchscript.py
│   ├── mlp_constitutive_ts.pt              your TorchScript file (about 26 KB)
│   ├── export_output.txt                   what the export printed
│   ├── verify_output.txt                   what the verify script printed
│   ├── README.md                           only if you used the course reference network: path, SHA-256s, modified or not, why
│   └── prompts_export.md
├── moose_run/
│   ├── plate_with_hole_nn_material.i       your copy (possibly modified)
│   ├── run_nn_simulation.sbatch
│   ├── nn_moose-<jobid>.out                the printout of each job: Task 2, and Task 4's three loads
│   ├── plate_nn_out.csv                    postprocessors, network run (and the _s1, _s2e-3, _s1e-4 CSVs of Task 4)
│   ├── plate_elastic_out.csv               postprocessors, elastic run
│   └── prompts_moose.md
├── comparison/
│   ├── comparison_three_methods.py
│   ├── comparison.png                      elastic | network | difference, four fields
│   ├── comparison_metrics.txt
│   └── prompts_comparison.md
├── extrapolation/
│   ├── comparison_metrics_s2e-3.txt, comparison_s2e-3.png
│   ├── comparison_metrics_s1e-4.txt, comparison_s1e-4.png
│   └── prompts_extrapolation.md            4a-4c answered
├── task5_square_plate/                     (optional, +15 bonus)
│   ├── plate_square_hole_nn.i
│   ├── nn_moose-<jobid>.out
│   ├── comparison_square.png
│   ├── comparison_square_metrics.txt
│   └── prompts_task5.md
└── reflection.md                           <= 500 words
```

Do **not** commit the Exodus files (`*.e`, 2.8 MB each at the default mesh, 6.6 MB in Task 5), the MOOSE `.log`
and `.time` files, or Task 5's grid CSVs (4–7 MB each); the postprocessor CSVs, the `.out` files, the figures and the
metrics are the record. The `.gitignore` line in "How to submit" keeps them out.

---

## Tasks

### Task 1 — Export your network and check the file (15 points)

**1a.** Set up the folders and copy your Lab 1 checkpoint and data next to the two scripts:

```bash
cd ~/me5475-<your-github-username> && git pull
mkdir -p labs/lab_5/<you>/{export,moose_run,comparison,extrapolation}
cd labs/lab_5/<you>/export
cp ../../../../module_5/examples/export_torchscript.py ../../../../module_5/examples/verify_torchscript.py .
cp ../../../lab_1/<you>/training/checkpoints/mlp_constitutive.pt .
cp ../../../lab_1/<you>/data_generation/data/single_element.csv .
source /project/me5475/setup5475.sh
```

**1b.** Export (seconds; the login node is fine):

```bash
python export_torchscript.py --checkpoint mlp_constitutive.pt | tee export_output.txt
```

It rebuilds your network from the checkpoint, wraps it with your Lab 1 standardization (so the file takes raw strain
and returns raw stress), traces it, and writes `mlp_constitutive_ts.pt`. That is a **new** file, not your checkpoint.
It then reloads the file and checks it against your network, in float32 and in float64 (MOOSE runs it in float64),
and stops with an error if either check fails.

**1c.** Check the file:

```bash
python verify_torchscript.py --model mlp_constitutive_ts.pt --data single_element.csv \
    --checkpoint mlp_constitutive.pt | tee verify_output.txt
```

It prints four checks: the stress at 10 test strains against plane-strain Hooke's law (E = 1, ν = 0.3), the stress at
zero strain, the autograd tangent MOOSE will use against the stiffness C, and the file against your eager model.
- **What decides the result:** the Hooke tests and the file-vs-model check.
  - With `--data`, a test strain outside the convex hull of all the strains in your CSV is reported but not judged.
    If fewer than 5 of the 10 tests are judged, the file is **not validated**, and the result is FAIL.
  - With `--checkpoint`, the file must reproduce your network in float32 and in float64.
  - Any non-finite stress or tangent is a FAIL.
- **Diagnostics only:** the stress at zero strain and the tangent are printed but do not decide the result.

The script ends with `RESULT: PASS` or `RESULT: FAIL`.

What to expect, from the course reference network (`measured_results.md` §2):
- `RESULT: PASS`, every test within 1.5 % of Hooke; one of the ten test strains (test 6) outside the hull of the
  CSV's 200 strains, so 9 judged;
- σ(0) of order 1e-6 (not exactly 0);
- the tangent at zero strain within 1.1 % of C (Frobenius norm), the same matrix Lab 1 Task 4 printed;
- the file identical to the eager model (difference 0 in float32 and in float64).

**1d.** Answer in `prompts_export.md`:
- Which of the ten test strains lie outside the hull of your CSV's strains?
- What ranges of ε_xx, ε_yy, γ_xy does your CSV actually cover (the verify script prints them)? Compare them with
  the range Lab 1's driver asks for, [−0.005, 0.005]. The CSV holds what MOOSE's element carried.
- The CSV also holds Lab 1's validation and test rows. With the defaults, Lab 1 trained on 140 of the 200 rows
  (a 70/15/15 split), so the hull the verify script uses is larger than the hull of the rows your network was
  trained on. With the course data, test 9 is inside the full hull but outside the 140 training rows' hull (§2).
  Why is the full-data hull still a useful check, and what can it miss?

### Task 2 — Run the plate with your network on ARCC (20 points)

**2a.** Copy the input file and the job script, and your TorchScript file:

```bash
cd ../moose_run
cp ../../../../module_5/examples/plate_with_hole_nn_material.i ../../../../module_5/examples/run_nn_simulation.sbatch .
cp ../export/mlp_constitutive_ts.pt .
```

**2b.** Read `plate_with_hole_nn_material.i` next to Lab 0's `plate_with_hole.i`. The mesh generator, the boundary
conditions and the solver are Lab 0's; the header lists what differs. Find each difference in the file: the
`[Materials]` block (both materials are there; `active` picks one), `sigma_inf`, `refine`, `nl_max_its`, the
postprocessors added for Lab 5, and the action's current name. The default mesh is `refine = 3` (12 288 elements);
`refine=4` on the command line gives Lab 0's finest mesh, at about four times the cost of the network run (§3).

**2c.** Submit:

```bash
sbatch run_nn_simulation.sbatch
squeue -u $USER
```

The job checks that the executable has the network material and that your file exists, then runs the plate with your
network (`plate_nn_out.*`) and with the elastic material (`plate_elastic_out.*`). Ours took 48–49 s for both runs
(the network run 43 s, the elastic run 3 s), plus the queue wait (§3).

**2d.** In `nn_moose-<jobid>.out`, find and record in `prompts_moose.md`:
- the Newton iterations of each run (`nl_its`, and the `Nonlinear |R|` lines). Why does the elastic run need one?
- `reaction_x_left` in both runs. What should it be, and why does it not depend on the material?
- the `min_/max_strain_*` postprocessors of the network run, against your CSV's ranges from 1d
  (γ_xy = 2 × strain_xy).

### Task 3 — Compare the two runs (25 points)

**3a.** Run the comparison (seconds):

```bash
cd ../comparison
cp ../../../../module_5/examples/comparison_three_methods.py .
python comparison_three_methods.py --elastic ../moose_run/plate_elastic_out.e --nn ../moose_run/plate_nn_out.e \
    --train-data ../export/single_element.csv
```

Both runs are on the same mesh, so the script compares displacements node by node and stresses element by element.
It reports:
- area-weighted relative L2 errors (for u, lumped nodal areas approximate the finite-element integral);
- the largest pointwise difference;
- every postprocessor, side by side;
- a **coverage proxy:** how many elements' final element-average strains lie inside the convex hull of all the
  strains in your CSV.

A proxy, because the material evaluates your network at quadrature points, whose strains differ from the element
average, and at every Newton iterate, which the script does not see. It writes `comparison.png` and
`comparison_metrics.txt`.

**3b.** Report the relative L2 errors of u, σ_xx, σ_yy, σ_xy and von Mises (`vonmises_pe`), and the peak σ_xx at the
hole (`max_stress_xx`) in both runs. Which field has the largest error, and which the smallest? (The course's numbers
are added to `measured_results.md` §3 after the lab closes, Wed Nov 11.)

**3c.** Compare the displacement error with the σ_xx error, and explain the difference. Two hints: compare the displacement
error with your network's stiffness error from Lab 1 Task 4 (is your network a little too soft, or too stiff?); and
ask how the stress field of a plate loaded only by tractions depends on E. Where in the plate are σ_yy and σ_xy
errors largest relative to their own size, and why are their relative errors larger than σ_xx's?

**3d.** Compare `max_vonmises_stress` with `max_vonmises_pe` in both runs. One of them differs between the runs by
much more than the other. Which, and why? (`module_5/readings/moose_material_interface_reference.md`, "σ_zz, plane
strain, and von Mises".)

**3e.** Cost: the run times (the `wall` lines in the `.out`), Newton iterations and residual evaluations of the two
runs. What does the network do at every quadrature point in every residual evaluation that the elastic material does
not?

### Task 4 — Outside the training data (15 points)

**4a.** Run Lab 0's actual load, σ∞ = 1, into separate files:

```bash
cd ../moose_run
TAG=_s1 sbatch run_nn_simulation.sbatch sigma_inf=1
```

`TAG` renames the outputs (`plate_nn_s1_out.*`, …); the extra argument goes to the run. Expect this job to **fail**,
in under a minute: with the course reference network, Newton took one step, the next linear solve failed with a zero
pivot, and the material then stopped the run because the network returned a non-finite stress (§4). The job stops
after the network run, so there is no elastic `_s1` run. Read the end of `plate_nn_s1_out.log` and explain the
failure. To see why, evaluate your network at growing strains (seconds, in `export/`):

```bash
python -c "
import torch; m = torch.jit.load('mlp_constitutive_ts.pt').double().eval()
for k in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
    print(k, m(torch.tensor([[k, 0.0, 0.0]], dtype=torch.float64)).tolist()[0])"
```

What does a tanh network return far outside its data along this strain direction, and what stiffness does it then
give Newton? What does this one-direction probe *not* tell you, about other strain states and about whether the
plate at σ∞ = 1 has an equilibrium with this material at all? (A failed Newton solve shows that Newton did not
converge with these settings, not that no solution exists.)

**4b.** Just outside the data, the solve still converges. Run σ∞ = 2e-3 and, once the job has finished, compare:

```bash
TAG=_s2e-3 sbatch run_nn_simulation.sbatch sigma_inf=2e-3
# ... wait until squeue -u $USER no longer lists the job ...
cd ../extrapolation
python ../comparison/comparison_three_methods.py --elastic ../moose_run/plate_elastic_s2e-3_out.e \
    --nn ../moose_run/plate_nn_s2e-3_out.e --train-data ../export/single_element.csv \
    --out comparison_s2e-3.png --metrics comparison_metrics_s2e-3.txt
```

With the course reference network the solve still converged, in 5 Newton iterations (§4). What fraction of your
plate is inside your CSV's hull now (the coverage line), which fields got worse compared with Task 3, by how much,
and where in the plate?

**4c.** Now the other direction, σ∞ = 1e-4 (`TAG=_s1e-4`, the same commands with `1e-4`). Check the coverage lines
of the metrics again. Did the displacement and stress errors shrink or grow compared with Task 3? Explain what
you see with σ(0) from Task 1c.

Avoid loads between 5e-3 and 1e-1 (the ones we tried) unless you want to watch Newton struggle: with no iteration
limit those runs stalled or failed their line search after 13–50 iterations and about 9 to 25 minutes each (§4).
The input file now stops Newton after 15 iterations.

### Task 5 (optional, +15 bonus) — Three methods on Lab 3's plate

Lab 3's plate (unit square minus a quarter hole, u₁ = 1 on the right edge) has a shared MOOSE reference and your Lab 3
PINN. Add your network:

**5a.** Make the folder and copy Lab 3's reference input under a new name:

```bash
mkdir -p ../task5_square_plate && cd ../task5_square_plate
cp /project/me5475/examples/plate_square_hole_reference.i plate_square_hole_nn.i
```

Replace its `[Materials]` block with:

```
[Materials]
  active = 'nn_stress'
  [nn_stress]
    type     = ComputeTorchScriptStress
    filename = mlp_constitutive_ts.pt
  []
  [elasticity_tensor]
    type           = ComputeIsotropicElasticityTensor
    youngs_modulus = 1.0
    poissons_ratio = 0.3
  []
  [elastic_stress]
    type = ComputeLinearElasticStress
  []
[]
```

Then copy the job script and your TorchScript file, and run with the displacement scaled to 1e-3 on the 48/96
mesh, starting Newton from the uniform stretch u_x = 1e-3·x:

```bash
cp ../moose_run/run_nn_simulation.sbatch ../export/mlp_constitutive_ts.pt .
INPUT=plate_square_hole_nn.i TAG=_sq sbatch run_nn_simulation.sbatch nt=48 nr=96 BCs/right_ux/value=1e-3 \
    ICs/ux/type=FunctionIC ICs/ux/variable=disp_x "ICs/ux/function=1e-3*x"
```

Ours took 1 min 29 s for both runs (the network run 78 s) (§6). The last three arguments are the initial guess. Without them Newton starts from u = 0 with
u_x = 1e-3 imposed on the right edge only, so the first iterate strains the elements next to that edge far outside the
network's data: our run then needed 15 Newton iterations and about 6 minutes, and at u_x = 5e-4 it failed (§6).
Explain in 5e why the elastic run does not care.

**5b.** Sample the network run on Lab 3's grid, and get the reference grid (Lab 3 Task 2a):

```bash
python /project/me5475/examples/extract_moose_reference.py --exodus plate_nn_sq_out.e --mode grid --nx 200 --ny 200 --output nn_grid.csv
python /project/me5475/examples/extract_moose_reference.py --mode grid --nx 200 --ny 200 --output reference_grid.csv
```

**5c.** Write your Lab 3 PINN's prediction on the same grid. At the end of your Lab 3 `comparison_script.py`
(Task 2b), after `pred = model.predict(...)`, add:

```python
pd.DataFrame({"x_1": ref["x_1"], "x_2": ref["x_2"], "u_1": pred[:, 0], "u_2": pred[:, 1],
              "sigma_11": pred[:, 2], "sigma_22": pred[:, 3], "sigma_12": pred[:, 4]}).to_csv("pinn_grid.csv", index=False)
```

Run it the way you ran it in Lab 3, and copy the `pinn_grid.csv` it writes into `task5_square_plate/`. (Use your
hard-BC network if you have it; say which one you used.)

**5d.** Compare all three (the network is scaled back by 1000; the reference is linear):

```bash
python ../comparison/comparison_three_methods.py --grid-reference reference_grid.csv --grid-nn nn_grid.csv \
    --nn-scale 1000 --grid-pinn pinn_grid.csv --out comparison_square.png --metrics comparison_square_metrics.txt
```

The script prints the relative L2 error of each method against the reference, field by field, and draws all three
with their differences. Also compare σ₁₁ at the top of the hole: the network run's `stress_xx_hole_top` (times 1000),
your PINN at (0, 0.1), and the reference, 3.2418 (`module_3/readings/measured_results.md` §1). Is the strain there
inside your CSV's hull? (`comparison_three_methods.py` checks only Lab 0's plate; compute it from the network
run's nodal `strain_*` fields, or reason from the reference's stresses.)

**5e.** In `prompts_task5.md`: which method is closer to the reference, for which field, and why? Which of the two
learned components (a material law, a whole solution) would you reuse on a plate of a different shape? And why does
the network run need a good initial guess when the elastic run does not?

### Task 6 — Reflection (20 points, ≤ 500 words)

Answer all four:

1. **Training error to simulation error.** How large were your displacement and stress errors in Task 3, and how do
   they follow from your Lab 1 network (its test error, its implied stiffness)?
2. **What the solver did with your network.** Newton iterations, run time, the σ_zz = 0 choice. Which of these would
   matter in a large 3-D simulation?
3. **Extrapolation.** What happened at σ∞ = 1 and why? What would you check before trusting a learned material
   inside a solver?
4. **Where this is worth it.** For linear elasticity the network adds cost and error. Name a material where a learned
   law would be worth deploying, and say what `ComputeTorchScriptStress` would need that it does not have (think of
   Lab 4's LSTM).

---

## Grading rubric (out of 100, +15 stretch)

| Component | Points |
|-----------|--------|
| Task 1: TorchScript file exported and checked; 1d answered | 15 |
| Task 2: both MOOSE runs on ARCC; 2d recorded | 20 |
| Task 3: comparison produced; 3b–3e answered | 25 |
| Task 4: extrapolation run and explained | 15 |
| Task 5 (optional): three methods on Lab 3's plate | +15 bonus |
| Task 6: reflection, all four questions answered concretely | 20 |
| Prompts logged in every sub-folder | 5 |

---

## Hints

- **Two files end in `.pt`.** `mlp_constitutive.pt` is your Lab 1 checkpoint (a dictionary for Python);
  `mlp_constitutive_ts.pt` is the TorchScript network MOOSE loads. Giving MOOSE the checkpoint fails at start-up
  (§5).
- **Run `verify_torchscript.py` before MOOSE.** A missing standardization fails every test there, in seconds. MOOSE
  does not catch it: our plate with such a file converged, without a warning, to strains more than twice the right
  ones (§5; `module_5/readings/torchscript_export_guide.md`).
- **Do not cast inside `forward`.** MOOSE converts your network to float64; an `.float()` in `forward` either makes
  every call fail or leaves Newton unable to reach its tolerance (§5).
- **Export on ARCC with the course environment.** The executable has libtorch 2.5.1, the course torch version. A file
  exported with torch 2.8.0 on a laptop also worked in our test (§5); other versions are untested.
- **If the job stops early,** read the end of `plate_nn_out.log` in `moose_run/`: the job prints its last lines, but the
  full MOOSE message is in the log. "A 'ComputeTorchScriptStress' is not a registered object" means the wrong
  executable; the job script names the right one.
- **The elastic run does not use your file,** so if only the network run fails, the problem is the file or its name.
- **Newton taking many iterations, or not converging, is information**, not a bug to hide: check the strain
  postprocessors against your CSV's ranges.

---

## What this lab leads into

- **Final project.** A learned material inside a solver is a natural final-project starting point: a different
  material model, a different geometry, a 3-D extension.
- **Stateful materials.** Lab 4 showed that path-dependent materials need history: a hidden state (LSTM) or internal
  variables with their own evolution law. Deploying one needs a material that stores and updates state at every
  quadrature point between steps. `ComputeTorchScriptStress` does not; building one is a project, not a lab.
- **Module 6** moves from learning a material law to learning whole fields (CNN surrogates), the other way to put a
  network into a simulation workflow.

---

## Academic integrity

Agents are encouraged. Logged prompts in each sub-folder are mandatory. Your network, runs and numbers must be your
own; copying a classmate's checkpoint, TorchScript file or result files is plagiarism. The course reference network
is the one permitted exception, with the disclosure described above. Discussing strategy with classmates is fine and
encouraged.

---

## How to submit

As in Labs 1–4: a pull request from branch `lab_5`, with your files under `labs/lab_5/<your-github-username>/`
(`module_0/readings/git_submission_guide.md` walks through it).

```
git checkout main && git pull
git checkout -b lab_5
# ... work in labs/lab_5/<your-github-username>/ ...
printf '*.e\n*.log\n*.time\n__pycache__/\n*_grid.csv\n' > labs/lab_5/<your-github-username>/.gitignore
git add labs/lab_5/<your-github-username>
git commit -m "Lab 5: what you did"
git push -u origin lab_5
# then open the pull request on github.com -- that PR is the submission
```
