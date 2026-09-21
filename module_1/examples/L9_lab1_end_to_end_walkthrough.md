# Lecture 9 — Lab 1, end to end, on ARCC

*Mon Sep 21, 2026. This class is a working session, with reading slides. We run the whole Lab 1
chain together, and you leave with it working.*

**Verified on MedicineBow (`mblog1`) on 2026-09-21.** Every command and every number below was run.
Measurements are recorded in `module_2/readings/measured_results.md`.

**Lab 1 is due Wed Sep 23.** The point of today is to make sure **everyone has a working
end-to-end workflow** — data generation, training, and prediction — and has built a complete MLP
through the in-class demo. Alongside it we provide a set of **reading slides** that expand on loss
functions, AdamW, gradient clipping and the learning-rate range test. That material is not optional:
**Lab 2 is already out and is due Wed Sep 30**, and it is built on it.

---

## What you are building

```
single_element_loadsweep.i          one MOOSE run   -> one (strain, stress) pair
        |
        v  generate_data.py loops it 200 times with random strains
data/single_element.csv             200 rows, 6 columns
        |
        v  mlp_constitutive.py
a trained MLP                       sigma = f(epsilon)
        |
        v  compare against C
Hooke's law, recovered              or not -- that is the deliverable
```

---

## 1 · Connect and set up

```bash
ssh <your-username>@medicinebow.arcc.uwyo.edu     # UW VPN if off campus
mkdir -p ~/me5475/L9 && cd ~/me5475/L9
```

```bash
source /project/me5475/setup5475.sh
module load gcc/14.2.0
```

**Both lines. Every new terminal.** The first gives you python, numpy, pandas and torch; it should
print `ME 5475 environment ready`. Check with `which python` — you want
`/project/me5475/envs/ml4sm/bin/python`.

> ### The second line is not optional, and the error it prevents is misleading
>
> Without `module load gcc/14.2.0`, running MOOSE **from Python** fails with:
>
> ```
> GLIBCXX_3.4.31 not found (required by .../libpetsc.so.3.20)
> RuntimeError: MOOSE returned non-zero exit code
> ```
>
> **There is nothing wrong with your input file.** `setup5475.sh` puts conda's `libstdc++` ahead of
> the one MOOSE was built against. MOOSE on its own works; Python on its own works; only the
> combination breaks. The module load puts a newer library first and it goes away.

---

## 2 · Get the files

**Either** from the shared course folder:

```bash
cp /project/me5475/examples/single_element_loadsweep.i .
cp /project/me5475/examples/generate_data.py .
cp /project/me5475/examples/mlp_constitutive.py .
```

**Or** from your own repository:

```bash
cd ~/me5475 && git clone git@github.com:me5475-uwyo/me5475-<your-github-username>.git
```

Both routes carry the same files. `/project/me5475/examples/` is read-only — copy before editing.

---

## 3 · One MOOSE run, and what it tells you

```bash
B=/project/me5475/software/rom_opt_arcc/rom_opt-opt
export OMP_NUM_THREADS=1
$B -i single_element_loadsweep.i Outputs/file_base=single Outputs/exodus=true
cat single.csv
```

Under a second. You get:

```
time,strain_xx_avg,strain_xy_avg,strain_yy_avg,stress_xx_avg,stress_xy_avg,stress_yy_avg
1,0.0005,-0.00025,0,0.00067307692307692,-0.00019230769230769,0.00028846153846154
```

**`moose-opt` is not a command here.** The binary lives at the path above and is not on `PATH`.

**Yellow deprecation warnings** about `Modules/TensorMechanics/Master` appear on every run. Harmless
— the course inputs predate a MOOSE syntax change.

### Check it by hand. This is the most important five minutes of the lab.

Plane-strain isotropic stiffness at E = 1.0, ν = 0.3:

```
[ 1.346154   0.576923   0.000000 ]
[ 0.576923   1.346154   0.000000 ]
[ 0.000000   0.000000   0.384615 ]
```

| | MOOSE printed | C·ε by hand |
|---|---|---|
| σ_xx | 6.7307692e-04 | 1.346154 × 5e-4 = 6.730769e-04 |
| σ_yy | 2.8846154e-04 | 0.576923 × 5e-4 = 2.884615e-04 |
| σ_xy | −1.9230769e-04 | 0.384615 × (−5e-4) = −1.923077e-04 |

**Where did −5e-4 come from?** MOOSE printed `strain_xy_avg = -2.5e-4`. The number that multiplies
`0.384615` is the **engineering** shear strain **γ_xy = 2ε_xy = −5e-4**.

**That factor of two is the single most common error in this lab.** Get it wrong and your learned
stiffness will be right in the top-left 2×2 block and wrong by exactly 2× in the (3,3) entry — and
nothing will crash.

---

## 4 · Two hundred runs

```bash
python generate_data.py --n 200 --out data/single_element.csv \
  --moose-exe $B
```

**About 3 minutes** (measured: 20 runs in 19 s). Strains are sampled uniformly in
[−0.005, +0.005] — small-strain, so the labels are exact, not noisy.

Do not submit this as 200 SLURM jobs. Each run is under a second; batch overhead would dominate.

```bash
wc -l data/single_element.csv     # expect 201: header + 200 rows
head -3 data/single_element.csv
```

---

## 5 · Train, and check against Hooke

```bash
python mlp_constitutive.py --data data/single_element.csv --layers 4 --hidden 32
```

Inputs are `eps_xx, eps_yy, gamma_xy`; outputs are `sigma_xx, sigma_yy, sigma_xy`. The architecture
is the one from L7: four `nn.Linear` layers, tanh between them, **no activation on the output** —
stress takes either sign and nothing should prevent that.

**The deliverable is not a low loss.** It is the comparison: recover the implied 3×3 stiffness from
your trained network and put it beside the matrix in §3. A 2339-parameter network has learned a
2-parameter map; the question is whether it learned *that* map.

---

## Extra · the plate with a hole, and ParaView

Not required for Lab 1. Worth ten minutes, and it is the M0 convergence study.

```bash
cp /project/me5475/examples/plate_with_hole.i .
for r in 0 1 2 3 4; do
  $B -i plate_with_hole.i Mesh/uniform_refine=$r Outputs/file_base=plate_r$r > /dev/null 2>&1
  echo -n "refine $r: "; tail -1 plate_r$r.csv
done
```

Ten seconds for all five. Measured:

| `uniform_refine` | elements | max σ_xx |
|---|---|---|
| 0 | 192 | 1.6221 |
| 1 | 768 | 2.0048 |
| 2 | 3 072 | 2.3841 |
| 3 | 12 288 | 2.6790 |
| 4 | 49 152 | 2.8762 |

The analytical stress concentration factor for a circular hole in an **infinite** plate under
uniaxial tension is **K_t = 3**. The sequence climbs toward it and does not arrive: the coarse mesh
cannot resolve the peak at the hole edge, and this plate is finite (rmax/rmin = 10, not ∞).

**To see it**, copy the Exodus file to your own machine and open it in ParaView — not over SSH,
where remote rendering usually defeats you:

```bash
# in a terminal on YOUR laptop
scp <user>@medicinebow.arcc.uwyo.edu:~/me5475/L9/plate_r4.e .
```

In ParaView: **Apply** → colour by `vonmises_stress` → **Rescale to Data Range** → *Surface With
Edges*. Open `plate_r0.e` beside it with the same colour range. Then try **Warp By Vector** on
`disp` with a large scale factor — the fastest way to see that what was solved for is a
*displacement* field, and stress is derived from it.

---

## If you get stuck

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: numpy` | environment not sourced **in this terminal** | `source /project/me5475/setup5475.sh` |
| `GLIBCXX_3.4.31 not found` | conda's libstdc++ shadows MOOSE's | `module load gcc/14.2.0` |
| `MOOSE executable not found: moose-opt` | the default `--moose-exe` does not exist here | pass the full path from §3 |
| yellow `Deprecation Warning` | course inputs predate a syntax change | ignore |
| `(3,3)` stiffness entry off by exactly 2 | γ_xy vs ε_xy | §3 |

Questions: Canvas Discussions, or office hours.
