# Lecture 9 on ARCC — running the learning-rate finder yourself

**Verified on MedicineBow, 2026-09-21** — every command below was run on login node `mblog1`
and produced the output shown.

Lecture 9 demonstrated the **LR-range test** — a thirty-second experiment that tells you roughly
what learning rate to train at, instead of guessing. This is the same path on MedicineBow.

You need the UW VPN if you are off campus.

---

## 1 · Connect VS Code to MedicineBow

Exactly as in `L8_on_arcc_walkthrough.md`: **Remote - SSH** extension, `F1` →
**Remote-SSH: Connect to Host…** → `<your-username>@medicinebow.arcc.uwyo.edu`, then
**File → Open Folder…** and make yourself `~/me5475/L9`.

---

## 2 · Get the files

**Either** copy them from the shared course folder — no GitHub account, no keys, no browser:

```bash
mkdir -p ~/me5475/L9 && cd ~/me5475/L9
cp /project/me5475/examples/generate_data_analytic.py .
cp /project/me5475/examples/lr_finder.py .
```

**Or** use your own repository:

```bash
cd ~/me5475
git clone git@github.com:me5475-uwyo/me5475-<your-github-username>.git
cd me5475-<your-github-username>
# the two files live in module_1/examples/ and module_2/examples/
```

`/project/me5475/examples/` is **read-only** — copy files out before editing.

---

## 3 · Activate the course environment

In the VS Code terminal (``Ctrl+` ``), **every time you open a new terminal**:

```bash
source /project/me5475/setup5475.sh
```

It prints what you actually got, and that report is the thing to paste if you need help:

```
ME 5475 environment ready
  python   3.11.15  (/project/me5475/envs/ml4sm)
  packages torch 2.5.1+cu121, numpy 2.4.4, matplotlib 3.10.9, pandas 3.0.3
```

Check it with `which python` — you want `/project/me5475/envs/ml4sm/bin/python`. If you get
`/usr/bin/python`, the source did not happen in this terminal. **Sourcing it more than once in the
same terminal is safe.**

---

## 4 · Cap your threads before running on the login node

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**Not optional.** The login node has 96 cores shared with the whole cluster. Without these, torch
starts 96 threads to do a few seconds of arithmetic and everyone else notices.

---

## 5 · Make the data

Lab 1's dataset, `single_element.csv`, comes from **MOOSE** — that is Lab 1's job, and you produce
it yourself. This lecture needs *a* stress–strain dataset in the same six columns, so there is a
second generator that does not need MOOSE:

```bash
python generate_data_analytic.py
```

It should print the plane-strain stiffness it used and write `data/single_element_analytic.csv`:

```
Plane-strain stiffness used (E = 1.0, nu = 0.3):
   [ 1.346154   0.576923   0.000000]
   [ 0.576923   1.346154   0.000000]
   [ 0.000000   0.000000   0.384615]
```

These numbers are **identical on ARCC and on a laptop**, down to the last digit of every data row.
They have to be: the script is closed-form and seeded, so there is nothing for a different
hardware or library version to change.

**This is not fake data.** One QUAD4 element under prescribed uniform strain has a closed-form
answer, σ = Cε, and MOOSE converges to exactly it. These are the rows MOOSE would print.

> **Look at `0.384615`.** That is G = E/(2(1+ν)) = 1/2.6, and it multiplies the **engineering**
> shear strain γ_xy = 2ε_xy — *not* the tensor component ε_xy. Dropping that factor of two is the
> single most common error in Lab 1. Here it is, in the answer key.

---

## 6 · Run the learning-rate finder

```bash
python lr_finder.py --data data/single_element_analytic.csv
```

**Always pass `--data`.** The script's default path assumes the repository layout, and it will not
resolve if you copied the files into a flat folder.

It sweeps 100 learning rates from 1e-7 to 1e-1, one optimizer step each, and plots loss against
rate. Output is `lr_finder.png` — open it from the VS Code file explorer.

---

## 7 · What the answer is, and what it is not

With the shipped seed the script prints:

```
Suggested learning rate: 1.000e-02
```

**Do not treat that as a measured optimum.** Try this:

```bash
python lr_finder.py --data data/single_element_analytic.csv --seed 7 --out lr_finder_seed7.png
```

It prints **no suggestion at all**. Same data, same script, different random initialisation.

And the reverse trap: with the default seed you get `1.000e-02` on ARCC *and* on a laptop, on two
different versions of torch. That looks like reproducibility. It is not — `1.000e-02` is one tenth
of `--lr-max`, the top of the sweep, which no version of torch is going to move.

Why: the test takes **one** optimizer step per learning rate, on a single network whose rate ramps
continuously. On a target as easy as linear elasticity the loss barely moves at small rates, so the
minimum lands near the top of the sweep — and the "divergence" the script backs off from is
sometimes just the end of the range, not a real wall.

**The honest reading: an LR-range test points you at a *decade*, not at a number.** That is still
far better than guessing, and it is the reason to run one. If you need a defensible value, take the
decade from here and then tune inside it on a validation set — which is what Module 2 is about, and
what Lab 2 has you do with Optuna.

---

## 8 · If something goes wrong

| symptom | what it means | what to do |
|---|---|---|
| `ModuleNotFoundError` on numpy, torch or pandas | you did not `source setup5475.sh` in **this** terminal | source it (section 3), then check `which python` says `/project/me5475/envs/ml4sm/bin/python` |
| `FileNotFoundError` on the CSV | section 5 not run, or you are in the wrong folder | re-run section 5; `ls data/` should show the file |
| no `Suggested learning rate` line | the seed found no divergence | **expected** — see section 7 |
| everything is very slow | thread caps not set | re-run section 4 |
