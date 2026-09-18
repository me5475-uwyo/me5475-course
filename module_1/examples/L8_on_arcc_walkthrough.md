# Lecture 8 on ARCC — repeating the demo yourself

Everything shown in Lecture 8 was run on MedicineBow through **VS Code Remote-SSH**. This is the
same path, step by step. Every command here was run on the cluster on **2026-09-18** and works.

You need the UW VPN if you are off campus.

---

## 1 · Connect VS Code to MedicineBow

1. Install the **Remote - SSH** extension (Microsoft) in VS Code.
2. `F1` → **Remote-SSH: Connect to Host…** → `xzhang16@medicinebow.arcc.uwyo.edu`
   (use *your* username, not the instructor's).
3. Wait for the bottom-left corner to read **SSH: medicinebow.arcc.uwyo.edu**.
4. **File → Open Folder…** → make yourself a working directory, e.g. `~/me5475/L8`.

The first connection installs a VS Code server into your home directory and takes a minute or two.
It is a one-time cost.

> **Check your home space before you start:** `du -sh ~` and `df -h ~`. The VS Code server is a few
> hundred MB. If home is tight, clean up old job outputs before connecting.

---

## 2 · Get the files

**Either** copy them from the shared course folder — no GitHub account, no keys, no browser:

```bash
mkdir -p ~/me5475/L8 && cd ~/me5475/L8
cp /project/me5475/examples/sine_mlp.py .
cp /project/me5475/examples/sine_mlp.sbatch .
```

**Or** use your own repository, if you have already set up the SSH key from Lab 1:

```bash
cd ~/me5475
git clone git@github.com:me5475-uwyo/me5475-<your-github-username>.git
cd me5475-<your-github-username>/module_1/examples
```

`/project/me5475/examples/` is **read-only** — copy files out before editing. `README.md` there
lists everything available.

---

## 3 · Activate the course environment

In the VS Code terminal (``Ctrl+` ``), **every time you open a new terminal**:

```bash
source /project/me5475/setup5475.sh
```

It should print:

```
ME 5475 environment ready
  python   3.11.15  (/project/me5475/envs/ml4sm)
  packages torch 2.5.1+cu121, numpy 2.4.4, matplotlib 3.10.9, pandas 3.0.3
```

It must be **sourced**, not executed. `./setup5475.sh` activates a throwaway shell and leaves your
prompt exactly as it was — the classic way this appears to work while doing nothing.

**Optional, for the Python extension:** `F1` → *Python: Select Interpreter* → *Enter interpreter
path* → `/project/me5475/envs/ml4sm/bin/python`. This only affects VS Code's own linting and the ▶
button; the terminal is what matters for everything below.

---

## 4 · Cap your threads before running on the login node

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**Do not skip this.** The login node has **96 cores and is shared with every other user on the
cluster**. Left unset, PyTorch starts 96 threads to fit 200 points. Four is plenty, and it makes
your timings reproducible.

The `.sbatch` files already do the equivalent from `SLURM_CPUS_PER_TASK`, so this line is only for
running directly in the terminal.

---

## 5 · Run it

```bash
cd ~/me5475/L8
python sine_mlp.py
```

Expect, on ARCC (torch 2.5.1):

```
epoch     0: loss = 5.589426e-01
epoch  4500: loss = 4.596591e-07

Wrote sine_mlp.png
Final dense-grid MSE: 4.1852e-07
```

About **17 seconds** on the login node with four threads.

Two things to notice:

- **The loss is not monotone.** It dips, rises a little, dips again. That is Adam's momentum
  carrying the iterate past the bottom before it settles — not a bug. The `semilogy` plot makes the
  trend obvious through the noise.
- **Your numbers may differ in the last digits** if you run on your laptop instead. The course
  environment is torch 2.5.1; a different version reorders floating-point operations slightly. The
  initial loss (`5.589426e-01`) should match exactly, because the seed fixes the initialisation.

**To see the figure:** click `sine_mlp.png` in the VS Code file tree. It opens inline over the SSH
connection — no X11, no downloading. This is why `sine_mlp.py` pins the `Agg` matplotlib backend:
it renders straight to a file, and a compute node has no display at all.

---

## 6 · Submit the same run as a batch job

```bash
sbatch sine_mlp.sbatch
squeue -u $USER
cat sine_mlp-<jobid>.out
```

The job runs on a **compute node** (`mbcpu-…`), not the login node. It takes about **11 seconds**
once it starts; queue wait is on top of that. It writes `sine_mlp.png` into the directory you
submitted from.

**One thing that will bite you if you write your own submit scripts.** Inside a batch script, call
the interpreter by its full path:

```bash
srun "$CONDA_PREFIX/bin/python" sine_mlp.py        # correct
srun python sine_mlp.py                            # fails, sometimes
```

Why: if you `source setup5475.sh` and *then* `sbatch`, the job inherits `CONDA_PREFIX` from your
shell. The `module purge` at the top of the script strips conda off `PATH`, but conda still believes
the environment is active — so the `conda activate` line quietly does nothing, bare `python`
resolves to the base interpreter, and the job dies with `ModuleNotFoundError: No module named
'matplotlib'`. It fails *only* when you submit from a shell where you already activated, which is
why it survives casual testing. The course scripts all use the full-path form.

---

## 7 · If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: torch` or `matplotlib` in the terminal | environment not activated in *this* terminal | `source /project/me5475/setup5475.sh` |
| same error in a **job output file** | bare `python` in your sbatch | use `srun "$CONDA_PREFIX/bin/python"` |
| `command not found: python` | same as above | as above |
| VS Code will not connect | VPN, or a stale server | connect the UW VPN; `F1` → *Remote-SSH: Kill VS Code Server on Host* |
| job sits in `PD` | the partition is busy | `squeue -p mb` to see the queue; it is not your fault |
| everything feels slow interactively | you skipped the thread caps | step 4 |

Still stuck: Canvas Discussions, or office hours. Paste the **exact** error text and the output of
`source /project/me5475/setup5475.sh` — that pair answers most questions immediately.
