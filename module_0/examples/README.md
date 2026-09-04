# Module 0 — runnable examples

Every file here runs. Two ways to get them onto ARCC; use whichever you like.

## Route A — copy from the shared course folder (fastest, on ARCC)

The same files are already on the cluster:

```bash
cp -r /project/me5475/examples ~/ME5475-examples
cd ~/ME5475-examples
```

Nothing to download, no login needed beyond being on ARCC.

## Route B — clone from GitHub (the backup, and good Git practice)

This repository is public, so cloning needs no password or token:

```bash
cd ~
git clone https://github.com/me5475-uwyo/me5475-course.git
cd me5475-course/module_0/examples
```

To pick up later changes without re-cloning:

```bash
cd ~/me5475-course
git pull
```

Route B is worth doing once even if Route A works, because `clone` and `pull` are two
of the three Git commands you will use all semester — and here there is nothing to
break. Your own graded work goes in your **private** repo, where you also `add`,
`commit` and `push`; practise here first.

## What each file is

| File | What it is |
|---|---|
| `hello.sbatch`, `hello.py` | your first SLURM job |
| `plate_with_hole.i` | MOOSE input, `uniform_refine = 0` by default |
| `run_convergence_study.sbatch` | SLURM array job, refine levels 0–4 (Lab 0) |
| `extract_stress.py` | parses the CSVs into a convergence table |
| `cantilever_beam.i`, `run_cantilever.sbatch`, `plot_beam_results.py` | Reading 4 |
| `fe1d_reference.py` | Reading 3; runs on the login node, no SLURM |
| `install_moose_prompt.md`, `run_moose_local.sh` | running MOOSE on your own machine |
| `plate_with_hole_schematic.svg` | the quarter-model geometry |

## Submitting your first job

```bash
cat hello.sbatch          # read it before you run it
sbatch hello.sbatch       # prints a job ID
squeue -u $USER           # watch it queue, then run
cat hello-<jobid>.out
```

The hostname it prints — something like `mbcpu-001` — is a **compute node**, not the
login node you connected to. That is the whole point of submitting a job.

The `.err` file will contain a short "The following modules were not unloaded" message
even when the run succeeds. That is normal.

Every `.sbatch` here already contains the exact module loads and library paths MOOSE
needs. **Copy those lines verbatim rather than typing paths by hand** — a mistyped path
is the most common reason a first job fails.
