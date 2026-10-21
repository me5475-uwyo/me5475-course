# L21 Live Demo on ARCC: Step by Step

*Course reading · ME 5475 · Module 4 · follow along in Lecture 21 (Wed Oct 21). You will train the lecture's ICNN twice
on MedicineBow, once with the stress term and once without, and then check whether each trained energy is
polyconvex. Every expected number below is recorded in M4 measured results, §4.1.*

**What you need:** a MedicineBow login and VS Code connected to it. If you have not connected VS Code yet, follow §1 of
*Running the L8 demo on ARCC* (Module 0 readings). Do it before class if you can.

**Rule:** every training run is a submitted job, never a run on the login node. On the `mb` partition the two jobs
below took about 75 seconds each once they started (they run in parallel), plus any time waiting in the queue.

---

## 1 · Make a folder and copy the three files

In a VS Code terminal on MedicineBow:

```
mkdir -p ~/me5475/L21 && cd ~/me5475/L21
cp /project/me5475/examples/icnn_hyperelastic.py .
cp /project/me5475/examples/icnn_rank_one_check.py .
cp /project/me5475/examples/icnn_demo.sbatch .
ls
```

You should see the three files. They are the same files as `module_4/examples/` in your course repository, so you
can also work from your repository after `git pull`.

## 2 · Submit both runs at once

```
sbatch icnn_demo.sbatch 1      # H1: fit the energy AND the stress P
sbatch icnn_demo.sbatch 0      # energy only
```

Each command prints `Submitted batch job <number>`. Note both numbers.

## 3 · Watch them

```
squeue -u $USER
```

- `PD` means the job is waiting for a node. When many people submit at once, this can take a few minutes; it is not
  your fault.
- `R` means it is running.
- When a job disappears from the list, it has finished.

To watch a running job print its progress (one line every 500 epochs, 40 lines in all), run the following, and
press Ctrl-C to stop watching. Stopping does not stop the job.

```
tail -f l21_icnn-<job number>.out
```

## 4 · Read the results

When both jobs have finished:

```
grep -A3 "Test psi" l21_icnn-*.out
```

What you should see (the same digits as the course's runs, because the seeds are fixed):

| run | figure | test ψ MSE | test P relative error |
|---|---|---|---|
| H¹ (`1`) | `icnn_h1.png` | 2.4689e-04 | 10.2221 % |
| energy only (`0`) | `icnn_energy_only.png` | 3.3190e-03 | 42.6326 % |

**Look at the figures:** click `icnn_h1.png` and `icnn_energy_only.png` in the VS Code file tree. Compare the **third
panel**, "Test set: stress". Fitting the slope as well as the value is what makes the stress right.

## 5 · The rank-one check

The bottom of each `.out` file is the output of `icnn_rank_one_check.py` for that trained model:

```
tail -11 l21_icnn-<job number>.out
```

| trained model | rank-one curvature below −10⁻⁶ | verdict printed |
|---|---|---|
| H¹ | 0 / 2 540 | no violation at these samples (this proves nothing) |
| energy only | 245 / 2 540 | NOT polyconvex |

The same architecture gives different answers depending on how it was trained. A clearly negative curvature proves a
model is not polyconvex; finding none proves nothing. Lines "1." of the output show that even the H¹ model decreases in
some of its inputs, so the sufficient construction is not what holds it up (L21 slides 13 and 17).

## 6 · If something goes wrong

| what you see | why | what to do |
|---|---|---|
| `cp: cannot stat '/project/me5475/examples/…'` | a typo in the path | copy the commands from this page |
| `Usage: sbatch icnn_demo.sbatch 1 …` in the `.out` file | the argument was neither `0` nor `1` (with no argument at all, the job runs H¹) | resubmit with `1` or `0` |
| `python: can't open file 'icnn_hyperelastic.py'` | the job was submitted from a different folder | `cd ~/me5475/L21` and resubmit |
| the job stays `PD` for a long time | the partition is busy | wait; `squeue -p mb` shows the queue. Results will be there after class |
| you want to stop a job | | `scancel <job number>` |

Your folder keeps everything: the two figures, the two `.pt` weight files, and the two `.out` files. You can rerun the
check on a saved model at any time inside a job; it takes about 5 seconds.

**Why 20 000 epochs?** At the old default of 5 000, the H¹ model is far from converged: ledger §4.1 records a 657 %
test stress error.
