# Module 6 — measured results

Every computed number in a Module 6 reading or in Lab 6 must appear here first, with what was run, where, and when.
A reading cites this file; it never carries a number that was not measured. If a value is wanted but unmeasured, the
reading says **(not measured)**.

`[fallback]` — measured by the Co-Worker (Opus 5.5) on ARCC MedicineBow, 2026-11-05, with the Lab 6 scripts as
released that day. ARCC work folders: `~/l6_smoke/` (instructor's home), `~/l6_e2e/` (the handout's command blocks
run end to end).

**Held until Lab 6 closes (Mon Nov 16, 11:59 PM MT):** values a student reports in Lab 6. With the default seeds
(42 + 1000 × worker), the ten workers produce the same 500 RVEs for everyone, so our dataset statistics, test errors,
baselines and sweep results are the students' answers. They are kept in
`team/reviews/2026-11-05_M6_lab6_evidence/measured_results_M6_HELD_until_Nov16.md` and are restored here after the
lab closes.

---

## §1 · Data generation (`data_generation.sbatch`, partition `mb`, 1 task × 2 CPUs per worker)

- **One MOOSE homogenization:** about 1.5 s of job time (a 4-RVE worker took 7 s in total, start-up included;
  job 25124855).
- **Ten workers × 50 RVEs (job 25124857):** each worker 46–51 s elapsed; all 500 RVEs succeeded (0 failures).
- **Stress fields (`--save-stress-fields`):** ten workers × 10 RVEs took 11–16 s each (job 25124885); the combined
  file is 5.4 MB per 100 RVEs.
- **File sizes:** `rve_dataset.npz` (500 RVEs, no fields) 2.05 MB.

## §2 · Training (`cnn_train.sbatch`, partitions `mb-l40s,mb-a30`, 1 GPU)

- **The scalar CNN runs on the GPU since the Nov 5 fix** (before it, the script never moved the model or data to the
  GPU, so the earlier jobs 25124870/71/74 ran on the CPU of a GPU node). The printout's `Device:` line says which ran.
- **CNN, default settings, on an A30 (jobs 25125051, 25125054, and the CPU-copy save check):** 97,761 parameters; each
  job 15–19 s elapsed, early stopping included. Checkpoint 0.40 MB, saved as CPU tensors (loads on a login node).
- **The same split in every run:** with `--split-seed 42` every run printed the same test rows (first five
  [473, 132, 472, 46, 40]), whatever `--seed` was.
- **Run-to-run variation:** on the GPU, runs with the same seeds did not give identical results (non-deterministic GPU
  kernels); the values are held with the other Task 2–3 results.
- **U-Net (stretch), 100-RVE field dataset:** 1,927,907 parameters; 23 s on a GPU (job 25124896). Checkpoint 7.8 MB.
  Only this 100-RVE run was made; the 500-RVE stretch task was not run end to end.

## §3 · Sweep (`cnn_optuna_sweep.sbatch`, partition `mb`, CPU, ten workers × 3 trials)

- **Job 25125052 (Nov 5, after the split fix):** every worker printed the same split (split seed 42, validation rows
  first five [140, 445, 83, 415, 180]); each worker 2 min 00 s to 4 min 39 s elapsed; 30 trials, 0 failed; no
  "database is locked" errors with reporting every 10 epochs. `analyze_study.py` (Module 2) wrote all four plots.
- An earlier sweep (job 25124873) let each worker use its own split; it is not a valid benchmark and is not used.
- **Retraining the best configuration** as a job (`sbatch cnn_train.sbatch modulus --lr … --out
  checkpoints/cnn_best.pt`, job 25125053): 15 s on an A30.

## §4 · Environment finding (Nov 5)

- A job script that runs `module purge`, `module load miniconda3` and `conda activate` fails with
  `ModuleNotFoundError` when it is submitted from a shell that had already sourced `setup5475.sh`: the base Python
  comes first (job 25124868 failed; the same script from a clean shell passed, job 25124870). Lab 2 found the same on
  2026-09-18. The Lab 6 job scripts now `source setup5475.sh` (training, sweep) or call the environment's Python by
  full path (data generation); both pass from a sourced shell (jobs 25124871–25124874).
