#!/bin/bash
# =============================================================================
# optuna_sweep.sbatch -- Module 2 / Lab 2 -- parallel HP sweep on ARCC
# =============================================================================
#
# Launches 10 persistent Optuna worker tasks. Each worker pulls trials from
# a shared SQLite database on the submit directory's filesystem. Together
# they run ~50 trials in parallel via TPE sampler + Median pruner.
#
# Before running for the first time, create the (empty) study from the login node:
#
#     optuna create-study --study-name lab2_sweep \
#         --storage sqlite:///$SLURM_SUBMIT_DIR/optuna.db \
#         --direction minimize
#
# Then submit:
#
#     sbatch optuna_sweep.sbatch
#     squeue -u $USER          # 10 array tasks should show up
#     # After ~10 minutes:
#     python analyze_study.py --storage sqlite:///optuna.db --study lab2_sweep
#
# ARCC setup confirmed 2026-06-06 (Phase 0 recon): account=me5475, partition=mb (CPU).
# ML env: /project/me5475/envs/ml4sm includes optuna.
# SQLite on VAST-based /project filesystem: tested OK for 10 concurrent writers.
# =============================================================================

#SBATCH --job-name=lab2_optuna
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --array=0-9                  # 10 worker tasks
#SBATCH --time=00:30:00              # each worker runs ~5 trials, ~1 min each
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=lab2-%A_%a.out
#SBATCH --error=lab2-%A_%a.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Activate shared ML conda environment.
# -----------------------------------------------------------------------------
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

# -----------------------------------------------------------------------------
# E-M1-PERF (instructor-approved 2026-08-22): bound the math libraries to the
# cores SLURM granted. Left unset, torch sees all 96 cores on the node and
# spawns that many threads inside this 2-core allocation, where they contend --
# 6 min 45 s against 22 s for the same job, measured on MedicineBow 2026-08-18.
# With ten array workers on shared nodes the contention compounds, and the
# lab's "~1 min per trial" budget depends on this being set.
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -----------------------------------------------------------------------------
# Move to submit directory.
# -----------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

echo "----- Worker $SLURM_ARRAY_TASK_ID start: $(date) -----"
echo "Running on: $(hostname)"

# Stagger worker starts to avoid SQLite race on study-table creation.
# Without this, multiple workers concurrently creating the same study
# can deadlock the shared SQLite DB on ARCC's NFS filesystem.
sleep $((SLURM_ARRAY_TASK_ID * 10))

# -----------------------------------------------------------------------------
# Run the worker. n-trials = how many trials THIS worker pulls.
# Total trials across the sweep = n_array_tasks * n_trials per worker.
# -----------------------------------------------------------------------------

# NOTE (2026-09-18, verified on MedicineBow): call the interpreter by its full
# path, not bare `python`. If you `source setup5475.sh` and THEN `sbatch`, the
# job inherits CONDA_PREFIX from your shell, `module purge` strips conda off
# PATH, and the `conda activate` below then no-ops because conda believes the
# env is already active -- so bare `python` resolves to the base interpreter and
# the job dies with ModuleNotFoundError. Reproduced as jobs 17587307 (clean
# shell, PASS) vs 17587308 (after sourcing, FAIL); this form passes from both.
# -----------------------------------------------------------------------------
# The dataset. Lab 2 runs from labs/lab_2/<handle>/sweep/, and the script's own
# default (../../module_1/examples/data/...) does not resolve from there -- every
# worker used to die with FileNotFoundError (found 2026-09-24). Copy your Lab 1
# CSV into ./data/ (lab_2.md step 2a), or point DATA elsewhere:
#     DATA=/path/to/your.csv sbatch optuna_sweep.sbatch
# -----------------------------------------------------------------------------
DATA="${DATA:-$SLURM_SUBMIT_DIR/data/single_element.csv}"
if [ ! -f "$DATA" ]; then
    echo "ERROR: dataset not found: $DATA" >&2
    echo "       Copy your Lab 1 CSV into $SLURM_SUBMIT_DIR/data/ (lab_2.md step 2a)." >&2
    exit 1
fi

srun "$CONDA_PREFIX/bin/python" optuna_sweep.py \
    --data "$DATA" \
    --study-name lab2_sweep \
    --storage "sqlite:///$SLURM_SUBMIT_DIR/optuna.db" \
    --n-trials 5 \
    --max-epochs 2000 \
    --seed $((42 + SLURM_ARRAY_TASK_ID))

echo "----- Worker $SLURM_ARRAY_TASK_ID end: $(date) -----"
