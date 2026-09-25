#!/bin/bash
# =============================================================================
# retrain_best.sbatch -- Module 2 / Lab 2 Task 4 -- run your retrain_best.py
# =============================================================================
#
# Task 4 trains two networks (the sweep's best configuration and Lab 1's
# default) on train+val and scores both on the sweep's test rows. That is
# training, so it runs as a batch job on a compute node, not on the login node.
#
# Submit from your verification/ folder, where your retrain_best.py lives:
#
#     cd labs/lab_2/<your-github-handle>/verification
#     sbatch retrain_best.sbatch
#
# Everything retrain_best.py prints is saved to retrain_log.txt (a deliverable)
# as well as to the job's .out file.
# =============================================================================

#SBATCH --job-name=lab2_retrain
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=retrain-%j.out
#SBATCH --error=retrain-%j.err

set -euo pipefail

# Activate the shared ML conda environment (same block as optuna_sweep.sbatch).
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
PY="$CONDA_PREFIX/bin/python"    # full path: see the note in optuna_sweep.sbatch

# Keep PyTorch's threads inside the cores SLURM gave us (the L8 lesson).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"
echo "----- Task 4 start: $(date) on $(hostname) -----"

[ -f retrain_best.py ] || { echo "ERROR: retrain_best.py not in $(pwd)" >&2; exit 1; }
[ -f ../sweep/optuna.db ] || { echo "ERROR: ../sweep/optuna.db not found -- run Task 2 first" >&2; exit 1; }

srun "$PY" retrain_best.py | tee retrain_log.txt

echo "----- Task 4 end: $(date) -----"
