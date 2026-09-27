#!/bin/bash
# =============================================================================
# pinn_optuna_sweep.sbatch -- Module 3 / Lab 3 Task 4 -- PINN HP sweep on ARCC
# =============================================================================
#
# Pre-stage the study from a login node:
#
#     optuna create-study --study-name pinn_lab3_sweep \
#         --storage "sqlite:///$(pwd)/pinn_opt.db" \
#         --direction minimize
#
# Then submit. Six workers, four trials each = 24 trials total.
# Walltime per worker: ~2 hours (each trial is up to ~10 min PINN training).
# =============================================================================

# ARCC setup confirmed 2026-06-06 (Phase 0 recon): GPU jobs use account=me5475, partition=mb-l40s,mb-a30.
# =============================================================================

#SBATCH --job-name=pinn_sweep
#SBATCH --account=me5475
#SBATCH --partition=mb-l40s,mb-a30
#SBATCH --array=0-5%2                # 6 workers, at most 2 GPUs at a time (course rule)
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=pinn_sweep-%A_%a.out
#SBATCH --error=pinn_sweep-%A_%a.err

set -euo pipefail

module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

cd $SLURM_SUBMIT_DIR

# Stagger worker starts to avoid SQLite race on study table creation.
sleep $((SLURM_ARRAY_TASK_ID * 10))

echo "----- Worker $SLURM_ARRAY_TASK_ID start: $(date) -----"

# Full interpreter path (see module_2/examples/optuna_sweep.sbatch) + thread caps.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun "$CONDA_PREFIX/bin/python" pinn_optuna_sweep.py \
    --study-name pinn_lab3_sweep \
    --storage "sqlite:///$SLURM_SUBMIT_DIR/pinn_opt.db" \
    --reference reference_grid.csv \
    --n-trials 4 \
    --seed $((42 + SLURM_ARRAY_TASK_ID))

echo "----- Worker $SLURM_ARRAY_TASK_ID end: $(date) -----"
