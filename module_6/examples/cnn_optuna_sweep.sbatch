#!/bin/bash
# =============================================================================
# cnn_optuna_sweep.sbatch -- Module 6 / Lab 6 Task 3 -- parallel Optuna sweep of the CNN on ARCC
# =============================================================================
#
# Ten array workers share one Optuna study in an SQLite file in the submit folder; each pulls
# N_TRIALS trials (default 3, so 30 in all). Same pattern as Lab 2's optuna_sweep.sbatch.
#
# Run it from the folder holding rve_dataset.npz, cnn_optuna_sweep.py and
# cnn_microstructure_to_modulus.py (the sweep imports the model from it).
# First create the empty study from the login node, in that folder:
#
#     source /project/me5475/setup5475.sh
#     optuna create-study --study-name lab6_cnn_sweep --storage sqlite:///cnn_opt.db --direction minimize
#
# Then:
#     sbatch cnn_optuna_sweep.sbatch                 # or: N_TRIALS=5 sbatch cnn_optuna_sweep.sbatch
#     squeue -u $USER
#     # when all ten finish:
#     python analyze_study.py --study-name lab6_cnn_sweep --storage sqlite:///cnn_opt.db
#
# CPU partition: one trial trains a small CNN on 350 images for at most 100 epochs.
# =============================================================================

#SBATCH --job-name=lab6_sweep
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --array=0-9                  # 10 workers
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=lab6_sweep-%A_%a.out
#SBATCH --error=lab6_sweep-%A_%a.err

set -euo pipefail
source /project/me5475/setup5475.sh     # the course Python; safe even if you sourced it before sbatch

# Bound math threads to the granted cores (unbounded, torch spawns one per node core and the
# workers contend; Lab 2 measured 6 min 45 s against 22 s for the same job).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"
N_TRIALS=${N_TRIALS:-3}
echo "----- Worker $SLURM_ARRAY_TASK_ID start: $(date) on $(hostname) -----"
sleep $((SLURM_ARRAY_TASK_ID * 10))     # stagger the first database access

srun python cnn_optuna_sweep.py \
    --data rve_dataset.npz \
    --study-name lab6_cnn_sweep \
    --storage sqlite:///cnn_opt.db \
    --n-trials "$N_TRIALS" \
    --seed $((42 + SLURM_ARRAY_TASK_ID))
echo "----- Worker $SLURM_ARRAY_TASK_ID end: $(date) -----"
