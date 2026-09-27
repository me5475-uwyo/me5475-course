#!/bin/bash
# =============================================================================
# pinn_inverse_ensemble.sbatch -- Module 3 / Lab 3 Task 5 -- ten inverse PINNs
# =============================================================================
#
# Runs plate_with_hole_inverse.py with seeds 0..9 as a SLURM array, one GPU each.
# Each seed changes several things at once: which 20 points are measured, the
# noise on them and on the force, the collocation points and the initial
# weights. The spread of the ten (E, nu) estimates is their COMBINED effect
# (Task 5e) -- not the uncertainty of one experiment.
#
# Submit from your task5_inverse/ folder, with plate_with_hole_inverse.py and
# measurements.csv (extract_moose_reference.py --mode random) next to this file:
#
#     sbatch pinn_inverse_ensemble.sbatch
#
# Each task writes inverse_results_seed_<N>.txt (one CSV row) and
# inverse_variables_seed_<N>.dat (E, nu during training). Aggregate the ten rows
# yourself (Task 5c).
# =============================================================================

#SBATCH --job-name=pinn_inverse
#SBATCH --account=me5475
#SBATCH --partition=mb-l40s,mb-a30
#SBATCH --array=0-9%2                # ten seeds, at most 2 GPUs at a time (course rule)
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=inverse-%A_%a.out
#SBATCH --error=inverse-%A_%a.err

set -euo pipefail

module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"
SEED=$SLURM_ARRAY_TASK_ID
echo "----- inverse seed $SEED start: $(date) on $(hostname) -----"
[ -f measurements.csv ] || { echo "ERROR: measurements.csv not in $(pwd) -- run extract_moose_reference.py --mode random first" >&2; exit 1; }

srun "$CONDA_PREFIX/bin/python" plate_with_hole_inverse.py \
    --measurements measurements.csv \
    --epochs 30000 \
    --seed "$SEED" \
    --save "inverse_results_seed_${SEED}.txt"

echo "----- inverse seed $SEED end: $(date) -----"
