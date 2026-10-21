#!/bin/bash
# =============================================================================
# icnn_demo.sbatch -- Module 4 / Lecture 21 live demo on ARCC
# =============================================================================
#
# Trains the ICNN of icnn_hyperelastic.py on the neo-Hookean energy, then checks
# the trained model with icnn_rank_one_check.py. One job per arm:
#
#   sbatch icnn_demo.sbatch 1     # H1: fit the energy AND the stress P (alpha = 1)
#   sbatch icnn_demo.sbatch 0     # energy only (alpha = 0)
#
# Submit both at once; they run in parallel, about 75 s each on 4 mb cores
# once they start (M4 measured results, section 4.1). Submit from the folder that
# holds icnn_hyperelastic.py and icnn_rank_one_check.py. Each job writes
#   icnn_h1.png / icnn_energy_only.png      loss, energy parity, stress parity
#   icnn_h1.pt  / icnn_energy_only.pt       the trained weights
#   l21_icnn-<jobid>.out                    test errors and the rank-one check
# =============================================================================

#SBATCH --job-name=l21_icnn
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=l21_icnn-%j.out

set -euo pipefail
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

ALPHA="${1:-1}"
case "$ALPHA" in
    1) TAG="h1" ;;
    0) TAG="energy_only" ;;
    *) echo "Usage: sbatch icnn_demo.sbatch 1   (H1)   or   sbatch icnn_demo.sbatch 0   (energy only)"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----  alpha = $ALPHA ($TAG)"
# Full interpreter path, not bare `python` (see module_3/examples/pinn_train.sbatch).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun "$CONDA_PREFIX/bin/python" -u icnn_hyperelastic.py --alpha "$ALPHA" --out "icnn_$TAG.png" --save-model "icnn_$TAG.pt"
echo
srun "$CONDA_PREFIX/bin/python" -u icnn_rank_one_check.py --model "icnn_$TAG.pt"
echo "----- Job end: $(date) -----"
