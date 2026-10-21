#!/bin/bash
# =============================================================================
# train_constitutive.sbatch -- Module 4 / Lab 4 -- train one constitutive model on ARCC
# =============================================================================
#
# Usage:
#   sbatch train_constitutive.sbatch                          # default: LSTM
#   sbatch train_constitutive.sbatch stateless                # stateless MLP baseline
#   sbatch train_constitutive.sbatch stateless_with_q         # MLP with eps_p input
#   sbatch train_constitutive.sbatch lstm                     # LSTM
#   sbatch train_constitutive.sbatch icnn                     # ICNN for hyperelasticity
#   sbatch train_constitutive.sbatch levelset                 # level-set yield surface (Lab 4 stretch)
# =============================================================================

# CPU partition: none of the five scripts moves anything to a GPU, and the
# models are small (run times: module_4/readings/measured_results.md).
# Submit from the folder that holds the script (and, for the first three
# variants, trajectories.npz).
# =============================================================================

#SBATCH --job-name=m4_train
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=m4_train-%j.out
#SBATCH --error=m4_train-%j.err

set -euo pipefail
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

VARIANT="${1:-lstm}"

case "$VARIANT" in
    stateless)        SCRIPT="stateless_mlp_baseline.py"; ARGS="--data trajectories.npz --epochs 2000" ;;
    stateless_with_q) SCRIPT="stateless_mlp_baseline.py"; ARGS="--data trajectories.npz --epochs 2000 --include-plastic-strain" ;;
    lstm)             SCRIPT="lstm_plasticity.py";        ARGS="--data trajectories.npz --epochs 2000 --hidden 32 --layers 2" ;;
    icnn)             SCRIPT="icnn_hyperelastic.py";      ARGS="--epochs 20000 --hidden 32" ;;
    levelset)         SCRIPT="levelset_yield_surface.py"; ARGS="--epochs 5000 --hidden 32 --n-samples 1000" ;;
    *)
        echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

cd $SLURM_SUBMIT_DIR
echo "----- Job start: $(date) on $(hostname) -----"
echo "Variant: $VARIANT  Script: $SCRIPT  Args: $ARGS"
# Full interpreter path, not bare `python`: if you `source setup5475.sh` and then
# `sbatch`, the job inherits CONDA_PREFIX and the `conda activate` above no-ops, so
# bare `python` would be the base interpreter (same fix as module_3/examples/pinn_train.sbatch).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun "$CONDA_PREFIX/bin/python" -u $SCRIPT $ARGS
echo "----- Job end: $(date) -----"
