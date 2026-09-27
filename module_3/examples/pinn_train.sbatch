#!/bin/bash
# =============================================================================
# pinn_train.sbatch -- Module 3 / Lab 3 -- single PINN training on ARCC
# =============================================================================
#
# Submits one plate-with-hole PINN training job. For ensemble or sweep,
# use pinn_optuna_sweep.sbatch or wrap this in a SLURM array.
#
# Usage:
#   sbatch pinn_train.sbatch                       # default: plate_with_hole_fixed.py
#   sbatch pinn_train.sbatch hard_bc               # plate_with_hole_hard_bc.py
#   sbatch pinn_train.sbatch parametric            # parametric version (longer)
#   SEED=3 sbatch pinn_train.sbatch inverse        # one inverse run (seed 3); for all ten
#                                                  #   seeds use pinn_inverse_ensemble.sbatch
#
# ARCC setup confirmed 2026-06-06 (Phase 0 recon): GPU jobs use account=me5475, partition=mb-l40s,mb-a30.
# ML env: /project/me5475/envs/ml4sm includes deepxde. No deepxde module on mb cluster.
# =============================================================================

#SBATCH --job-name=pinn
#SBATCH --account=me5475
#SBATCH --partition=mb-l40s,mb-a30
#SBATCH --time=01:00:00              # 1 hour walltime; PINN takes 10-30 min typically
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=pinn-%j.out
#SBATCH --error=pinn-%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Activate shared ML conda environment (PyTorch + DeepXDE).
# -----------------------------------------------------------------------------
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

# -----------------------------------------------------------------------------
# Pick the script based on the first command-line argument.
# -----------------------------------------------------------------------------
VARIANT="${1:-fixed}"

case "$VARIANT" in
    fixed)
        SCRIPT="plate_with_hole_fixed.py"
        ARGS="--epochs 50000 --seed ${SEED:-42}"
        ;;
    hard_bc)
        SCRIPT="plate_with_hole_hard_bc.py"
        ARGS="--epochs 50000 --seed ${SEED:-42}"
        ;;
    parametric)
        SCRIPT="plate_with_hole_parametric.py"
        ARGS="--epochs 100000"
        ;;
    inverse)
        SCRIPT="plate_with_hole_inverse.py"
        ARGS="--measurements measurements.csv --epochs 30000 --seed ${SEED:-0} --save inverse_results_seed_${SEED:-0}.txt"
        ;;
    *)
        echo "Unknown variant: $VARIANT. Use one of: fixed, hard_bc, parametric, inverse"
        exit 1
        ;;
esac

cd $SLURM_SUBMIT_DIR

echo "----- Job start: $(date) -----"
echo "Hostname: $(hostname)"
echo "Variant: $VARIANT"
echo "Script: $SCRIPT"
echo "Args: $ARGS"
"$CONDA_PREFIX/bin/python" --version
"$CONDA_PREFIX/bin/python" -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
"$CONDA_PREFIX/bin/python" -c "import deepxde; print('DeepXDE:', deepxde.__version__)"

# Full interpreter path, not bare `python`: if you `source setup5475.sh` and then
# `sbatch`, the job inherits CONDA_PREFIX and the `conda activate` above no-ops, so
# bare `python` would be the base interpreter (same fix as module_2/examples/*.sbatch).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun "$CONDA_PREFIX/bin/python" $SCRIPT $ARGS

echo "----- Job end: $(date) -----"
