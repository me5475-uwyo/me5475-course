#!/bin/bash
# =============================================================================
# pinn_diffusion.sbatch -- Module 3 / L13 -- the two diffusion warm-ups, as a job
# =============================================================================
#
# Usage (from the folder that holds this file and the two scripts):
#   sbatch pinn_diffusion.sbatch                  # 1-D: pinn_diffusion_1d.py
#   sbatch pinn_diffusion.sbatch 2d               # 2-D: pinn_diffusion_2d.py
#   sbatch pinn_diffusion.sbatch 2d --seed 7      # anything after 1d/2d goes to the script
#
# Watch it while it runs -- the loss table appears row by row:
#   squeue -u $USER
#   tail -F pinn-diffusion-<jobid>.out            # Ctrl-C stops watching, not the job
#
# CPU partition, four cores, no GPU: networks this small do not need one, and
# the GPUs are the scarce, shared resource. Measured on ARCC, 2026-09-26
# (module_3/readings/measured_results.md, section 7): 1-D about half a minute,
# 2-D two to three minutes, start to finish.
#
# The plot (pinn_diffusion_1d.png / _2d.png) lands next to the .out file.
# =============================================================================

#SBATCH --job-name=pinn-diffusion
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=pinn-diffusion-%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Activate the shared course environment (PyTorch + DeepXDE).
# -----------------------------------------------------------------------------
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

# -----------------------------------------------------------------------------
# 1d or 2d; everything after it is passed to the script unchanged.
# -----------------------------------------------------------------------------
DIM="${1:-1d}"
if [ "$#" -gt 0 ]; then shift; fi
case "$DIM" in
    1d) SCRIPT="pinn_diffusion_1d.py" ;;
    2d) SCRIPT="pinn_diffusion_2d.py" ;;
    *)  echo "Unknown dimension: $DIM. Use 1d or 2d."; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname)"
echo "Script: $SCRIPT $*"

# Full interpreter path, not bare `python` (same reason as pinn_train.sbatch: a
# job submitted after `source setup5475.sh` would otherwise use the base python).
# Threads capped at the cores SLURM gave the job; unbuffered, so `tail -f` shows
# each loss row as it is printed rather than all at the end.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
srun "$CONDA_PREFIX/bin/python" "$SCRIPT" "$@"

echo "----- Job end: $(date)"
