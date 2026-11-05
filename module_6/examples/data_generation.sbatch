#!/bin/bash
# =============================================================================
# data_generation.sbatch -- Module 6 / Lab 6 -- generate RVE dataset on ARCC
# =============================================================================
#
# A SLURM array job that splits the 500-RVE dataset across 10 workers.
# Each worker runs ~50 MOOSE simulations sequentially and writes its own
# partial .npz file. Aggregation happens after all workers complete.
#
# Usage:
#   sbatch data_generation.sbatch
#   # After all 10 array tasks finish (sacct -j <jobid>):
#   source /project/me5475/setup5475.sh
#   python aggregate_parts.py          # -> rve_dataset.npz (keeps stress_fields and the run constants)
# =============================================================================

# ARCC setup confirmed 2026-06-06 (Phase 0 recon): account=me5475, partition=mb (CPU).
# MOOSE binary: /project/me5475/software/rom_opt_arcc/rom_opt-opt (hash 437fbe5082)
# ML env provides numpy/pandas; MOOSE env provides runtime libs.
# =============================================================================

#SBATCH --job-name=m6_rve
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --array=0-9                  # 10 workers, 50 RVEs each = 500 total
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=m6_rve-%A_%a.out
#SBATCH --error=m6_rve-%A_%a.err

set -euo pipefail
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"
export MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt

cd $SLURM_SUBMIT_DIR

WORKER_ID=$SLURM_ARRAY_TASK_ID
N_PER_WORKER=50
SEED=$((42 + WORKER_ID * 1000))

echo "----- Worker $WORKER_ID start: $(date) -----"
# Call the course Python by its full path, not bare "python": if you sourced setup5475.sh before
# sbatch, "module purge" + "conda activate" above leave the base Python first and the job fails
# with ModuleNotFoundError (the same fix as Lab 2's optuna_sweep.sbatch, verified 2026-09-18).
PY=/project/me5475/envs/ml4sm/bin/python
srun "$PY" generate_microstructures.py \
    --moose-exe "$MOOSE_EXE" \
    --n $N_PER_WORKER \
    --out "rve_dataset_part_${WORKER_ID}.npz" \
    --seed $SEED
echo "----- Worker $WORKER_ID end: $(date) -----"
