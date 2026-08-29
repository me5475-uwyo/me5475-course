#!/bin/bash
# =============================================================================
# run_convergence_study.sbatch -- Lab 0 main task starter template
# =============================================================================
#
# Submits 5 MOOSE runs of plate_with_hole.i on ARCC, one per refinement level,
# using a SLURM array job. Each task overrides Mesh/uniform_refine and writes
# output to a level-tagged file base so the 5 runs don't overwrite each other.
#
# Students should treat this as a STARTER -- they will be asked to inspect it
# in their lead-review agent loop and confirm the resource requests are
# sensible, the file_base scheme is correct, and the parsing in extract_stress.py
# matches the output naming and reads both peak-stress postprocessors.
#
# Usage on ARCC:
#   sbatch run_convergence_study.sbatch
#   squeue -u $USER                    # monitor all 5 array tasks
#   ls plate_with_hole_refine_*_out.csv  # five CSVs after completion
#
# ARCC setup confirmed 2026-06-06 (Phase 0 recon): account=me5475, partition=mb.
# MOOSE binary: /project/me5475/software/rom_opt_arcc/rom_opt-opt (hash 437fbe5082, TensorMechanics)
# moose_test-opt lacks TensorMechanics; mowry-opt has a version-mismatch error.
# Full module stack required: gcc/14.2.0 openmpi/5.0.5 hdf5/1.14.3... miniconda3/24.3.0
# =============================================================================

#SBATCH --job-name=plate_conv
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --array=0-4                 # five refinement levels: 0,1,2,3,4
#SBATCH --time=00:30:00             # generous; finest level should be < 10 min
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # 4 MPI ranks per task for moderate parallelism
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=conv-%A_%a.out
#SBATCH --error=conv-%A_%a.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Activate MOOSE runtime environment and set executable path.
# -----------------------------------------------------------------------------
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"

MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt

# -----------------------------------------------------------------------------
# Pull the refinement level from the SLURM array task ID.
# Tag output files so the 5 runs don't collide.
# -----------------------------------------------------------------------------
REFINE_LEVEL=${SLURM_ARRAY_TASK_ID}
FILE_BASE="plate_with_hole_refine_${REFINE_LEVEL}_out"

echo "----- Task $REFINE_LEVEL start: $(date) -----"
echo "Running on: $(hostname)"
echo "Refinement level: $REFINE_LEVEL"
echo "Output file base: $FILE_BASE"

srun --ntasks=$SLURM_NTASKS "$MOOSE_EXE" \
    -i plate_with_hole.i \
    Mesh/uniform_refine="$REFINE_LEVEL" \
    Outputs/file_base="$FILE_BASE"

RESULT_CSV="${FILE_BASE}.csv"
test -s "$RESULT_CSV"
echo "Postprocessor results (max_vonmises_stress, max_stress_xx, num_elements):"
tail -n 2 "$RESULT_CSV"

echo "----- Task $REFINE_LEVEL end: $(date) -----"
