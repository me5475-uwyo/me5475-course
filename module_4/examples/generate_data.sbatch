#!/bin/bash
# =============================================================================
# generate_data.sbatch -- Module 4 / Lab 4 Task 1 -- build trajectories.npz on ARCC
# =============================================================================
#
# Runs generate_plasticity_data.py with the course MOOSE build. The driver calls
# MOOSE once per trajectory (200 short runs, one after another), so it runs as a
# job, never on the login node. On ARCC, MOOSE is not on PATH and needs the
# module stack below -- that is all this script adds.
#
# Usage (from the folder that holds this file, j2_cyclic_loading.i and
# generate_plasticity_data.py):
#   sbatch generate_data.sbatch                              # --n 200 --out trajectories.npz
#   sbatch generate_data.sbatch --n 20 --out test20.npz      # any driver options
# Watch it:   squeue -u $USER
#
# Output, next to this file:  trajectories.npz (or your --out) and
# gen-data-<jobid>.out (the driver's progress lines, then the file summary).
# Time and memory: module_4/readings/measured_results.md, section 1.
#
# One CPU core: each MOOSE run is a single element. Same module stack as Lab 0's
# run_cantilever.sbatch; the new course MOOSE build (me5475-opt).
# =============================================================================

#SBATCH --job-name=gen-data             # name shown in squeue
#SBATCH --account=me5475                # course allocation
#SBATCH --partition=mb                  # CPU partition
#SBATCH --time=00:30:00                 # wall-clock limit; 200 runs take minutes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=gen-data-%j.out        # %j = job id; the log lands here

set -euo pipefail

for f in generate_plasticity_data.py j2_cyclic_loading.i; do
  [ -f "$f" ] || { echo "no $f in $(pwd) -- submit from the folder that holds it" >&2; exit 2; }
done

# --- activate the course MOOSE runtime (same module stack as Lab 0) ----------
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"

MOOSE_EXE=/project/me5475/software/moose_torch/me5475/me5475-opt

# No options given: the Lab 4 command.
[ $# -gt 0 ] || set -- --n 200 --out trajectories.npz

echo "----- start: $(date) on $(hostname) -----"
echo "driver options: $*"
# Full interpreter path, not bare `python` (see module_3/examples/pinn_train.sbatch);
# -u so the progress lines reach the log as they are printed.
"$CONDA_PREFIX/bin/python" -u generate_plasticity_data.py --moose-exe "$MOOSE_EXE" "$@"
echo "----- end: $(date) -----"
