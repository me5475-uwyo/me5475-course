#!/bin/bash
# =============================================================================
# burgers_data.sbatch -- Module 7 / Lab 7 -- generate 1-D Burgers (u0, u(x,1)) pairs on ARCC
# =============================================================================
#
# Usage (from the folder holding generate_burgers_data.py):
#   sbatch burgers_data.sbatch                         # Task 1: 1000 pairs at N=128 -> burgers_data.npz
#   sbatch burgers_data.sbatch --n 100 --N 256 --seed 7 --out test_N256.npz   # Task 4: a test set at N=256
# Arguments after the script name replace the defaults below. Pure NumPy, CPU only (partition mb).
# Time per pair grows with N (the solver shrinks its time step on finer grids to stay stable): about 0.1 s at
# N <= 128, 0.4 s at N = 256, 2 s at N = 512 on a laptop (module_7/readings/measured_results.md).
# =============================================================================

#SBATCH --job-name=m7_data
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=m7_data-%j.out

set -euo pipefail
source /project/me5475/setup5475.sh

ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--n 1000 --N 128 --seed 42 --out burgers_data.npz)

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----"
SECONDS=0
srun python generate_burgers_data.py "${ARGS[@]}"
echo "data: ${SECONDS} s"
echo "----- Job end: $(date) -----"
