#!/bin/bash
# =============================================================================
# deeponet_demo.sbatch -- Module 7 / L31 demo -- Burgers data, then DeepONet, on ARCC
# =============================================================================
#
# Usage (from a folder holding this file, generate_burgers_data.py and deeponet_burgers.py):
#   sbatch deeponet_demo.sbatch
# One CPU job (neither script uses a GPU):
#   1. generate_burgers_data.py makes 1000 (u0, u(x,1)) pairs on 128 points  -> burgers_data.npz   (1.5-2 min)
#   2. deeponet_burgers.py trains the DeepONet for 500 epochs               -> deeponet_burgers.{pt,png} (2-3.5 min)
# Output: deeponet_demo-<jobid>.out. Measured Nov 12-13, 2026: 3.5 to 5.5 minutes, depending on the node, plus queue.
# CPU partition: mb (account me5475).
# =============================================================================

#SBATCH --job-name=m7_deeponet
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=deeponet_demo-%j.out

set -euo pipefail
# The course Python (the same line as the Lab 6 job scripts).
source /project/me5475/setup5475.sh

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----"
SECONDS=0
srun python generate_burgers_data.py --n 1000 --N 128 --seed 42 --out burgers_data.npz
echo "data: ${SECONDS} s"
SECONDS=0
srun python deeponet_burgers.py --data burgers_data.npz --epochs 500 --out deeponet_burgers.pt
echo "training: ${SECONDS} s"
echo "----- Job end: $(date) -----"
