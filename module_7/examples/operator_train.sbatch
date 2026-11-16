#!/bin/bash
# =============================================================================
# operator_train.sbatch -- Module 7 / Lab 7 -- train the DeepONet or the FNO on ARCC
# =============================================================================
#
# Usage (from the folder holding the data file and the training scripts):
#   sbatch operator_train.sbatch deeponet                       # -> deeponet_burgers.{pt,png}
#   sbatch operator_train.sbatch fno                            # -> fno_burgers.{pt,png}
#   sbatch operator_train.sbatch fno --modes 8 --out fno_m8.pt  # anything after the variant goes to the script
#   sbatch operator_train.sbatch fno-micro --data rve_dataset.npz --unet-ckpt unet_stress.pt   # Task 5 (stretch)
# Both Burgers scripts run on the CPU (they never move tensors to a GPU), so this is a CPU job on partition mb.
# For deeponet and fno the data file defaults to burgers_data.npz (Task 1); pass --data <file> after the variant to
# change it. fno-micro has no default: give it your Lab 6 field dataset with --data.
# =============================================================================

#SBATCH --job-name=m7_op
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=m7_op-%j.out
#SBATCH --error=m7_op-%j.err

set -euo pipefail
# The course Python. Not "module purge; conda activate": that sequence fails when submitted from a shell that has
# already sourced setup5475.sh (found in Lab 6, Nov 5).
source /project/me5475/setup5475.sh

VARIANT="${1:-deeponet}"
shift || true                       # the remaining arguments are passed to the script

case "$VARIANT" in
    deeponet)  SCRIPT="deeponet_burgers.py";   DEFAULTS=(--data burgers_data.npz --epochs 500) ;;
    fno)       SCRIPT="fno_burgers.py";        DEFAULTS=(--data burgers_data.npz --epochs 500) ;;
    fno-micro) SCRIPT="fno_microstructure.py"; DEFAULTS=() ;;
    *) echo "Unknown variant: $VARIANT (try: deeponet, fno, fno-micro)"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----"
SECONDS=0
srun python "$SCRIPT" ${DEFAULTS[@]+"${DEFAULTS[@]}"} "$@"
echo "training: ${SECONDS} s"
echo "----- Job end: $(date) -----"
