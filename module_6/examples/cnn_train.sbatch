#!/bin/bash
# =============================================================================
# cnn_train.sbatch -- Module 6 / Lab 6 -- train CNN on ARCC
# =============================================================================

#
# Usage (from the folder holding rve_dataset.npz and the training script):
#   sbatch cnn_train.sbatch modulus                       # defaults -> checkpoints/cnn_modulus.{pt,png}
#   sbatch cnn_train.sbatch unet                          # stretch  -> checkpoints/unet_stress.{pt,png}
#   sbatch cnn_train.sbatch modulus --lr 7e-4 --depth 3 --out checkpoints/cnn_best.pt
#                                                         # anything after the variant goes to the script;
#                                                         # a later --out overrides the default one
# Both scripts use the GPU when the job has one; the printout's "Device:" line says which ran.
# GPU partitions: mb-l40s, mb-a30 (account me5475).
# =============================================================================

#SBATCH --job-name=m6_cnn
#SBATCH --account=me5475
#SBATCH --partition=mb-l40s,mb-a30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=m6_cnn-%j.out
#SBATCH --error=m6_cnn-%j.err

set -euo pipefail
# The course Python. Not "module purge; conda activate": if you sourced setup5475.sh before sbatch,
# that sequence leaves the base Python first and the job fails with "No module named 'matplotlib'"
# (reproduced Nov 5, 2026: jobs 25124868 fail, 25124870 clean shell pass, 25124871 this line pass).
source /project/me5475/setup5475.sh

VARIANT="${1:-modulus}"
shift || true                       # the remaining arguments are passed to the script

case "$VARIANT" in
    modulus)
        SCRIPT="cnn_microstructure_to_modulus.py"
        ARGS="--data rve_dataset.npz --epochs 200 --out checkpoints/cnn_modulus.pt"
        ;;
    unet)
        SCRIPT="unet_microstructure_to_stress.py"
        ARGS="--data rve_dataset.npz --epochs 200 --out checkpoints/unet_stress.pt"
        ;;
    *)
        echo "Unknown variant: $VARIANT (try: modulus, unet)"
        exit 1
        ;;
esac

cd $SLURM_SUBMIT_DIR
echo "----- Job start: $(date) -----"
srun python $SCRIPT $ARGS "$@"
echo "----- Job end: $(date) -----"
