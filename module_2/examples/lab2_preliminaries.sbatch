#!/bin/bash
# =============================================================================
# lab2_preliminaries.sbatch -- Module 2 / Lab 2 Task 1 -- the three preliminaries
# =============================================================================
#
# Runs Lab 2's Task 1 as a batch job on a compute node -- the LR finder and the
# reproducibility check train small networks, and training does not belong on
# the login node:
#   1a  lr_finder.py                 -> lr_finder_output.png
#   1b  loss_curve_gallery.py        -> loss_curve_gallery.png
#   1c  reproducibility_checklist.py -> reproducibility_check.txt
#
# Submit from your preliminaries/ folder, with the three scripts copied there:
#
#     cd labs/lab_2/<your-github-handle>/preliminaries
#     sbatch lab2_preliminaries.sbatch
#
# The LR finder reads your Lab 1 dataset. By default that is
#     ../../../lab_1/<your-github-handle>/data_generation/data/single_element.csv
# with <your-github-handle> taken from the folder you submit from. Point it
# elsewhere with:   DATA=/path/to/your.csv sbatch lab2_preliminaries.sbatch
# =============================================================================

#SBATCH --job-name=lab2_prelim
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=prelim-%j.out
#SBATCH --error=prelim-%j.err

set -euo pipefail

# Activate the shared ML conda environment (same block as optuna_sweep.sbatch).
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
PY="$CONDA_PREFIX/bin/python"    # full path: see the note in optuna_sweep.sbatch

# Keep PyTorch's threads inside the cores SLURM gave us (the L8 lesson).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"
echo "----- Task 1 start: $(date) on $(hostname) -----"

HANDLE="$(basename "$(dirname "$SLURM_SUBMIT_DIR")")"
DATA="${DATA:-$SLURM_SUBMIT_DIR/../../../lab_1/$HANDLE/data_generation/data/single_element.csv}"
if [ ! -f "$DATA" ]; then
    echo "ERROR: Lab 1 dataset not found: $DATA" >&2
    echo "       Submit from labs/lab_2/<handle>/preliminaries/, or set DATA=/path/to/your.csv" >&2
    exit 1
fi
for f in lr_finder.py loss_curve_gallery.py reproducibility_checklist.py; do
    [ -f "$f" ] || { echo "ERROR: $f not in $(pwd) -- copy it from module_2/examples/ first" >&2; exit 1; }
done

echo "== 1a LR finder on $DATA"
srun "$PY" lr_finder.py --data "$DATA" --out lr_finder_output.png

echo "== 1b loss-curve gallery"
srun "$PY" loss_curve_gallery.py --out loss_curve_gallery.png

echo "== 1c reproducibility checklist"
srun "$PY" reproducibility_checklist.py | tee reproducibility_check.txt

echo "----- Task 1 end: $(date) -----"
