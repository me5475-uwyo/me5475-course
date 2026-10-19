#!/bin/bash
# [fallback] ME-5475 after-L19 examples. ARCC validation recorded in measured_results.md Section 7.
# Submit from module_4/examples. Optional arguments: model (or all), then dt.
# Resource requests are limits, not measured run costs.
#SBATCH --job-name=constitutive
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=constitutive-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from module_4/examples}"
model="${1:-all}"
dt="${2:-0.01}"
case "$model" in
  all) models=(drucker_prager kinematic voce hill) ;;
  drucker_prager|kinematic|voce|hill) models=("$model") ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
case "$dt" in
  0.01|0.005) ;;
  *) echo "For this walkthrough choose dt=0.01 or 0.005" >&2; exit 2 ;;
esac
for name in "${models[@]}"; do
  test -f "constitutive_${name}.i"
done

source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"
MOOSE_EXE=/project/me5475/software/moose_torch/me5475/me5475-opt
test -x "$MOOSE_EXE"

run_dir="${PWD}/constitutive_runs_${SLURM_JOB_ID:?}"
mkdir "$run_dir"
"$MOOSE_EXE" --version > "$run_dir/version.txt" 2>&1
for name in "${models[@]}"; do
  input="constitutive_${name}.i"
  cp "$input" "$run_dir/"
  output="$run_dir/$name"
  echo "Checking and running $name with dt=$dt"
  "$MOOSE_EXE" -i "$input" --check-input "Executioner/dt=$dt" \
    "Outputs/file_base=$output" > "$output.check.log" 2>&1
  "$MOOSE_EXE" -i "$input" "Executioner/dt=$dt" \
    "Outputs/file_base=$output" > "$output.run.log" 2>&1
  test -s "$output.csv"
done
echo "Runs completed. Inspect logs and CSVs in $run_dir; completion is not physical validation."
