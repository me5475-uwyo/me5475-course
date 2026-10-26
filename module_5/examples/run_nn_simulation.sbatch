#!/bin/bash
# =============================================================================
# run_nn_simulation.sbatch -- Module 5 / Lab 5 Task 2 -- the plate with a network as its material
# =============================================================================
#
# Runs plate_with_hole_nn_material.i twice, one after the other, on 4 MPI ranks:
#   1. with the network (ComputeTorchScriptStress)     -> plate_nn_out.{csv,e}
#   2. with Lab 0's elastic material (the reference)   -> plate_elastic_out.{csv,e}
# Same mesh, same load, same solver; only the [Materials] choice differs.
#
# Usage (from the folder holding the input file and your TorchScript file):
#   sbatch run_nn_simulation.sbatch                          # default model mlp_constitutive_ts.pt
#   MODEL=my_net.pt sbatch run_nn_simulation.sbatch          # another TorchScript file
#   sbatch run_nn_simulation.sbatch sigma_inf=1              # extra MOOSE overrides go to BOTH runs
#   TAG=_s1 sbatch run_nn_simulation.sbatch sigma_inf=1      # ... into plate_nn_s1_out, plate_elastic_s1_out
#
# Measured: module_5/readings/measured_results.md (run times, Newton iterations, numbers).
# Course executable with ComputeTorchScriptStress: team build of 2026-10-03 (libtorch 2.5.1, the
# version of the course Python environment, so files exported there load here).
# =============================================================================

#SBATCH --job-name=nn_moose
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4                   # MPI ranks for MOOSE
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=nn_moose-%j.out
#SBATCH --error=nn_moose-%j.err

set -euo pipefail

# --- the course MOOSE runtime (the same block as every course MOOSE job script) ---
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"   # miniconda's older libstdc++ must not win
MOOSE_EXE=/project/me5475/software/moose_torch/me5475/me5475-opt

INPUT=${INPUT:-plate_with_hole_nn_material.i}
MODEL=${MODEL:-mlp_constitutive_ts.pt}
TAG=${TAG:-}
cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----"

# --- pre-flight checks: fail in seconds, not after a queue wait and a crash ---
[ -x "$MOOSE_EXE" ] || { echo "ERROR: $MOOSE_EXE not found or not executable."; exit 1; }
# grep -c reads the whole list; grep -q would stop early, and with pipefail the cut-off pipe fails the check
if ! "$MOOSE_EXE" --registry 2>/dev/null | grep -c ComputeTorchScriptStress > /dev/null; then
    echo "ERROR: $MOOSE_EXE has no ComputeTorchScriptStress -- wrong executable?"; exit 1
fi
[ -f "$INPUT" ] || { echo "ERROR: input file $INPUT not found in $(pwd)."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: TorchScript file $MODEL not found in $(pwd). Run export_torchscript.py first."; exit 1; }
echo "OK: executable has ComputeTorchScriptStress; input $INPUT; model $MODEL ($(stat -c %s "$MODEL") bytes)"
echo "Extra MOOSE arguments for both runs: ${*:-(none)}"

run() {   # run <file_base> <extra MOOSE arguments...>
    local base=$1; shift
    echo; echo "===== $base ====="
    /usr/bin/time -o "$base.time" -f "wall %e s" srun --ntasks="$SLURM_NTASKS" "$MOOSE_EXE" -i "$INPUT" \
        --color off "Outputs/file_base=$base" "$@" > "$base.log" 2>&1 \
        || { echo "ERROR: MOOSE failed for $base; the end of $base.log:"; tail -30 "$base.log"; exit 1; }
    echo "$base: $(tail -1 "$base.time")   (MOOSE log: $base.log)"
    grep -E "Nonlinear \|R\|" "$base.log" | tail -12 || true
    echo "--- $base.csv ---"; cat "$base.csv"
}

# 1. the network
run "plate_nn${TAG}_out" "Materials/nn_stress/filename=$MODEL" "$@"
# 2. the elastic reference: the same file, Lab 0's material
run "plate_elastic${TAG}_out" "Materials/active='elasticity_tensor elastic_stress'" "$@"

echo; echo "----- Job end: $(date) -----"
ls -l plate_*"${TAG}"_out.*
