#!/bin/bash
# =============================================================================
# run_nn_simulation_commented.sbatch -- Module 5 / Lab 5 Task 2 -- the plate with a network as its material
# =============================================================================
#
# The COMMENTED copy of run_nn_simulation.sbatch: the same commands, with more comments (L27).
# Either file runs the same job; submit this one with  sbatch run_nn_simulation_commented.sbatch
#
# Getting new course files: they arrive on the branch main (HEAD only means "where you are now").
# First run git status, and commit the lab files you mean to keep on your lab branch (git add <those files>).
#   Existing lab:  git checkout main && git pull           # get the course's latest
#                  ls module_5/examples                    # check the new files are there
#                  git checkout lab_5 && git merge main    # bring them into your lab branch
#   New lab:       git checkout main && git pull && git checkout -b lab_6    # branch from an updated main
# If a merge reports a conflict, resolve it before continuing.
# More: module_0/readings/git_basics_and_course_repo_primer.md (git basics, branches, this repository).
#
# Runs plate_with_hole_nn_material.i twice, one after the other, on 4 MPI ranks:
#   1. with the network (ComputeTorchScriptStress)     -> plate_nn_out.{csv,e}
#   2. with Lab 0's elastic material (the reference)   -> plate_elastic_out.{csv,e}
# Same mesh, same load, same solver; only the [Materials] choice differs.
#
# Usage (from the folder holding the input file and your TorchScript file):
#   sbatch run_nn_simulation_commented.sbatch                       # default model mlp_constitutive_ts.pt
#   MODEL=my_net.pt sbatch run_nn_simulation_commented.sbatch       # another TorchScript file
#   sbatch run_nn_simulation_commented.sbatch sigma_inf=1           # extra MOOSE overrides go to BOTH runs
#   TAG=_s1 sbatch run_nn_simulation_commented.sbatch sigma_inf=1   # ... into plate_nn_s1_out, plate_elastic_s1_out
#
# One sbatch = ONE job. Inside it, each srun (one per run below) is a job STEP:
#   squeue -u $USER          shows the one job while it waits or runs
#   sacct -j <jobid>         lists its steps afterwards: <jobid>.0 (the network), <jobid>.1 (elastic)
#
# What you get in the submit folder:
#   nn_moose-<jobid>.out     this script's printout: per run, the wall time, the |R| lines and the CSV
#   nn_moose-<jobid>.err     error messages; if the job ends within seconds, check .err AND .out (this
#                            script's own checks print to .out); MOOSE's detailed errors are in the run's .log
#   plate_*_out.log          MOOSE's full log for each run
#   plate_*_out.e / .csv     the fields (open in ParaView) and the postprocessor values
#   plate_*_out.time         the wall time of each run
#
# Measured: module_5/readings/measured_results.md (run times, Newton iterations, numbers).
# Course executable with ComputeTorchScriptStress: team build of 2026-10-03 (libtorch 2.5.1, the
# version of the course Python environment, so files exported there load here).
# =============================================================================

# --- Slurm requests. #SBATCH lines must come before the first command; Slurm reads them, bash ignores them.
#     %j in a file name becomes the job number (job 12345 writes nn_moose-12345.out).
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

# -e exits on unhandled command failures (tests and &&/|| lists have exceptions).
# -u errors on expansion of unset variables; pipefail makes a pipeline fail if any command in it fails.
# Expected failures below are handled explicitly by if/! or ||.
set -euo pipefail

# --- the course MOOSE runtime (the same block as every course MOOSE job script) ---
# With sbatch's default export settings, the job inherits your exported environment variables.
# Purge the loaded modules, then load the specified course versions to reduce module-state differences.
# On the Nov 4 ARCC rehearsal, arcc/1.0 and slurm/latest remained as sticky modules.
# This does not clear unrelated inherited variables; MODEL, INPUT, TAG and other settings can still matter.
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
# MOOSE was compiled with gcc 14 and needs a C++ runtime (libstdc++.so.6) that provides gcc 14's symbols;
# the conda environment brings an older copy. Ask gcc where its copy lives, keep the folder (dirname), and put
# that folder FIRST in LD_LIBRARY_PATH: the loader searches left to right, so the first match wins. In the
# Nov 4 rehearsal, omitting this let conda's older library win, and MOOSE stopped at start-up with
# "version `GLIBCXX_3.4.32' not found". ${VAR:-} means "VAR, or empty if VAR is unset or empty": with set -u,
# a bare $LD_LIBRARY_PATH would stop the script on a node where it is not set.
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"   # miniconda's older libstdc++ must not win
MOOSE_EXE=/project/me5475/software/moose_torch/me5475/me5475-opt

# Defaults you can override when you submit, e.g. MODEL=my_net.pt sbatch ... (sbatch passes your environment
# variables into the job). ${VAR:-default} uses default when VAR is unset or empty; otherwise VAR's value.
# TAG is empty unless you set it; it goes into the output names, so runs with different tags don't overwrite
# each other (reusing a tag overwrites).
INPUT=${INPUT:-plate_with_hole_nn_material.i}
MODEL=${MODEL:-mlp_constitutive_ts.pt}
TAG=${TAG:-}
cd "$SLURM_SUBMIT_DIR"           # the folder you ran sbatch from: inputs are read and outputs written here
echo "----- Job start: $(date) on $(hostname) -----"

# --- pre-flight checks: fail in seconds, not after a queue wait and a crash ---
# [ -x f ] / [ -f f ]: f exists and is executable / is a regular file. "A || { ...; }" runs the braces only if A
# failed: print a clear message, then exit 1 (sacct shows the job as FAILED).
[ -x "$MOOSE_EXE" ] || { echo "ERROR: $MOOSE_EXE not found or not executable."; exit 1; }
# Is the course material compiled into this executable? --registry lists every object type it knows.
# grep -c reads the whole list; grep -q would stop early, and with pipefail the cut-off pipe fails the check
if ! "$MOOSE_EXE" --registry 2>/dev/null | grep -c ComputeTorchScriptStress > /dev/null; then
    echo "ERROR: $MOOSE_EXE has no ComputeTorchScriptStress -- wrong executable?"; exit 1
fi
[ -f "$INPUT" ] || { echo "ERROR: input file $INPUT not found in $(pwd)."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: TorchScript file $MODEL not found in $(pwd). Run export_torchscript.py first."; exit 1; }
echo "OK: executable has ComputeTorchScriptStress; input $INPUT; model $MODEL ($(stat -c %s "$MODEL") bytes)"
echo "Extra MOOSE arguments for both runs: ${*:-(none)}"

# run <file_base> <MOOSE arguments...>: one MOOSE run as one job step.
#   local base=$1; shift   takes the output name; "$@" is then the rest, passed to MOOSE on the command line
#   srun                   starts MOOSE on the job's 4 MPI ranks; /usr/bin/time writes the wall time to $base.time
#   Outputs/file_base      gives each run its own file names, so the second run doesn't overwrite the first
#   > "$base.log" 2>&1     MOOSE's full log goes to its own file; this script then prints only a summary
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

# The same input file twice; the material is chosen by a run-time flag, not by editing the file.
# Here (outside the function) "$@" is what you typed after the script's name, e.g. sigma_inf=1: both runs get it.
# 1. the network
#    The input already says active = 'nn_stress'; this flag only sets which TorchScript file it loads.
run "plate_nn${TAG}_out" "Materials/nn_stress/filename=$MODEL" "$@"
# 2. the elastic reference: the same file, Lab 0's material
#    Overrides active to Lab 0's elastic pair. The value has a space, so it is quoted: the inner single quotes
#    reach MOOSE, the outer double quotes keep it one argument.
run "plate_elastic${TAG}_out" "Materials/active='elasticity_tensor elastic_stress'" "$@"

echo; echo "----- Job end: $(date) -----"
ls -l plate_*"${TAG}"_out.*
