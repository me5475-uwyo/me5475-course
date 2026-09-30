#!/bin/bash
# =============================================================================
# run_moose_diffusion.sbatch -- Module 3 / L13 -- the two diffusion problems in MOOSE
# =============================================================================
#
# Usage (from the folder that holds this file and the two input files):
#   sbatch run_moose_diffusion.sbatch               # 1-D: diffusion_1d.i
#   sbatch run_moose_diffusion.sbatch 2d            # 2-D: diffusion_2d.i
#   sbatch run_moose_diffusion.sbatch 1d nx=100     # anything after 1d/2d is passed
#                                                   # to MOOSE as a command-line override
# Watch it:   squeue -u $USER          (it finishes in seconds)
#
# Everything lands next to this file:
#   moose-diffusion-<jobid>.out     MOOSE's log, then the checks (l2_error, u_mid / u_centre)
#   diffusion_1d_out.csv            the same checks as a table
#   diffusion_1d_out_u_line_*.csv   u at 101 points along the line, for plotting
#   diffusion_1d_out.e              the full solution (Exodus; open in ParaView)
# (diffusion_2d_... for the 2-D case.)  A second run of the same case in the same
# folder overwrites them, and two such jobs running AT THE SAME TIME write over each
# other: one folder per run, or give each its own Outputs/file_base=<name>. (1d and
# 2d write different files, so those two can share a folder.)
#
# One CPU core: these meshes are tiny. Same MOOSE build and module stack as
# Lab 0's run_cantilever.sbatch.
# module_3/readings/moose_diffusion_walkthrough.md walks through every step.
# =============================================================================

#SBATCH --job-name=moose-diffusion      # name shown in squeue
#SBATCH --account=me5475                # course allocation
#SBATCH --partition=mb                  # CPU partition
#SBATCH --time=00:10:00                 # wall-clock limit; the job needs seconds
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # one MPI rank
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=moose-diffusion-%j.out # %j = job id; the log lands here

set -euo pipefail

# --- which problem: 1d (default) or 2d; the rest are MOOSE overrides ----------
CASE="${1:-1d}"
case "$CASE" in
  1d|2d) [ $# -gt 0 ] && shift ;;
  *) echo "usage: sbatch run_moose_diffusion.sbatch [1d|2d] [override ...]   e.g.  2d nx=100" >&2
     exit 2 ;;
esac
INPUT="diffusion_${CASE}.i"
[ -f "$INPUT" ] || { echo "no $INPUT in $(pwd) -- submit from the folder that holds it" >&2; exit 2; }

# --- activate the course MOOSE runtime (same stack as Lab 0) -----------------
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"

MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt

echo "----- start: $(date) on $(hostname) -----"
echo "input: $INPUT   overrides: ${*:-none}"
# This run's own postprocessor CSV, by name: <input>_out.csv, or <name>.csv when the
# overrides include Outputs/file_base=<name> (the last one given counts).
BASE="${INPUT%.i}_out"
for a in "$@"; do case "$a" in Outputs/file_base=*) BASE="${a#Outputs/file_base=}" ;; esac; done
BASE="${BASE//\'/}"; BASE="${BASE//\"/}"
CSV="$BASE.csv"
MARK=".moose-start-$SLURM_JOB_ID"; touch "$MARK"     # to tell a fresh CSV from an older one
srun --ntasks=$SLURM_NTASKS "$MOOSE_EXE" -i "$INPUT" --color off "$@"

# Print that CSV only if this run wrote it; never "some recent CSV" -- another job in
# the same folder may have written one too.
echo
if [ -f "$CSV" ] && [ "$CSV" -nt "$MARK" ]; then echo "Checks ($CSV):"; cat "$CSV"
else echo "Checks: this run wrote no new $CSV; see MOOSE's postprocessor table above."; fi
rm -f "$MARK"
echo "----- end: $(date) -----"
