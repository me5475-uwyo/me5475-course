#!/bin/bash
# =============================================================================
# run_plate_reference.sbatch -- Module 3 / L14 -- the Lab 3 plate, solved with MOOSE
# =============================================================================
#
# Usage (from the folder that holds this file and plate_square_hole_reference.i):
#   sbatch run_plate_reference.sbatch                     # coarse mesh, nt=24 nr=48 (seconds)
#   sbatch run_plate_reference.sbatch nt=48 nr=96         # any mesh level: MOOSE overrides
#   sbatch --mem=48G run_plate_reference.sbatch nt=192 nr=384
#                                                         # the shared reference level (~2 min);
#                                                         # you do not need to: it is staged at
#                                                         # /project/me5475/examples/lab3_reference/
# Watch it:   squeue -u $USER
#
# Everything lands next to this file:
#   moose-plate-<jobid>.out                 MOOSE's log, then the checks
#   plate_square_hole_reference_out.csv     the checks (postprocessors) as a table
#   plate_square_hole_reference_out.e       the full solution: displacements and
#                                           nodal stresses (ParaView, or
#                                           extract_moose_reference.py --exodus ...)
# A second run with the same output name in the same folder overwrites them, and two
# such jobs running AT THE SAME TIME write over each other: one folder per mesh, or
# give each run its own Outputs/file_base=<name> (its checks are then <name>.csv).
#
# Same MOOSE build and module stack as Lab 0's run_cantilever.sbatch.
# module_3/readings/moose_plate_walkthrough.md walks through every step.
# =============================================================================

#SBATCH --job-name=moose-plate          # name shown in squeue
#SBATCH --account=me5475                # course allocation
#SBATCH --partition=mb                  # CPU partition
#SBATCH --time=00:15:00                 # wall-clock limit
#SBATCH --nodes=1
#SBATCH --ntasks=8                      # 8 MPI ranks (the reference's measured setup)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                        # enough below the reference level; see --mem above
#SBATCH --output=moose-plate-%j.out     # %j = job id; the log lands here

set -euo pipefail

INPUT=plate_square_hole_reference.i
[ -f "$INPUT" ] || { echo "no $INPUT in $(pwd) -- submit from the folder that holds it" >&2; exit 2; }
# No overrides given: run the coarse level, not the file's 192/384 reference default.
[ $# -gt 0 ] || set -- nt=24 nr=48

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
echo "input: $INPUT   overrides: $*"
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
if [ -f "$CSV" ] && [ "$CSV" -nt "$MARK" ]; then echo "Checks ($CSV, one column per postprocessor):"; cat "$CSV"
else echo "Checks: this run wrote no new $CSV; see MOOSE's postprocessor table above."; fi
rm -f "$MARK"
echo "----- end: $(date) -----"
