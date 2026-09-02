#!/bin/bash
# =============================================================================
# Submit the cantilever-beam MOOSE job on ARCC MedicineBow.
#
#   ssh <netid>@medicinebow.arcc.uwyo.edu
#   mkdir -p ~/ME5475/examples          # course folder, one per student
#   # copy cantilever_beam.i, run_cantilever.sbatch, plot_beam_results.py here
#   cd ~/ME5475/examples
#   sbatch run_cantilever.sbatch        # returns a job id
#   squeue -u $USER                     # watch it
#
# SLURM runs the job from the directory you submitted from, so the input file
# and all outputs stay together in ~/ME5475/examples.
# module_0/readings/moose_cantilever_arcc.md walks through every block of this file.
# =============================================================================

#SBATCH --job-name=cantilever       # name shown in squeue
#SBATCH --account=me5475            # course allocation — billing/authorization
#SBATCH --partition=mb              # CPU partition on MedicineBow
#SBATCH --time=00:10:00             # wall-clock limit; this job needs seconds
#SBATCH --nodes=1                   # everything on one node
#SBATCH --ntasks=2                  # 2 MPI ranks — plenty for 1,000 elements
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=cantilever-%j.out  # %j = job id; stdout lands here
#SBATCH --error=cantilever-%j.err

set -euo pipefail

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
srun --ntasks=$SLURM_NTASKS "$MOOSE_EXE" -i cantilever_beam.i

echo "Postprocessors (tip_disp_y, max_vonmises_stress, num_elements):"
tail -n 2 cantilever_beam_out.csv
echo "----- end: $(date) -----"
