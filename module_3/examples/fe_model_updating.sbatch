#!/bin/bash
# =============================================================================
# fe_model_updating.sbatch -- Module 3, companion to L18 -- E and nu by FE model updating
# -----------------------------------------------------------------------------
#   sbatch fe_model_updating.sbatch                     # seed 0, displacements + force
#   sbatch fe_model_updating.sbatch --seed 1
#   sbatch fe_model_updating.sbatch --seed 0 --no-force # displacements only (slide 5's question)
#   sbatch fe_model_updating.sbatch --nt 48 --nr 96     # a finer inversion mesh
# Any extra arguments go to fe_model_updating.py.  Needs measurements.csv in this folder
# (extract_moose_reference.py --mode random), as for plate_with_hole_inverse.py.
# Output: femu-<jobid>.out, and femu_seed<k>_<data>_<mesh>_<jobid>.json with the results.
# Resources are sized to what the runs used (sacct, 2026-10-09), so the job can start in a
# gap in the queue: per MOOSE solve, peak memory about 0.56 GB (24/48) and 1.4 GB (48/96);
# whole job about 1-2 min (24/48) and 3.5 min (48/96).  Only those two meshes are covered: the 192/384
# mesh that made the data needs about 48 GB (run_plate_reference.sbatch).
# module_3/readings/fe_model_updating_walkthrough.md walks through it.
# =============================================================================
#SBATCH --job-name=femu
#SBATCH --account=me5475                # course allocation
#SBATCH --partition=mb                  # CPU partition: MOOSE needs no GPU
#SBATCH --time=00:10:00                 # measured: 1-2 min (24/48), 3.5 min (48/96)
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # the coarse inversion mesh solves in seconds on 1 rank
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                        # measured peak: 0.56 GB (24/48), 1.4 GB (48/96)
#SBATCH --output=femu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f measurements.csv ] || { echo "ERROR: measurements.csv not in $(pwd) -- run extract_moose_reference.py --mode random first" >&2; exit 1; }

# --- the course MOOSE runtime (same stack as run_plate_reference.sbatch) ------
source /apps/s/lmod/lmod/init/bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"

echo "----- start: $(date) on $(hostname) -----"
"$CONDA_PREFIX/bin/python" fe_model_updating.py --measurements measurements.csv "$@"
echo "----- end: $(date) -----"
