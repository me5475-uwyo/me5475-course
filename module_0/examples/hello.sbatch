#!/bin/bash
# =============================================================================
# hello.sbatch -- Module 0 / Lecture 3 -- first ARCC job
# =============================================================================
#
# Submits hello.py as a 1-core, 5-minute batch job on ARCC.
# Demonstrated live in Lecture 3 after instructor-requested account onboarding.
#
# Usage:
#   sbatch hello.sbatch
#   squeue -u $USER
#   cat hello-<jobid>.out
#
# ARCC setup confirmed 2026-06-06 (Phase 0 recon): account=me5475, partition=mb,
# miniconda3/24.3.0 for Python (no standalone python/3.11 module on mb cluster).
# =============================================================================

#SBATCH --job-name=hello
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=hello-%j.out
#SBATCH --error=hello-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>

# -----------------------------------------------------------------------------
# Load miniconda3 module; conda base provides Python 3.12.
# -----------------------------------------------------------------------------
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh

echo "----- Job start: $(date) -----"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo

srun python hello.py

echo
echo "----- Job end: $(date) -----"
