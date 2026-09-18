#!/bin/bash
# =============================================================================
# sine_mlp.sbatch -- Module 1 / Lecture 8 -- your second ARCC job
# =============================================================================
#
# Runs sine_mlp.py (fit y = sin(pi x) with a small MLP) as a CPU batch job.
# This is the L8 in-class submission; Lab 1's variant is train_mlp.sbatch,
# which trains mlp_constitutive.py instead and asks for a longer walltime.
#
# Usage:
#   sbatch sine_mlp.sbatch
#   squeue -u $USER
#   cat sine_mlp-<jobid>.out
#
# sine_mlp.py writes sine_mlp.png next to itself. It pins the Agg matplotlib
# backend, so it needs no display and works over SSH.
#
# ARCC setup confirmed 2026-06-06 (Phase 0 recon): account=me5475, partition=mb,
# miniconda3/24.3.0 (no pytorch module on MedicineBow -- the conda env is the
# only path to PyTorch).
# =============================================================================

#SBATCH --job-name=sine_mlp
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=00:15:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=sine_mlp-%j.out

module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm

# Keep the math libraries inside the cores SLURM actually gave us. Left unset,
# torch sees all 96 cores on the node and spawns that many threads inside a
# 2-core cgroup -- the threads then fight for the same two cores and the job
# runs minutes instead of seconds.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}


# NOTE (2026-09-18, verified on MedicineBow): call the interpreter by its full
# path, not bare `python`. If you `source setup5475.sh` and THEN `sbatch`, the
# job inherits CONDA_PREFIX from your shell, `module purge` strips conda off
# PATH, and the `conda activate` below then no-ops because conda believes the
# env is already active -- so bare `python` resolves to the base interpreter and
# the job dies with ModuleNotFoundError. Reproduced as jobs 17587307 (clean
# shell, PASS) vs 17587308 (after sourcing, FAIL); this form passes from both.
srun "$CONDA_PREFIX/bin/python" sine_mlp.py
