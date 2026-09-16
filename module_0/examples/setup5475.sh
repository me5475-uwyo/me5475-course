#!/usr/bin/env bash
# =============================================================================
# setup5475.sh — one command for a working ME 5475 environment on ARCC MedicineBow
# =============================================================================
#
# WHY THIS MUST BE SOURCED, NOT RUN
# ---------------------------------
# `conda activate` changes the shell it runs in.  If you execute this file
# (./setup5475.sh) bash makes a NEW shell, activates the environment there, then
# throws that shell away — and your prompt is exactly where it started.  So:
#
#     source /project/me5475/setup5475.sh          # works
#     ./setup5475.sh                               # silently does nothing useful
#
# NORMAL USE — nothing to install
# ------------------------------
# Log in, then run this as your second command.  That is the whole workflow:
#
#     ssh <netid>@medicinebow.arcc.uwyo.edu
#     source /project/me5475/setup5475.sh
#
# Nothing is installed and no dotfile is edited, so there is nothing to get
# wrong and nothing that can drift between one person's account and another's.
#
# OPTIONAL SHORTHAND
# ------------------
# If you would rather type one word, add a wrapper to your ~/.bashrc ONCE:
#
#     echo 'setup5475() { source /project/me5475/setup5475.sh; }' >> ~/.bashrc
#     source ~/.bashrc
#
# after which `setup5475` alone does it.  This is a convenience, not a
# requirement — the `source` line above always works.
#
# DO NOT put `setup5475` itself in your ~/.bashrc to run automatically.  It would
# fire on every non-interactive connection too — scp, rsync, git-over-ssh and
# SLURM prologues — slowing them down and occasionally breaking them outright.
# Activate deliberately, when you are about to run something.
#
# IN A BATCH SCRIPT
# -----------------
# Source it the same way inside .sbatch files, instead of repeating the module
# and conda lines in every one:
#
#     source /project/me5475/setup5475.sh
#
# Set SETUP5475_QUIET=1 beforehand to suppress the summary in job output.
# =============================================================================

# --- single place to change if ARCC moves anything -------------------------
ME5475_CONDA_MODULE="${ME5475_CONDA_MODULE:-miniconda3/24.3.0}"
ME5475_CONDA_SH="${ME5475_CONDA_SH:-/apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh}"
ME5475_ENV="${ME5475_ENV:-/project/me5475/envs/ml4sm}"
ME5475_ACCOUNT="${ME5475_ACCOUNT:-me5475}"

# --- refuse to run un-sourced, rather than pretending to work --------------
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "setup5475: this script must be SOURCED, not executed." >&2
    echo "           conda activate only affects the shell it runs in, and" >&2
    echo "           executing this file creates a throwaway shell." >&2
    echo >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_me5475_fail() { echo "setup5475: $*" >&2; return 1; }

# --- modules ---------------------------------------------------------------
module purge 2>/dev/null
if ! module load "$ME5475_CONDA_MODULE" 2>/dev/null; then
    _me5475_fail "could not 'module load $ME5475_CONDA_MODULE'.
           Run 'module avail miniconda' to see what this cluster offers, then
           tell the instructor — the version in this script needs updating."
    return 1 2>/dev/null || exit 1
fi

# --- conda -----------------------------------------------------------------
if [ ! -f "$ME5475_CONDA_SH" ]; then
    _me5475_fail "conda profile script not found at:
             $ME5475_CONDA_SH
           The miniconda install has moved. Tell the instructor."
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$ME5475_CONDA_SH"

if [ ! -d "$ME5475_ENV" ]; then
    _me5475_fail "course environment not found at:
             $ME5475_ENV
           Check you are on MedicineBow and that you are in the me5475 project."
    return 1 2>/dev/null || exit 1
fi
if ! conda activate "$ME5475_ENV" 2>/dev/null; then
    _me5475_fail "'conda activate $ME5475_ENV' failed."
    return 1 2>/dev/null || exit 1
fi

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$ME5475_ACCOUNT}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$ME5475_ACCOUNT}"

# --- report what you actually got ------------------------------------------
# The point of printing this is that "it didn't work" becomes a thing you can
# paste to the instructor instead of describe.
if [ -z "${SETUP5475_QUIET:-}" ]; then
    python - <<'PYCHECK'
import importlib, sys, os
need = ("torch", "numpy", "matplotlib", "pandas")
have, miss = [], []
for m in need:
    try:
        mod = importlib.import_module(m)
        have.append(f"{m} {getattr(mod, '__version__', '?')}")
    except Exception:
        miss.append(m)
print("ME 5475 environment ready")
print(f"  python   {sys.version.split()[0]}  ({sys.prefix})")
print(f"  packages {', '.join(have) if have else 'none found'}")
if miss:
    print(f"  MISSING  {', '.join(miss)}")
    print("  -> Do NOT pip install into the shared environment.")
    print("     Report this to the instructor; it affects everyone.")
PYCHECK
fi
