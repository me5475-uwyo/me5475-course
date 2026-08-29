#!/usr/bin/env bash
# =============================================================================
# run_moose_local.sh -- run the plate-with-hole example on a laptop
# =============================================================================
#
# Usage:
#   ./run_moose_local.sh              # default: uniform_refine = 0
#   ./run_moose_local.sh 2            # override refinement level via CLI
#
# Prerequisites:
#   - MOOSE installed via the INL conda path (see Lecture 3 install prompt).
#   - The 'moose' conda environment is activated, OR moose-opt is on PATH.
# =============================================================================

set -euo pipefail

REFINE_LEVEL="${1:-0}"
INPUT_FILE="plate_with_hole.i"

if ! command -v moose-opt &> /dev/null; then
    echo "ERROR: moose-opt not found on PATH."
    echo "       Activate your MOOSE conda environment: 'conda activate moose'"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: $INPUT_FILE not found in $(pwd)"
    exit 1
fi

echo "Running $INPUT_FILE with uniform_refine = $REFINE_LEVEL"
echo "MOOSE version: $(moose-opt --version 2>&1 | head -1)"

moose-opt -i "$INPUT_FILE" Mesh/uniform_refine="$REFINE_LEVEL"

echo
echo "Done. Outputs:"
ls -la plate_with_hole_out.e plate_with_hole_out.csv 2>/dev/null || echo "(no outputs found - check for errors above)"
