#!/bin/bash
# =============================================================================
# gnn_train.sbatch -- Module 8 / Lab 8 -- convert to graphs, train a GNN, or run the MeshGraphNet demo, on ARCC
# =============================================================================
#
# Usage (from the folder holding the scripts and, for convert, your Lab 6 rve_dataset.npz):
#   sbatch gnn_train.sbatch convert                    # rve_dataset.npz -> microstructure_graphs.pt
#   sbatch gnn_train.sbatch gin                        # -> checkpoints/gin_modulus.{pt,png}
#   sbatch gnn_train.sbatch gat                        # -> checkpoints/gat_modulus.{pt,png}
#   sbatch gnn_train.sbatch gin --hidden 32 --out checkpoints/gin_h32.pt   # anything after the variant goes to
#                                                                            # the script; a later --out wins
#   sbatch gnn_train.sbatch meshgraph                  # Task 5 (stretch) -> meshgraphnet_demo.{pt,png}
# All of these run on the CPU (the scripts never move tensors to a GPU), so this is a CPU job on partition mb.
# =============================================================================

#SBATCH --job-name=m8_gnn
#SBATCH --account=me5475
#SBATCH --partition=mb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=m8_gnn-%j.out
#SBATCH --error=m8_gnn-%j.err

set -euo pipefail
# The course Python (PyTorch Geometric included). Not "module purge; conda activate": that sequence fails when
# submitted from a shell that has already sourced setup5475.sh (found in Lab 6, Nov 5).
source /project/me5475/setup5475.sh

VARIANT="${1:-gin}"
shift || true                       # the remaining arguments are passed to the script

case "$VARIANT" in
    convert)   SCRIPT="microstructure_to_graph.py"; DEFAULTS=(--in rve_dataset.npz --out microstructure_graphs.pt) ;;
    gin)       SCRIPT="gnn_microstructure.py"; DEFAULTS=(--graphs microstructure_graphs.pt --architecture gin) ;;
    gat)       SCRIPT="gnn_microstructure.py"; DEFAULTS=(--graphs microstructure_graphs.pt --architecture gat) ;;
    meshgraph) SCRIPT="meshgraphnet_demo.py";  DEFAULTS=() ;;
    *) echo "Unknown variant: $VARIANT (try: convert, gin, gat, meshgraph)"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
echo "----- Job start: $(date) on $(hostname) -----"
SECONDS=0
srun python "$SCRIPT" ${DEFAULTS[@]+"${DEFAULTS[@]}"} "$@"
echo "elapsed: ${SECONDS} s"
echo "----- Job end: $(date) -----"
