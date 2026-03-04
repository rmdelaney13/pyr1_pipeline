#!/bin/bash
# Collect Boltz2 binary PDB files for MD candidates and push to GitHub.
# Run on Alpine after md_candidates_lca_top100.csv has been generated locally
# and pushed/synced to the repo.
#
# Usage:
#   bash ml_modelling/analysis/boltz_LCA/collect_md_pdbs.sh

set -euo pipefail

REPO=/projects/ryde3462/software/pyr1_pipeline
BOLTZ_DIR=/scratch/alpine/ryde3462/boltz_lca/output_lca_binary
CSV=$REPO/ml_modelling/analysis/boltz_LCA/md_candidates_lca_top100.csv
OUT_DIR=$REPO/ml_modelling/analysis/boltz_LCA/md_candidate_pdbs

if [ ! -f "$CSV" ]; then
    echo "ERROR: CSV not found at $CSV"
    echo "Generate it locally first: python plot_strategy_h_figures.py"
    exit 1
fi

mkdir -p "$OUT_DIR"

copied=0
missing=0

# Skip header, read pair_id from column 1
tail -n +2 "$CSV" | cut -d',' -f1 | while read pair_id; do
    src="$BOLTZ_DIR/boltz_results_${pair_id}/predictions/${pair_id}/${pair_id}_model_0.pdb"
    if [ -f "$src" ]; then
        cp "$src" "$OUT_DIR/${pair_id}.pdb"
        copied=$((copied + 1))
    else
        echo "WARNING: missing $src"
        missing=$((missing + 1))
    fi
done

n_copied=$(ls "$OUT_DIR"/*.pdb 2>/dev/null | wc -l)
echo "Copied $n_copied PDB files to $OUT_DIR"

if [ "$n_copied" -eq 0 ]; then
    echo "ERROR: No PDBs found. Check BOLTZ_DIR path."
    exit 1
fi

cd "$REPO"
git add ml_modelling/analysis/boltz_LCA/md_candidate_pdbs/
git add ml_modelling/analysis/boltz_LCA/md_candidates_lca_top100.csv
git commit -m "Add LCA top-100 MD candidate PDBs and CSV for Strategy H filter"
git push

echo "Done. $n_copied PDB files committed and pushed."
