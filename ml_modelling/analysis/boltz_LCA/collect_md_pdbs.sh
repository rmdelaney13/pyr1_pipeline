#!/bin/bash
# Collect Boltz2 binary PDB files for MD candidates and push to GitHub.
# Run on Alpine after md_candidates_lca_top100.csv has been generated locally
# and pushed/synced to the repo.
#
# Usage:
#   bash ml_modelling/analysis/boltz_LCA/collect_md_pdbs.sh

set -euo pipefail

REPO=/projects/ryde3462/software/pyr1_pipeline
BOLTZ_BASE=/scratch/alpine/ryde3462/boltz_lca
CSV=$REPO/ml_modelling/analysis/boltz_LCA/md_candidates_lca_top100.csv
OUT_DIR=$REPO/ml_modelling/analysis/boltz_LCA/md_candidate_pdbs

# MSA-mode Boltz binary PDB outputs for LCA (exclude template predictions)
BOLTZ_DIRS=(
    "$BOLTZ_BASE/output_lca_binary"
    "$BOLTZ_BASE/lca_msa_binary_output/output_tier1_binary"
    "$BOLTZ_BASE/lca_msa_binary_output/output_tier4_binary"
)

if [ ! -f "$CSV" ]; then
    echo "ERROR: CSV not found at $CSV"
    echo "Generate it locally first: python plot_strategy_h_figures.py"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Searching ${#BOLTZ_DIRS[@]} Boltz output directories:"
for d in "${BOLTZ_DIRS[@]}"; do
    if [ -d "$d" ]; then
        echo "  OK: $d"
    else
        echo "  MISSING: $d"
    fi
done
echo ""

# Skip header, read pair_id from column 1
tail -n +2 "$CSV" | cut -d',' -f1 | while read pair_id; do
    found=0
    for boltz_dir in "${BOLTZ_DIRS[@]}"; do
        src="$boltz_dir/boltz_results_${pair_id}/predictions/${pair_id}/${pair_id}_model_0.pdb"
        if [ -f "$src" ]; then
            cp "$src" "$OUT_DIR/${pair_id}.pdb"
            found=1
            break
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "WARNING: missing $pair_id (not found in any directory)"
    fi
done

n_copied=$(ls "$OUT_DIR"/*.pdb 2>/dev/null | wc -l)
echo ""
echo "Copied $n_copied PDB files to $OUT_DIR"

if [ "$n_copied" -eq 0 ]; then
    echo "ERROR: No PDBs found. Check BOLTZ_DIRS paths."
    exit 1
fi

cd "$REPO"
git add -f ml_modelling/analysis/boltz_LCA/md_candidate_pdbs/
git add -f ml_modelling/analysis/boltz_LCA/md_candidates_lca_top100.csv
git commit -m "Add LCA MD candidate PDBs (120 designs: top100 + negative controls)"
git push

echo "Done. $n_copied PDB files committed and pushed."
