#!/bin/bash
# ============================================================================
# Aggregate Boltz2 ternary results for ALL three LCA ligands
# ============================================================================
# Run after all ternary SLURM jobs complete (from run_boltz_lca_ternary.sh).
# Produces:
#   - Per-ligand ternary CSVs
#   - Merged binary+ternary CSVs (joined by pair_id / name)
#   - Quick analysis with labels for AUC computation
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/aggregate_boltz_lca_ternary.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"

module load anaconda
source activate boltz_env

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Aggregating Boltz2 ternary results"
echo "============================================"

# ── LCA ternary ──
echo ""
echo "--- Lithocholic Acid (ternary) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --ternary-dir "${SCRATCH}/output_lca_ternary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_ternary_results.csv"

# ── GlycoLCA ternary ──
echo ""
echo "--- GlycoLithocholic Acid (ternary) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --ternary-dir "${SCRATCH}/output_glca_ternary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_ternary_results.csv"

# ── LCA-3-S ternary ──
echo ""
echo "--- Lithocholic Acid 3-Sulfate (ternary) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --ternary-dir "${SCRATCH}/output_lca3s_ternary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_ternary_results.csv"

# ── Merge binary + ternary per ligand ──
echo ""
echo "============================================"
echo "Merging binary + ternary results"
echo "============================================"

python -c "
import csv
import sys
from pathlib import Path

results_dir = '${RESULTS_DIR}'

for ligand in ['lca', 'glca', 'lca3s']:
    binary_csv = Path(results_dir) / f'boltz_{ligand}_binary_results.csv'
    ternary_csv = Path(results_dir) / f'boltz_{ligand}_ternary_results.csv'
    merged_csv = Path(results_dir) / f'boltz_{ligand}_merged_results.csv'

    if not binary_csv.exists():
        print(f'  {ligand}: binary results not found, skipping merge')
        continue
    if not ternary_csv.exists():
        print(f'  {ligand}: ternary results not found, skipping merge')
        continue

    # Read binary results (keyed by name)
    binary = {}
    with open(binary_csv) as f:
        reader = csv.DictReader(f)
        binary_fields = list(reader.fieldnames)
        for row in reader:
            binary[row['name']] = row

    # Read ternary results (keyed by name)
    ternary = {}
    with open(ternary_csv) as f:
        reader = csv.DictReader(f)
        ternary_fields = list(reader.fieldnames)
        for row in reader:
            ternary[row['name']] = row

    # Merge: binary fields + ternary fields (skip duplicate 'name')
    all_names = sorted(set(list(binary.keys()) + list(ternary.keys())))
    all_fields = binary_fields.copy()
    for f in ternary_fields:
        if f not in all_fields:
            all_fields.append(f)

    with open(merged_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for name in all_names:
            row = {'name': name}
            if name in binary:
                row.update(binary[name])
            if name in ternary:
                row.update(ternary[name])
            writer.writerow(row)

    print(f'  {ligand}: merged {len(binary)} binary + {len(ternary)} ternary -> {len(all_names)} rows')
    print(f'    -> {merged_csv}')
"

# ── Quick analysis with labels ──
echo ""
echo "============================================"
echo "Quick analysis (with labels for AUC)"
echo "============================================"

LABELS_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"

for LIGAND in lca glca lca3s; do
    MERGED="${RESULTS_DIR}/boltz_${LIGAND}_merged_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    if [ -f "$MERGED" ]; then
        echo ""
        echo "--- ${LIGAND} (merged binary+ternary) ---"
        if [ -f "$LABELS" ]; then
            python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" "$MERGED" --labels "$LABELS" 2>/dev/null || true
        else
            echo "  (no labels CSV found at ${LABELS})"
        fi
    fi
done

echo ""
echo "============================================"
echo "Results written to: ${RESULTS_DIR}/"
echo "============================================"
echo ""
echo "Per-ligand ternary results:"
ls -la "${RESULTS_DIR}/"*_ternary_results.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Merged binary+ternary results:"
ls -la "${RESULTS_DIR}/"*_merged_results.csv 2>/dev/null || echo "  (none found)"
