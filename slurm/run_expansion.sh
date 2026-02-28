#!/bin/bash
# ============================================================================
# Iterative Neural Expansion for Bile Acid Designs
# ============================================================================
#
# Re-entrant master script: auto-detects current phase and does the next step.
# Run repeatedly for each round, waiting for SLURM jobs between phases.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_expansion.sh <ligand> <round>
#
# Examples:
#   bash slurm/run_expansion.sh ca 0     # Score initial predictions
#   bash slurm/run_expansion.sh ca 1     # Phase A: select + MPNN
#   bash slurm/run_expansion.sh ca 1     # Phase B: convert + Boltz (after MPNN)
#   bash slurm/run_expansion.sh ca 1     # Phase C: score + merge (after Boltz)
#
# All 4 ligands in parallel:
#   for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 0; done
#   for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 1; done
#
# ============================================================================

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────

if [ $# -lt 2 ]; then
    echo "Usage: bash slurm/run_expansion.sh <ligand> <round>"
    echo "  ligand: ca, cdca, udca, dca"
    echo "  round:  0 (score initial), 1-4 (expansion rounds)"
    exit 1
fi

LIGAND="${1,,}"   # lowercase
ROUND="$2"

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462"
EXPANSION_ROOT="${SCRATCH}/expansion_${LIGAND}"
ROUND_DIR="${EXPANSION_ROOT}/round_${ROUND}"

# Initial Boltz output from run_boltz_bile_acids.sh
INITIAL_BOLTZ_DIR="${SCRATCH}/boltz_bile_acids/output_${LIGAND}_binary"

# Reference PDB for H-bond geometry
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

# WT MSA for Boltz predictions
WT_MSA="${SCRATCH}/boltz_lca/wt_prediction/boltz_results_pyr1_wt_lca/msa/pyr1_wt_lca_unpaired_tmp_env/uniref.a3m"

# Boltz settings
DIFFUSION_SAMPLES=5
BATCH_SIZE=20

# Expansion settings
TOP_N=100

# SMILES map
declare -A SMILES_MAP=(
    ["ca"]='C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1([C@H](C[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)O)C'
    ["cdca"]='C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C'
    ["udca"]='C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C'
    ["dca"]='C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1([C@H](C[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C'
)

SMILES="${SMILES_MAP[$LIGAND]:-}"
if [ -z "$SMILES" ]; then
    echo "ERROR: Unknown ligand '$LIGAND'. Use: ca, cdca, udca, dca"
    exit 1
fi

# ── Activate environment ─────────────────────────────────────────────────────

module load anaconda
source activate boltz_env

echo "============================================"
echo "Expansion: ${LIGAND^^} round ${ROUND}"
echo "============================================"
echo ""

# ── Round 0: Score initial predictions ───────────────────────────────────────

if [ "$ROUND" -eq 0 ]; then
    mkdir -p "$ROUND_DIR"
    SCORES="${ROUND_DIR}/scores.csv"

    if [ -f "$SCORES" ]; then
        NROWS=$(tail -n +2 "$SCORES" | wc -l)
        echo "Round 0 already complete: ${SCORES} (${NROWS} designs)"
        exit 0
    fi

    if [ ! -d "$INITIAL_BOLTZ_DIR" ]; then
        echo "ERROR: Initial Boltz output not found: $INITIAL_BOLTZ_DIR"
        echo "Run slurm/run_boltz_bile_acids.sh first and wait for completion."
        exit 1
    fi

    echo "Scoring initial ${LIGAND^^} predictions..."
    python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
        --binary-dir "$INITIAL_BOLTZ_DIR" \
        --ref-pdb "$REF_PDB" \
        --out "$SCORES"

    NROWS=$(tail -n +2 "$SCORES" | wc -l)
    echo ""
    echo "Round 0 complete: ${NROWS} designs scored -> ${SCORES}"
    exit 0
fi

# ── Rounds 1+: Expansion ────────────────────────────────────────────────────

mkdir -p "$ROUND_DIR"

# Get previous round's cumulative scores (or round 0 scores)
PREV_ROUND=$((ROUND - 1))
if [ "$PREV_ROUND" -eq 0 ]; then
    PREV_SCORES="${EXPANSION_ROOT}/round_0/scores.csv"
else
    PREV_SCORES="${EXPANSION_ROOT}/round_${PREV_ROUND}/cumulative_scores.csv"
fi

if [ ! -f "$PREV_SCORES" ]; then
    echo "ERROR: Previous round scores not found: $PREV_SCORES"
    echo "Run round ${PREV_ROUND} first."
    exit 1
fi

# Collect all Boltz output dirs (initial + all previous expansion rounds)
BOLTZ_DIRS=("$INITIAL_BOLTZ_DIR")
for r in $(seq 1 $PREV_ROUND); do
    EXPANSION_BOLTZ="${EXPANSION_ROOT}/round_${r}/boltz_output"
    if [ -d "$EXPANSION_BOLTZ" ]; then
        BOLTZ_DIRS+=("$EXPANSION_BOLTZ")
    fi
done

# ── Phase detection ──────────────────────────────────────────────────────────

SELECTED_DIR="${ROUND_DIR}/selected_pdbs"
MANIFEST="${ROUND_DIR}/selected_manifest.txt"
MPNN_DIR="${ROUND_DIR}/mpnn_output"
EXPANSION_CSV="${ROUND_DIR}/expansion.csv"
BOLTZ_INPUT_DIR="${ROUND_DIR}/boltz_inputs"
BOLTZ_OUTPUT_DIR="${ROUND_DIR}/boltz_output"
NEW_SCORES="${ROUND_DIR}/new_scores.csv"
CUMULATIVE="${ROUND_DIR}/cumulative_scores.csv"

# ── Phase C: Score + Merge ───────────────────────────────────────────────────
if [ -d "$BOLTZ_OUTPUT_DIR" ] && [ ! -f "$CUMULATIVE" ]; then
    echo "Phase C: Score new predictions + merge"
    echo ""

    # Check that Boltz output has some results
    N_RESULTS=$(find "$BOLTZ_OUTPUT_DIR" -name "*_model_0.pdb" 2>/dev/null | wc -l)
    if [ "$N_RESULTS" -eq 0 ]; then
        echo "WARNING: No Boltz predictions found yet in $BOLTZ_OUTPUT_DIR"
        echo "Are the Boltz jobs still running? Check: squeue -u \$USER"
        exit 1
    fi
    echo "Found $N_RESULTS Boltz predictions"

    # Score new predictions
    echo "Scoring new predictions..."
    python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
        --binary-dir "$BOLTZ_OUTPUT_DIR" \
        --ref-pdb "$REF_PDB" \
        --out "$NEW_SCORES"

    # Merge with previous cumulative
    echo ""
    echo "Merging scores..."
    python "${PROJECT_ROOT}/scripts/expansion_merge.py" \
        --previous "$PREV_SCORES" \
        --new "$NEW_SCORES" \
        --out "$CUMULATIVE"

    echo ""
    echo "============================================"
    echo "Round ${ROUND} COMPLETE for ${LIGAND^^}"
    echo "============================================"
    echo "  Cumulative scores: ${CUMULATIVE}"
    NROWS=$(tail -n +2 "$CUMULATIVE" | wc -l)
    echo "  Total designs: ${NROWS}"
    echo ""
    echo "Next: bash slurm/run_expansion.sh ${LIGAND} $((ROUND + 1))"
    exit 0
fi

# ── Phase B: Convert MPNN → CSV → YAML → submit Boltz ───────────────────────
if [ -d "$MPNN_DIR" ] && [ ! -d "$BOLTZ_INPUT_DIR" ]; then
    echo "Phase B: Convert MPNN output + submit Boltz predictions"
    echo ""

    # Check MPNN output has FASTA files
    N_FASTAS=$(find "$MPNN_DIR" -name "*.fa" 2>/dev/null | wc -l)
    if [ "$N_FASTAS" -eq 0 ]; then
        echo "WARNING: No FASTA files found yet in $MPNN_DIR"
        echo "Are the MPNN jobs still running? Check: squeue -u \$USER"
        exit 1
    fi
    echo "Found $N_FASTAS MPNN FASTA files"

    # Convert MPNN FASTAs to Boltz CSV
    echo "Converting MPNN output to Boltz CSV..."
    DEDUP_ARGS=()
    if [ -f "$PREV_SCORES" ]; then
        DEDUP_ARGS+=(--existing-csv "$PREV_SCORES")
    fi

    python "${PROJECT_ROOT}/scripts/expansion_mpnn_to_csv.py" \
        --mpnn-dir "$MPNN_DIR" \
        --ligand-name "${LIGAND^^}" \
        --ligand-smiles "$SMILES" \
        --round "$ROUND" \
        --out "$EXPANSION_CSV" \
        "${DEDUP_ARGS[@]}"

    # Generate Boltz YAMLs
    echo ""
    echo "Generating Boltz YAML inputs..."
    python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
        "$EXPANSION_CSV" \
        --out-dir "$BOLTZ_INPUT_DIR" \
        --mode binary \
        --msa "$WT_MSA" \
        --affinity

    BOLTZ_MANIFEST="${BOLTZ_INPUT_DIR}/manifest.txt"
    TOTAL=$(wc -l < "$BOLTZ_MANIFEST")
    ARRAY_MAX=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "${LIGAND^^} round ${ROUND}: ${TOTAL} YAMLs -> array=0-${ARRAY_MAX} (batch=${BATCH_SIZE})"

    # Submit Boltz
    echo ""
    echo "Submitting Boltz predictions..."
    JOB_ID=$(sbatch --array=0-${ARRAY_MAX} \
        --job-name="boltz_exp_${LIGAND}_r${ROUND}" \
        "${PROJECT_ROOT}/slurm/submit_boltz.sh" \
        "$BOLTZ_MANIFEST" "$BOLTZ_OUTPUT_DIR" "$BATCH_SIZE" "$DIFFUSION_SAMPLES" \
        | awk '{print $NF}')

    echo ""
    echo "============================================"
    echo "Phase B complete: Boltz job ${JOB_ID} submitted"
    echo "============================================"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "After completion: bash slurm/run_expansion.sh ${LIGAND} ${ROUND}"
    exit 0
fi

# ── Phase A: Select top N + submit MPNN ──────────────────────────────────────
if [ ! -d "$SELECTED_DIR" ]; then
    echo "Phase A: Select top ${TOP_N} designs + submit MPNN"
    echo ""

    BOLTZ_DIR_ARGS=()
    for bd in "${BOLTZ_DIRS[@]}"; do
        BOLTZ_DIR_ARGS+=(--boltz-dirs "$bd")
    done

    python "${PROJECT_ROOT}/scripts/expansion_select.py" \
        --scores "$PREV_SCORES" \
        "${BOLTZ_DIR_ARGS[@]}" \
        --out-dir "$SELECTED_DIR" \
        --top-n "$TOP_N"

    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not created. Check expansion_select.py output."
        exit 1
    fi

    TOTAL=$(wc -l < "$MANIFEST")
    echo ""
    echo "Submitting LigandMPNN array job (${TOTAL} PDBs)..."

    mkdir -p "$MPNN_DIR"
    JOB_ID=$(sbatch --array=1-${TOTAL} \
        --job-name="mpnn_${LIGAND}_r${ROUND}" \
        "${PROJECT_ROOT}/slurm/submit_mpnn_expansion.sh" \
        "$MANIFEST" "$MPNN_DIR" \
        | awk '{print $NF}')

    echo ""
    echo "============================================"
    echo "Phase A complete: MPNN job ${JOB_ID} submitted"
    echo "============================================"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "After completion: bash slurm/run_expansion.sh ${LIGAND} ${ROUND}"
    exit 0
fi

# ── If we got here, the round state is ambiguous ─────────────────────────────
echo "Round ${ROUND} state:"
echo "  selected_pdbs: $([ -d "$SELECTED_DIR" ] && echo 'exists' || echo 'missing')"
echo "  mpnn_output:   $([ -d "$MPNN_DIR" ] && echo 'exists' || echo 'missing')"
echo "  boltz_inputs:  $([ -d "$BOLTZ_INPUT_DIR" ] && echo 'exists' || echo 'missing')"
echo "  boltz_output:  $([ -d "$BOLTZ_OUTPUT_DIR" ] && echo 'exists' || echo 'missing')"
echo "  cumulative:    $([ -f "$CUMULATIVE" ] && echo 'exists' || echo 'missing')"
echo ""
echo "Cannot determine next phase. Check SLURM job status: squeue -u \$USER"
echo "If stuck, delete the relevant directory to re-run that phase."
