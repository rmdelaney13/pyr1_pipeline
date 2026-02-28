#!/bin/bash
# ============================================================================
# Iterative Neural Expansion for Bile Acid Designs
# ============================================================================
#
# Re-entrant master script: auto-detects current phase and does the next step.
# Run repeatedly for each round, waiting for SLURM jobs between phases.
#
# Supports two sequence design methods:
#   ligandmpnn (default) - 3 phases: A (select+MPNN), B (Boltz), C (score)
#   lasermpnn            - 4 phases: A (select), A' (prep+LASErMPNN), B (Boltz), C (score)
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_expansion.sh <ligand> <round> [method]
#
# Examples:
#   bash slurm/run_expansion.sh ca 0                  # Score initial predictions
#   bash slurm/run_expansion.sh ca 1                  # LigandMPNN (default)
#   bash slurm/run_expansion.sh ca 1 lasermpnn        # LASErMPNN
#
# All 4 ligands in parallel:
#   for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 0; done
#   for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 1; done
#
# ============================================================================

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────

if [ $# -lt 2 ]; then
    echo "Usage: bash slurm/run_expansion.sh <ligand> <round> [method]"
    echo "  ligand: ca, cdca, udca, dca"
    echo "  round:  0 (score initial), 1-4 (expansion rounds)"
    echo "  method: ligandmpnn (default), lasermpnn"
    exit 1
fi

LIGAND="${1,,}"   # lowercase
ROUND="$2"
METHOD="${3:-ligandmpnn}"

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462"
EXPANSION_ROOT="${SCRATCH}/expansion/${METHOD}/${LIGAND}"
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

# LASErMPNN-specific settings
LASER_BATCH_SIZE=20      # PDBs per GPU array task
DESIGNS_PER_INPUT=3      # sequences per PDB (match LigandMPNN)

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

# Method-specific job name patterns
if [ "$METHOD" = "lasermpnn" ]; then
    DESIGN_JOB_PREFIX="laser"
    BOLTZ_JOB_PREFIX="boltz_lexp"
else
    DESIGN_JOB_PREFIX="mpnn"
    BOLTZ_JOB_PREFIX="boltz_exp"
fi

# ── Activate environment ─────────────────────────────────────────────────────

module load anaconda
source activate boltz_env

echo "============================================"
echo "Expansion: ${LIGAND^^} round ${ROUND} (${METHOD})"
echo "============================================"
echo ""

# ── Round 0: Score initial predictions ───────────────────────────────────────

if [ "$ROUND" -eq 0 ]; then
    mkdir -p "$ROUND_DIR"
    SCORES="${ROUND_DIR}/scores.csv"

    if [ -f "$SCORES" ] || [ -L "$SCORES" ]; then
        NROWS=$(tail -n +2 "$(readlink -f "$SCORES")" | wc -l)
        echo "Round 0 already complete: ${SCORES} (${NROWS} designs)"
        exit 0
    fi

    # For lasermpnn, try to symlink from ligandmpnn's round 0 (same initial data)
    if [ "$METHOD" = "lasermpnn" ]; then
        LIGMPNN_SCORES="${SCRATCH}/expansion/ligandmpnn/${LIGAND}/round_0/scores.csv"
        if [ -f "$LIGMPNN_SCORES" ]; then
            ln -sf "$LIGMPNN_SCORES" "$SCORES"
            NROWS=$(tail -n +2 "$LIGMPNN_SCORES" | wc -l)
            echo "Round 0: symlinked ${NROWS} designs from LigandMPNN pipeline"
            echo "  -> ${LIGMPNN_SCORES}"
            exit 0
        fi
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

# ── Phase detection variables ────────────────────────────────────────────────

SELECTED_DIR="${ROUND_DIR}/selected_pdbs"
MANIFEST="${ROUND_DIR}/selected_manifest.txt"
EXPANSION_CSV="${ROUND_DIR}/expansion.csv"
BOLTZ_INPUT_DIR="${ROUND_DIR}/boltz_inputs"
BOLTZ_OUTPUT_DIR="${ROUND_DIR}/boltz_output"
NEW_SCORES="${ROUND_DIR}/new_scores.csv"
CUMULATIVE="${ROUND_DIR}/cumulative_scores.csv"

# Method-specific design output directory
if [ "$METHOD" = "lasermpnn" ]; then
    DESIGN_DIR="${ROUND_DIR}/laser_output"
    PREPPED_DIR="${ROUND_DIR}/prepped_pdbs"
    PREPPED_MANIFEST="${ROUND_DIR}/prepped_manifest.txt"
else
    DESIGN_DIR="${ROUND_DIR}/mpnn_output"
fi

# ── Phase C: Score + Merge ───────────────────────────────────────────────────
if [ -d "$BOLTZ_OUTPUT_DIR" ] && [ ! -f "$CUMULATIVE" ]; then
    echo "Phase C: Score new predictions + merge"
    echo ""

    # Check for running Boltz jobs for this ligand/round
    RUNNING_BOLTZ=$(squeue -u "$USER" -n "${BOLTZ_JOB_PREFIX}_${LIGAND}_r${ROUND}" -h 2>/dev/null | wc -l)
    if [ "$RUNNING_BOLTZ" -gt 0 ]; then
        echo "BLOCKED: ${RUNNING_BOLTZ} Boltz jobs still running for ${LIGAND^^} round ${ROUND}"
        echo "Wait for completion: squeue -u \$USER -n ${BOLTZ_JOB_PREFIX}_${LIGAND}_r${ROUND}"
        exit 1
    fi

    # Check completeness: compare expected (from manifest) vs actual PDBs
    N_RESULTS=$(find "$BOLTZ_OUTPUT_DIR" -name "*_model_0.pdb" 2>/dev/null | wc -l)
    if [ "$N_RESULTS" -eq 0 ]; then
        echo "WARNING: No Boltz predictions found in $BOLTZ_OUTPUT_DIR"
        echo "Are the Boltz jobs still running? Check: squeue -u \$USER"
        exit 1
    fi

    BOLTZ_MANIFEST="${BOLTZ_INPUT_DIR}/manifest.txt"
    if [ -f "$BOLTZ_MANIFEST" ]; then
        N_EXPECTED=$(wc -l < "$BOLTZ_MANIFEST")
        echo "Found $N_RESULTS / $N_EXPECTED Boltz predictions"
        if [ "$N_RESULTS" -lt "$N_EXPECTED" ]; then
            PCTG=$(( N_RESULTS * 100 / N_EXPECTED ))
            echo "  (${PCTG}% complete — some jobs may have failed)"
        fi
    else
        echo "Found $N_RESULTS Boltz predictions"
    fi

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
    echo "Round ${ROUND} COMPLETE for ${LIGAND^^} (${METHOD})"
    echo "============================================"
    echo "  Cumulative scores: ${CUMULATIVE}"
    NROWS=$(tail -n +2 "$CUMULATIVE" | wc -l)
    echo "  Total designs: ${NROWS}"
    echo ""
    echo "Next: bash slurm/run_expansion.sh ${LIGAND} $((ROUND + 1)) ${METHOD}"
    exit 0
fi

# ── Phase B: Convert designs → CSV → YAML → submit Boltz ────────────────────
if [ -d "$DESIGN_DIR" ] && [ ! -d "$BOLTZ_INPUT_DIR" ]; then
    echo "Phase B: Convert ${METHOD} output + submit Boltz predictions"
    echo ""

    # Check for running design jobs
    RUNNING_DESIGN=$(squeue -u "$USER" -n "${DESIGN_JOB_PREFIX}_${LIGAND}_r${ROUND}" -h 2>/dev/null | wc -l)
    if [ "$RUNNING_DESIGN" -gt 0 ]; then
        echo "BLOCKED: ${RUNNING_DESIGN} ${METHOD} jobs still running for ${LIGAND^^} round ${ROUND}"
        echo "Wait for completion: squeue -u \$USER -n ${DESIGN_JOB_PREFIX}_${LIGAND}_r${ROUND}"
        exit 1
    fi

    # Collect existing expansion CSVs for cross-round dedup
    DEDUP_ARGS=()
    for r in $(seq 1 $((ROUND - 1))); do
        EXP_CSV="${EXPANSION_ROOT}/round_${r}/expansion.csv"
        if [ -f "$EXP_CSV" ]; then
            DEDUP_ARGS+=(--existing-csv "$EXP_CSV")
        fi
    done

    if [ "$METHOD" = "lasermpnn" ]; then
        # Check LASErMPNN output completeness
        N_FASTAS=$(find "$DESIGN_DIR" -name "designs.fasta" 2>/dev/null | wc -l)
        if [ "$N_FASTAS" -eq 0 ]; then
            echo "WARNING: No designs.fasta files found in $DESIGN_DIR"
            echo "Check LASErMPNN job logs for errors."
            exit 1
        fi
        echo "Found $N_FASTAS LASErMPNN designs.fasta files"

        # Convert LASErMPNN FASTAs to Boltz CSV
        echo "Converting LASErMPNN output to Boltz CSV..."
        python "${PROJECT_ROOT}/scripts/laser_fasta_to_csv.py" \
            --laser-dir "$DESIGN_DIR" \
            --ligand-name "${LIGAND^^}" \
            --ligand-smiles "$SMILES" \
            --round "$ROUND" \
            --out "$EXPANSION_CSV" \
            ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"}
    else
        # Check MPNN output completeness
        N_FASTAS=$(find "$DESIGN_DIR" -name "*.fa" 2>/dev/null | wc -l)
        if [ "$N_FASTAS" -eq 0 ]; then
            echo "WARNING: No FASTA files found in $DESIGN_DIR"
            echo "Check MPNN job logs for errors."
            exit 1
        fi
        N_EXPECTED_MPNN=$(wc -l < "$MANIFEST")
        echo "Found $N_FASTAS / $N_EXPECTED_MPNN MPNN FASTA files"
        if [ "$N_FASTAS" -lt "$N_EXPECTED_MPNN" ]; then
            echo "  (some MPNN jobs may have failed — proceeding with available output)"
        fi

        # Convert MPNN FASTAs to Boltz CSV
        echo "Converting MPNN output to Boltz CSV..."
        python "${PROJECT_ROOT}/scripts/expansion_mpnn_to_csv.py" \
            --mpnn-dir "$DESIGN_DIR" \
            --ligand-name "${LIGAND^^}" \
            --ligand-smiles "$SMILES" \
            --round "$ROUND" \
            --out "$EXPANSION_CSV" \
            ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"}
    fi

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
    echo "${LIGAND^^} ${METHOD} round ${ROUND}: ${TOTAL} YAMLs -> array=0-${ARRAY_MAX} (batch=${BATCH_SIZE})"

    # Submit Boltz
    echo ""
    echo "Submitting Boltz predictions..."
    JOB_ID=$(sbatch --array=0-${ARRAY_MAX} \
        --job-name="${BOLTZ_JOB_PREFIX}_${LIGAND}_r${ROUND}" \
        "${PROJECT_ROOT}/slurm/submit_boltz.sh" \
        "$BOLTZ_MANIFEST" "$BOLTZ_OUTPUT_DIR" "$BATCH_SIZE" "$DIFFUSION_SAMPLES" \
        | awk '{print $NF}')

    echo ""
    echo "============================================"
    echo "Phase B complete: Boltz job ${JOB_ID} submitted"
    echo "============================================"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "After completion: bash slurm/run_expansion.sh ${LIGAND} ${ROUND} ${METHOD}"
    exit 0
fi

# ── Phase A' (LASErMPNN only): Prep PDBs + submit LASErMPNN ─────────────────
if [ "$METHOD" = "lasermpnn" ] && [ -d "$SELECTED_DIR" ] && [ ! -d "$PREPPED_DIR" ]; then
    echo "Phase A': Prep PDBs + submit LASErMPNN"
    echo ""

    # Protonate ligand, set B-factors, add water
    echo "Preparing PDBs for LASErMPNN..."
    python "${PROJECT_ROOT}/scripts/prep_boltz_for_laser.py" \
        --input-dir "$SELECTED_DIR" \
        --output-dir "$PREPPED_DIR" \
        --smiles "$SMILES" \
        --ref-pdb "$REF_PDB" \
        --add-water

    if [ ! -f "$PREPPED_MANIFEST" ]; then
        echo "ERROR: Prepped manifest not created. Check prep_boltz_for_laser.py output."
        exit 1
    fi

    TOTAL=$(wc -l < "$PREPPED_MANIFEST")
    ARRAY_MAX=$(( (TOTAL + LASER_BATCH_SIZE - 1) / LASER_BATCH_SIZE - 1 ))
    echo ""
    echo "Submitting LASErMPNN array job (${TOTAL} PDBs, batch=${LASER_BATCH_SIZE})..."

    mkdir -p "$DESIGN_DIR"
    JOB_ID=$(sbatch --array=0-${ARRAY_MAX} \
        --job-name="${DESIGN_JOB_PREFIX}_${LIGAND}_r${ROUND}" \
        "${PROJECT_ROOT}/slurm/submit_laser_expansion.sh" \
        "$PREPPED_MANIFEST" "$DESIGN_DIR" "$LASER_BATCH_SIZE" "$DESIGNS_PER_INPUT" \
        | awk '{print $NF}')

    echo ""
    echo "============================================"
    echo "Phase A' complete: LASErMPNN job ${JOB_ID} submitted"
    echo "============================================"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "After completion: bash slurm/run_expansion.sh ${LIGAND} ${ROUND} ${METHOD}"
    exit 0
fi

# ── Phase A: Select top N [+ submit LigandMPNN] ─────────────────────────────
if [ ! -d "$SELECTED_DIR" ]; then
    echo "Phase A: Select top ${TOP_N} designs"
    echo ""

    python "${PROJECT_ROOT}/scripts/expansion_select.py" \
        --scores "$PREV_SCORES" \
        --boltz-dirs "${BOLTZ_DIRS[@]}" \
        --out-dir "$SELECTED_DIR" \
        --top-n "$TOP_N"

    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not created. Check expansion_select.py output."
        exit 1
    fi

    TOTAL=$(wc -l < "$MANIFEST")

    if [ "$METHOD" = "ligandmpnn" ]; then
        # LigandMPNN: submit immediately (combined Phase A)
        echo ""
        echo "Submitting LigandMPNN array job (${TOTAL} PDBs)..."

        mkdir -p "$DESIGN_DIR"
        JOB_ID=$(sbatch --array=1-${TOTAL} \
            --job-name="${DESIGN_JOB_PREFIX}_${LIGAND}_r${ROUND}" \
            "${PROJECT_ROOT}/slurm/submit_mpnn_expansion.sh" \
            "$MANIFEST" "$DESIGN_DIR" \
            | awk '{print $NF}')

        echo ""
        echo "============================================"
        echo "Phase A complete: MPNN job ${JOB_ID} submitted"
        echo "============================================"
    else
        # LASErMPNN: Phase A only selects; Phase A' preps and submits
        echo ""
        echo "============================================"
        echo "Phase A complete: ${TOTAL} PDBs selected"
        echo "============================================"
    fi

    echo ""
    echo "Next: bash slurm/run_expansion.sh ${LIGAND} ${ROUND} ${METHOD}"
    exit 0
fi

# ── If we got here, the round state is ambiguous ─────────────────────────────
echo "Round ${ROUND} state (${METHOD}):"
echo "  selected_pdbs: $([ -d "$SELECTED_DIR" ] && echo 'exists' || echo 'missing')"
if [ "$METHOD" = "lasermpnn" ]; then
    echo "  prepped_pdbs:  $([ -d "$PREPPED_DIR" ] && echo 'exists' || echo 'missing')"
    echo "  laser_output:  $([ -d "$DESIGN_DIR" ] && echo 'exists' || echo 'missing')"
else
    echo "  mpnn_output:   $([ -d "$DESIGN_DIR" ] && echo 'exists' || echo 'missing')"
fi
echo "  boltz_inputs:  $([ -d "$BOLTZ_INPUT_DIR" ] && echo 'exists' || echo 'missing')"
echo "  boltz_output:  $([ -d "$BOLTZ_OUTPUT_DIR" ] && echo 'exists' || echo 'missing')"
echo "  cumulative:    $([ -f "$CUMULATIVE" ] && echo 'exists' || echo 'missing')"
echo ""
echo "Cannot determine next phase. Check SLURM job status: squeue -u \$USER"
echo "If stuck, delete the relevant directory to re-run that phase."
