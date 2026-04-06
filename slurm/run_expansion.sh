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
if [ -n "${EXPANSION_ROOT_OVERRIDE:-}" ]; then
    EXPANSION_ROOT="$EXPANSION_ROOT_OVERRIDE"
else
    EXPANSION_ROOT="${SCRATCH}/expansion/${METHOD}/${LIGAND}"
    # CDCA design campaign uses separate paths
    if [ "$LIGAND" = "cdca" ]; then
        EXPANSION_ROOT="${SCRATCH}/CDCA/design/expansion/${METHOD}"
    fi
fi

ROUND_DIR="${EXPANSION_ROOT}/round_${ROUND}"

# Initial Boltz output from run_boltz_bile_acids.sh
if [ -n "${INITIAL_BOLTZ_DIR_OVERRIDE:-}" ]; then
    INITIAL_BOLTZ_DIR="$INITIAL_BOLTZ_DIR_OVERRIDE"
else
    INITIAL_BOLTZ_DIR="${SCRATCH}/boltz_bile_acids/output_${LIGAND}_binary"
    if [ "$LIGAND" = "cdca" ] && [ -d "${SCRATCH}/CDCA/design/boltz_output" ]; then
        INITIAL_BOLTZ_DIR="${SCRATCH}/CDCA/design/boltz_output"
    fi
fi

# Reference PDB for H-bond geometry
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

# Reference PDB for ligand geometry / stereochemistry check
# Uses ligands/<lig>_*.pdb — same references as analyze_expansion_readiness.py
if [ -n "${REF_LIGAND_PDB_OVERRIDE:-}" ]; then
    REF_LIGAND_PDB="$REF_LIGAND_PDB_OVERRIDE"
else
    REF_LIGAND_PDB=""
    REF_LIGAND_CANDIDATES=("${PROJECT_ROOT}/ligands/${LIGAND}_"*.pdb)
    if [ -f "${REF_LIGAND_CANDIDATES[0]}" ]; then
        REF_LIGAND_PDB="${REF_LIGAND_CANDIDATES[0]}"
    fi
fi

# Ternary reference PDB for HAB1 clash check (chain C = HAB1)
REF_TERNARY_PDB="${SCRATCH}/boltz_lca/wt_ternary/boltz_results_pyr1_wt_lca_hab1/predictions/pyr1_wt_lca_hab1/pyr1_wt_lca_hab1_model_0.pdb"

# WT MSA for Boltz predictions
WT_MSA="${SCRATCH}/boltz_lca/wt_prediction/boltz_results_pyr1_wt_lca/msa/pyr1_wt_lca_unpaired_tmp_env/uniref.a3m"

# Boltz settings
DIFFUSION_SAMPLES=5
BATCH_SIZE=25

# Expansion settings
TOP_N=500

# Campaign-specific MPNN omit/bias configs (overridable via env vars)
MPNN_OMIT_JSON="${MPNN_OMIT_JSON_OVERRIDE:-${PROJECT_ROOT}/design/mpnn/expansion_omit.json}"
MPNN_BIAS_JSON="${MPNN_BIAS_JSON_OVERRIDE:-${PROJECT_ROOT}/design/mpnn/expansion_bias.json}"
if [ "$LIGAND" = "cdca" ] && [ -z "${MPNN_OMIT_JSON_OVERRIDE:-}" ]; then
    MPNN_OMIT_JSON="${PROJECT_ROOT}/campaigns/CDCA/mpnn/expansion_omit_boltz.json"
    MPNN_BIAS_JSON="${PROJECT_ROOT}/campaigns/CDCA/mpnn/expansion_bias_boltz.json"
fi

# LASErMPNN-specific settings
LASER_BATCH_SIZE=50      # PDBs per GPU array task (split across 2 GPUs)
# Sequences per parent PDB: 5 early, 3 standard (round 6+)
if [ "$ROUND" -le 5 ]; then
    DESIGNS_PER_INPUT=5
else
    DESIGNS_PER_INPUT=3
fi
GEOMETRY_WEIGHT=2.0      # weight for H-bond geometry in total_score
UNSAT_PENALTY=0.5        # penalty per unsatisfied polar contact in total_score
BACKBONE_NOISE=0.3       # Gaussian noise (Å) on backbone atoms during MPNN design

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
    GEOM_ARGS=""
    if [ -n "$REF_LIGAND_PDB" ] && [ -f "$REF_LIGAND_PDB" ]; then
        GEOM_ARGS="--ref-ligand-pdb $REF_LIGAND_PDB"
        if [ -n "${REF_LIGAND_CHAIN:-}" ]; then
            GEOM_ARGS="$GEOM_ARGS --ref-ligand-chain $REF_LIGAND_CHAIN"
        fi
        echo "  Ligand geometry check enabled: $REF_LIGAND_PDB"
    fi
    if [ -n "$REF_TERNARY_PDB" ] && [ -f "$REF_TERNARY_PDB" ]; then
        GEOM_ARGS="$GEOM_ARGS --ref-ternary-pdb $REF_TERNARY_PDB"
        echo "  HAB1 clash check enabled: $REF_TERNARY_PDB"
    fi
    python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
        --binary-dir "$INITIAL_BOLTZ_DIR" \
        --ref-pdb "$REF_PDB" \
        $GEOM_ARGS \
        --geometry-weight "$GEOMETRY_WEIGHT" \
        --unsat-penalty "$UNSAT_PENALTY" \
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
    GEOM_ARGS=""
    if [ -n "$REF_LIGAND_PDB" ] && [ -f "$REF_LIGAND_PDB" ]; then
        GEOM_ARGS="--ref-ligand-pdb $REF_LIGAND_PDB"
    fi
    if [ -n "$REF_TERNARY_PDB" ] && [ -f "$REF_TERNARY_PDB" ]; then
        GEOM_ARGS="$GEOM_ARGS --ref-ternary-pdb $REF_TERNARY_PDB"
    fi
    python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
        --binary-dir "$BOLTZ_OUTPUT_DIR" \
        --ref-pdb "$REF_PDB" \
        $GEOM_ARGS \
        --geometry-weight "$GEOMETRY_WEIGHT" \
        --unsat-penalty "$UNSAT_PENALTY" \
        --out "$NEW_SCORES"

    # Merge with previous cumulative
    echo ""
    echo "Merging scores..."
    python "${PROJECT_ROOT}/scripts/expansion_merge.py" \
        --previous "$PREV_SCORES" \
        --new "$NEW_SCORES" \
        --out "$CUMULATIVE" \
        --new-round "$ROUND"

    echo ""
    echo "============================================"
    echo "Round ${ROUND} COMPLETE for ${LIGAND^^} (${METHOD})"
    echo "============================================"
    echo "  Cumulative scores: ${CUMULATIVE}"
    NROWS=$(tail -n +2 "$CUMULATIVE" | wc -l)
    echo "  Total designs: ${NROWS}"

    # ── Recluster if cluster-aware expansion is enabled ──
    if [ -n "${CLUSTER_CSV_OVERRIDE:-}" ] || [ -n "${CLUSTER_CUTOFF:-}" ]; then
        CLUSTER_CUTOFF="${CLUSTER_CUTOFF:-2.0}"
        CLUSTER_WORK="${EXPANSION_ROOT}/cluster_analysis"
        CLUSTER_SHARD_DIR="${CLUSTER_WORK}/shards"
        CLUSTER_RESULT_DIR="${CLUSTER_WORK}/results"
        CLUSTER_NAME_LIST="${CLUSTER_WORK}/design_names.txt"

        echo ""
        echo "Reclustering all cumulative passers..."
        mkdir -p "$CLUSTER_SHARD_DIR" "$CLUSTER_RESULT_DIR" "${CLUSTER_WORK}/logs"

        # Generate passers-only name list
        python3 -c "
import csv
with open('${CUMULATIVE}') as f:
    names = [row['name'] for row in csv.DictReader(f) if row.get('pass_all') == '1']
with open('${CLUSTER_NAME_LIST}', 'w') as f:
    f.write('\n'.join(names) + '\n')
print(f'  {len(names)} passing designs for clustering')
"
        N_DESIGNS=$(wc -l < "$CLUSTER_NAME_LIST")
        SHARD_SIZE=1000
        N_SHARDS=$(( (N_DESIGNS + SHARD_SIZE - 1) / SHARD_SIZE ))
        MAX_ARRAY_IDX=$(( N_SHARDS - 1 ))

        # Collect all boltz dirs for PDB lookup
        ALL_BOLTZ_DIRS="$INITIAL_BOLTZ_DIR"
        for r in $(seq 1 $ROUND); do
            EXP_BOLTZ="${EXPANSION_ROOT}/round_${r}/boltz_output"
            if [ -d "$EXP_BOLTZ" ]; then
                ALL_BOLTZ_DIRS="${ALL_BOLTZ_DIRS} ${EXP_BOLTZ}"
            fi
        done

        # Clear old shards
        rm -f "${CLUSTER_SHARD_DIR}"/*.npz

        STAGE1_ID=$(sbatch --parsable <<CLUSTERBATCH
#!/bin/bash
#SBATCH --job-name=${LIGAND}_align
#SBATCH --account=ucb671_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-${MAX_ARRAY_IDX}
#SBATCH --output=${CLUSTER_WORK}/logs/align_%a.out
#SBATCH --error=${CLUSTER_WORK}/logs/align_%a.err

cd "\$SLURM_SUBMIT_DIR"
module purge
source ~/.bashrc
conda activate pyrosetta

# Try flat boltz dirs (multiple)
for BDIR in ${ALL_BOLTZ_DIRS}; do
    BOLTZ_ARGS="\${BOLTZ_ARGS:+\$BOLTZ_ARGS }--boltz-dir \$BDIR"
done

python ${PROJECT_ROOT}/scripts/cluster_align_shard.py \
    --boltz-dir ${ALL_BOLTZ_DIRS} \
    --ref-pdb ${REF_PDB} \
    --name-list ${CLUSTER_NAME_LIST} \
    --shard-index \${SLURM_ARRAY_TASK_ID} \
    --shard-size ${SHARD_SIZE} \
    --out-dir ${CLUSTER_SHARD_DIR}
CLUSTERBATCH
)

        STAGE2_ID=$(sbatch --parsable --dependency=afterok:${STAGE1_ID} <<CLUSTERBATCH
#!/bin/bash
#SBATCH --job-name=${LIGAND}_cluster
#SBATCH --account=ucb671_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=${CLUSTER_WORK}/logs/merge.out
#SBATCH --error=${CLUSTER_WORK}/logs/merge.err

cd "\$SLURM_SUBMIT_DIR"
module purge
source ~/.bashrc
conda activate pyrosetta

python ${PROJECT_ROOT}/scripts/cluster_merge_and_plot.py \
    --shard-dir ${CLUSTER_SHARD_DIR} \
    --scores-csv ${CUMULATIVE} \
    --cutoffs 1.0 1.5 2.0 2.5 4.0 \
    --min-cluster-size 5 \
    --out-dir ${CLUSTER_RESULT_DIR}
CLUSTERBATCH
)

        echo "  Recluster jobs: align=${STAGE1_ID} (array 0-${MAX_ARRAY_IDX}), merge=${STAGE2_ID}"
        echo "  Results: ${CLUSTER_RESULT_DIR}/clusters_${CLUSTER_CUTOFF}A.csv"
        echo ""
        echo "Wait for clustering, then: bash slurm/run_expansion.sh ${LIGAND} $((ROUND + 1)) ${METHOD}"
    else
        echo ""
        echo "Next: bash slurm/run_expansion.sh ${LIGAND} $((ROUND + 1)) ${METHOD}"
    fi
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

# ── Phase A (pre-seeded): Submit MPNN for externally seeded selected_pdbs ────
# Handles case where selected_pdbs/ was created by seed_expansion_from_filtered.py
# or similar, but MPNN hasn't been submitted yet.
if [ "$METHOD" = "ligandmpnn" ] && [ -d "$SELECTED_DIR" ] && [ ! -d "$DESIGN_DIR" ]; then
    echo "Phase A: Submit LigandMPNN (selected_pdbs pre-seeded)"
    echo ""

    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not found: $MANIFEST"
        echo "selected_pdbs/ exists but no manifest. Re-seed or create manifest."
        exit 1
    fi

    TOTAL=$(wc -l < "$MANIFEST")
    echo "Submitting LigandMPNN array job (${TOTAL} PDBs)..."

    mkdir -p "$DESIGN_DIR"
    JOB_ID=$(sbatch --array=1-${TOTAL} \
        --job-name="${DESIGN_JOB_PREFIX}_${LIGAND}_r${ROUND}" \
        "${PROJECT_ROOT}/slurm/submit_mpnn_expansion.sh" \
        "$MANIFEST" "$DESIGN_DIR" "$MPNN_OMIT_JSON" "$MPNN_BIAS_JSON" "$DESIGNS_PER_INPUT" "$BACKBONE_NOISE" \
        | awk '{print $NF}')

    echo ""
    echo "============================================"
    echo "Phase A complete: MPNN job ${JOB_ID} submitted"
    echo "============================================"
    echo ""
    echo "Next: bash slurm/run_expansion.sh ${LIGAND} ${ROUND} ${METHOD}"
    exit 0
fi

# ── Phase A: Select top N [+ submit LigandMPNN] ─────────────────────────────
if [ ! -d "$SELECTED_DIR" ]; then
    echo "Phase A: Select top ${TOP_N} designs"
    echo ""

    # Gates are embedded in scored CSV (pass_all column).
    # Selection uses pass_all + diversity + binding mode stratification.
    SELECT_ARGS=(
        --scores "$PREV_SCORES"
        --boltz-dirs "${BOLTZ_DIRS[@]}"
        --out-dir "$SELECTED_DIR"
        --top-n "$TOP_N"
        --diverse --diverse-fraction 0.5
    )

    # Cluster-aware selection (overrides binding-mode stratification)
    CLUSTER_CUTOFF="${CLUSTER_CUTOFF:-2.0}"
    CLUSTER_CSV="${CLUSTER_CSV_OVERRIDE:-}"
    # Auto-detect cluster CSV from expansion cluster_analysis dir
    if [ -z "$CLUSTER_CSV" ]; then
        AUTO_CLUSTER="${EXPANSION_ROOT}/cluster_analysis/results/clusters_${CLUSTER_CUTOFF}A.csv"
        if [ -f "$AUTO_CLUSTER" ]; then
            CLUSTER_CSV="$AUTO_CLUSTER"
        fi
    fi

    if [ -n "$CLUSTER_CSV" ] && [ -f "$CLUSTER_CSV" ]; then
        echo "Cluster-aware selection: $CLUSTER_CSV"
        SELECT_ARGS+=(
            --cluster-csv "$CLUSTER_CSV"
            --cluster-require-oh
            --cluster-min-pass 3
        )
    else
        # Fall back to binding-mode stratification
        SELECT_ARGS+=(--binding-mode-stratify --mode-quotas "normal:$((TOP_N/2)),flipped:$((TOP_N/2))")
    fi

    # Campaign-specific selection options
    if [ "$LIGAND" = "cdca" ]; then
        SELECT_ARGS+=(--exclude-pocket-aa 83:WFY 117:Y 159:DE)
        SELECT_ARGS+=(--min-diverse-score 1.5)
        SELECT_ARGS+=(--hamming-dedup 2)
        # Adaptive contact balance: equal allocation across contact classes,
        # rare classes get ALL their designs, surplus redistributes.
        # normal mode -> stratify by OH contact (core sterol OH in pocket)
        # flipped mode -> stratify by COO contact (carboxylate in pocket)
        SELECT_ARGS+=(
            --adaptive-contact-balance
            --adaptive-normal-classes "92,117,120,160,110,122"
            --adaptive-flipped-classes "94,167,163,122,59,120"
            --require-oh-contact
            --require-coo-contact
        )
    fi

    python "${PROJECT_ROOT}/scripts/expansion_select.py" "${SELECT_ARGS[@]}"

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
            "$MANIFEST" "$DESIGN_DIR" "$MPNN_OMIT_JSON" "$MPNN_BIAS_JSON" "$DESIGNS_PER_INPUT" "$BACKBONE_NOISE" \
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
