#!/bin/bash
#SBATCH --job-name=pyrosetta_relax_score
#SBATCH --output=relax_score_%A_%a.out
#SBATCH --error=relax_score_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --array=1-900  # Adjust based on actual chunking

# === Configuration ===
CHUNK_SIZE=2  # Number of PDBs to process per array task
INPUT_DIR="/scratch/alpine/ryde3462/cholic_acid_desgin/mpnn_output_2"
OUTPUT_DIR="/scratch/alpine/ryde3462/cholic_acid_design/relax_2"
PYROSETTA_SCRIPT="projects/ryde3462/software/ligand_alignment/scripts/20250605_relax_general.py"
LIGAND_PARAMS="/projects/ryde3462/bile_acids/CA/conformers/0/0.params"

mkdir -p "$OUTPUT_DIR"

# === Load PDB list ===
mapfile -t PDB_FILES < <(find "$INPUT_DIR" -type f -path "*/packed/*.pdb" | sort)
TOTAL=${#PDB_FILES[@]}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE - 1 ))
if (( END >= TOTAL )); then END=$(( TOTAL - 1 )); fi

echo "Task $SLURM_ARRAY_TASK_ID processing files $START to $END out of $TOTAL"

for i in $(seq $START $END); do
  PDB="${PDB_FILES[$i]}"
  BASE=$(basename "$PDB" .pdb)

  # Remap water chain TP3 → D
  TMP="$OUTPUT_DIR/${BASE}_tmp.pdb"
  awk 'BEGIN { OFS=""; }
       /^ATOM/ || /^HETATM/ {
         resn = substr($0,18,3);
         if (resn=="TP3") {
           printf "%sD%s\n", substr($0,1,21), substr($0,23)
         } else {
           print $0
         }
         next
       }
       { print $0 }' "$PDB" > "$TMP"

  OUTP="$OUTPUT_DIR/${BASE}_relaxed.pdb"
  MERGED="$OUTPUT_DIR/${BASE}_relaxed.merged.sc"

  echo "→ [$SLURM_ARRAY_TASK_ID:$i] Processing $BASE"

  python "$PYROSETTA_SCRIPT" "$TMP" "$OUTP" "$LIGAND_PARAMS"

  if [[ ! -f "$MERGED" ]]; then
    echo "ERROR: merged score file not found: $MERGED" >&2
    continue
  fi

  echo "✔ Done: $MERGED"
  rm "$TMP"
done
