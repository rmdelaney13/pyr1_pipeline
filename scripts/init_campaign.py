#!/usr/bin/env python
"""
Bootstrap a new PYR1 ligand design campaign.

Creates the campaign directory tree, generates MPNN JSON files (with automatic
PDB→Boltz numbering offset), and produces MPNN SLURM shell scripts.

Usage
-----
# Option A  – from CLI args (creates config.txt if it doesn't exist):
python scripts/init_campaign.py --name DCA \
    --smiles "C[C@H](CCC(=O)O)..." \
    --omit "115:F 157:R" \
    --bias "139:K=3.0 79:D=1.0 157:H=1.5"

# Option B  – from an existing config file:
python scripts/init_campaign.py campaigns/DCA/config.txt

Idempotent: safe to re-run.  Overwrites generated artifacts (JSONs, scripts).
"""
import argparse
import json
import os
import sys
from configparser import ConfigParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPE_ROOT_DEFAULT = "/projects/ryde3462/software/pyr1_pipeline"
SCRATCH_PREFIX = "/scratch/alpine/ryde3462"

DESIGN_POSITIONS_PDB = [59, 79, 81, 90, 92, 106, 108, 115, 118, 120,
                        139, 157, 158, 161, 162, 165]

LIGANDMPNN_RUN = "/projects/ryde3462/software/LigandMPNN/run.py"
LIGANDMPNN_CKPT = (
    "/projects/ryde3462/software/LigandMPNN/model_params/"
    "ligandmpnn_v_32_020_25.pt"
)

SLURM_PARTITION = "amilan"
SLURM_ACCOUNT = "ucb472_asc2"
SLURM_QOS = "normal"

TEMPLATE_PDB_PRE = "docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
TEMPLATE_PDB_POST = "docking/ligand_alignment/files_for_PYR1_docking/3QN1_nolig_H2O.pdb"


# ---------------------------------------------------------------------------
# PDB ↔ Boltz offset
# ---------------------------------------------------------------------------

def pdb_to_boltz(pos: int) -> int:
    """3QN1 crystal is missing 2 residues near position 70.
    Boltz uses full sequence → positions >= 72 get +2."""
    return pos + 2 if pos >= 72 else pos


# ---------------------------------------------------------------------------
# Parsing helpers for OmitAA / BiasAA config strings
# ---------------------------------------------------------------------------

def parse_omit(raw: str) -> dict:
    """Parse 'OmitAA = 115:F 157:R' → {115: 'F', 157: 'R'}."""
    result = {}
    if not raw or not raw.strip():
        return result
    for token in raw.split():
        pos_str, aas = token.split(":", 1)
        result[int(pos_str)] = aas
    return result


def parse_bias(raw: str) -> dict:
    """Parse 'BiasAA = 139:K=3.0 79:D=1.0,E=0.5' →
    {139: {'K': 3.0}, 79: {'D': 1.0, 'E': 0.5}}."""
    result = {}
    if not raw or not raw.strip():
        return result
    for token in raw.split():
        pos_str, rest = token.split(":", 1)
        pos = int(pos_str)
        pairs = {}
        for item in rest.split(","):
            aa, weight = item.split("=", 1)
            pairs[aa.strip()] = float(weight)
        result[pos] = pairs
    return result


# ---------------------------------------------------------------------------
# JSON generation
# ---------------------------------------------------------------------------

def omit_to_json(omit_dict: dict, offset_fn=None) -> dict:
    """Convert {pos: 'AAs'} → {"A<pos>": "AAs"} with optional numbering offset."""
    out = {}
    for pos, aas in sorted(omit_dict.items()):
        p = offset_fn(pos) if offset_fn else pos
        out[f"A{p}"] = aas
    return out


def bias_to_json(bias_dict: dict, offset_fn=None) -> dict:
    """Convert {pos: {AA: wt}} → {"A<pos>": {AA: wt}} with optional offset."""
    out = {}
    for pos, pairs in sorted(bias_dict.items()):
        p = offset_fn(pos) if offset_fn else pos
        out[f"A{p}"] = {aa: wt for aa, wt in pairs.items()}
    return out


def write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# MPNN shell script generation
# ---------------------------------------------------------------------------

MPNN_SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=LigandMPNN_{ligand}_{variant}
#SBATCH --output=LigandMPNN_{ligand}_{variant}_%A_%a.out
#SBATCH --error=LigandMPNN_{ligand}_{variant}_%A_%a.err
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=1-__ARRAY_COUNT__

module load anaconda
conda activate ligandmpnn_env

PDB_DIR="__PDB_DIR__"
OUTPUT_BASE="__OUTPUT_BASE__"
MODEL_SCRIPT="{mpnn_run}"

OMIT_JSON="{omit_json}"
BIAS_JSON="{bias_json}"

mapfile -t pdb_files < <(ls "${{PDB_DIR}}"/*.pdb | sort)
TOTAL_FILES=${{#pdb_files[@]}}
echo "Total number of PDB files: ${{TOTAL_FILES}}"

GROUP_SIZE=1
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * GROUP_SIZE ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE - 1 ))
if [ $END_INDEX -ge $TOTAL_FILES ]; then
    END_INDEX=$(( TOTAL_FILES - 1 ))
fi

echo "SLURM_ARRAY_TASK_ID: ${{SLURM_ARRAY_TASK_ID}}"
echo "Processing files from index ${{START_INDEX}} to ${{END_INDEX}}"

for (( i=START_INDEX; i<=END_INDEX; i++ )); do
    PDB_FILE="${{pdb_files[$i]}}"
    PDB_BASENAME=$(basename "${{PDB_FILE}}" .pdb)
    OUT_FOLDER="${{OUTPUT_BASE}}/${{PDB_BASENAME}}_mpnn"

    echo "Processing file: ${{PDB_FILE}}"
    mkdir -p "${{OUT_FOLDER}}"

    cd /projects/ryde3462/software/LigandMPNN
    python "${{MODEL_SCRIPT}}" \\
        --seed 111 \\
        --model_type "ligand_mpnn" \\
        --pdb_path "${{PDB_FILE}}" \\
        --redesigned_residues "{design_residues}" \\
        --out_folder "${{OUT_FOLDER}}" \\
        --number_of_batches 1 \\
        --batch_size {batch_size} \\
        --temperature 0.3 \\
        --omit_AA_per_residue "${{OMIT_JSON}}" \\
        --bias_AA_per_residue "${{BIAS_JSON}}" \\
        --pack_side_chains 0 \\
        --checkpoint_ligand_mpnn "{mpnn_ckpt}" \\
        --pack_with_ligand_context 1

    echo "LigandMPNN completed for ${{PDB_FILE}}. Output: ${{OUT_FOLDER}}"
done

echo "All files in batch (job ${{SLURM_ARRAY_TASK_ID}}) have been processed."
"""


def write_mpnn_script(path: str, ligand: str, variant: str,
                      omit_json: str, bias_json: str,
                      design_positions: list, batch_size: int = 20):
    """Write an MPNN SLURM submit script for a given numbering variant."""
    residues_str = " ".join(f"A{p}" for p in design_positions)
    content = MPNN_SCRIPT_TEMPLATE.format(
        ligand=ligand,
        variant=variant,
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        qos=SLURM_QOS,
        mpnn_run=LIGANDMPNN_RUN,
        omit_json=omit_json,
        bias_json=bias_json,
        design_residues=residues_str,
        batch_size=batch_size,
        mpnn_ckpt=LIGANDMPNN_CKPT,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Config template generation
# ---------------------------------------------------------------------------

def generate_config(campaign_dir: str, ligand: str, smiles: str,
                    omit_raw: str, bias_raw: str, triplets_raw: str,
                    pipe_root: str):
    """Fill the template config with actual ligand values."""
    template_path = os.path.join(pipe_root, "campaigns", "template", "config.txt")
    if not os.path.exists(template_path):
        # Fallback: use the template from the same repo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.normpath(
            os.path.join(script_dir, "..", "campaigns", "template", "config.txt")
        )
    if not os.path.exists(template_path):
        print(f"ERROR: Template config not found at {template_path}")
        sys.exit(1)

    with open(template_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    content = content.replace("__LIGAND__", ligand)
    content = content.replace("__SMILES__", smiles)

    # Inject omit/bias/triplets into config
    content = content.replace("OmitAA =", f"OmitAA = {omit_raw}" if omit_raw else "OmitAA =")
    content = content.replace("BiasAA =", f"BiasAA = {bias_raw}" if bias_raw else "BiasAA =")
    if triplets_raw:
        content = content.replace(
            "TargetAtomTriplets        =",
            f"TargetAtomTriplets        = {triplets_raw}")

    config_path = os.path.join(campaign_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  wrote {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(campaign_dir: str, pipe_root: str, ligand: str):
    """Print status checklist."""
    print("\n--- Validation Checklist ---")
    checks = [
        ("Config",
         os.path.join(campaign_dir, "config.txt")),
        ("MPNN omit (rosetta)",
         os.path.join(campaign_dir, "mpnn", "omit_rosetta.json")),
        ("MPNN omit (boltz)",
         os.path.join(campaign_dir, "mpnn", "omit_boltz.json")),
        ("MPNN bias (rosetta)",
         os.path.join(campaign_dir, "mpnn", "bias_rosetta.json")),
        ("MPNN bias (boltz)",
         os.path.join(campaign_dir, "mpnn", "bias_boltz.json")),
        ("MPNN script (rosetta)",
         os.path.join(campaign_dir, "scripts", "mpnn_rosetta.sh")),
        ("MPNN script (boltz)",
         os.path.join(campaign_dir, "scripts", "mpnn_boltz.sh")),
        ("Conformers SDF",
         os.path.join(campaign_dir, "conformers", "conformers_final.sdf")),
        ("Rosetta params",
         os.path.join(campaign_dir, "conformers", "0", "0.params")),
        ("Template PDB (pre)",
         os.path.join(pipe_root, TEMPLATE_PDB_PRE)),
        ("Template PDB (post)",
         os.path.join(pipe_root, TEMPLATE_PDB_POST)),
    ]
    all_ok = True
    for label, path in checks:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status:7s}] {label}: {path}")

    if not all_ok:
        print("\nSome files are missing. Generate conformers before docking:")
        print(f"  python -m ligand_conformers --input \"<SMILES>\" --input-type smiles \\")
        print(f"      --outdir {os.path.join(campaign_dir, 'conformers')} --num-confs 500 -k 10")
    return all_ok


# ---------------------------------------------------------------------------
# Print workflow commands
# ---------------------------------------------------------------------------

def print_workflow(campaign_dir: str, ligand: str, smiles: str, pipe_root: str):
    conf_dir = os.path.join(campaign_dir, "conformers")
    config = os.path.join(campaign_dir, "config.txt")
    abs_config = os.path.join(pipe_root, "campaigns", ligand, "config.txt")

    print("\n--- Workflow Commands ---")
    print(f"""
# 1. Generate conformers (if not yet done)
python -m ligand_conformers \\
    --input "{smiles}" --input-type smiles \\
    --outdir {conf_dir} \\
    --num-confs 500 -k 10

# 2. Prepare docking table
python docking/scripts/run_docking_from_sdf.py \\
    {config} --prepare-only

# 3. SLURM docking (500 arrays x 1 perturbation)
sbatch --array=0-499 docking/scripts/submit_docking_workflow.sh \\
    {abs_config}

# 4. Cluster
python docking/scripts/cluster_docked_post_array.py \\
    {config}

# 5. Design
python design/scripts/run_design_pipeline.py \\
    {config} --wait
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a new PYR1 ligand design campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_file", nargs="?", default=None,
        help="Path to an existing campaign config.txt (Option B)")
    parser.add_argument(
        "--name", default=None,
        help="Ligand name (e.g., DCA, UDCA). Used for directory + file naming.")
    parser.add_argument(
        "--smiles", default=None,
        help="SMILES string (protonated acid form, e.g., C(=O)O not C(=O)[O-])")
    parser.add_argument(
        "--omit", default="",
        help='Omit spec in PDB numbering: "115:F 157:R"')
    parser.add_argument(
        "--bias", default="",
        help='Bias spec in PDB numbering: "139:K=3.0 79:D=1.0"')
    parser.add_argument(
        "--triplets", default="",
        help='TargetAtomTriplets for alignment, e.g. "O2-C11-C9; O2-C9-C11"')
    parser.add_argument(
        "--pipe-root", default=None,
        help=f"Pipeline root (default: {PIPE_ROOT_DEFAULT})")
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="MPNN batch size per parent dock (default: 20)")
    args = parser.parse_args()

    # Resolve PIPE_ROOT
    if args.pipe_root:
        pipe_root = args.pipe_root
    else:
        # Try to infer from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(script_dir, ".."))
        if os.path.isdir(os.path.join(candidate, "campaigns")):
            pipe_root = candidate
        else:
            pipe_root = PIPE_ROOT_DEFAULT

    # ---- Option B: read from existing config ----
    if args.config_file and os.path.isfile(args.config_file):
        config = ConfigParser()
        with open(args.config_file, "r", encoding="utf-8-sig") as fh:
            config.read_file(fh)

        campaign_sec = config["campaign"] if "campaign" in config else {}
        ligand = campaign_sec.get("LigandName", "").strip()
        smiles = campaign_sec.get("SMILES", "").strip()
        omit_raw = campaign_sec.get("OmitAA", "").strip()
        bias_raw = campaign_sec.get("BiasAA", "").strip()
        triplets_raw = ""  # already in config file

        if not ligand:
            print("ERROR: [campaign] LigandName is required in config")
            sys.exit(1)
        if not smiles:
            print("ERROR: [campaign] SMILES is required in config")
            sys.exit(1)

        campaign_dir = os.path.dirname(os.path.abspath(args.config_file))

        # Override pipe_root from config if available
        if "DEFAULT" in config and "PIPE_ROOT" in config["DEFAULT"]:
            raw = config["DEFAULT"]["PIPE_ROOT"].strip()
            if raw:
                pipe_root = raw

    # ---- Option A: from CLI args ----
    elif args.name and args.smiles:
        ligand = args.name
        smiles = args.smiles
        omit_raw = args.omit
        bias_raw = args.bias
        triplets_raw = args.triplets
        campaign_dir = os.path.join(pipe_root, "campaigns", ligand)
    else:
        parser.print_help()
        print("\nERROR: Provide either --name + --smiles (Option A) "
              "or an existing config file path (Option B).")
        sys.exit(1)

    print(f"=== Initializing campaign: {ligand} ===")
    print(f"  PIPE_ROOT:    {pipe_root}")
    print(f"  Campaign dir: {campaign_dir}")
    print(f"  SMILES:       {smiles}")

    # Create directory tree
    for subdir in ["mpnn", "scripts", "conformers"]:
        os.makedirs(os.path.join(campaign_dir, subdir), exist_ok=True)

    # Generate config.txt if it doesn't exist (Option A)
    config_path = os.path.join(campaign_dir, "config.txt")
    if not os.path.exists(config_path):
        generate_config(campaign_dir, ligand, smiles, omit_raw, bias_raw,
                        triplets_raw, pipe_root)
    else:
        print(f"  config.txt already exists — skipping generation")

    # Parse omit / bias
    omit = parse_omit(omit_raw)
    bias = parse_bias(bias_raw)

    if omit:
        print(f"  Omit (PDB): {omit}")
    else:
        print("  Omit: (none)")
    if bias:
        print(f"  Bias (PDB): {bias}")
    else:
        print("  Bias: (none)")

    # Generate MPNN JSON files (4 files: rosetta + boltz × omit + bias)
    print("\nGenerating MPNN JSON files...")

    write_json(
        os.path.join(campaign_dir, "mpnn", "omit_rosetta.json"),
        omit_to_json(omit))
    write_json(
        os.path.join(campaign_dir, "mpnn", "omit_boltz.json"),
        omit_to_json(omit, offset_fn=pdb_to_boltz))
    write_json(
        os.path.join(campaign_dir, "mpnn", "bias_rosetta.json"),
        bias_to_json(bias))
    write_json(
        os.path.join(campaign_dir, "mpnn", "bias_boltz.json"),
        bias_to_json(bias, offset_fn=pdb_to_boltz))

    # Generate MPNN shell scripts (2 variants)
    print("\nGenerating MPNN shell scripts...")

    boltz_positions = [pdb_to_boltz(p) for p in DESIGN_POSITIONS_PDB]

    # Use POSIX paths for JSON refs (scripts run on Linux)
    mpnn_dir_posix = pipe_root + "/campaigns/" + ligand + "/mpnn"

    write_mpnn_script(
        path=os.path.join(campaign_dir, "scripts", "mpnn_rosetta.sh"),
        ligand=ligand,
        variant="rosetta",
        omit_json=mpnn_dir_posix + "/omit_rosetta.json",
        bias_json=mpnn_dir_posix + "/bias_rosetta.json",
        design_positions=DESIGN_POSITIONS_PDB,
        batch_size=args.batch_size,
    )
    write_mpnn_script(
        path=os.path.join(campaign_dir, "scripts", "mpnn_boltz.sh"),
        ligand=ligand,
        variant="boltz",
        omit_json=mpnn_dir_posix + "/omit_boltz.json",
        bias_json=mpnn_dir_posix + "/bias_boltz.json",
        design_positions=boltz_positions,
        batch_size=args.batch_size,
    )

    # Validate
    all_ok = validate(campaign_dir, pipe_root, ligand)

    # Print workflow
    print_workflow(campaign_dir, ligand, smiles, pipe_root)

    if all_ok:
        print("Campaign ready!")
    else:
        print("Campaign scaffolded — generate conformers to complete setup.")


if __name__ == "__main__":
    main()
