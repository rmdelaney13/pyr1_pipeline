#!/usr/bin/env python3
"""
Generate Boltz prediction YAML inputs from variant signatures + ligand SMILES.

Reads a tier CSV (pair_id, ligand_name, ligand_smiles, variant_name, variant_signature, ...)
or a simple CSV (name, variant_signature, ligand_smiles).

Usage:
    # Tier CSV format (auto-detected):
    python prepare_boltz_yamls.py tier1_strong_binders.csv --out-dir ./boltz_inputs
    python prepare_boltz_yamls.py tier4_LCA_screen.csv --out-dir ./boltz_inputs --ligand-filter "Lithocholic Acid" --max-rows 500

    # With template + pocket constraints:
    python prepare_boltz_yamls.py tier1.csv --out-dir ./boltz_inputs --template 3QN1.pdb --force-template --pocket-constraint

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# WT PYR1 sequence (181 residues) — 3QN1 stabilized construct.
WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

# HAB1 PP2C sequence (337 residues, matches predict_wt_ternary.sh)
HAB1_SEQUENCE = (
    "GAMGRSVYELDCIPLWGTVSIQGNRSEMEDAFAVSPHFLKLPIKMLMGDHEGMSPSLTHLTGHFFGVY"
    "DGHGGHKVADYCRDRLHFALAEEIERIKDELCKRNTGEGRQVQWDKVFTSCFLTVDGEIEGKIGRAVVG"
    "SSDKVLEAVASETVGSTAVVALVCSSHIVVSNCGDSRAVLFRGKEAMPLSVDHKPDREDEYARIENAGGK"
    "VIQWQGARVFGVLAMSRSIGDRYLKPYVIPEPEVTFMPRSREDECLILASDGLWDVMNNQEVCEIARRR"
    "ILMWHKKNGAPPLAERGKGIDPACQAAADYLSMLALQKGSKDNISIIVIDLKAQRKFKTRT"
)

# PYR1 pocket residues (1-indexed) for pocket constraints
POCKET_RESIDUES = [59, 62, 79, 81, 83, 88, 90, 92, 106, 108, 110, 115, 116, 118, 120, 139, 157, 158, 159, 161, 162, 165]


def parse_variant_signature(signature: str) -> Dict[int, str]:
    """Parse variant signature into {position: target_amino_acid} dict."""
    if not signature or str(signature).strip() in ('', 'nan', 'None'):
        return {}

    mutations = {}
    normalized = str(signature).replace('_', ';').replace(' ', ';')
    normalized = re.sub(r';+', ';', normalized)

    for mut in normalized.split(';'):
        mut = mut.strip()
        if not mut:
            continue
        match = re.match(r'^([A-Z])?(\d+)([A-Z])$', mut)
        if match:
            _, pos, target_aa = match.groups()
            mutations[int(pos)] = target_aa
        else:
            print(f"WARNING: Could not parse mutation: '{mut}' in '{signature}'", file=sys.stderr)

    return mutations


def thread_mutations(wt_sequence: str, variant_signature: str) -> str:
    """Thread mutations onto WT sequence."""
    mutations = parse_variant_signature(variant_signature)
    if not mutations:
        return wt_sequence

    seq_list = list(wt_sequence)
    for pos, target_aa in mutations.items():
        idx = pos - 1
        if 0 <= idx < len(seq_list):
            seq_list[idx] = target_aa
        else:
            print(f"WARNING: Position {pos} out of range for sequence length {len(seq_list)}", file=sys.stderr)

    return ''.join(seq_list)


def patch_a3m_query(a3m_path: str, variant_signature: str, output_path: str) -> bool:
    """Create a modified .a3m where the query sequence matches the variant.

    In .a3m format the first sequence is the query (all uppercase, no gaps).
    We replace only the mutated positions so Boltz accepts the MSA as matching
    the input sequence.  Homolog rows are left untouched.

    Returns True if mutations were applied, False if variant is WT (file copied).
    """
    mutations = parse_variant_signature(variant_signature)
    if not mutations:
        shutil.copy2(a3m_path, output_path)
        return False

    with open(a3m_path) as f:
        content = f.read()

    lines = content.split('\n')

    # Locate the query: first header line, then contiguous non-header lines
    header_idx = None
    seq_start = None
    seq_end = None

    for i, line in enumerate(lines):
        if line.startswith('>') and header_idx is None:
            header_idx = i
            seq_start = i + 1
        elif seq_start is not None and seq_end is None:
            if line.startswith('>') or (line.strip() == '' and i > seq_start):
                seq_end = i
                break

    if seq_end is None:
        seq_end = len(lines)

    # Reconstruct the query sequence (may span multiple lines)
    query_seq = ''.join(lines[seq_start:seq_end])

    # Apply mutations (1-indexed positions → 0-indexed into uppercase-only query)
    mutated = list(query_seq)
    applied = 0
    for pos, aa in mutations.items():
        idx = pos - 1
        if 0 <= idx < len(mutated):
            mutated[idx] = aa
            applied += 1
        else:
            print(f"WARNING: MSA patch position {pos} out of range (query len={len(mutated)})",
                  file=sys.stderr)

    mutated_seq = ''.join(mutated)

    # Rebuild: header + mutated query (single line) + rest of file
    new_lines = lines[:seq_start] + [mutated_seq] + lines[seq_end:]

    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))

    return True


def generate_yaml(
    name: str,
    sequence: str,
    ligand_smiles: str,
    mode: str = "binary",
    msa_path: str = None,
    hab1_msa_path: str = None,
    template_path: str = None,
    force_template: bool = False,
    template_threshold: float = 2.0,
    pocket_constraint: bool = False,
    affinity: bool = False,
    msa_server: bool = False,
) -> str:
    """Generate a Boltz YAML input string."""
    lines = ["version: 1", "sequences:"]

    # Protein A (PYR1)
    lines.append("  - protein:")
    lines.append("      id: A")
    lines.append(f"      sequence: \"{sequence}\"")
    if msa_path:
        lines.append(f"      msa: {msa_path}")
    elif not msa_server:
        # Explicit empty MSA (single-sequence mode).
        # Omit msa: line entirely when --msa-server is used so Boltz
        # generates MSAs on-the-fly via --use_msa_server CLI flag.
        lines.append("      msa: empty")

    # Ligand B
    lines.append("  - ligand:")
    lines.append("      id: B")
    lines.append(f"      smiles: \"{ligand_smiles}\"")

    # Protein C (HAB1) for ternary mode
    if mode == "ternary":
        lines.append("  - protein:")
        lines.append("      id: C")
        lines.append(f"      sequence: \"{HAB1_SEQUENCE}\"")
        if hab1_msa_path:
            lines.append(f"      msa: {hab1_msa_path}")
        elif not msa_server:
            lines.append("      msa: empty")

    # Templates (top-level section per Boltz schema)
    if template_path:
        tmpl_ext = Path(template_path).suffix.lower()
        tmpl_key = "cif" if tmpl_ext == ".cif" else "pdb"
        lines.append("templates:")
        lines.append(f"  - {tmpl_key}: {template_path}")
        lines.append("    chain_id: A")
        if force_template:
            lines.append("    force: true")
            lines.append(f"    threshold: {template_threshold}")
        # Ternary: also template HAB1 (chain C) from the same complex CIF
        if mode == "ternary":
            lines.append(f"  - {tmpl_key}: {template_path}")
            lines.append("    chain_id: C")
            if force_template:
                lines.append("    force: true")
                lines.append(f"    threshold: {template_threshold}")

    # Constraints
    if pocket_constraint:
        lines.append("constraints:")
        lines.append("  - pocket:")
        lines.append("      binder: B")
        lines.append("      contacts:")
        for res in POCKET_RESIDUES:
            lines.append(f"        - [A, {res}]")
        lines.append("      max_distance: 6.0")

    # Affinity prediction
    if affinity:
        lines.append("properties:")
        lines.append("  - affinity:")
        lines.append("      binder: B")

    return '\n'.join(lines) + '\n'


def read_tier_csv(csv_path: str, ligand_filter: str = None) -> List[dict]:
    """Read a tier CSV and return normalized rows."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Auto-detect format: tier CSV has 'pair_id', simple CSV has 'name'
        is_tier = 'pair_id' in fieldnames

        for row in reader:
            if is_tier:
                name = row['pair_id'].strip()
                signature = row.get('variant_signature', '').strip()
                smiles = row['ligand_smiles'].strip()
                ligand_name = row.get('ligand_name', '').strip()
            else:
                name = row['name'].strip()
                signature = row.get('variant_signature', '').strip()
                smiles = row['ligand_smiles'].strip()
                ligand_name = row.get('ligand_name', '').strip()

            if ligand_filter and ligand_name != ligand_filter:
                continue

            rows.append({
                'name': name,
                'variant_signature': signature,
                'ligand_smiles': smiles,
                'ligand_name': ligand_name,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate Boltz YAML inputs from variant CSV")
    parser.add_argument("input_csv", nargs='+', help="One or more CSVs (tier format or simple format)")
    parser.add_argument("--out-dir", required=True, help="Output directory for YAML files")
    parser.add_argument("--mode", choices=["binary", "ternary"], default="binary",
                        help="Prediction mode (default: binary)")
    parser.add_argument("--msa", default=None, help="Path to pre-computed PYR1 .a3m MSA file")
    parser.add_argument("--msa-server", action="store_true",
                        help="Omit msa: line from YAML so Boltz uses --use_msa_server on-the-fly. "
                             "Mutually exclusive with --msa.")
    parser.add_argument("--hab1-msa", default=None, help="Path to pre-computed HAB1 .a3m MSA file (ternary mode)")
    parser.add_argument("--template", default=None, help="Path to template PDB/CIF (format auto-detected)")
    parser.add_argument("--force-template", action="store_true",
                        help="Force backbone to stay near template")
    parser.add_argument("--template-threshold", type=float, default=2.0,
                        help="Backbone distance threshold in Angstroms (default: 2.0)")
    parser.add_argument("--pocket-constraint", action="store_true",
                        help="Add pocket constraint to guide ligand placement")
    parser.add_argument("--affinity", action="store_true",
                        help="Request affinity prediction")
    parser.add_argument("--smiles-override", default=None,
                        help="Override all ligand SMILES with this string (e.g., canonical SMILES)")
    parser.add_argument("--ligand-filter", default=None,
                        help="Only include rows matching this ligand_name (e.g., 'Lithocholic Acid')")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to use (randomly sampled if CSV has more)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for downsampling (default: 42)")

    args = parser.parse_args()

    if args.msa and args.msa_server:
        print("ERROR: --msa and --msa-server are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all input CSVs
    all_rows = []
    for csv_path in args.input_csv:
        rows = read_tier_csv(csv_path, ligand_filter=args.ligand_filter)
        print(f"Read {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)

    # Downsample if needed
    if args.max_rows and len(all_rows) > args.max_rows:
        random.seed(args.seed)
        all_rows = random.sample(all_rows, args.max_rows)
        print(f"Downsampled to {args.max_rows} rows (seed={args.seed})")

    print(f"Total: {len(all_rows)} predictions to generate")

    if args.smiles_override:
        print(f"SMILES override: {args.smiles_override}")

    # If --msa provided, create per-variant patched MSAs in a subdirectory
    msa_dir = None
    if args.msa:
        msa_dir = out_dir / "msa"
        msa_dir.mkdir(exist_ok=True)
        print(f"Will patch MSA query sequences into {msa_dir}/")

    count = 0
    patched = 0
    for row in all_rows:
        sequence = thread_mutations(WT_PYR1_SEQUENCE, row['variant_signature'])
        smiles = args.smiles_override if args.smiles_override else row['ligand_smiles']

        # Per-variant MSA: patch query sequence to match mutant
        msa_path = args.msa  # default: use as-is (WT or None)
        if args.msa and row['variant_signature']:
            variant_msa = msa_dir / f"{row['name']}.a3m"
            was_patched = patch_a3m_query(args.msa, row['variant_signature'], str(variant_msa))
            msa_path = str(variant_msa)
            if was_patched:
                patched += 1

        yaml_content = generate_yaml(
            name=row['name'],
            sequence=sequence,
            ligand_smiles=smiles,
            mode=args.mode,
            msa_path=msa_path,
            hab1_msa_path=args.hab1_msa,
            template_path=args.template,
            force_template=args.force_template,
            template_threshold=args.template_threshold,
            pocket_constraint=args.pocket_constraint,
            affinity=args.affinity,
            msa_server=args.msa_server,
        )

        out_path = out_dir / f"{row['name']}.yaml"
        with open(out_path, 'w') as yf:
            yf.write(yaml_content)
        count += 1

    if msa_dir:
        print(f"Patched {patched} MSA files ({count - patched} WT copies) in {msa_dir}")

    print(f"Generated {count} YAML files in {out_dir}")

    # Write manifest for SLURM array indexing
    manifest_path = out_dir / "manifest.txt"
    yamls = sorted(out_dir.glob("*.yaml"))
    with open(manifest_path, 'w') as mf:
        for y in yamls:
            mf.write(str(y) + '\n')
    print(f"Manifest written to {manifest_path} ({len(yamls)} entries)")


if __name__ == "__main__":
    main()
