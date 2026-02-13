#!/usr/bin/env python3
import os
import json
import argparse
import sys
import copy

def parse_fasta(fasta_path):
    """
    Simple FASTA parser that yields (header, sequence) tuples.
    Header is taken as the first word after '>' (no spaces).
    Sequence is concatenated lines (no whitespace).
    """
    sequences = []
    with open(fasta_path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(seq_lines)))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        # Add last record
        if header is not None:
            sequences.append((header, ''.join(seq_lines)))
    return sequences

def extract_smiles_from_sdf(sdf_path):
    """Extract canonical SMILES from an SDF file using RDKit."""
    try:
        from rdkit import Chem
    except ImportError:
        print("ERROR: RDKit is required for --sdf. Install with: conda install -c conda-forge rdkit",
              file=sys.stderr)
        sys.exit(1)
    supplier = Chem.SDMolSupplier(str(sdf_path))
    mol = next(iter(supplier))
    if mol is None:
        print(f"ERROR: Could not read molecule from {sdf_path}", file=sys.stderr)
        sys.exit(1)
    return Chem.MolToSmiles(mol)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def main(template_path, fasta_path, outdir, smiles=None):
    # 1. Load the template JSON
    try:
        template = load_json(template_path)
    except Exception as e:
        print(f"Error: could not read template JSON '{template_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Parse the FASTA file
    sequences = parse_fasta(fasta_path)
    if not sequences:
        print(f"Error: no sequences found in FASTA '{fasta_path}'", file=sys.stderr)
        sys.exit(1)

    # 3. Verify that template has a "sequences" list and a protein with id "A"
    if "sequences" not in template or not isinstance(template["sequences"], list):
        print(f"Error: template JSON does not contain a 'sequences' list", file=sys.stderr)
        sys.exit(1)

    # Find index of chain A entry
    idx_A = None
    for idx, entry in enumerate(template["sequences"]):
        if "protein" in entry and entry["protein"].get("id") == "A":
            idx_A = idx
            break
    if idx_A is None:
        print(f"Error: template JSON has no 'protein' entry with id 'A' in 'sequences'", file=sys.stderr)
        sys.exit(1)

    # 3b. If SMILES provided, update ligand entry in template before generating
    if smiles:
        updated_ligand = False
        for entry in template["sequences"]:
            if "ligand" in entry and "smiles" in entry["ligand"]:
                old_smiles = entry["ligand"]["smiles"]
                entry["ligand"]["smiles"] = smiles
                updated_ligand = True
                print(f"Ligand SMILES: {old_smiles} -> {smiles}")
                break
        if not updated_ligand:
            print("Warning: --smiles provided but no ligand entry found in template", file=sys.stderr)

    # 4. Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # 5. For each (header, seq), produce a JSON
    for header, seq in sequences:
        # Make a deep copy of the template object
        new_json = copy.deepcopy(template)

        # Replace chain A sequence
        new_json["sequences"][idx_A]["protein"]["sequence"] = seq

        # Update the top-level "name" field to match the FASTA header
        new_json["name"] = header

        # Write out to outdir/<header>.json
        out_path = os.path.join(outdir, f"{header}.json")
        try:
            write_json(new_json, out_path)
            print(f"Wrote {out_path}")
        except Exception as e:
            print(f"Warning: could not write JSON for '{header}': {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold3 JSON files by replacing chain A sequence (and name) from a FASTA file."
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the template JSON (e.g. alphafold_input_pyr.json)."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the FASTA file containing new chain-A sequences."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where individual JSON files will be written."
    )

    smiles_group = parser.add_mutually_exclusive_group()
    smiles_group.add_argument(
        "--smiles",
        help="SMILES string for the ligand (replaces SMILES in template)."
    )
    smiles_group.add_argument(
        "--sdf",
        help="SDF file to extract ligand SMILES from (requires RDKit)."
    )

    args = parser.parse_args()

    # Resolve SMILES
    smiles = args.smiles
    if args.sdf:
        smiles = extract_smiles_from_sdf(args.sdf)
        print(f"Extracted SMILES from {args.sdf}: {smiles}")

    main(args.template, args.fasta, args.outdir, smiles=smiles)

