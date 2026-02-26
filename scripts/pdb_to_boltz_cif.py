#!/usr/bin/env python3
"""Convert a PDB to a Boltz-compatible mmCIF using gemmi.

Boltz uses gemmi.make_structure_from_block() to parse templates. PyMOL CIF
exports lack required metadata, causing IndexError. This script uses gemmi
to read the PDB and write a proper mmCIF that Boltz can parse.

Requires: gemmi (installed in boltz_env on Alpine)

Usage:
    python pdb_to_boltz_cif.py input.pdb output.cif
"""

import argparse
import sys

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi not installed. Run in boltz_env: pip install gemmi", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PDB to Boltz-compatible mmCIF via gemmi")
    parser.add_argument("input_pdb", help="Input PDB file")
    parser.add_argument("output_cif", help="Output CIF file")
    args = parser.parse_args()

    # Read PDB with gemmi
    structure = gemmi.read_structure(args.input_pdb)

    if len(structure) == 0:
        print("ERROR: No models found in PDB", file=sys.stderr)
        sys.exit(1)

    # Set up entity info so gemmi writes proper mmCIF metadata
    structure.setup_entities()
    structure.assign_subchains()

    n_atoms = sum(1 for model in structure for chain in model for res in chain for atom in res)
    n_chains = sum(1 for chain in structure[0])

    print(f"Read {n_atoms} atoms, {n_chains} chain(s) from {args.input_pdb}")

    # Write as mmCIF
    structure.make_mmcif_document().write_file(args.output_cif)

    # Verify it's parseable
    block = gemmi.cif.read(args.output_cif)[0]
    verify = gemmi.make_structure_from_block(block)
    if len(verify) == 0 or len(verify[0]) == 0:
        print("WARNING: Verification failed - output CIF has no models/chains", file=sys.stderr)
        sys.exit(1)

    n_verify_chains = sum(1 for chain in verify[0])
    print(f"Verified: {args.output_cif} has {len(verify)} model(s), {n_verify_chains} chain(s)")
    print("CIF is Boltz-compatible.")


if __name__ == "__main__":
    main()
