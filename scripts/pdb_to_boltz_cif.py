#!/usr/bin/env python3
"""Convert a single-chain PDB to a Boltz-compatible mmCIF with required metadata.

PyMOL/ChimeraX CIF exports often lack _entity, _entity_poly, _entity_poly_seq
sections that Boltz's mmCIF parser requires. This script reads a PDB and writes
a complete CIF.

Usage:
    python pdb_to_boltz_cif.py input.pdb output.cif
"""

import argparse
import sys
from collections import OrderedDict

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def parse_pdb(pdb_path):
    """Parse PDB ATOM records, return atoms list and sequence."""
    atoms = []
    sequence = OrderedDict()  # resnum -> (resname, chain)

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            atom_serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            alt_loc = line[16:17].strip() or '.'
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip() or 'A'
            res_seq = int(line[22:26].strip())
            ins_code = line[26:27].strip() or '?'
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occ = float(line[54:60]) if len(line) > 54 else 1.00
            bfac = float(line[60:66]) if len(line) > 60 else 0.00
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            charge = line[78:80].strip() if len(line) > 78 else '0'
            if not charge or charge == '':
                charge = '0'
            # Clean up charge
            try:
                int(charge)
            except ValueError:
                charge = '0'

            record_type = 'ATOM' if line.startswith('ATOM') else 'HETATM'

            atoms.append({
                'record': record_type,
                'serial': atom_serial,
                'name': atom_name,
                'alt': alt_loc,
                'resname': res_name,
                'chain': chain_id,
                'resseq': res_seq,
                'icode': ins_code,
                'x': x, 'y': y, 'z': z,
                'occ': occ, 'bfac': bfac,
                'element': element,
                'charge': charge,
            })

            if res_name in THREE_TO_ONE:
                sequence[res_seq] = (res_name, chain_id)

    return atoms, sequence


def write_cif(atoms, sequence, entry_id, out_path):
    """Write a Boltz-compatible mmCIF file."""
    chain_id = atoms[0]['chain'] if atoms else 'A'
    one_letter_seq = ''.join(THREE_TO_ONE.get(rn, 'X') for rn, _ in sequence.values())

    with open(out_path, 'w') as f:
        f.write(f"data_{entry_id}\n")
        f.write(f"_entry.id {entry_id}\n")
        f.write("#\n")

        # _struct
        f.write(f"_struct.entry_id {entry_id}\n")
        f.write(f"_struct.title 'PYR1 template structure'\n")
        f.write("#\n")

        # _entity
        f.write("loop_\n")
        f.write("_entity.id\n")
        f.write("_entity.type\n")
        f.write("_entity.pdbx_description\n")
        f.write("1 polymer 'PYR1 receptor'\n")
        f.write("#\n")

        # _entity_poly
        f.write("loop_\n")
        f.write("_entity_poly.entity_id\n")
        f.write("_entity_poly.type\n")
        f.write("_entity_poly.pdbx_seq_one_letter_code\n")
        f.write("_entity_poly.pdbx_strand_id\n")
        f.write(f"1 'polypeptide(L)' '{one_letter_seq}' {chain_id}\n")
        f.write("#\n")

        # _entity_poly_seq
        f.write("loop_\n")
        f.write("_entity_poly_seq.entity_id\n")
        f.write("_entity_poly_seq.num\n")
        f.write("_entity_poly_seq.mon_id\n")
        for i, (resseq, (resname, _)) in enumerate(sequence.items(), 1):
            f.write(f"1 {i} {resname}\n")
        f.write("#\n")

        # _atom_site
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.label_atom_id\n")
        f.write("_atom_site.label_alt_id\n")
        f.write("_atom_site.label_comp_id\n")
        f.write("_atom_site.label_asym_id\n")
        f.write("_atom_site.label_entity_id\n")
        f.write("_atom_site.label_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        f.write("_atom_site.occupancy\n")
        f.write("_atom_site.B_iso_or_equiv\n")
        f.write("_atom_site.pdbx_formal_charge\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.pdbx_PDB_model_num\n")

        for i, atom in enumerate(atoms, 1):
            f.write(
                f"{atom['record']}   {i}   {atom['element']} {atom['name']}  "
                f" {atom['alt']} {atom['resname']} {atom['chain']} 1 "
                f"{atom['resseq']} {atom['icode']} "
                f"{atom['x']:.3f} {atom['y']:.3f} {atom['z']:.3f} "
                f"{atom['occ']:.2f}  {atom['bfac']:.2f} {atom['charge']} "
                f"{atom['chain']} 1\n"
            )
        f.write("#\n")

    print(f"Wrote {len(atoms)} atoms, {len(sequence)} residues to {out_path}")
    print(f"Sequence ({len(one_letter_seq)} aa): {one_letter_seq[:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Convert PDB to Boltz-compatible mmCIF")
    parser.add_argument("input_pdb", help="Input PDB file")
    parser.add_argument("output_cif", help="Output CIF file")
    parser.add_argument("--entry-id", default=None, help="Entry ID (default: from filename)")
    args = parser.parse_args()

    entry_id = args.entry_id or args.input_pdb.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    atoms, sequence = parse_pdb(args.input_pdb)

    if not atoms:
        print("ERROR: No ATOM records found", file=sys.stderr)
        sys.exit(1)

    write_cif(atoms, sequence, entry_id, args.output_cif)


if __name__ == "__main__":
    main()
