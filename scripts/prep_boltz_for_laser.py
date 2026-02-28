#!/usr/bin/env python3
"""
Prepare Boltz output PDBs for LASErMPNN inference.

Three transformations:
  1. Protonate ligand — RDKit adds explicit Hs with 3D coords from SMILES template
  2. Set B-factors — 0.0 on 16 designable pocket positions, 1.0 on everything else
  3. Add conserved water — superpose from 3QN1 reference (optional, --add-water)

Usage:
    python prep_boltz_for_laser.py \
        --input-dir /scratch/.../selected_pdbs \
        --output-dir /scratch/.../prepped_pdbs \
        --smiles "C[C@H](CCC(=O)O)..." \
        --ref-pdb /projects/.../3QN1_H2O.pdb \
        --add-water

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import prody as pr
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
    print("  This script requires prody and rdkit (available in boltz_env).", file=sys.stderr)
    sys.exit(1)

# Suppress ProDy warnings
pr.confProDy(verbosity='warning')

# 16 designable pocket positions (Boltz/natural numbering, 181-residue PYR1)
POCKET_RESNUMS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]


def protonate_ligand(pdb_path: str, smiles: str):
    """Extract ligand from PDB, protonate with RDKit, return modified ProDy AtomGroup + CONECT records.

    Adapted from NISE protonate_and_add_conect_records.py.
    Returns (protonated_ligand_ag, conect_lines) or (None, None) on failure.
    """
    prot = pr.parsePDB(pdb_path)
    if prot is None:
        return None, None, None

    prot_only = prot.select('protein')
    if prot_only is None:
        return None, None, None
    prot_only = prot_only.copy()

    # Select ligand (non-protein, non-water heavy atoms)
    lig_sel = prot.select('(not protein) and (not water) and not element H')
    if lig_sel is None:
        return None, None, None
    lig = lig_sel.copy()

    # Get ligand residue name
    resnames = set(lig.getResnames())
    if len(resnames) != 1:
        print(f"  WARNING: Multiple ligand residue names: {resnames}", file=sys.stderr)
    tlc = resnames.pop()

    # Write ligand to PDB string for RDKit
    sstream = io.StringIO()
    pr.writePDBStream(sstream, lig)
    lig_pdb_block = sstream.getvalue()

    # Parse with RDKit
    pdb_mol = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=True, sanitize=False)
    if pdb_mol is None:
        return None, None, None

    smi_mol = Chem.MolFromSmiles(smiles)
    if smi_mol is None:
        return None, None, None

    # Assign bond orders from SMILES template
    try:
        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    except Exception as e:
        print(f"  WARNING: Bond order assignment failed: {e}", file=sys.stderr)
        return None, None, None

    # Add explicit hydrogens with 3D coordinates
    pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)

    # Get CONECT records from RDKit PDB output
    rdkit_pdb_block = Chem.MolToPDBBlock(pdb_mol, flavor=(4 | 8))
    rdkit_conect = [line for line in rdkit_pdb_block.split('\n') if line.startswith('CONECT')]

    # Parse the protonated ligand back into ProDy
    rdkit_stream = io.StringIO(rdkit_pdb_block)
    modlig = pr.parsePDBStream(rdkit_stream)
    if modlig is None:
        return None, None, None

    # Standardize naming
    modlig.setResnames(tlc)
    modlig.setResnums([1] * len(modlig))
    modlig.setChids('B')
    modlig.setOccupancies([1.0] * len(modlig))
    modlig.setBetas([0.0] * len(modlig))

    # Rename atoms sequentially by element (C1, C2, N1, H1, H2, ...)
    new_atomnames = []
    dd = defaultdict(int)
    elements = [''.join(c for c in name if not c.isdigit()) for name in modlig.getNames()]
    for element in elements:
        dd[element] += 1
        new_atomnames.append(f"{element}{dd[element]}")
    modlig.setNames(new_atomnames)

    return prot_only, modlig, rdkit_conect


def set_bfactors(protein_ag, pocket_resnums=POCKET_RESNUMS):
    """Set B-factors: 1.0 (fixed) everywhere, 0.0 (designable) at pocket positions."""
    protein_ag.setBetas([1.0] * len(protein_ag))
    sel_str = 'resnum ' + ' '.join(str(r) for r in pocket_resnums)
    designable = protein_ag.select(sel_str)
    if designable is not None:
        designable.setBetas([0.0] * designable.numAtoms())
        return designable.numAtoms()
    return 0


def superpose_water(protein_ag, ref_pdb_path: str):
    """Superpose conserved water from reference PDB onto the predicted structure.

    Aligns protein CAs, transforms reference water into predicted frame.
    Returns water AtomGroup or None.
    """
    ref = pr.parsePDB(ref_pdb_path)
    if ref is None:
        return None

    # Get reference water (TP3 D 1 or HOH)
    ref_water = ref.select('resname TP3 or resname HOH')
    if ref_water is None:
        return None

    # Get CA atoms for alignment — match by residue number since
    # 3QN1 (res 8-181) and Boltz (res 1-181) have different ranges
    ref_ca = ref.select('protein and name CA')
    pred_ca = protein_ag.select('name CA')
    if ref_ca is None or pred_ca is None:
        return None

    # Find common residue numbers
    ref_resnums = set(ref_ca.getResnums())
    pred_resnums = set(pred_ca.getResnums())
    common = sorted(ref_resnums & pred_resnums)
    if len(common) < 20:
        return None

    resnum_sel = 'resnum ' + ' '.join(str(r) for r in common)
    ref_ca_matched = ref.select(f'protein and name CA and ({resnum_sel})')
    pred_ca_matched = protein_ag.select(f'name CA and ({resnum_sel})')
    if ref_ca_matched is None or pred_ca_matched is None:
        return None
    if len(ref_ca_matched) != len(pred_ca_matched):
        return None

    # Align reference CAs onto predicted CAs to get transformation
    # calcTransformation(mobile, target) gives T such that T(mobile) ≈ target
    try:
        transformation = pr.calcTransformation(ref_ca_matched, pred_ca_matched)
    except Exception:
        return None

    water_copy = ref_water.copy()
    transformation.apply(water_copy)

    # Standardize water naming
    water_copy.setResnames(['HOH'] * len(water_copy))
    water_copy.setChids(['W'] * len(water_copy))
    water_copy.setResnums([1] * len(water_copy))
    water_copy.setBetas([0.0] * len(water_copy))
    water_copy.setOccupancies([1.0] * len(water_copy))

    return water_copy


def write_prepped_pdb(output_path, protein_ag, ligand_ag, conect_lines, water_ag=None):
    """Write merged PDB with protein + ligand + optional water + CONECT records."""
    # Merge components
    merged = protein_ag + ligand_ag
    if water_ag is not None:
        merged = merged + water_ag

    # Write PDB
    final_stream = io.StringIO()
    pr.writePDBStream(final_stream, merged)
    pdb_text = final_stream.getvalue()

    # Count protein atoms (including TER line) for CONECT offset
    prot_len = len(protein_ag)
    if 'TER ' in pdb_text:
        prot_len += 1

    # Offset CONECT atom indices to account for protein atoms
    offset_conect = []
    for conect in conect_lines:
        parts = conect.split()
        keyword = parts[0]
        indices = parts[1:]
        offset_indices = [str(int(idx) + prot_len).rjust(5) for idx in indices]
        offset_conect.append(keyword + ''.join(offset_indices))

    # Write output: PDB content (without END) + CONECT records + END
    with open(output_path, 'w') as f:
        # Remove trailing END if present
        pdb_text = pdb_text.rsplit('END', 1)[0]
        f.write(pdb_text)
        if offset_conect:
            f.write('\n'.join(offset_conect))
            f.write('\n')
        f.write('END\n')


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Boltz PDBs for LASErMPNN (protonate, B-factors, water)")
    parser.add_argument("--input-dir", required=True,
                        help="Directory of Boltz PDBs to prep")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for prepped PDBs")
    parser.add_argument("--smiles", required=True,
                        help="Ligand SMILES string for protonation")
    parser.add_argument("--ref-pdb", default=None,
                        help="Reference PDB for water superposition (e.g. 3QN1_H2O.pdb)")
    parser.add_argument("--add-water", action="store_true",
                        help="Add conserved water from reference PDB")
    parser.add_argument("--pocket-residues", default=None,
                        help="Space-separated pocket residue numbers (default: 16 PYR1 positions)")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pocket_residues:
        pocket_resnums = [int(x) for x in args.pocket_residues.split()]
    else:
        pocket_resnums = POCKET_RESNUMS

    if args.add_water and not args.ref_pdb:
        print("ERROR: --add-water requires --ref-pdb", file=sys.stderr)
        sys.exit(1)

    # Find PDB files
    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"ERROR: No PDB files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Preparing {len(pdb_files)} PDBs for LASErMPNN")
    print(f"  Pocket positions: {pocket_resnums}")
    print(f"  Add water: {args.add_water}")
    print()

    successful = 0
    failed = 0
    manifest_lines = []

    for pdb_path in pdb_files:
        name = pdb_path.stem

        # Step 1: Protonate ligand
        prot_ag, lig_ag, conect = protonate_ligand(str(pdb_path), args.smiles)
        if prot_ag is None or lig_ag is None:
            print(f"  FAILED: {name} (protonation error)")
            failed += 1
            continue

        # Step 2: Set B-factors
        n_designable = set_bfactors(prot_ag, pocket_resnums)

        # Step 3: Optionally add water
        water_ag = None
        if args.add_water and args.ref_pdb:
            water_ag = superpose_water(prot_ag, args.ref_pdb)
            if water_ag is None:
                print(f"  WARNING: {name} — water superposition failed, proceeding without water")

        # Write output
        out_path = output_dir / pdb_path.name
        write_prepped_pdb(str(out_path), prot_ag, lig_ag, conect, water_ag)

        manifest_lines.append(str(out_path))
        successful += 1

    # Write manifest
    manifest_path = output_dir.parent / "prepped_manifest.txt"
    with open(manifest_path, 'w') as f:
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
