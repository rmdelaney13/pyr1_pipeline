#!/usr/bin/env python3
"""
Validate SMILES in the CSV and calculate rotatable bonds.

Usage:
    python validate_ligand_smiles.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
except ImportError:
    print("ERROR: RDKit is required")
    sys.exit(1)


def calculate_conformers(n_rot_bonds):
    """3^N heuristic for conformer count."""
    if n_rot_bonds == 0:
        return 50
    return min(max(3 ** n_rot_bonds, 50), 2000)


def main():
    csv_path = Path("ligand_smiles_signature.csv")

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    print("=" * 80)
    print("VALIDATING LIGAND SMILES")
    print("=" * 80)

    # Track ligands
    ligands_with_smiles = {}
    ligands_without_smiles = defaultdict(int)
    invalid_smiles = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            ligand_name = row['ligand_name']
            smiles = row['ligand_smiles_or_ligand_ID'].strip()
            variant = row['PYR1_variant_name']

            # Count variants
            if not smiles:
                ligands_without_smiles[ligand_name] += 1
                continue

            # Only process unique ligands
            if ligand_name in ligands_with_smiles:
                continue

            # Try to validate SMILES
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                invalid_smiles.append({
                    'ligand_name': ligand_name,
                    'smiles': smiles,
                    'reason': 'Invalid SMILES - RDKit cannot parse'
                })
                continue

            # Calculate properties
            n_rot_bonds = Descriptors.NumRotatableBonds(mol)
            n_heavy = Descriptors.HeavyAtomCount(mol)
            mw = Descriptors.MolWt(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            logp = Descriptors.MolLogP(mol)
            n_conformers = calculate_conformers(n_rot_bonds)

            ligands_with_smiles[ligand_name] = {
                'smiles': smiles,
                'mol': mol,
                'n_rotatable_bonds': n_rot_bonds,
                'n_heavy_atoms': n_heavy,
                'mol_weight': mw,
                'formula': formula,
                'logp': logp,
                'recommended_conformers': n_conformers,
            }

    # Print results
    print(f"\n{'SUMMARY'}")
    print("=" * 80)
    print(f"Ligands WITH valid SMILES: {len(ligands_with_smiles)}")
    print(f"Ligands WITHOUT SMILES: {len(ligands_without_smiles)}")
    print(f"Ligands with INVALID SMILES: {len(invalid_smiles)}")

    # Valid SMILES
    if ligands_with_smiles:
        print("\n" + "=" * 80)
        print("VALID SMILES ✓")
        print("=" * 80)
        print(f"{'Ligand':<35} {'Rot.Bonds':<12} {'Conformers':<12} {'MW':<10}")
        print("-" * 80)

        # Sort by rotatable bonds
        sorted_ligands = sorted(ligands_with_smiles.items(),
                                key=lambda x: x[1]['n_rotatable_bonds'],
                                reverse=True)

        for name, props in sorted_ligands:
            print(f"{name:<35} {props['n_rotatable_bonds']:<12} "
                  f"{props['recommended_conformers']:<12} {props['mol_weight']:<10.1f}")
            print(f"  Formula: {props['formula']}")
            print(f"  SMILES: {props['smiles'][:70]}{'...' if len(props['smiles']) > 70 else ''}")
            print()

    # Invalid SMILES
    if invalid_smiles:
        print("\n" + "=" * 80)
        print("INVALID SMILES ✗")
        print("=" * 80)
        for item in invalid_smiles:
            print(f"Ligand: {item['ligand_name']}")
            print(f"  SMILES: {item['smiles']}")
            print(f"  Reason: {item['reason']}")
            print()

    # Missing SMILES
    if ligands_without_smiles:
        print("\n" + "=" * 80)
        print(f"MISSING SMILES ({len(ligands_without_smiles)} ligands)")
        print("=" * 80)
        sorted_missing = sorted(ligands_without_smiles.items(),
                                key=lambda x: x[1],
                                reverse=True)

        for name, count in sorted_missing:
            print(f"  - {name} ({count} variants)")

    # Write validated output
    if ligands_with_smiles:
        output_path = Path("ligand_smiles_validated.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'ligand_name', 'smiles', 'formula', 'mol_weight',
                'n_heavy_atoms', 'n_rotatable_bonds', 'recommended_conformers',
                'logp'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for name, props in sorted_ligands:
                writer.writerow({
                    'ligand_name': name,
                    'smiles': props['smiles'],
                    'formula': props['formula'],
                    'mol_weight': f"{props['mol_weight']:.2f}",
                    'n_heavy_atoms': props['n_heavy_atoms'],
                    'n_rotatable_bonds': props['n_rotatable_bonds'],
                    'recommended_conformers': props['recommended_conformers'],
                    'logp': f"{props['logp']:.2f}",
                })

        print("\n" + "=" * 80)
        print(f"✓ Validated SMILES written to: {output_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
