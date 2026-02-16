#!/usr/bin/env python3
"""
Merge all ligand data sources (original CSV + fetched from PubChem).

Usage:
    python merge_ligand_data.py
"""

import csv
from pathlib import Path
from typing import Dict, List


def main():
    # Input files
    original_csv = Path("ligand_properties_with_conformer_counts.csv")
    fetched_csv = Path("ligand_smiles_fetched_from_pubchem.csv")
    original_signatures = Path("ligand_smiles_signature.csv")

    # Output file
    output_csv = Path("ligand_complete_dataset.csv")

    # Check files exist
    if not original_csv.exists():
        print(f"ERROR: {original_csv} not found")
        return

    all_ligands = {}

    # Read original (with SMILES from CSV)
    print(f"Reading {original_csv}...")
    with open(original_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ligands[row['ligand_name']] = row

    # Read fetched from PubChem
    if fetched_csv.exists():
        print(f"Reading {fetched_csv}...")
        with open(fetched_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_ligands[row['ligand_name']] = row
    else:
        print(f"WARNING: {fetched_csv} not found - skipping")

    # Read original signatures to get variant info
    print(f"Reading {original_signatures} for variant details...")
    variant_map = {}  # ligand_name -> list of (variant_name, signature)

    with open(original_signatures, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ligand_name = row['ligand_name']
            variant_name = row['PYR1_variant_name']
            signature = row['PYR1_variant_signature']

            if ligand_name not in variant_map:
                variant_map[ligand_name] = []
            variant_map[ligand_name].append((variant_name, signature))

    # Combine data
    combined = []
    for ligand_name, data in all_ligands.items():
        # Add variant details
        if ligand_name in variant_map:
            data['n_variants'] = len(variant_map[ligand_name])
            data['variant_names'] = "; ".join([v[0] for v in variant_map[ligand_name]])
        combined.append(data)

    # Sort by rotatable bonds (descending)
    combined.sort(key=lambda x: int(x.get('n_rotatable_bonds', 0)), reverse=True)

    # Write output
    if combined:
        fieldnames = list(combined[0].keys())
        # Reorder for readability
        priority_fields = [
            'ligand_name', 'smiles', 'smiles_source', 'n_variants',
            'n_rotatable_bonds', 'recommended_conformers', 'mol_weight',
            'n_heavy_atoms', 'formula', 'molecular_formula'
        ]
        other_fields = [f for f in fieldnames if f not in priority_fields]
        fieldnames = [f for f in priority_fields if f in fieldnames] + other_fields

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(combined)

        print(f"\n✓ Merged {len(combined)} ligands")
        print(f"  Output: {output_csv}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total ligands with SMILES: {len(combined)}")

        sources = {}
        for lig in combined:
            source = lig.get('smiles_source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        print("\nSMILES sources:")
        for source, count in sources.items():
            print(f"  - {source}: {count}")

        print("\nTop 10 by rotatable bonds:")
        print("-" * 70)
        print(f"{'Ligand':<30} {'Rot.Bonds':<12} {'Rec.Confs':<12}")
        print("-" * 70)
        for lig in combined[:10]:
            print(f"{lig['ligand_name']:<30} "
                  f"{lig.get('n_rotatable_bonds', 'N/A'):<12} "
                  f"{lig.get('recommended_conformers', 'N/A'):<12}")

        print("=" * 70)
        print("\nNEXT STEPS:")
        print("1. Review the complete dataset:", output_csv)
        print("2. Use this data to generate conformers with ligand_conformers module")
        print("3. Run docking simulations with your pyr1_pipeline")
        print("=" * 70)


if __name__ == "__main__":
    main()
