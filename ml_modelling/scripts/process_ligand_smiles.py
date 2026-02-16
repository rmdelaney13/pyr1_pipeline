#!/usr/bin/env python3
"""
Process ligand SMILES from CSV, calculate rotatable bonds, and determine
optimal conformer counts for docking.

Usage:
    python process_ligand_smiles.py
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError:
    print("ERROR: RDKit is required. Install with:")
    print("  conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("WARNING: pandas not found. Install for better output:")
    print("  pip install pandas")
    pd = None


# ---------------------------------------------------------------------------
# Conformer count heuristic based on rotatable bonds
# ---------------------------------------------------------------------------

def calculate_num_conformers(n_rotatable_bonds: int, base: int = 3) -> int:
    """
    Calculate number of conformers to generate based on rotatable bonds.

    Common heuristic: 3^N where N is number of rotatable bonds, capped at
    a reasonable maximum.

    Parameters
    ----------
    n_rotatable_bonds : int
        Number of rotatable bonds in the molecule
    base : int
        Base for exponential calculation (default: 3)

    Returns
    -------
    int
        Recommended number of conformers to generate
    """
    if n_rotatable_bonds == 0:
        return 50  # rigid molecules still need some sampling

    # 3^N heuristic with caps
    num_confs = base ** n_rotatable_bonds

    # Apply reasonable bounds
    if num_confs < 50:
        return 50
    elif num_confs > 2000:
        return 2000  # cap at 2000 for computational feasibility
    else:
        return num_confs


# ---------------------------------------------------------------------------
# SMILES processing and molecular property calculation
# ---------------------------------------------------------------------------

def get_mol_properties(smiles: str) -> Optional[Dict]:
    """
    Calculate molecular properties from SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule

    Returns
    -------
    dict or None
        Dictionary with properties, or None if SMILES is invalid
    """
    if not smiles or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_rot_bonds = Descriptors.NumRotatableBonds(mol)
    n_heavy_atoms = Descriptors.HeavyAtomCount(mol)
    mol_weight = Descriptors.MolWt(mol)
    n_hbd = Descriptors.NumHDonors(mol)
    n_hba = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)

    # Calculate recommended conformer count
    num_conformers = calculate_num_conformers(n_rot_bonds)

    return {
        "smiles": smiles,
        "mol_weight": round(mol_weight, 2),
        "n_heavy_atoms": n_heavy_atoms,
        "n_rotatable_bonds": n_rot_bonds,
        "n_hbd": n_hbd,
        "n_hba": n_hba,
        "logp": round(logp, 2),
        "recommended_conformers": num_conformers,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


# ---------------------------------------------------------------------------
# PubChem lookup for missing SMILES
# ---------------------------------------------------------------------------

def fetch_smiles_from_pubchem(ligand_name: str) -> Optional[str]:
    """
    Fetch SMILES from PubChem by compound name.

    Parameters
    ----------
    ligand_name : str
        Name of the compound

    Returns
    -------
    str or None
        Canonical SMILES if found, else None
    """
    try:
        import pubchempy as pcp
    except ImportError:
        print("INFO: pubchempy not available for automatic lookup.")
        print("  Install with: pip install pubchempy")
        return None

    try:
        compounds = pcp.get_compounds(ligand_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"  PubChem lookup failed for {ligand_name}: {e}")

    return None


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path) -> List[Dict]:
    """
    Process the ligand CSV and extract unique ligands with properties.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file

    Returns
    -------
    list of dict
        List of ligand records with properties
    """
    ligand_data = {}  # ligand_name -> properties

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            ligand_name = row['ligand_name'].strip()
            smiles_or_id = row['ligand_smiles_or_ligand_ID'].strip()
            variant_name = row['PYR1_variant_name'].strip()

            # Skip if we've already processed this ligand
            if ligand_name in ligand_data:
                # Add variant to the list
                ligand_data[ligand_name]['variants'].append(variant_name)
                continue

            # Try to parse as SMILES
            props = get_mol_properties(smiles_or_id)

            if props:
                # Valid SMILES found
                ligand_data[ligand_name] = {
                    'ligand_name': ligand_name,
                    'smiles': smiles_or_id,
                    'smiles_source': 'csv',
                    'variants': [variant_name],
                    **props
                }
            else:
                # No valid SMILES - mark for manual lookup
                ligand_data[ligand_name] = {
                    'ligand_name': ligand_name,
                    'smiles': None,
                    'smiles_source': 'missing',
                    'variants': [variant_name],
                    'csv_id': smiles_or_id if smiles_or_id else 'N/A',
                }

    # Convert to list and count variants
    results = []
    for ligand_name, data in ligand_data.items():
        data['n_variants'] = len(data['variants'])
        # Keep only unique variants
        data['variants'] = list(set(data['variants']))
        results.append(data)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def write_output(ligands: List[Dict], output_path: Path):
    """
    Write processed ligand data to CSV.

    Parameters
    ----------
    ligands : list of dict
        Processed ligand records
    output_path : Path
        Path to output CSV file
    """
    # Separate ligands with and without SMILES
    with_smiles = [lig for lig in ligands if lig['smiles'] is not None]
    without_smiles = [lig for lig in ligands if lig['smiles'] is None]

    # Sort by rotatable bonds (descending) for with_smiles
    with_smiles.sort(key=lambda x: x.get('n_rotatable_bonds', 0), reverse=True)

    # Write complete data
    fieldnames = [
        'ligand_name', 'smiles', 'smiles_source', 'n_variants',
        'mol_weight', 'n_heavy_atoms', 'n_rotatable_bonds',
        'recommended_conformers', 'formula', 'n_hbd', 'n_hba', 'logp'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(with_smiles)

    print(f"\n✓ Wrote {len(with_smiles)} ligands with SMILES to: {output_path}")

    # Write missing SMILES separately
    if without_smiles:
        missing_path = output_path.parent / f"{output_path.stem}_missing_smiles.csv"
        with open(missing_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ligand_name', 'csv_id', 'n_variants'])
            writer.writeheader()
            for lig in without_smiles:
                writer.writerow({
                    'ligand_name': lig['ligand_name'],
                    'csv_id': lig.get('csv_id', 'N/A'),
                    'n_variants': lig['n_variants']
                })

        print(f"⚠ {len(without_smiles)} ligands missing SMILES written to: {missing_path}")
        print("\nLigands missing SMILES:")
        for lig in without_smiles:
            print(f"  - {lig['ligand_name']} ({lig['n_variants']} variants)")

    return with_smiles, without_smiles


def print_summary(with_smiles: List[Dict], without_smiles: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total unique ligands: {len(with_smiles) + len(without_smiles)}")
    print(f"  - With SMILES: {len(with_smiles)}")
    print(f"  - Missing SMILES: {len(without_smiles)}")

    if with_smiles:
        print("\n" + "-" * 70)
        print("LIGANDS BY ROTATABLE BONDS (top 10):")
        print("-" * 70)
        print(f"{'Ligand':<30} {'Rot.Bonds':<12} {'Rec.Confs':<12} {'MW':<8}")
        print("-" * 70)
        for lig in with_smiles[:10]:
            print(f"{lig['ligand_name']:<30} "
                  f"{lig['n_rotatable_bonds']:<12} "
                  f"{lig['recommended_conformers']:<12} "
                  f"{lig['mol_weight']:<8.1f}")

        # Statistics
        rot_bonds = [lig['n_rotatable_bonds'] for lig in with_smiles]
        print("\n" + "-" * 70)
        print("ROTATABLE BOND STATISTICS:")
        print("-" * 70)
        print(f"  Min:    {min(rot_bonds)}")
        print(f"  Max:    {max(rot_bonds)}")
        print(f"  Mean:   {sum(rot_bonds) / len(rot_bonds):.1f}")
        print(f"  Median: {sorted(rot_bonds)[len(rot_bonds)//2]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Input and output paths
    input_csv = Path(__file__).parent / "ligand_smiles_signature.csv"
    output_csv = Path(__file__).parent / "ligand_properties_with_conformer_counts.csv"

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)

    print("=" * 70)
    print("LIGAND SMILES PROCESSOR")
    print("=" * 70)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")

    # Process CSV
    print("\nProcessing ligands...")
    ligands = process_csv(input_csv)

    # Write outputs
    with_smiles, without_smiles = write_output(ligands, output_csv)

    # Print summary
    print_summary(with_smiles, without_smiles)

    # Provide guidance for missing SMILES
    if without_smiles:
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR MISSING SMILES:")
        print("=" * 70)
        print("1. Search PubChem (https://pubchem.ncbi.nlm.nih.gov/)")
        print("2. Check other databases (ChEMBL, DrugBank, etc.)")
        print("3. Manually add SMILES to the CSV")
        print("4. Re-run this script to calculate properties")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
