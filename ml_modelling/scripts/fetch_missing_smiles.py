#!/usr/bin/env python3
"""
Automatically fetch missing SMILES from PubChem.

Usage:
    python fetch_missing_smiles.py
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import pubchempy as pcp
except ImportError:
    print("ERROR: pubchempy is required. Install with:")
    print("  pip install pubchempy")
    exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("WARNING: RDKit not available - will skip validation")
    RDKIT_AVAILABLE = False


def fetch_smiles_from_pubchem(ligand_name: str, retry: int = 2) -> Optional[Dict]:
    """
    Fetch SMILES and basic info from PubChem.

    Parameters
    ----------
    ligand_name : str
        Name of the compound
    retry : int
        Number of retry attempts

    Returns
    -------
    dict or None
        Dictionary with CID, SMILES, and IUPAC name if found
    """
    search_terms = [
        ligand_name,  # Original name
        ligand_name.replace('-', ' '),  # Replace hyphens with spaces
        ligand_name.replace(' ', '-'),  # Replace spaces with hyphens
    ]

    for attempt in range(retry):
        for search_term in search_terms:
            try:
                # Try by name first
                compounds = pcp.get_compounds(search_term, 'name')

                if not compounds:
                    # Try as synonym
                    continue

                if compounds:
                    compound = compounds[0]  # Take first match

                    # Validate SMILES with RDKit if available
                    smiles = compound.canonical_smiles
                    if not smiles:
                        print(f"    ⚠ No SMILES for {ligand_name}, skipping")
                        continue

                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            print(f"    ⚠ Invalid SMILES for {ligand_name}, skipping")
                            continue

                    return {
                        'cid': compound.cid,
                        'smiles': smiles,
                        'iupac_name': str(compound.iupac_name) if compound.iupac_name else ligand_name,
                        'molecular_formula': str(compound.molecular_formula) if compound.molecular_formula else '',
                        'molecular_weight': float(compound.molecular_weight) if compound.molecular_weight else 0.0,
                    }

            except Exception as e:
                if attempt < retry - 1:
                    print(f"    Retry {attempt + 1}/{retry} for {search_term}")
                    time.sleep(1)
                else:
                    print(f"    ✗ Failed to fetch {ligand_name} (tried: {search_term}): {e}")

    return None


def get_mol_properties(smiles: str) -> Dict:
    """Calculate RDKit properties if available."""
    if not RDKIT_AVAILABLE:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    n_rot_bonds = Descriptors.NumRotatableBonds(mol)

    # 3^N heuristic for conformer count
    if n_rot_bonds == 0:
        num_conformers = 50
    else:
        num_conformers = min(max(3 ** n_rot_bonds, 50), 2000)

    return {
        'n_rotatable_bonds': n_rot_bonds,
        'n_heavy_atoms': Descriptors.HeavyAtomCount(mol),
        'recommended_conformers': num_conformers,
        'logp': round(Descriptors.MolLogP(mol), 2),
        'n_hbd': Descriptors.NumHDonors(mol),
        'n_hba': Descriptors.NumHAcceptors(mol),
    }


def load_manual_lookup() -> Dict[str, str]:
    """Load manual SMILES lookup if available."""
    manual_file = Path("manual_smiles_lookup.csv")
    lookup = {}

    if manual_file.exists():
        with open(manual_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lookup[row['ligand_name']] = row['smiles']
        print(f"✓ Loaded {len(lookup)} manual SMILES entries\n")

    return lookup


def main():
    # Paths
    input_csv = Path("ligand_properties_with_conformer_counts_missing_smiles.csv")
    output_csv = Path("ligand_smiles_fetched_from_pubchem.csv")
    failed_csv = Path("ligand_smiles_fetch_failed.csv")

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found")
        print("Run process_ligand_smiles.py first")
        return

    print("=" * 70)
    print("FETCHING MISSING SMILES")
    print("=" * 70)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print()

    # Load manual lookup
    manual_lookup = load_manual_lookup()

    # Read missing ligands
    missing_ligands = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        missing_ligands = list(reader)

    print(f"Found {len(missing_ligands)} ligands to fetch\n")

    # Fetch SMILES
    results = []
    failed = []

    for i, lig in enumerate(missing_ligands, 1):
        ligand_name = lig['ligand_name']
        print(f"[{i}/{len(missing_ligands)}] Processing: {ligand_name}")

        data = None
        smiles_source = None

        # Check manual lookup first
        if ligand_name in manual_lookup:
            smiles = manual_lookup[ligand_name]
            print(f"    ✓ Found in manual lookup")

            # Validate with RDKit
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"    ✗ Invalid SMILES in manual lookup")
                    failed.append({
                        'ligand_name': ligand_name,
                        'n_variants': lig['n_variants'],
                        'reason': 'Invalid SMILES in manual lookup'
                    })
                    print()
                    continue

            data = {'smiles': smiles, 'cid': '', 'iupac_name': ligand_name,
                    'molecular_formula': '', 'molecular_weight': ''}
            smiles_source = 'manual'
        else:
            # Try PubChem
            data = fetch_smiles_from_pubchem(ligand_name)
            smiles_source = 'pubchem'

        if data:
            # Add RDKit properties
            rdkit_props = get_mol_properties(data['smiles'])

            result = {
                'ligand_name': ligand_name,
                'smiles': data['smiles'],
                'smiles_source': smiles_source,
                'pubchem_cid': data.get('cid', ''),
                'iupac_name': data.get('iupac_name', ligand_name),
                'n_variants': lig['n_variants'],
                'molecular_formula': data.get('molecular_formula', ''),
                'mol_weight': data.get('molecular_weight', ''),
                **rdkit_props
            }
            results.append(result)

            if smiles_source == 'pubchem':
                print(f"    ✓ Found CID {data['cid']}: {data['smiles'][:50]}...")
            else:
                print(f"    SMILES: {data['smiles'][:50]}...")

            if rdkit_props:
                print(f"      Rotatable bonds: {rdkit_props['n_rotatable_bonds']}, "
                      f"Recommended conformers: {rdkit_props['recommended_conformers']}")
        else:
            failed.append({
                'ligand_name': ligand_name,
                'n_variants': lig['n_variants'],
                'reason': 'Not found in PubChem or manual lookup'
            })
            print(f"    ✗ Not found")

        # Be nice to PubChem servers (only if we queried)
        if smiles_source == 'pubchem':
            time.sleep(0.5)
        print()

    # Write results
    if results:
        fieldnames = [
            'ligand_name', 'smiles', 'smiles_source', 'pubchem_cid', 'iupac_name',
            'n_variants', 'molecular_formula', 'mol_weight', 'n_rotatable_bonds',
            'n_heavy_atoms', 'recommended_conformers', 'logp', 'n_hbd', 'n_hba'
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        print("=" * 70)
        print(f"✓ Successfully fetched {len(results)} ligands")
        print(f"  Output: {output_csv}")

    # Write failed
    if failed:
        with open(failed_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ligand_name', 'n_variants', 'reason'])
            writer.writeheader()
            writer.writerows(failed)

        print(f"✗ Failed to fetch {len(failed)} ligands")
        print(f"  Output: {failed_csv}")
        print("\nFailed ligands:")
        for lig in failed:
            print(f"  - {lig['ligand_name']} ({lig['n_variants']} variants)")

    print("=" * 70)
    print("\nNEXT STEPS:")
    print("1. Review fetched SMILES in:", output_csv)
    print("2. For failed ligands, manually search:")
    print("   - PubChem with alternative names")
    print("   - ChEMBL (https://www.ebi.ac.uk/chembl/)")
    print("   - DrugBank (https://go.drugbank.com/)")
    print("3. Run merge_ligand_data.py to combine all SMILES")
    print("=" * 70)


if __name__ == "__main__":
    main()
