#!/usr/bin/env python3
"""
Generate CDCA (chenodeoxycholic acid) conformers for docking.

Uses the ligand_conformers package (RDKit ETKDGv3 + MMFF94s).
Outputs 10 diverse conformers as SDF + PDB files.

Usage:
    python scripts/generate_cdca_conformers.py [output_dir]

Default output: /projects/ryde3462/pyr1_pipeline/conformers/CDCA/
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligand_conformers.config import ConformerConfig
from ligand_conformers.core import generate_conformer_set

CDCA_SMILES = (
    "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2"
    "[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C"
)


def main():
    if len(sys.argv) > 1:
        outdir = Path(sys.argv[1])
    else:
        outdir = PROJECT_ROOT / "conformers" / "CDCA"

    cfg = ConformerConfig(
        num_confs=500,
        seed=42,
        k_final=10,
        selection_policy="diverse",
        cluster_rmsd_cutoff=1.25,
        ligand_id="CDCA",
    )

    print(f"Generating CDCA conformers → {outdir}")
    print(f"SMILES: {CDCA_SMILES}")

    result = generate_conformer_set(
        input_spec={"type": "smiles", "value": CDCA_SMILES},
        outdir=outdir,
        cfg=cfg,
    )

    if result.success:
        print(f"\nSuccess: {result.num_clusters} clusters, "
              f"{len(result.selected_ids)} conformers selected")
        print(f"Output: {result.outdir}")
    else:
        print(f"\nFailed: {result.errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
