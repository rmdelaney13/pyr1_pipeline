#!/usr/bin/env python3
"""
Convert tested_designs_nonbinders_CDCA.csv (residue-column format) into a
Boltz-compatible CSV for binary prediction with prepare_boltz_yamls.py.

Reads columns like res59, res81, ... and diffs against WT PYR1 to produce
variant signatures. Outputs a simple-format CSV:
    name, variant_signature, ligand_smiles, ligand_name

Usage:
    python scripts/prepare_cdca_nonbinder_boltz_csv.py
"""

import csv
import sys
from pathlib import Path

WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

CDCA_SMILES = (
    "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2"
    "[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_CSV = PROJECT_ROOT / "ml_modelling" / "data" / "tested_designs_nonbinders_CDCA.csv"
OUTPUT_CSV = PROJECT_ROOT / "ml_modelling" / "data" / "boltz_cdca_nonbinders_binary.csv"


def main():
    # Read input CSV
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Extract position columns (res59, res81, ...)
        pos_cols = [c for c in fieldnames if c.startswith("res")]
        positions = {c: int(c.replace("res", "")) for c in pos_cols}
        print(f"Design positions ({len(positions)}): {sorted(positions.values())}")

        # WT residues at these positions
        wt_at_pos = {pos: WT_PYR1_SEQUENCE[pos - 1] for pos in positions.values()}
        print(f"WT residues: { {p: wt_at_pos[p] for p in sorted(wt_at_pos)} }")

        rows = list(reader)

    # Build output rows
    out_rows = []
    for row in rows:
        name = row["target"]
        mutations = []
        for col, pos in sorted(positions.items(), key=lambda x: x[1]):
            aa = row[col].strip().upper()
            if aa != wt_at_pos[pos]:
                mutations.append(f"{pos}{aa}")

        signature = ";".join(mutations)
        out_rows.append({
            "name": name,
            "variant_signature": signature,
            "ligand_smiles": CDCA_SMILES,
            "ligand_name": "CDCA",
        })

    # Write output CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "variant_signature", "ligand_smiles", "ligand_name"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {OUTPUT_CSV}")

    # Print a few examples
    print("\nFirst 3 variant signatures:")
    for r in out_rows[:3]:
        n_muts = len(r["variant_signature"].split(";")) if r["variant_signature"] else 0
        print(f"  {r['name']}: {r['variant_signature']}  ({n_muts} mutations)")


if __name__ == "__main__":
    main()
