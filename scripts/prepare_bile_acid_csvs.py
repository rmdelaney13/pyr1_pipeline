#!/usr/bin/env python3
"""
Convert per-ligand bile acid FASTA files to tier CSVs for Boltz YAML generation.

Reads per-ligand FASTA files (e.g. CA_aggregated.fasta) and:
1. Extracts ligand name from filename (e.g. CA_aggregated.fasta -> CA)
2. Diffs each variant sequence against WT PYR1 to extract variant signatures
3. Maps ligand names to canonical SMILES
4. Outputs per-ligand CSVs compatible with prepare_boltz_yamls.py

Input: Per-ligand FASTA files with PYR1 variant sequences
    >design_name
    MASELTPEERSELKNSIAEFHTYQLDPGQCSSLHAQRIHAPPDLVW...

Output: Per-ligand CSVs in tier format:
    pair_id,ligand_name,ligand_smiles,variant_name,variant_signature

Usage:
    python prepare_bile_acid_csvs.py \
        CA_aggregated.fasta CDCA_aggregated.fasta UDCA_aggregated.fasta DCA_aggregated.fasta \
        --out-dir ./bile_acid_csvs

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import sys
from pathlib import Path

# WT PYR1 sequence (181 residues) — must match prepare_boltz_yamls.py
WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

# Canonical SMILES for each bile acid
LIGAND_SMILES = {
    "CA":   "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1([C@H](C[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)O)C",
    "CDCA": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C",
    "UDCA": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C",
    "DCA":  "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1([C@H](C[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@H](C4)O)C)O)C",
}


def sequence_to_signature(variant_seq: str, wt_seq: str = WT_PYR1_SEQUENCE) -> str:
    """Diff variant sequence against WT to produce variant signature string.

    Returns e.g. '59V;81D;83L;92M' for positions that differ from WT.
    Positions are 1-indexed, matching the convention in prepare_boltz_yamls.py.
    """
    if len(variant_seq) != len(wt_seq):
        print(f"WARNING: Length mismatch: variant={len(variant_seq)}, WT={len(wt_seq)}",
              file=sys.stderr)
        return ""

    mutations = []
    for i, (wt_aa, var_aa) in enumerate(zip(wt_seq, variant_seq)):
        if wt_aa != var_aa:
            mutations.append(f"{i + 1}{var_aa}")

    return ";".join(mutations)


def read_fasta(fasta_path: str):
    """Read a FASTA file and yield (header, sequence) tuples."""
    header = None
    seq_parts = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        yield header, "".join(seq_parts)


def ligand_from_filename(fasta_path: str) -> str:
    """Extract ligand name from filename (e.g. CA_aggregated.fasta -> CA)."""
    stem = Path(fasta_path).stem  # e.g. CA_aggregated
    ligand = stem.split("_")[0].upper()
    return ligand


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-ligand FASTA files to Boltz tier CSVs")
    parser.add_argument("fasta_files", nargs="+",
                        help="Per-ligand FASTA files (ligand name extracted from filename, "
                             "e.g. CA_aggregated.fasta -> CA)")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-ligand CSVs")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for fasta_path in args.fasta_files:
        ligand = ligand_from_filename(fasta_path)

        if ligand not in LIGAND_SMILES:
            print(f"ERROR: Unknown ligand '{ligand}' from {fasta_path}. "
                  f"Expected one of: {', '.join(LIGAND_SMILES.keys())}", file=sys.stderr)
            sys.exit(1)

        rows = []
        seen_seqs = set()
        skipped = 0
        duplicates = 0
        for header, seq in read_fasta(fasta_path):
            if len(seq) != len(WT_PYR1_SEQUENCE):
                print(f"WARNING: {header} has length {len(seq)} "
                      f"(expected {len(WT_PYR1_SEQUENCE)}), skipping", file=sys.stderr)
                skipped += 1
                continue

            if seq in seen_seqs:
                duplicates += 1
                continue
            seen_seqs.add(seq)

            signature = sequence_to_signature(seq)
            rows.append({
                "header": header,
                "variant_signature": signature,
            })

        if skipped:
            print(f"{ligand}: skipped {skipped} sequences with wrong length")
        if duplicates:
            print(f"{ligand}: removed {duplicates} duplicate sequences")

        if not rows:
            print(f"{ligand}: 0 valid sequences in {fasta_path}, skipping")
            continue

        out_path = out_dir / f"boltz_{ligand.lower()}_binary.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pair_id", "ligand_name", "ligand_smiles",
                "variant_name", "variant_signature",
            ])

            for i, row in enumerate(rows, 1):
                pair_id = f"pair_{i:04d}"
                writer.writerow([
                    pair_id,
                    ligand,
                    LIGAND_SMILES[ligand],
                    f"{ligand.lower()}_{i:04d}",
                    row["variant_signature"],
                ])

        print(f"{ligand}: {len(rows)} sequences -> {out_path}")
        total += len(rows)

    print(f"\nTotal: {total} predictions across {len(args.fasta_files)} ligands")


if __name__ == "__main__":
    main()
