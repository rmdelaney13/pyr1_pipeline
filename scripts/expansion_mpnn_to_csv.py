#!/usr/bin/env python3
"""
Convert LigandMPNN FASTA outputs to a Boltz-compatible CSV for prediction.

Reads FASTA files from LigandMPNN output directories, skips the native (first)
sequence, diffs each designed sequence against WT PYR1 to get variant signatures,
deduplicates, and outputs a CSV compatible with prepare_boltz_yamls.py.

Usage:
    python expansion_mpnn_to_csv.py \
        --mpnn-dir /scratch/.../round_1/mpnn_output \
        --ligand-name CA \
        --ligand-smiles "C[C@H](CCC(=O)O)..." \
        --round 1 \
        --out /scratch/.../round_1/expansion.csv \
        --existing-csv /scratch/.../round_0/scores.csv

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


def sequence_to_signature(variant_seq: str, wt_seq: str = WT_PYR1_SEQUENCE) -> str:
    """Diff variant sequence against WT to produce variant signature.

    Returns e.g. '59V;81D;83L;92M' for positions that differ.
    Positions are 1-indexed, matching prepare_boltz_yamls.py convention.
    """
    if len(variant_seq) != len(wt_seq):
        return ""

    mutations = []
    for i, (wt_aa, var_aa) in enumerate(zip(wt_seq, variant_seq)):
        if wt_aa != var_aa:
            mutations.append(f"{i + 1}{var_aa}")

    return ";".join(mutations)


def signature_to_sequence(signature: str, wt_seq: str = WT_PYR1_SEQUENCE) -> str:
    """Reconstruct full sequence from variant signature like '59V;81D;83L'.

    Inverse of sequence_to_signature(). Applies mutations to WT sequence.
    """
    seq = list(wt_seq)
    if not signature:
        return wt_seq
    for mut in signature.split(';'):
        mut = mut.strip()
        if not mut:
            continue
        pos = int(mut[:-1]) - 1  # 1-indexed to 0-indexed
        aa = mut[-1]
        if 0 <= pos < len(seq):
            seq[pos] = aa
    return ''.join(seq)


def collect_existing_sequences(csv_paths: list) -> set:
    """Collect full sequences from existing CSVs for deduplication.

    Reconstructs full sequences from variant_signature column when available.
    This enables cross-round dedup: MPNN often produces the same pocket
    sequence from different parents across rounds.
    """
    sequences = set()
    for csv_path in csv_paths:
        p = Path(csv_path)
        if not p.exists():
            continue
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig = row.get('variant_signature', '')
                if sig:
                    sequences.add(signature_to_sequence(sig))
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Convert LigandMPNN FASTA outputs to Boltz CSV")
    parser.add_argument("--mpnn-dir", required=True,
                        help="LigandMPNN output directory (contains */seqs/*.fa)")
    parser.add_argument("--ligand-name", required=True,
                        help="Ligand name (e.g. CA, CDCA)")
    parser.add_argument("--ligand-smiles", required=True,
                        help="Ligand SMILES string")
    parser.add_argument("--round", type=int, required=True,
                        help="Expansion round number (for naming)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    parser.add_argument("--existing-csv", nargs='*', default=[],
                        help="Existing scores CSVs for deduplication")

    args = parser.parse_args()
    mpnn_dir = Path(args.mpnn_dir)
    lig = args.ligand_name.upper()
    rnd = args.round

    # Collect existing sequences for cross-round deduplication
    existing_seqs = collect_existing_sequences(args.existing_csv)
    if existing_seqs:
        print(f"Loaded {len(existing_seqs)} existing sequences for cross-round dedup")

    # Find all MPNN FASTA outputs
    fasta_files = sorted(mpnn_dir.glob("*_mpnn/seqs/*.fa"))
    if not fasta_files:
        # Try alternate patterns
        fasta_files = sorted(mpnn_dir.glob("*/seqs/*.fa"))
    if not fasta_files:
        fasta_files = sorted(mpnn_dir.glob("*_model_0_mpnn/seqs/*.fa"))
    if not fasta_files:
        print(f"ERROR: No FASTA files found in {mpnn_dir}", file=sys.stderr)
        print(f"  Searched: {mpnn_dir}/*_mpnn/seqs/*.fa", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} MPNN FASTA files")

    rows = []
    seen_seqs = set(existing_seqs)  # seed with all previously-predicted sequences
    skipped_native = 0
    skipped_length = 0
    skipped_dup_intra = 0
    skipped_dup_cross = 0

    for fa_path in fasta_files:
        # Parent PDB name: extract from directory structure
        # e.g., mpnn_output/{name}_model_0_mpnn/seqs/{name}_model_0.fa
        parent_dir_name = fa_path.parent.parent.name  # e.g., ca_0001_model_0_mpnn
        # Strip _mpnn suffix to get parent PDB name
        parent_name = parent_dir_name
        if parent_name.endswith("_mpnn"):
            parent_name = parent_name[:-5]

        is_first = True
        sample_idx = 0
        for header, seq in read_fasta(str(fa_path)):
            if is_first:
                # Skip native/input sequence (first entry)
                is_first = False
                skipped_native += 1
                continue

            sample_idx += 1

            if len(seq) != len(WT_PYR1_SEQUENCE):
                skipped_length += 1
                continue

            if seq in seen_seqs:
                # Distinguish intra-round vs cross-round duplicates
                if seq in existing_seqs:
                    skipped_dup_cross += 1
                else:
                    skipped_dup_intra += 1
                continue

            # Generate the variant name for Boltz prediction
            variant_name = f"{lig.lower()}_exp_r{rnd}_{parent_name}_s{sample_idx}"

            seen_seqs.add(seq)
            signature = sequence_to_signature(seq)

            rows.append({
                "variant_name": variant_name,
                "variant_signature": signature,
                "parent": parent_name,
            })

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "ligand_name", "ligand_smiles",
            "variant_name", "variant_signature",
        ])

        for i, row in enumerate(rows, 1):
            pair_id = f"exp_r{rnd}_{i:04d}"
            writer.writerow([
                pair_id,
                lig,
                args.ligand_smiles,
                row["variant_name"],
                row["variant_signature"],
            ])

    print(f"\nResults:")
    print(f"  FASTA files processed: {len(fasta_files)}")
    print(f"  Native sequences skipped: {skipped_native}")
    print(f"  Wrong-length sequences: {skipped_length}")
    print(f"  Intra-round duplicates: {skipped_dup_intra}")
    print(f"  Cross-round duplicates: {skipped_dup_cross}")
    print(f"  New unique sequences: {len(rows)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
