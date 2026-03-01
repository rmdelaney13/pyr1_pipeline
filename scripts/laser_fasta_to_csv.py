#!/usr/bin/env python3
"""
Convert LASErMPNN FASTA outputs to a Boltz-compatible CSV for prediction.

Reads designs.fasta files from LASErMPNN batch output directories, diffs each
designed sequence against WT PYR1 to get variant signatures, deduplicates, and
outputs a CSV compatible with prepare_boltz_yamls.py.

Key differences from expansion_mpnn_to_csv.py (LigandMPNN version):
  - LASErMPNN does NOT output native sequence as first entry — all are designs
  - FASTA header format: >{stem}_design_{idx}_{offset}_segment_... score=...
  - Files at: laser_output/{input_stem}/designs.fasta

Usage:
    python laser_fasta_to_csv.py \
        --laser-dir /scratch/.../round_1/laser_output \
        --ligand-name CA \
        --ligand-smiles "C[C@H](CCC(=O)O)..." \
        --round 1 \
        --out /scratch/.../round_1/expansion.csv \
        --existing-csv /scratch/.../prev_round/expansion.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import re
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


def parse_laser_header(header: str) -> dict:
    """Parse LASErMPNN FASTA header.

    Expected format:
      {pdb_stem}_design_{idx}_{offset}_segment_{seg}_chain_{chain} score={val}

    Returns dict with 'parent', 'design_idx', 'score'.
    """
    result = {'parent': '', 'design_idx': 0, 'score': None}

    # Split score from the rest
    parts = header.split(' score=')
    name_part = parts[0].strip()
    if len(parts) > 1:
        try:
            result['score'] = float(parts[1].strip())
        except ValueError:
            pass

    # Extract parent PDB name: everything before _design_
    design_match = re.match(r'^(.+?)_design_(\d+)', name_part)
    if design_match:
        result['parent'] = design_match.group(1)
        result['design_idx'] = int(design_match.group(2))
    else:
        # Fallback: use the whole name part
        result['parent'] = name_part

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert LASErMPNN FASTA outputs to Boltz CSV")
    parser.add_argument("--laser-dir", required=True,
                        help="LASErMPNN output directory (contains */designs.fasta)")
    parser.add_argument("--ligand-name", required=True,
                        help="Ligand name (e.g. CA, CDCA)")
    parser.add_argument("--ligand-smiles", required=True,
                        help="Ligand SMILES string")
    parser.add_argument("--round", type=int, required=True,
                        help="Expansion round number (for naming)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    parser.add_argument("--existing-csv", nargs='*', default=[],
                        help="Existing expansion CSVs for deduplication")

    args = parser.parse_args()
    laser_dir = Path(args.laser_dir)
    lig = args.ligand_name.upper()
    rnd = args.round

    # Collect existing sequences for cross-round deduplication
    existing_seqs = collect_existing_sequences(args.existing_csv)
    if existing_seqs:
        print(f"Loaded {len(existing_seqs)} existing sequences for cross-round dedup")

    # Find all LASErMPNN FASTA outputs
    # Batch inference creates: laser_output/{input_stem}/designs.fasta
    fasta_files = sorted(laser_dir.glob("*/designs.fasta"))
    if not fasta_files:
        # Try direct .fasta files
        fasta_files = sorted(laser_dir.glob("*.fasta"))
    if not fasta_files:
        print(f"ERROR: No FASTA files found in {laser_dir}", file=sys.stderr)
        print(f"  Searched: {laser_dir}/*/designs.fasta", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} LASErMPNN FASTA files")

    rows = []
    seen_seqs = set(existing_seqs)  # seed with all previously-predicted sequences
    skipped_length = 0
    skipped_dup_intra = 0
    skipped_dup_cross = 0
    skipped_wt = 0
    total_designs = 0

    for fa_path in fasta_files:
        # Parent PDB name from directory name
        parent_dir_name = fa_path.parent.name

        for header, seq in read_fasta(str(fa_path)):
            total_designs += 1

            # Extract sequence for chain A only (LASErMPNN may output multi-chain)
            # The FASTA header contains segment/chain info
            # For PYR1 we want chain A (protein), skip chain B (ligand)
            if '_chain_B' in header or '_chain_W' in header:
                continue

            if len(seq) != len(WT_PYR1_SEQUENCE):
                skipped_length += 1
                continue

            # Skip if identical to WT
            if seq == WT_PYR1_SEQUENCE:
                skipped_wt += 1
                continue

            if seq in seen_seqs:
                if seq in existing_seqs:
                    skipped_dup_cross += 1
                else:
                    skipped_dup_intra += 1
                continue

            # Parse header for design index
            parsed = parse_laser_header(header)
            design_idx = parsed['design_idx']

            # Use directory name as parent (strip _model_0 suffix if present for cleaner names)
            parent_name = parent_dir_name

            # Generate variant name
            variant_name = f"{lig.lower()}_laser_r{rnd}_{parent_name}_d{design_idx}"

            seen_seqs.add(seq)
            signature = sequence_to_signature(seq)

            rows.append({
                "variant_name": variant_name,
                "variant_signature": signature,
                "parent": parent_name,
                "laser_nll": parsed['score'],
            })

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "ligand_name", "ligand_smiles",
            "variant_name", "variant_signature", "laser_nll",
        ])

        for i, row in enumerate(rows, 1):
            pair_id = f"laser_r{rnd}_{i:04d}"
            nll = row["laser_nll"]
            nll_str = f"{nll:.4f}" if nll is not None and nll != float('-inf') and nll != float('inf') else str(nll) if nll is not None else ""
            writer.writerow([
                pair_id,
                lig,
                args.ligand_smiles,
                row["variant_name"],
                row["variant_signature"],
                nll_str,
            ])

    print(f"\nResults:")
    print(f"  FASTA files processed: {len(fasta_files)}")
    print(f"  Total design entries: {total_designs}")
    print(f"  WT sequences skipped: {skipped_wt}")
    print(f"  Wrong-length sequences: {skipped_length}")
    print(f"  Intra-round duplicates: {skipped_dup_intra}")
    print(f"  Cross-round duplicates: {skipped_dup_cross}")
    print(f"  New unique sequences: {len(rows)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
