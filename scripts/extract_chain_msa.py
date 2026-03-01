#!/usr/bin/env python3
"""
Extract per-chain MSA from Boltz multi-chain .a3m files.

Boltz stores multi-chain MSAs with chains separated by '/' in each alignment row.
For a 3-chain complex (PYR1 + ligand + HAB1), each sequence line looks like:
    PYR1_ALIGNMENT/LIGAND_PLACEHOLDER/HAB1_ALIGNMENT

This script extracts a single chain's alignment into a standalone .a3m file,
suitable for use with prepare_boltz_yamls.py --hab1-msa.

Usage:
    # Extract HAB1 (chain index 2) from WT ternary MSA:
    python scripts/extract_chain_msa.py \
        /scratch/alpine/ryde3462/boltz_lca/wt_ternary/boltz_results_pyr1_wt_lca_hab1/msa/pyr1_wt_lca_hab1_unpaired_tmp_env/uniref.a3m \
        --chain-index 2 \
        --out /scratch/alpine/ryde3462/boltz_lca/hab1_msa/hab1.a3m

    # Can also merge multiple source .a3m files:
    python scripts/extract_chain_msa.py \
        /path/to/uniref.a3m /path/to/bfd.a3m \
        --chain-index 2 \
        --out hab1_merged.a3m
"""

import argparse
import sys
from pathlib import Path


def parse_multi_chain_a3m(a3m_path: str, chain_index: int):
    """Parse a Boltz multi-chain .a3m and extract one chain's alignments.

    Boltz .a3m format:
        - Header lines start with '>'
        - Sequence lines may contain '/' separators between chains
        - A sequence line can span multiple actual lines (until next '>')
        - Gap-only alignments for a chain indicate no homolog for that chain

    Args:
        a3m_path: Path to input .a3m file
        chain_index: 0-based index of the chain to extract

    Yields:
        (header, sequence) tuples for the extracted chain
    """
    with open(a3m_path) as f:
        lines = f.read().splitlines()

    # Parse into (header, sequence) pairs
    entries = []
    current_header = None
    current_seq_parts = []

    for line in lines:
        if line.startswith('>'):
            if current_header is not None:
                entries.append((current_header, ''.join(current_seq_parts)))
            current_header = line
            current_seq_parts = []
        elif line.strip():
            current_seq_parts.append(line.strip())

    if current_header is not None:
        entries.append((current_header, ''.join(current_seq_parts)))

    if not entries:
        return

    # Check if this is a multi-chain .a3m (has '/' separators)
    first_seq = entries[0][1]
    if '/' not in first_seq:
        # Single-chain .a3m — if chain_index == 0, yield as-is
        if chain_index == 0:
            for header, seq in entries:
                yield header, seq
        else:
            print(f"WARNING: {a3m_path} is single-chain but chain_index={chain_index}",
                  file=sys.stderr)
        return

    # Multi-chain: split each sequence by '/' and extract the target chain
    n_chains = first_seq.count('/') + 1
    if chain_index >= n_chains:
        print(f"ERROR: chain_index={chain_index} but only {n_chains} chains in {a3m_path}",
              file=sys.stderr)
        return

    for header, seq in entries:
        parts = seq.split('/')
        if len(parts) != n_chains:
            # Skip malformed entries
            continue

        chain_seq = parts[chain_index]

        # Skip entries where this chain has no alignment (all gaps)
        ungapped = chain_seq.replace('-', '')
        if not ungapped:
            continue

        yield header, chain_seq


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-chain MSA from Boltz multi-chain .a3m files")
    parser.add_argument("input_a3m", nargs='+',
                        help="One or more Boltz multi-chain .a3m files")
    parser.add_argument("--chain-index", type=int, required=True,
                        help="0-based chain index to extract (e.g., 2 for HAB1 in PYR1+lig+HAB1)")
    parser.add_argument("--out", required=True,
                        help="Output .a3m file path")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Maximum number of sequences to keep (default: all)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_entries = []
    seen_seqs = set()

    for a3m_path in args.input_a3m:
        if not Path(a3m_path).exists():
            print(f"WARNING: {a3m_path} not found, skipping", file=sys.stderr)
            continue

        for header, seq in parse_multi_chain_a3m(a3m_path, args.chain_index):
            # Deduplicate by ungapped sequence
            ungapped = seq.replace('-', '').upper()
            if ungapped not in seen_seqs:
                seen_seqs.add(ungapped)
                all_entries.append((header, seq))

    if not all_entries:
        print("ERROR: No sequences extracted", file=sys.stderr)
        sys.exit(1)

    # Apply max_seqs limit (keep query + top homologs)
    if args.max_seqs and len(all_entries) > args.max_seqs:
        all_entries = all_entries[:args.max_seqs]

    # Write output .a3m
    with open(out_path, 'w') as f:
        for header, seq in all_entries:
            f.write(header + '\n')
            f.write(seq + '\n')

    query_len = len(all_entries[0][1].replace('-', ''))
    print(f"Extracted {len(all_entries)} sequences for chain index {args.chain_index}")
    print(f"Query length: {query_len} aa")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
