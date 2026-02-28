#!/usr/bin/env python3
"""
Analyze pocket mutation frequencies across expansion rounds.

For each ligand and round, loads the top N designs from cumulative scores,
maps them back to variant signatures, and shows amino acid frequencies at
each of the 16 pocket positions. Tracks how the pocket consensus evolves
across rounds of LigandMPNN expansion.

Usage:
    python expansion_pocket_analysis.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --initial-csv-dir /scratch/alpine/ryde3462/boltz_bile_acids/csvs \
        --top-n 100 \
        --out pocket_analysis.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# WT PYR1 sequence (181 residues)
WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

# 16 mutable pocket positions (1-indexed, Boltz/AF3 numbering)
POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]

RANK_COLUMN = 'binary_total_score'


def wt_at_position(pos):
    """Get WT amino acid at 1-indexed position."""
    return WT_PYR1_SEQUENCE[pos - 1]


def parse_signature(signature):
    """Parse variant signature into {position: AA} dict."""
    mutations = {}
    if not signature:
        return mutations
    for mut in signature.split(';'):
        mut = mut.strip()
        if not mut:
            continue
        pos = int(mut[:-1])
        aa = mut[-1]
        mutations[pos] = aa
    return mutations


def pocket_from_signature(signature):
    """Extract pocket amino acids from variant signature.

    Returns dict {pos: AA} for all 16 pocket positions.
    Positions not in signature get WT amino acid.
    """
    mutations = parse_signature(signature)
    pocket = {}
    for pos in POCKET_POSITIONS:
        pocket[pos] = mutations.get(pos, wt_at_position(pos))
    return pocket


def build_signature_lookup(initial_csv_dir, expansion_root, ligand):
    """Build name→variant_signature lookup from all source CSVs."""
    lookup = {}

    # Initial bile acid CSV
    initial_csv = Path(initial_csv_dir) / f"boltz_{ligand}_binary.csv"
    if initial_csv.exists():
        with open(initial_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('pair_id', '').strip()
                sig = row.get('variant_signature', '').strip()
                if name:
                    lookup[name] = sig

    # Expansion CSVs from each round
    exp_root = Path(expansion_root) / ligand
    for rnd in range(1, 20):
        exp_csv = exp_root / f"round_{rnd}" / "expansion.csv"
        if not exp_csv.exists():
            break
        with open(exp_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('pair_id', '').strip()
                sig = row.get('variant_signature', '').strip()
                if name:
                    lookup[name] = sig

    return lookup


def load_top_n(scores_path, n, rank_col=RANK_COLUMN):
    """Load scores CSV and return top N rows by rank column."""
    rows = []
    with open(scores_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(rank_col, '')
            try:
                row['_rank_val'] = float(val)
                rows.append(row)
            except (ValueError, TypeError):
                continue
    rows.sort(key=lambda r: r['_rank_val'], reverse=True)
    return rows[:n]


def format_freq_table(position_freqs, top_n_aas=3):
    """Format frequency table for a single position.

    Returns string like 'V:45 L:30 I:20 (5 other)'
    """
    total = sum(position_freqs.values())
    if total == 0:
        return '-'
    sorted_aas = position_freqs.most_common()
    parts = []
    shown = 0
    for aa, count in sorted_aas[:top_n_aas]:
        pct = count * 100 // total
        parts.append(f"{aa}:{pct}%")
        shown += count
    remaining = total - shown
    if remaining > 0:
        parts.append(f"+{len(sorted_aas) - top_n_aas}")
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pocket mutation frequencies across expansion rounds")
    parser.add_argument("--expansion-root", required=True,
                        help="Root directory (e.g. /scratch/.../expansion/ligandmpnn)")
    parser.add_argument("--initial-csv-dir", required=True,
                        help="Directory with initial bile acid CSVs "
                             "(e.g. /scratch/.../boltz_bile_acids/csvs)")
    parser.add_argument("--ligands", nargs='+', default=['ca', 'cdca', 'udca', 'dca'],
                        help="Ligands to analyze (default: ca cdca udca dca)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top designs to analyze (default: 100)")
    parser.add_argument("--max-round", type=int, default=10,
                        help="Max round number to check (default: 10)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path for per-position frequencies")

    args = parser.parse_args()

    all_csv_rows = []

    for lig in args.ligands:
        lig_dir = Path(args.expansion_root) / lig
        if not lig_dir.exists():
            print(f"WARNING: {lig_dir} not found, skipping", file=sys.stderr)
            continue

        # Build signature lookup
        lookup = build_signature_lookup(args.initial_csv_dir, args.expansion_root, lig)
        print(f"\n{'='*70}")
        print(f"  {lig.upper()} — Pocket mutations in top {args.top_n}")
        print(f"{'='*70}")
        print(f"  Signature lookup: {len(lookup)} designs mapped")

        # Print WT reference
        wt_pocket = ''.join(wt_at_position(p) for p in POCKET_POSITIONS)
        pos_header = '  '.join(f"{p:>3d}" for p in POCKET_POSITIONS)
        print(f"\n  Positions: {pos_header}")
        print(f"  WT:        {'  '.join(f'  {wt_at_position(p)}' for p in POCKET_POSITIONS)}")

        for rnd in range(0, args.max_round + 1):
            round_dir = lig_dir / f"round_{rnd}"
            if not round_dir.exists():
                break

            if rnd == 0:
                scores_path = round_dir / "scores.csv"
            else:
                scores_path = round_dir / "cumulative_scores.csv"
            if not scores_path.exists():
                break

            top = load_top_n(scores_path, args.top_n)

            # Get pocket mutations for each top design
            position_freqs = {pos: Counter() for pos in POCKET_POSITIONS}
            unmapped = 0

            for row in top:
                name = row.get('name', '')
                sig = lookup.get(name)
                if sig is None:
                    unmapped += 1
                    continue
                pocket = pocket_from_signature(sig)
                for pos in POCKET_POSITIONS:
                    position_freqs[pos][pocket[pos]] += 1

            mapped = len(top) - unmapped

            # Print round header
            print(f"\n  ── Round {rnd} (top {mapped} mapped" +
                  (f", {unmapped} unmapped" if unmapped else "") + ") ──")

            # Print consensus (most common AA at each position)
            consensus = []
            for pos in POCKET_POSITIONS:
                if position_freqs[pos]:
                    top_aa, top_count = position_freqs[pos].most_common(1)[0]
                    pct = top_count * 100 // mapped if mapped else 0
                    wt = wt_at_position(pos)
                    marker = '*' if top_aa != wt else ' '
                    consensus.append(f"{top_aa}{marker}")
                else:
                    consensus.append('  ')
            print(f"  Consensus: {'  '.join(f'{c:>3s}' for c in consensus)}")

            # Print detailed frequency for each position
            for pos in POCKET_POSITIONS:
                freq = position_freqs[pos]
                wt = wt_at_position(pos)
                if not freq:
                    continue

                # Only show positions with interesting variation
                top_aa, top_count = freq.most_common(1)[0]
                if len(freq) == 1 and top_aa == wt:
                    continue  # All WT, skip

                freq_str = format_freq_table(freq)
                changed = '**' if top_aa != wt else '  '
                print(f"  {changed} Pos {pos:3d} (WT={wt}): {freq_str}")

            # CSV output rows
            for pos in POCKET_POSITIONS:
                freq = position_freqs[pos]
                wt = wt_at_position(pos)
                for aa, count in freq.most_common():
                    all_csv_rows.append({
                        'ligand': lig.upper(),
                        'round': rnd,
                        'position': pos,
                        'wt_aa': wt,
                        'aa': aa,
                        'count': count,
                        'fraction': round(count / mapped, 4) if mapped else 0,
                        'is_wt': aa == wt,
                    })

    # Write CSV
    if args.out and all_csv_rows:
        fieldnames = ['ligand', 'round', 'position', 'wt_aa', 'aa',
                      'count', 'fraction', 'is_wt']
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nWrote {len(all_csv_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
