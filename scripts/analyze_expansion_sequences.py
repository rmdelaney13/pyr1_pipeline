#!/usr/bin/env python3
"""
Analyze expansion sequences split by binding mode (COO vs OH).

For a given round's scores CSV + Boltz output dirs:
  1. Classify each design as COO or OH binding mode
  2. Apply the 6 hard filters
  3. Extract pocket sequences (16 positions)
  4. Generate sequence logos per mode (matplotlib + logomaker)
  5. Print top designs per mode with key metrics
  6. Copy top PDBs into output directory for inspection

Usage:
    python scripts/analyze_expansion_sequences.py \
        --scores /scratch/alpine/ryde3462/CDCA/design/expansion/ligandmpnn/round_0/scores.csv \
        --boltz-dirs /scratch/alpine/ryde3462/CDCA/design/boltz_output \
        --ligand cdca \
        --top-n 10 \
        --out-dir /scratch/alpine/ryde3462/CDCA/design/expansion/sequence_analysis

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# WT PYR1 sequence (181 residues, 1-indexed)
WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

# 16 mutable pocket positions (1-indexed, Boltz numbering)
POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122,
                    141, 159, 160, 163, 164, 167]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# Hard filters (same as run_expansion.sh Phase A)
HARD_FILTERS = [
    ("binary_hbond_distance", "<", 4.0),
    ("binary_hbond_distance", ">", 1.8),
    ("binary_plddt_ligand", ">", 0.65),
    ("binary_coo_to_r116_dist", ">", 4.0),
    ("binary_ligand_distorted", "<", 1),
    ("binary_hab1_clash_dist", ">", 2.0),
    ("binary_latch_rmsd", "<", 1.0),
]


def wt_at_position(pos):
    return WT_PYR1_SEQUENCE[pos - 1]


def classify_binding_mode(row):
    """Classify as COO or OH binding mode based on which O is closer to water."""
    try:
        coo_dist = float(row.get('binary_coo_to_water_dist', 999))
        oh_dist = float(row.get('binary_oh_to_water_dist', 999))
    except (ValueError, TypeError):
        return 'unknown'
    if coo_dist >= 99 and oh_dist >= 99:
        return 'unknown'
    if coo_dist < oh_dist and coo_dist < 4.0:
        return 'COO'
    if oh_dist <= coo_dist and oh_dist < 4.0:
        return 'OH'
    return 'unknown'


def find_pdb_for_name(name, boltz_dirs):
    """Locate the Boltz output PDB for a given prediction name."""
    for d in boltz_dirs:
        d = Path(d)
        pdb = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
        if pdb.exists():
            return pdb
        cif = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        if cif.exists():
            return cif
    return None


def get_pocket_seq_from_pdb(pdb_path):
    """Extract 16-residue pocket sequence from a PDB."""
    THREE_TO_ONE = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    residues = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[21] != 'A':
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resnum in POCKET_POSITIONS:
                    residues[resnum] = THREE_TO_ONE.get(resname, 'X')
    except Exception:
        return None
    if len(residues) < len(POCKET_POSITIONS):
        return None
    return ''.join(residues.get(p, 'X') for p in POCKET_POSITIONS)


def apply_filters(rows):
    """Apply hard filters. Returns passing rows."""
    for col, op, val in HARD_FILTERS:
        before = len(rows)
        filtered = []
        for r in rows:
            v = r.get(col)
            if v is None or v == '':
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                continue
            if op == '<' and fv < val:
                filtered.append(r)
            elif op == '>' and fv > val:
                filtered.append(r)
        rows = filtered
        print("  %-40s %d -> %d" % (
            "%s%s%.1f" % (col, op, val), before, len(rows)))
    return rows


def build_frequency_matrix(sequences):
    """Build position x AA frequency matrix from pocket sequences."""
    n_pos = len(POCKET_POSITIONS)
    freq = np.zeros((n_pos, 20))
    for seq in sequences:
        for i, aa in enumerate(seq):
            if aa in AA_ORDER:
                freq[i, AA_ORDER.index(aa)] += 1
    # Normalize to frequencies
    totals = freq.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    freq /= totals
    return freq


def make_logo(freq_matrix, title, outpath, positions=None):
    """Generate a sequence logo using logomaker."""
    try:
        import logomaker
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: logomaker or matplotlib not available, skipping logo")
        return

    import pandas as pd

    if positions is None:
        positions = POCKET_POSITIONS

    # Build DataFrame with integer index (logomaker requirement)
    # Use 0..N-1 index, then relabel ticks to show actual positions
    df = pd.DataFrame(freq_matrix, index=list(range(len(positions))),
                      columns=AA_ORDER)

    # Convert to information content (bits)
    info = logomaker.transform_matrix(df, from_type='probability',
                                       to_type='information')

    fig, ax = plt.subplots(figsize=(14, 3))
    logo = logomaker.Logo(info, ax=ax, color_scheme='chemistry')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Information (bits)', fontsize=10)
    ax.set_xlabel('Pocket position', fontsize=10)

    # Relabel x-axis with actual position numbers
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], fontsize=8)

    # Add WT reference on top axis
    wt_pocket = [wt_at_position(p) for p in positions]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(positions)))
    ax2.set_xticklabels(wt_pocket, fontsize=8, color='gray')
    ax2.set_xlabel('WT residue', fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Logo saved: %s" % outpath)


def print_top_table(rows, boltz_dirs, n=10, label=""):
    """Print a table of top designs with key metrics."""
    cols = [
        ('name', 40),
        ('binary_total_score', 8),
        ('binary_plddt_ligand', 6),
        ('binary_hbond_distance', 6),
        ('binary_latch_rmsd', 6),
        ('binary_hab1_clash_dist', 6),
    ]
    header = "%-40s %8s %6s %6s %6s %6s  pocket_seq" % (
        'name', 'score', 'pLDDT', 'hbond', 'latch', 'clash')
    print("\n  Top %d %s:" % (n, label))
    print("  " + header)
    print("  " + "-" * len(header) + "--------")

    for r in rows[:n]:
        name = r.get('name', '?')[:40]
        score = r.get('binary_total_score', '')
        plddt = r.get('binary_plddt_ligand', '')
        hbond = r.get('binary_hbond_distance', '')
        latch = r.get('binary_latch_rmsd', '')
        clash = r.get('binary_hab1_clash_dist', '')
        # Get pocket sequence
        pdb = find_pdb_for_name(r.get('name', ''), boltz_dirs)
        pocket = get_pocket_seq_from_pdb(str(pdb)) if pdb else '?'

        def fmt(v, w):
            try:
                return "%*.3f" % (w, float(v))
            except (ValueError, TypeError):
                return "%*s" % (w, '-')

        print("  %-40s %s %s %s %s %s  %s" % (
            name, fmt(score, 8), fmt(plddt, 6), fmt(hbond, 6),
            fmt(latch, 6), fmt(clash, 6), pocket or '?'))


def print_position_summary(sequences, label):
    """Print per-position AA frequency summary."""
    position_freqs = {i: Counter() for i in range(len(POCKET_POSITIONS))}
    for seq in sequences:
        for i, aa in enumerate(seq):
            position_freqs[i][aa] += 1

    n = len(sequences)
    print("\n  Position frequencies (%s, n=%d):" % (label, n))
    pos_header = '  '.join("%3d" % p for p in POCKET_POSITIONS)
    wt_header = '  '.join("  %s" % wt_at_position(p) for p in POCKET_POSITIONS)
    print("  Positions: %s" % pos_header)
    print("  WT:        %s" % wt_header)

    for i, pos in enumerate(POCKET_POSITIONS):
        freq = position_freqs[i]
        wt = wt_at_position(pos)
        if not freq:
            continue
        top_aa, top_count = freq.most_common(1)[0]
        if len(freq) == 1 and top_aa == wt:
            continue
        parts = []
        for aa, count in freq.most_common(5):
            pct = count * 100 // n
            marker = '*' if aa != wt else ''
            parts.append("%s:%d%%%s" % (aa, pct, marker))
        changed = '**' if top_aa != wt else '  '
        print("  %s Pos %3d (WT=%s): %s" % (changed, pos, wt, ' '.join(parts)))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expansion sequences split by binding mode")
    parser.add_argument("--scores", required=True,
                        help="Scores CSV (round 0 scores.csv or cumulative)")
    parser.add_argument("--boltz-dirs", nargs='+', required=True,
                        help="Boltz output directories to search for PDBs")
    parser.add_argument("--ligand", default="cdca",
                        help="Ligand name (default: cdca)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top PDBs to copy per mode (default: 10)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for logos, tables, and PDBs")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip hard filters (analyze all scored designs)")
    parser.add_argument("--score-column", default="binary_total_score",
                        help="Column to rank by (default: binary_total_score)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boltz_dirs = args.boltz_dirs

    # Load scores
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(args.score_column, '')
            try:
                row['_score'] = float(val)
            except (ValueError, TypeError):
                continue
            rows.append(row)

    rows.sort(key=lambda r: r['_score'], reverse=True)
    print("Loaded %d scored designs from %s" % (len(rows), args.scores))

    # Apply filters
    if not args.no_filter:
        print("\nApplying hard filters:")
        rows = apply_filters(rows)
        print("  Passing: %d designs" % len(rows))

    if not rows:
        print("ERROR: No designs pass filters")
        sys.exit(1)

    # Classify binding modes
    mode_rows = {'COO': [], 'OH': [], 'unknown': []}
    for r in rows:
        mode = classify_binding_mode(r)
        r['_mode'] = mode
        mode_rows[mode].append(r)

    print("\nBinding mode breakdown:")
    for mode in ['COO', 'OH', 'unknown']:
        print("  %s: %d designs" % (mode, len(mode_rows[mode])))

    # For each mode: extract sequences, make logos, print tops, copy PDBs
    for mode in ['COO', 'OH']:
        mrows = mode_rows[mode]
        if not mrows:
            print("\n--- %s: no designs, skipping ---" % mode)
            continue

        print("\n" + "=" * 70)
        print("  %s binding mode (%d designs)" % (mode, len(mrows)))
        print("=" * 70)

        # Extract pocket sequences
        sequences = []
        rows_with_seq = []
        for r in mrows:
            pdb = find_pdb_for_name(r.get('name', ''), boltz_dirs)
            if pdb:
                seq = get_pocket_seq_from_pdb(str(pdb))
                if seq:
                    r['_pocket_seq'] = seq
                    sequences.append(seq)
                    rows_with_seq.append(r)

        print("  Extracted %d / %d pocket sequences" % (
            len(sequences), len(mrows)))

        if not sequences:
            continue

        # Print top table
        print_top_table(rows_with_seq, boltz_dirs, n=args.top_n,
                        label="%s mode" % mode)

        # Position frequency summary
        print_position_summary(sequences, mode)

        # Unique sequences
        unique_seqs = set(sequences)
        print("\n  Unique pocket sequences: %d / %d (%.0f%% unique)" % (
            len(unique_seqs), len(sequences),
            100 * len(unique_seqs) / len(sequences)))

        # Sequence logo
        freq_matrix = build_frequency_matrix(sequences)
        logo_path = out_dir / ("logo_%s_%s.png" % (args.ligand, mode.lower()))
        make_logo(freq_matrix,
                  "%s %s-mode pocket (n=%d)" % (args.ligand.upper(), mode,
                                                 len(sequences)),
                  str(logo_path))

        # Copy top PDBs
        pdb_dir = out_dir / ("top_%s_%s" % (args.ligand, mode.lower()))
        pdb_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for r in rows_with_seq[:args.top_n]:
            pdb = find_pdb_for_name(r.get('name', ''), boltz_dirs)
            if pdb and pdb.exists():
                dest = pdb_dir / pdb.name
                shutil.copy2(str(pdb), str(dest))
                copied += 1
        print("  Copied %d PDBs to %s" % (copied, pdb_dir))

    # Also save a combined summary CSV
    summary_path = out_dir / ("sequence_summary_%s.csv" % args.ligand)
    fieldnames = ['name', 'mode', 'pocket_seq', 'binary_total_score',
                  'binary_plddt_ligand', 'binary_hbond_distance',
                  'binary_latch_rmsd', 'binary_hab1_clash_dist',
                  'binary_oh_to_water_dist', 'binary_coo_to_water_dist']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if '_pocket_seq' not in r:
                pdb = find_pdb_for_name(r.get('name', ''), boltz_dirs)
                if pdb:
                    r['_pocket_seq'] = get_pocket_seq_from_pdb(str(pdb)) or ''
            out_row = {}
            for k in fieldnames:
                if k == 'mode':
                    out_row[k] = r.get('_mode', '')
                elif k == 'pocket_seq':
                    out_row[k] = r.get('_pocket_seq', '')
                else:
                    out_row[k] = r.get(k, '')
            writer.writerow(out_row)

    print("\nSummary CSV: %s" % summary_path)
    print("Done.")


if __name__ == "__main__":
    main()
