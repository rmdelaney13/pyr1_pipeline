#!/usr/bin/env python3
"""
Track metric improvement across expansion rounds for top N designs.

For each ligand and round, loads the cumulative scores CSV, selects the
top N designs by binary_total_score, and computes:
  - Summary statistics (mean, median, max) for key metrics
  - Round attribution: how many of the top N originated from each round
  - Score at fixed ranks (1st, 10th, 50th, 100th)

Usage:
    python expansion_round_analysis.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --ligands ca cdca udca dca \
        --top-n 100 \
        --out round_analysis.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median


METRICS = [
    'binary_total_score',
    'binary_boltz_score',
    'binary_iptm',
    'binary_confidence_score',
    'binary_plddt_ligand',
    'binary_plddt_pocket',
    'binary_hbond_distance',
    'binary_hbond_angle',
    'binary_geometry_score',
    'binary_affinity_probability_binary',
]

RANK_COLUMN = 'binary_total_score'


def origin_round(name):
    """Determine which expansion round a design originated from.

    Round 0 names: 'pair_0001' (from initial Boltz predictions)
    Round N names: '{lig}_exp_r{N}_{parent}_s{sample}' (from expansion)
    """
    m = re.search(r'_exp_r(\d+)_', name)
    if m:
        return int(m.group(1))
    return 0


def load_scores(csv_path):
    """Load scores CSV and return list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_top_n(rows, n, rank_col=RANK_COLUMN):
    """Sort rows by rank_col descending and return top N."""
    scored = []
    for row in rows:
        val = row.get(rank_col, '')
        try:
            row['_rank_val'] = float(val)
            scored.append(row)
        except (ValueError, TypeError):
            continue
    scored.sort(key=lambda r: r['_rank_val'], reverse=True)
    return scored[:n]


def summarize_metrics(rows, metrics):
    """Compute mean, median, max for each metric across rows."""
    summary = {}
    for m in metrics:
        vals = []
        for row in rows:
            v = row.get(m, '')
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                continue
        if vals:
            summary[f'{m}_mean'] = round(mean(vals), 4)
            summary[f'{m}_median'] = round(median(vals), 4)
            summary[f'{m}_max'] = round(max(vals), 4)
        else:
            summary[f'{m}_mean'] = None
            summary[f'{m}_median'] = None
            summary[f'{m}_max'] = None
    return summary


def round_attribution(top_rows, max_round):
    """Count how many of the top N designs came from each round."""
    origins = [origin_round(row.get('name', '')) for row in top_rows]
    counts = Counter(origins)
    result = {}
    for r in range(0, max_round + 1):
        result[f'from_r{r}'] = counts.get(r, 0)
    return result


def scores_at_ranks(top_rows, ranks, rank_col='_rank_val'):
    """Get score at specific rank positions (1-indexed)."""
    result = {}
    for r in ranks:
        idx = r - 1
        if idx < len(top_rows):
            result[f'score_at_rank_{r}'] = round(top_rows[idx][rank_col], 4)
        else:
            result[f'score_at_rank_{r}'] = None
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Track metric improvement across expansion rounds")
    parser.add_argument("--expansion-root", required=True,
                        help="Root directory (e.g. /scratch/.../expansion/ligandmpnn)")
    parser.add_argument("--ligands", nargs='+', default=['ca', 'cdca', 'udca', 'dca'],
                        help="Ligands to analyze (default: ca cdca udca dca)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top designs to summarize (default: 100)")
    parser.add_argument("--max-round", type=int, default=10,
                        help="Max round number to check (default: 10)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: print to stdout only)")

    args = parser.parse_args()
    root = Path(args.expansion_root)

    all_results = []

    for lig in args.ligands:
        lig_dir = root / lig
        if not lig_dir.exists():
            print(f"WARNING: {lig_dir} not found, skipping", file=sys.stderr)
            continue

        print(f"\n{'='*70}")
        print(f"  {lig.upper()} — Top {args.top_n} designs per round")
        print(f"{'='*70}")

        # Track highest round seen for attribution columns
        highest_round = 0

        for rnd in range(0, args.max_round + 1):
            round_dir = lig_dir / f"round_{rnd}"
            if not round_dir.exists():
                break

            # Round 0 uses scores.csv, round 1+ uses cumulative_scores.csv
            if rnd == 0:
                scores_path = round_dir / "scores.csv"
            else:
                scores_path = round_dir / "cumulative_scores.csv"

            if not scores_path.exists():
                break

            highest_round = rnd
            rows = load_scores(scores_path)
            top = get_top_n(rows, args.top_n)
            summary = summarize_metrics(top, METRICS)
            attribution = round_attribution(top, rnd)
            rank_scores = scores_at_ranks(top, [1, 10, 50, 100])

            result = {
                'ligand': lig.upper(),
                'round': rnd,
                'total_designs': len(rows),
                'top_n': len(top),
            }
            result.update(attribution)
            result.update(rank_scores)
            result.update(summary)
            all_results.append(result)

            # Print metrics line
            ts_mean = summary.get('binary_total_score_mean')
            ts_max = summary.get('binary_total_score_max')
            iptm_mean = summary.get('binary_iptm_mean')
            plddt_lig = summary.get('binary_plddt_ligand_mean')
            pbind = summary.get('binary_affinity_probability_binary_mean')
            hbond_d = summary.get('binary_hbond_distance_mean')
            geom = summary.get('binary_geometry_score_mean')

            print(f"  R{rnd}: {len(rows):5d} designs | "
                  f"total={ts_mean or 0:.3f} (max {ts_max or 0:.3f}) | "
                  f"ipTM={iptm_mean or 0:.3f} | "
                  f"pLDDT_lig={plddt_lig or 0:.3f} | "
                  f"P(bind)={pbind or 0:.3f} | "
                  f"hbond_d={hbond_d or 0:.2f}A | "
                  f"geom={geom or 0:.3f}")

            # Print round attribution
            attr_parts = []
            for r in range(0, rnd + 1):
                count = attribution.get(f'from_r{r}', 0)
                if count > 0:
                    attr_parts.append(f"R{r}:{count}")
            print(f"       Top {len(top)} from: {', '.join(attr_parts)}")

            # Print score at fixed ranks
            s1 = rank_scores.get('score_at_rank_1')
            s10 = rank_scores.get('score_at_rank_10')
            s50 = rank_scores.get('score_at_rank_50')
            s100 = rank_scores.get('score_at_rank_100')
            rank_parts = []
            if s1 is not None:
                rank_parts.append(f"#1={s1:.3f}")
            if s10 is not None:
                rank_parts.append(f"#10={s10:.3f}")
            if s50 is not None:
                rank_parts.append(f"#50={s50:.3f}")
            if s100 is not None:
                rank_parts.append(f"#100={s100:.3f}")
            print(f"       Scores at rank: {', '.join(rank_parts)}")

    # Write CSV
    if args.out and all_results:
        # Collect all fieldnames across all results
        fieldnames = []
        for r in all_results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWrote {len(all_results)} rows to {out_path}")

    if not all_results:
        print("No data found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
