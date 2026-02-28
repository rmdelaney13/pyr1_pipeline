#!/usr/bin/env python3
"""
Merge expansion round scores into a cumulative CSV.

Takes the previous round's cumulative scores and the new round's scores,
concatenates them, deduplicates by name, and re-sorts by binary_total_score.

Usage:
    python expansion_merge.py \
        --previous /scratch/.../round_0/scores.csv \
        --new /scratch/.../round_1/new_scores.csv \
        --out /scratch/.../round_1/cumulative_scores.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge expansion round scores into cumulative CSV")
    parser.add_argument("--previous", required=True,
                        help="Previous round's cumulative scores CSV")
    parser.add_argument("--new", required=True,
                        help="New round's scores CSV")
    parser.add_argument("--out", required=True,
                        help="Output cumulative CSV")
    parser.add_argument("--score-column", default="binary_total_score",
                        help="Column to sort by (default: binary_total_score)")

    args = parser.parse_args()

    # Collect all fieldnames and rows
    all_rows = []
    all_fieldnames = []
    seen_names = set()

    for csv_path, label in [(args.previous, "previous"), (args.new, "new")]:
        p = Path(csv_path)
        if not p.exists():
            print(f"WARNING: {label} CSV not found: {csv_path}", file=sys.stderr)
            continue

        with open(p) as f:
            reader = csv.DictReader(f)
            for fn in reader.fieldnames:
                if fn not in all_fieldnames:
                    all_fieldnames.append(fn)

            count = 0
            dups = 0
            for row in reader:
                name = row.get('name', '')
                if name in seen_names:
                    dups += 1
                    continue
                seen_names.add(name)
                all_rows.append(row)
                count += 1

            print(f"  {label}: {count} rows loaded" +
                  (f" ({dups} duplicates skipped)" if dups else ""))

    # Sort by score descending
    def sort_key(row):
        val = row.get(args.score_column, '')
        try:
            return float(val)
        except (ValueError, TypeError):
            return -999.0

    all_rows.sort(key=sort_key, reverse=True)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCumulative: {len(all_rows)} designs -> {out_path}")
    if all_rows:
        top_score = sort_key(all_rows[0])
        bot_score = sort_key(all_rows[-1])
        print(f"  Score range: {top_score:.4f} - {bot_score:.4f}")


if __name__ == "__main__":
    main()
