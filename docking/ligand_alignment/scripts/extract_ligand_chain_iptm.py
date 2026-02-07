#!/usr/bin/env python3
"""
extract_ligand_chain_iptm.py

Scan an AlphaFold3 inference directory for all '*_summary_confidences.json' files,
extract the specified chain_iptm element (default index 1, i.e. second chain),
and write out a CSV with one row per target.

Usage:
  ./extract_ligand_chain_iptm.py \
    /path/to/inference_root \
    /path/to/output.csv \
    --ligand_chain_idx 1
"""
import os
import json
import glob
import argparse
import csv

def main(inference_dir, output_csv, ligand_idx):
    # Find all summary JSONs recursively
    pattern = os.path.join(inference_dir, '**', '*_summary_confidences.json')
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No summary JSON files found under {inference_dir}")
        return

    records = []
    for path in sorted(files):
        target = os.path.basename(path).split('_summary_confidences.json')[0]
        try:
            data = json.load(open(path))
            chain_iptm = data.get('chain_iptm', [])
            # Extract the ligand score at the given index
            ligand_score = None
            if isinstance(chain_iptm, list) and len(chain_iptm) > ligand_idx:
                ligand_score = chain_iptm[ligand_idx]
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
            ligand_score = None
        records.append((target, ligand_score))

    # Write out CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['target', 'ligand_chain_iptm'])
        for tgt, score in records:
            writer.writerow([tgt, score])

    print(f"Wrote {len(records)} records to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract ligand chain_iptm scores from AF3 summary JSONs'
    )
    parser.add_argument('inference_dir', help='Root of AF3 inference directories')
    parser.add_argument('output_csv', help='Path to write aggregated CSV')
    parser.add_argument(
        '--ligand_chain_idx', type=int, default=1,
        help='0-based index of ligand chain in chain_iptm array (default=1)'
    )
    args = parser.parse_args()
    main(args.inference_dir, args.output_csv, args.ligand_chain_idx)

