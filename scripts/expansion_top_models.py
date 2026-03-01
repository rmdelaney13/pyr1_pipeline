#!/usr/bin/env python3
"""
Extract top N models per ligand from expansion results.

Copies PDB files and writes a summary CSV with scores for the best
designs from the final expansion round.

Usage:
    python expansion_top_models.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --initial-boltz-dir /scratch/alpine/ryde3462/boltz_bile_acids \
        --out-dir /scratch/alpine/ryde3462/expansion/ligandmpnn/top_models \
        --top-n 6

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List, Optional


RANK_COLUMN = 'binary_total_score'

DISPLAY_COLUMNS = [
    'name', 'binary_total_score', 'binary_boltz_score', 'binary_iptm',
    'binary_confidence_score', 'binary_plddt_ligand', 'binary_plddt_pocket',
    'binary_hbond_distance', 'binary_hbond_angle', 'binary_geometry_score',
    'binary_affinity_probability_binary',
]


def find_pdb(name: str, boltz_dirs: List[Path]) -> Optional[Path]:
    """Locate Boltz output PDB for a given prediction name."""
    for d in boltz_dirs:
        pdb = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
        if pdb.exists():
            return pdb
        cif = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        if cif.exists():
            return cif
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract top N models per ligand from expansion results")
    parser.add_argument("--expansion-root", required=True,
                        help="Expansion root (e.g. /scratch/.../expansion/ligandmpnn)")
    parser.add_argument("--initial-boltz-dir", required=True,
                        help="Initial Boltz output parent dir (e.g. /scratch/.../boltz_bile_acids)")
    parser.add_argument("--ligands", nargs='+', default=['ca', 'cdca', 'udca', 'dca'])
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for PDBs and summary CSV")
    parser.add_argument("--max-round", type=int, default=10)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_top_rows = []

    for lig in args.ligands:
        lig_dir = Path(args.expansion_root) / lig
        if not lig_dir.exists():
            print(f"WARNING: {lig_dir} not found, skipping", file=sys.stderr)
            continue

        # Find latest completed round
        scores_path = None
        latest_round = -1
        for rnd in range(0, args.max_round + 1):
            rd = lig_dir / f"round_{rnd}"
            if not rd.exists():
                break
            if rnd == 0:
                candidate = rd / "scores.csv"
            else:
                candidate = rd / "cumulative_scores.csv"
            if candidate.exists():
                scores_path = candidate
                latest_round = rnd

        if scores_path is None:
            print(f"WARNING: No scores found for {lig.upper()}", file=sys.stderr)
            continue

        # Collect all Boltz output directories
        boltz_dirs = [Path(args.initial_boltz_dir) / f"output_{lig}_binary"]
        for rnd in range(1, latest_round + 1):
            bd = lig_dir / f"round_{rnd}" / "boltz_output"
            if bd.exists():
                boltz_dirs.append(bd)

        # Load and rank
        rows = []
        with open(scores_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(RANK_COLUMN, '')
                try:
                    row['_rank_val'] = float(val)
                    rows.append(row)
                except (ValueError, TypeError):
                    continue
        rows.sort(key=lambda r: r['_rank_val'], reverse=True)
        top = rows[:args.top_n]

        # Print and copy
        print(f"\n{'='*70}")
        print(f"  {lig.upper()} — Top {len(top)} (from round {latest_round}, "
              f"{len(rows)} total designs)")
        print(f"{'='*70}")

        lig_out = out_dir / lig.upper()
        lig_out.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(top, 1):
            name = row.get('name', '')
            total = row.get('binary_total_score', '')
            iptm = row.get('binary_iptm', '')
            plddt_lig = row.get('binary_plddt_ligand', '')
            pbind = row.get('binary_affinity_probability_binary', '')
            hbond_d = row.get('binary_hbond_distance', '')
            geom = row.get('binary_geometry_score', '')

            # Find and copy PDB
            pdb_path = find_pdb(name, boltz_dirs)
            pdb_status = ''
            if pdb_path:
                dest = lig_out / f"{lig.upper()}_top{i}_{name}{pdb_path.suffix}"
                shutil.copy2(pdb_path, dest)
                pdb_status = f"-> {dest.name}"
            else:
                pdb_status = "PDB NOT FOUND"

            print(f"  #{i:2d}  {name:<25s}  total={total:>6s}  ipTM={iptm:>6s}  "
                  f"pLDDT_lig={plddt_lig:>6s}  P(bind)={pbind:>6s}  "
                  f"hbond_d={hbond_d:>5s}  geom={geom:>6s}  {pdb_status}")

            # Store for CSV
            csv_row = {'ligand': lig.upper(), 'rank': i}
            for col in DISPLAY_COLUMNS:
                csv_row[col] = row.get(col, '')
            csv_row['pdb_file'] = dest.name if pdb_path else ''
            csv_row['source_round'] = latest_round
            all_top_rows.append(csv_row)

    # Write summary CSV
    if all_top_rows:
        summary_path = out_dir / "top_models_summary.csv"
        fieldnames = ['ligand', 'rank', 'source_round', 'pdb_file'] + DISPLAY_COLUMNS
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_top_rows)
        print(f"\nSummary: {summary_path}")
        print(f"PDBs copied to: {out_dir}/{{CA,CDCA,UDCA,DCA}}/")


if __name__ == "__main__":
    main()
