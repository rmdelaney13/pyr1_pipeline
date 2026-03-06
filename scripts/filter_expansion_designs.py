#!/usr/bin/env python3
"""Filter expansion Boltz2 designs using relaxed Strategy H gates.

Loads per-ligand scored CSVs (from analyze_boltz_output.py), applies
pLDDT ligand and H-bond distance gates, ranks by pocket pLDDT, selects
top N designs, and reports R116 flip statistics.

The COO-to-R116 distance detects the binary prediction artifact where
the carboxylate salt-bridges R116 (latch), which is impossible in vivo
when HAB1 Trp385 (lock) occupies that site.

Usage:
    python scripts/filter_expansion_designs.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --ligands ca cdca dca \
        --gate-plddt 0.75 \
        --gate-hbond 4.5 \
        --top-n 100 \
        --extract-sequences

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import sys
from pathlib import Path


def load_scored_csv(csv_path):
    """Load analyze_boltz_output.py results, converting numeric fields."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for key in row:
                if row[key] == '' or row[key] == 'None' or row[key] is None:
                    row[key] = None
                else:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass  # keep as string
            rows.append(row)
    return rows


def extract_sequence_from_pdb(pdb_path, chain='A'):
    """Extract protein sequence from PDB chain A."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    seq = []
    seen_resids = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            if ch != chain:
                continue
            resname = line[17:20].strip()
            resid = line[22:27].strip()
            if resid in seen_resids:
                continue
            seen_resids.add(resid)
            aa = three_to_one.get(resname)
            if aa:
                seq.append(aa)
    return ''.join(seq)


def find_pdb_path(expansion_root, ligand, name):
    """Find the PDB file for a given prediction name across round dirs."""
    lig_dir = Path(expansion_root) / ligand
    for round_dir in sorted(lig_dir.glob("round_*")):
        boltz_dir = round_dir / "boltz_output"
        if not boltz_dir.exists():
            continue
        pdb = (boltz_dir / f"boltz_results_{name}" / "predictions"
               / name / f"{name}_model_0.pdb")
        if pdb.exists():
            return pdb
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter expansion designs with relaxed Strategy H")
    parser.add_argument(
        "--expansion-root", required=True,
        help="Root of expansion directory (e.g., /scratch/.../expansion/ligandmpnn)")
    parser.add_argument(
        "--ligands", nargs='+', default=["ca", "cdca", "dca"],
        help="Ligands to process (default: ca cdca dca)")
    parser.add_argument(
        "--gate-plddt", type=float, default=0.75,
        help="Minimum binary pLDDT ligand gate (default: 0.75)")
    parser.add_argument(
        "--gate-hbond", type=float, default=4.5,
        help="Maximum binary H-bond distance gate in Angstroms (default: 4.5)")
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of top designs to select per ligand (default: 100)")
    parser.add_argument(
        "--r116-flip-threshold", type=float, default=5.0,
        help="COO-to-R116 distance threshold for flip detection (default: 5.0 A)")
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: <expansion-root>/filtered)")
    parser.add_argument(
        "--extract-sequences", action="store_true",
        help="Extract protein sequences from Boltz PDB files")

    args = parser.parse_args()

    root = Path(args.expansion_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Strategy H (relaxed) gates:")
    print(f"  pLDDT_ligand >= {args.gate_plddt}")
    print(f"  H-bond dist  <= {args.gate_hbond} A")
    print(f"  Rank by: binary_plddt_pocket (descending)")
    print(f"  Top N: {args.top_n} per ligand")
    print(f"  R116 flip threshold: < {args.r116_flip_threshold} A")
    print()

    all_top = []

    for lig in args.ligands:
        scored_csv = root / lig / "boltz_scored.csv"
        if not scored_csv.exists():
            print(f"WARNING: {scored_csv} not found - skipping {lig.upper()}")
            print(f"  Run analyze_boltz_output.py first (see score_expansion_boltz.sh)")
            print()
            continue

        rows = load_scored_csv(str(scored_csv))
        total = len(rows)

        print(f"{'=' * 60}")
        print(f"{lig.upper()}: {total} total predictions")
        print(f"{'=' * 60}")

        # ── Apply gates ──
        gated = []
        fail_plddt = 0
        fail_hbond = 0
        fail_both = 0
        fail_missing = 0

        for row in rows:
            plddt_lig = row.get('binary_plddt_ligand')
            hbond_dist = row.get('binary_hbond_distance')

            if plddt_lig is None or hbond_dist is None:
                fail_missing += 1
                continue

            pass_plddt = plddt_lig >= args.gate_plddt
            pass_hbond = hbond_dist <= args.gate_hbond

            if pass_plddt and pass_hbond:
                gated.append(row)
            elif not pass_plddt and not pass_hbond:
                fail_both += 1
            elif not pass_plddt:
                fail_plddt += 1
            else:
                fail_hbond += 1

        print(f"  Gate results:")
        print(f"    Pass both:    {len(gated):>5} ({100 * len(gated) / max(total, 1):.1f}%)")
        print(f"    Fail pLDDT:   {fail_plddt:>5}")
        print(f"    Fail H-bond:  {fail_hbond:>5}")
        print(f"    Fail both:    {fail_both:>5}")
        if fail_missing:
            print(f"    Missing data: {fail_missing:>5}")

        # ── Rank by pocket pLDDT ──
        gated.sort(
            key=lambda r: r.get('binary_plddt_pocket') or 0,
            reverse=True,
        )
        top = gated[:args.top_n]

        print(f"\n  Top {len(top)} by pocket pLDDT:")

        if top:
            pocket_plddts = [r['binary_plddt_pocket'] for r in top
                             if r.get('binary_plddt_pocket') is not None]
            hbond_dists = [r['binary_hbond_distance'] for r in top
                           if r.get('binary_hbond_distance') is not None]
            plddt_ligs = [r['binary_plddt_ligand'] for r in top
                          if r.get('binary_plddt_ligand') is not None]

            if pocket_plddts:
                mean_pp = sum(pocket_plddts) / len(pocket_plddts)
                print(f"    Pocket pLDDT: {min(pocket_plddts):.3f} - "
                      f"{max(pocket_plddts):.3f} (mean {mean_pp:.3f})")
            if plddt_ligs:
                print(f"    Ligand pLDDT: {min(plddt_ligs):.3f} - "
                      f"{max(plddt_ligs):.3f}")
            if hbond_dists:
                print(f"    H-bond dist:  {min(hbond_dists):.2f} - "
                      f"{max(hbond_dists):.2f} A")

            # ipTM summary
            iptms = [r.get('binary_iptm') for r in top
                     if r.get('binary_iptm') is not None]
            if iptms:
                mean_iptm = sum(iptms) / len(iptms)
                print(f"    ipTM:         {min(iptms):.3f} - "
                      f"{max(iptms):.3f} (mean {mean_iptm:.3f})")

            # ── R116 flip statistics ──
            # Three categories based on COO-to-R116 distance:
            #   < 5 A:  salt bridge (COO directly engaging R116 guanidinium)
            #   5-10 A: COO-up (COO facing R116 side, not salt-bridging)
            #   > 10 A: OH-up / normal (COO away from R116)
            r116_dists = [r.get('binary_coo_to_r116_dist') for r in top
                          if r.get('binary_coo_to_r116_dist') is not None]
            if r116_dists:
                n_salt = sum(1 for d in r116_dists if d < 5.0)
                n_coo_up = sum(1 for d in r116_dists if 5.0 <= d < 10.0)
                n_oh_up = sum(1 for d in r116_dists if d >= 10.0)
                n_total = len(r116_dists)
                mean_r116 = sum(r116_dists) / n_total
                print(f"\n  R116 orientation analysis (COO-to-R116 distance):")
                print(f"    < 5 A  (salt bridge): {n_salt:>4} / {n_total} "
                      f"({100 * n_salt / n_total:.1f}%)")
                print(f"    5-10 A (COO-up):      {n_coo_up:>4} / {n_total} "
                      f"({100 * n_coo_up / n_total:.1f}%)")
                print(f"    > 10 A (OH-up):       {n_oh_up:>4} / {n_total} "
                      f"({100 * n_oh_up / n_total:.1f}%)")
                print(f"    Range: {min(r116_dists):.2f} - "
                      f"{max(r116_dists):.2f} A (mean {mean_r116:.2f})")
            else:
                print(f"\n  R116 flip: no coo_to_r116_dist data available")

        # ── Extract sequences ──
        if args.extract_sequences and top:
            print(f"\n  Extracting sequences from PDB files...")
            n_found = 0
            for row in top:
                pdb_path = find_pdb_path(str(root), lig, row['name'])
                if pdb_path:
                    row['sequence'] = extract_sequence_from_pdb(str(pdb_path))
                    n_found += 1
                else:
                    row['sequence'] = ''
            print(f"    Extracted {n_found} / {len(top)} sequences")

        # ── Tag rows and save ──
        for i, row in enumerate(top):
            row['ligand'] = lig.upper()
            row['rank'] = i + 1
        all_top.extend(top)

        if top:
            out_csv = out_dir / f"top{args.top_n}_{lig}.csv"
            # Key columns first, then everything else
            primary_fields = [
                'rank', 'ligand', 'name',
                'binary_plddt_pocket', 'binary_plddt_ligand',
                'binary_hbond_distance', 'binary_coo_to_r116_dist',
                'binary_iptm', 'binary_confidence_score',
                'binary_hbond_angle',
                'binary_coo_to_water_dist', 'binary_oh_to_water_dist',
                'binary_flip_score',
            ]
            if args.extract_sequences:
                primary_fields.append('sequence')

            # Add remaining binary_ columns
            extra = sorted(k for k in top[0].keys()
                           if k.startswith('binary_') and k not in primary_fields)
            fields = primary_fields + extra

            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields,
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(top)
            print(f"\n  Saved: {out_csv}")

        print()

    # ── Combined output ──
    if all_top:
        combined_csv = out_dir / f"combined_top{args.top_n}_all_ligands.csv"
        fields = [
            'rank', 'ligand', 'name',
            'binary_plddt_pocket', 'binary_plddt_ligand',
            'binary_hbond_distance', 'binary_coo_to_r116_dist',
            'binary_iptm', 'binary_confidence_score',
        ]
        if args.extract_sequences:
            fields.append('sequence')

        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_top)

        print(f"Combined output: {combined_csv}")
        print(f"Total top designs: {len(all_top)}")

        # Per-ligand counts
        for lig in args.ligands:
            n = sum(1 for r in all_top
                    if r.get('ligand', '').lower() == lig)
            print(f"  {lig.upper()}: {n}")

        # Combined FASTA
        if args.extract_sequences:
            fasta_path = out_dir / f"top{args.top_n}_all_ligands.fasta"
            n_written = 0
            with open(fasta_path, 'w') as f:
                for row in all_top:
                    seq = row.get('sequence', '')
                    if seq:
                        lig_tag = row.get('ligand', 'UNK')
                        name = row.get('name', 'unknown')
                        f.write(f">{lig_tag}_{name}\n{seq}\n")
                        n_written += 1
            print(f"FASTA: {fasta_path} ({n_written} sequences)")

    else:
        print("No designs passed filtering for any ligand.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
