#!/usr/bin/env python3
"""
Quick analysis: how many designs pass OH satisfaction with corrected mapping.

Loads cumulative scores, applies hard filters, then checks OH satisfaction
for each passing design. Reports counts by binding mode and pos-83 residue.

Usage:
    python scripts/check_oh_satisfaction.py \
        --scores /scratch/alpine/ryde3462/CDCA/design/expansion/ligandmpnn/round_1/cumulative_scores.csv \
        --boltz-dirs /scratch/alpine/ryde3462/CDCA/design/boltz_output \
                     /scratch/alpine/ryde3462/CDCA/design/expansion/ligandmpnn/round_1/boltz_output \
        --ligand cdca
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from filter_expansion_designs import WATER_MEDIATED_OH


POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122,
                    141, 159, 160, 163, 164, 167]

HARD_FILTERS = [
    ("binary_hbond_distance", "<", 4.0),
    ("binary_hbond_distance", ">", 1.8),
    ("binary_plddt_ligand", ">", 0.65),
    ("binary_coo_to_r116_dist", ">", 4.0),
    ("binary_ligand_distorted", "<", 1),
    ("binary_hab1_clash_dist", ">", 2.0),
    ("binary_latch_rmsd", "<", 1.0),
]


def find_pdb(name, boltz_dirs):
    for d in boltz_dirs:
        d = Path(d)
        pdb = d / ("boltz_results_%s" % name) / "predictions" / name / ("%s_model_0.pdb" % name)
        if pdb.exists():
            return pdb
    return None


def classify_binding_mode(row):
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


def get_res83(pdb_path):
    """Get the amino acid at position 83 from PDB."""
    THREE_TO_ONE = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[21] != 'A' or line[12:16].strip() != 'CA':
                    continue
                resnum = int(line[22:26].strip())
                if resnum == 83:
                    return THREE_TO_ONE.get(line[17:20].strip(), 'X')
    except Exception:
        pass
    return '?'


def compute_oh_satisfaction(pdb_path, water_oh_indices, hbond_cutoff=3.5):
    """Check OH satisfaction. Returns dict with per-OH details."""
    protein_acceptors = []
    ligand_oxygens = []
    ligand_carbons = []

    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = atom_name[0]

                if ch == 'A':
                    if elem in ('O', 'N'):
                        protein_acceptors.append(np.array([x, y, z]))
                elif ch == 'B':
                    coord = np.array([x, y, z])
                    if elem == 'O':
                        ligand_oxygens.append((coord, atom_name))
                    elif elem == 'C':
                        ligand_carbons.append((coord, atom_name))
    except Exception:
        return None

    if not ligand_oxygens or not protein_acceptors:
        return None

    protein_coords = np.array(protein_acceptors)

    # Identify carboxylate oxygens
    coo_indices = set()
    for c_coord, _ in ligand_carbons:
        bonded = [i for i, (o_coord, _) in enumerate(ligand_oxygens)
                  if np.linalg.norm(o_coord - c_coord) < 1.65]
        if len(bonded) == 2:
            coo_indices.update(bonded)

    results = []
    n_oh = 0
    n_satisfied = 0
    n_unsatisfied = 0

    for i, (coord, aname) in enumerate(ligand_oxygens):
        if i in coo_indices:
            continue
        is_water = i in water_oh_indices
        dists = np.linalg.norm(protein_coords - coord, axis=1)
        min_dist = float(dists.min())
        satisfied = min_dist <= hbond_cutoff

        if not is_water:
            n_oh += 1
            if satisfied:
                n_satisfied += 1
            else:
                n_unsatisfied += 1

        results.append({
            'index': i,
            'atom': aname,
            'water_mediated': is_water,
            'min_protein_dist': min_dist,
            'satisfied': satisfied,
        })

    return {
        'n_oh': n_oh,
        'n_satisfied': n_satisfied,
        'n_unsatisfied': n_unsatisfied,
        'details': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--boltz-dirs", nargs='+', required=True)
    parser.add_argument("--ligand", default="cdca")
    args = parser.parse_args()

    water_set = WATER_MEDIATED_OH.get(args.ligand.lower(), set())
    print("Ligand: %s" % args.ligand.upper())
    print("Water-mediated OH indices: %s" % water_set)

    # Load and filter
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('binary_total_score', '')
            try:
                row['_score'] = float(val)
            except (ValueError, TypeError):
                continue
            rows.append(row)

    rows.sort(key=lambda r: r['_score'], reverse=True)
    print("Loaded %d designs" % len(rows))

    # Apply hard filters
    print("\nApplying hard filters:")
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
        print("  %-40s %d -> %d" % ("%s%s%.1f" % (col, op, val), before, len(rows)))
    print("  Passing: %d" % len(rows))

    # Check OH satisfaction for each
    print("\nChecking OH satisfaction (this reads each PDB)...")
    mode_res83_stats = defaultdict(lambda: defaultdict(lambda: {'sat': 0, 'unsat': 0, 'fail': 0}))
    total_sat = 0
    total_unsat = 0
    total_fail = 0

    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            print("  %d / %d..." % (i + 1, len(rows)))

        mode = classify_binding_mode(r)
        r['_mode'] = mode

        pdb = find_pdb(r['name'], args.boltz_dirs)
        if not pdb:
            total_fail += 1
            continue

        res83 = get_res83(str(pdb))
        r['_res83'] = res83

        oh = compute_oh_satisfaction(str(pdb), water_set)
        if oh is None:
            total_fail += 1
            continue

        r['_n_unsat'] = oh['n_unsatisfied']

        if oh['n_unsatisfied'] == 0:
            total_sat += 1
            mode_res83_stats[mode][res83]['sat'] += 1
        else:
            total_unsat += 1
            mode_res83_stats[mode][res83]['unsat'] += 1

    print("\n" + "=" * 70)
    print("OH Satisfaction Summary (corrected: O43=water, O44=must satisfy)")
    print("=" * 70)
    print("  All OH satisfied:    %d / %d (%.1f%%)" % (
        total_sat, len(rows), 100 * total_sat / len(rows) if rows else 0))
    print("  Some OH unsatisfied: %d / %d (%.1f%%)" % (
        total_unsat, len(rows), 100 * total_unsat / len(rows) if rows else 0))
    if total_fail:
        print("  Failed/no PDB:       %d" % total_fail)

    # Breakdown by mode
    for mode in ['COO', 'OH', 'unknown']:
        stats = mode_res83_stats.get(mode)
        if not stats:
            continue
        mode_sat = sum(v['sat'] for v in stats.values())
        mode_unsat = sum(v['unsat'] for v in stats.values())
        mode_total = mode_sat + mode_unsat
        print("\n  --- %s mode (%d designs) ---" % (mode, mode_total))
        print("  Satisfied: %d (%.1f%%)" % (mode_sat, 100 * mode_sat / mode_total if mode_total else 0))
        print("  Unsatisfied: %d (%.1f%%)" % (mode_unsat, 100 * mode_unsat / mode_total if mode_total else 0))

        # By residue at position 83
        print("\n  By residue at position 83:")
        print("  %5s  %5s  %5s  %5s  %6s" % ('Res83', 'Sat', 'Unsat', 'Total', '%Sat'))
        all_res = sorted(stats.keys(), key=lambda k: -(stats[k]['sat'] + stats[k]['unsat']))
        for res in all_res:
            s = stats[res]['sat']
            u = stats[res]['unsat']
            t = s + u
            pct = 100 * s / t if t else 0
            print("  %5s  %5d  %5d  %5d  %5.1f%%" % (res, s, u, t, pct))


if __name__ == "__main__":
    main()
