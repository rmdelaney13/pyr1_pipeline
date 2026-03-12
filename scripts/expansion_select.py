#!/usr/bin/env python3
"""
Select top N designs and copy their Boltz PDBs to a staging dir for MPNN redesign.

Supports three selection modes:
  1. Pure top-N by score (default, original behavior)
  2. OH-aware: prefer designs with all protein-contacting OHs satisfied
  3. Diverse: split N between top-score picks and Hamming-diverse picks

Usage:
    # Original behavior:
    python expansion_select.py \
        --scores /scratch/.../scores.csv \
        --boltz-dirs /scratch/.../output_ca_binary /scratch/.../round_1/boltz_output \
        --out-dir /scratch/.../round_1/selected_pdbs \
        --top-n 100

    # OH-aware + diverse:
    python expansion_select.py \
        --scores /scratch/.../scores.csv \
        --boltz-dirs /scratch/.../output_ca_binary \
        --out-dir /scratch/.../round_1/selected_pdbs \
        --top-n 100 \
        --ligand ca \
        --prefer-oh-satisfied \
        --diverse --diverse-fraction 0.5

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))


def find_pdb_for_name(name: str, boltz_dirs: List[str]) -> Optional[Path]:
    """Locate the Boltz output PDB for a given prediction name.

    Searches: boltz_dir/boltz_results_{name}/predictions/{name}/{name}_model_0.pdb
    """
    for d in boltz_dirs:
        d = Path(d)
        # Direct path
        pdb = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
        if pdb.exists():
            return pdb
        # Try CIF
        cif = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        if cif.exists():
            return cif
    return None


def compute_oh_unsatisfied(pdb_path, water_oh_indices=None, gate_residue=88):
    """Count unsatisfied protein-contacting OHs for a PDB.

    Water-mediated OH is identified geometrically: the hydroxyl O closest
    to the gate residue (PRO88 CA) is excluded. water_oh_indices is
    deprecated and ignored.

    Returns (n_protein_oh, n_unsatisfied, flipped) or (None, None, None).
    flipped=True means the water-mediated OH is closer to the carboxylate
    than the pocket-facing OH, indicating the 7α-OH is at the gate (wrong
    orientation for ternary complex).
    """
    import numpy as np

    protein_acceptors = []
    ligand_oxygens = []
    ligand_carbons = []
    gate_ca = None

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
                    resnum = int(line[22:26].strip())
                    if resnum == gate_residue and atom_name == 'CA':
                        gate_ca = np.array([x, y, z])
                elif ch == 'B':
                    coord = np.array([x, y, z])
                    if elem == 'O':
                        ligand_oxygens.append((coord, atom_name))
                    elif elem == 'C':
                        ligand_carbons.append((coord, atom_name))
    except Exception:
        return None, None

    if not ligand_oxygens or not protein_acceptors:
        return None, None, None

    protein_coords = np.array(protein_acceptors)

    # Identify carboxylate oxygens
    coo_indices = set()
    for c_coord, _ in ligand_carbons:
        bonded = [i for i, (o_coord, _) in enumerate(ligand_oxygens)
                  if np.linalg.norm(o_coord - c_coord) < 1.65]
        if len(bonded) == 2:
            coo_indices.update(bonded)

    # Collect hydroxyl oxygens
    hydroxyl_os = [(i, coord) for i, (coord, _) in enumerate(ligand_oxygens)
                   if i not in coo_indices]

    # Geometrically identify water-mediated OH: closest to gate CA
    water_idx = None
    if gate_ca is not None and len(hydroxyl_os) > 1:
        gate_dists = [(np.linalg.norm(coord - gate_ca), i)
                      for i, coord in hydroxyl_os]
        gate_dists.sort()
        water_idx = gate_dists[0][1]

    # Check if the water-mediated OH is the one closer to the carboxylate
    # (i.e. the 7α-OH, which should face the pocket, not the gate).
    # On the steroid scaffold, 7α-OH (C7) is closer to the carboxylate (C24)
    # than 3α-OH (C3). If the gate-nearest OH is also COO-nearest, it's flipped.
    flipped = False
    if water_idx is not None and len(hydroxyl_os) == 2 and coo_indices:
        coo_coords = [ligand_oxygens[i][0] for i in coo_indices]
        coo_centroid = np.mean(coo_coords, axis=0)
        water_coord = ligand_oxygens[water_idx][0]
        other_idx = [i for i, _ in hydroxyl_os if i != water_idx][0]
        other_coord = ligand_oxygens[other_idx][0]
        water_to_coo = float(np.linalg.norm(water_coord - coo_centroid))
        other_to_coo = float(np.linalg.norm(other_coord - coo_centroid))
        if water_to_coo < other_to_coo:
            flipped = True

    # Count unsatisfied hydroxyl OHs (skip water-mediated)
    n_protein_oh = 0
    n_unsatisfied = 0
    for i, coord in hydroxyl_os:
        if i == water_idx:
            continue
        n_protein_oh += 1
        dists = np.linalg.norm(protein_coords - coord, axis=1)
        if float(dists.min()) > 3.5:
            n_unsatisfied += 1

    return n_protein_oh, n_unsatisfied, flipped


def get_pocket_seq(pdb_path, pocket_positions):
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
                if resnum in pocket_positions:
                    residues[resnum] = THREE_TO_ONE.get(resname, 'X')
    except Exception:
        return None
    if len(residues) < len(pocket_positions):
        return None
    return ''.join(residues.get(p, 'X') for p in sorted(pocket_positions))


def hamming_dist(s1, s2):
    """Hamming distance between two equal-length strings."""
    return sum(a != b for a, b in zip(s1, s2))


def select_diverse(rows, boltz_dirs, n_total, diverse_frac,
                   pocket_positions, min_hamming=3):
    """Select n_total designs: (1-diverse_frac)*n by score, rest by diversity.

    Diversity picks are chosen greedily to maximize minimum Hamming distance
    from already-selected pocket sequences.
    """
    n_score = int(n_total * (1 - diverse_frac))
    n_diverse = n_total - n_score

    # Score picks (already sorted by score)
    score_picks = rows[:n_score]
    selected_set = set(r['name'] for r in score_picks)

    # Build pocket sequences for score picks
    selected_pockets = []
    for r in score_picks:
        pdb_path = find_pdb_for_name(r['name'], boltz_dirs)
        if pdb_path:
            seq = get_pocket_seq(str(pdb_path), pocket_positions)
            if seq:
                selected_pockets.append(seq)

    # Candidate pool for diversity picks (rest of the list)
    candidates = []
    for r in rows[n_score:]:
        if r['name'] in selected_set:
            continue
        pdb_path = find_pdb_for_name(r['name'], boltz_dirs)
        if pdb_path is None:
            continue
        seq = get_pocket_seq(str(pdb_path), pocket_positions)
        if seq is None:
            continue
        candidates.append((r, seq, str(pdb_path)))

    # Greedy diversity selection
    diverse_picks = []
    for _ in range(n_diverse):
        if not candidates:
            break

        best_row = None
        best_min_dist = -1
        best_idx = -1

        for ci, (r, seq, pdb) in enumerate(candidates):
            if not selected_pockets:
                min_d = 16  # max possible
            else:
                min_d = min(hamming_dist(seq, sp) for sp in selected_pockets)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_row = r
                best_idx = ci

        if best_row is None or best_min_dist < 1:
            break

        diverse_picks.append(best_row)
        _, seq, _ = candidates[best_idx]
        selected_pockets.append(seq)
        candidates.pop(best_idx)

    print(f"  Diverse selection: {n_score} score picks + "
          f"{len(diverse_picks)} diversity picks")
    if diverse_picks and selected_pockets:
        # Report min Hamming distance of diversity picks
        print(f"  Min Hamming distance of diversity picks: {min_hamming}")

    return score_picks + diverse_picks


def classify_binding_mode(row):
    """Classify a design as COO or OH binding mode.

    Uses binary_coo_to_water_dist and binary_oh_to_water_dist from scores CSV.
    COO: carboxylate oxygen closer to water. OH: hydroxyl oxygen closer.
    Returns 'COO', 'OH', or 'unknown'.
    """
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


def parse_mode_quotas(quota_str):
    """Parse 'COO:100,OH:100' into dict {'COO': 100, 'OH': 100}."""
    quotas = {}
    for part in quota_str.split(','):
        mode, count = part.strip().split(':')
        quotas[mode.strip().upper()] = int(count.strip())
    return quotas


def select_stratified(rows, boltz_dirs, quotas, diverse, diverse_frac,
                      pocket_positions):
    """Select designs stratified by binding mode with per-mode quotas.

    Within each mode, applies score-only or score+diversity selection.
    Underfilled quotas redistribute to other modes.
    """
    # Classify all rows
    mode_buckets = defaultdict(list)
    for r in rows:
        mode = classify_binding_mode(r)
        r['_binding_mode'] = mode
        mode_buckets[mode].append(r)

    print(f"\n  Binding mode distribution:")
    for mode in ['COO', 'OH', 'unknown']:
        print(f"    {mode}: {len(mode_buckets[mode])}")

    # First pass: fill each mode's quota
    selected = []
    remaining_quota = 0
    modes_with_surplus = []

    for mode, quota in quotas.items():
        bucket = mode_buckets.get(mode, [])
        if diverse and len(bucket) > 0:
            picks = select_diverse(
                bucket, boltz_dirs, min(quota, len(bucket)),
                diverse_frac, pocket_positions)
        else:
            picks = bucket[:min(quota, len(bucket))]

        selected.extend(picks)
        shortfall = quota - len(picks)
        if shortfall > 0:
            remaining_quota += shortfall
            print(f"    {mode}: filled {len(picks)}/{quota} "
                  f"(shortfall {shortfall})")
        else:
            modes_with_surplus.append(mode)
            print(f"    {mode}: filled {len(picks)}/{quota}")

    # Redistribute shortfall to modes with surplus
    if remaining_quota > 0 and modes_with_surplus:
        selected_names = set(r['name'] for r in selected)
        extra_per_mode = remaining_quota // len(modes_with_surplus)
        leftover = remaining_quota % len(modes_with_surplus)

        for i, mode in enumerate(modes_with_surplus):
            n_extra = extra_per_mode + (1 if i < leftover else 0)
            if n_extra == 0:
                continue
            bucket = [r for r in mode_buckets[mode]
                      if r['name'] not in selected_names]
            extra = bucket[:n_extra]
            selected.extend(extra)
            if extra:
                print(f"    {mode}: +{len(extra)} redistributed")

    # Also fill from 'unknown' if still short
    total_quota = sum(quotas.values())
    if len(selected) < total_quota and mode_buckets.get('unknown'):
        selected_names = set(r['name'] for r in selected)
        n_need = total_quota - len(selected)
        unknown = [r for r in mode_buckets['unknown']
                   if r['name'] not in selected_names]
        extra = unknown[:n_need]
        selected.extend(extra)
        if extra:
            print(f"    unknown: +{len(extra)} to fill remaining quota")

    return selected


def main():
    from filter_expansion_designs import POCKET_POSITIONS

    parser = argparse.ArgumentParser(
        description="Select top N designs and copy PDBs for MPNN redesign")
    parser.add_argument("--scores", required=True,
                        help="Scores CSV from analyze_boltz_output.py")
    parser.add_argument("--boltz-dirs", nargs='+', action='append', required=True,
                        help="Boltz output directories to search for PDBs "
                             "(can be repeated: --boltz-dirs A B --boltz-dirs C)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for selected PDBs")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top designs to select (default: 100)")
    parser.add_argument("--score-column", default="binary_total_score",
                        help="Column to rank by (default: binary_total_score)")
    parser.add_argument("--ligand", default=None,
                        help="Ligand name (ca/cdca/dca) for OH-aware selection")
    parser.add_argument("--prefer-oh-satisfied", action="store_true",
                        help="Sort OH-satisfied designs first. Designs with "
                             "all protein-contacting OHs satisfied are ranked "
                             "above those with unsatisfied OHs at equal score. "
                             "Requires --ligand.")
    parser.add_argument("--diverse", action="store_true",
                        help="Use diverse parent selection: split top-N between "
                             "score picks and Hamming-diverse picks")
    parser.add_argument("--diverse-fraction", type=float, default=0.5,
                        help="Fraction of top-N to fill with diversity picks "
                             "(default: 0.5 = 50/50 split)")
    parser.add_argument("--binding-mode-stratify", action="store_true",
                        help="Stratify selection by binding mode (COO vs OH). "
                             "Uses binary_coo_to_water_dist and binary_oh_to_water_dist "
                             "columns to classify each design.")
    parser.add_argument("--mode-quotas", default=None,
                        help="Per-mode quotas, e.g. 'COO:100,OH:100'. "
                             "Underfilled quotas redistribute to other modes. "
                             "Requires --binding-mode-stratify.")
    parser.add_argument("--filter", nargs='+', default=[], metavar="EXPR",
                        help="Hard filters as 'column<value' or 'column>value'. "
                             "Applied before selection. "
                             "E.g. --filter 'binary_hbond_distance<4.0' "
                             "'binary_plddt_ligand>0.65' 'binary_coo_to_r116_dist>4.0'")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefer_oh_satisfied and not args.ligand:
        parser.error("--prefer-oh-satisfied requires --ligand")

    # Flatten boltz_dirs: action='append' + nargs='+' gives list of lists
    boltz_dirs = [d for group in args.boltz_dirs for d in group]

    print(f"Searching {len(boltz_dirs)} Boltz output directories:")
    for d in boltz_dirs:
        exists = Path(d).is_dir()
        print(f"  {'OK' if exists else 'MISSING'}: {d}")

    # Read and sort scores
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_val = row.get(args.score_column)
            if score_val is None or score_val == '':
                continue
            try:
                row['_sort_score'] = float(score_val)
            except ValueError:
                continue
            rows.append(row)

    rows.sort(key=lambda r: r['_sort_score'], reverse=True)
    print(f"\nLoaded {len(rows)} scored designs")

    # Apply hard filters
    if args.filter:
        import re
        for expr in args.filter:
            m = re.match(r'^(.+?)([<>])(.+)$', expr)
            if not m:
                parser.error(f"Invalid filter expression: {expr}")
            col, op, val = m.group(1), m.group(2), float(m.group(3))
            before = len(rows)
            if op == '<':
                rows = [r for r in rows if r.get(col) and float(r[col]) < val]
            else:
                rows = [r for r in rows if r.get(col) and float(r[col]) > val]
            print(f"  Filter {expr}: {before} -> {len(rows)}")
        print(f"  After all filters: {len(rows)} designs")

    # OH-aware sorting: within similar scores, prefer OH-satisfied
    if args.prefer_oh_satisfied and args.ligand:
        print(f"\nChecking OH satisfaction (ligand={args.ligand.upper()}, "
              f"water-mediated: closest OH to PRO88)...")

        n_flipped = 0
        for r in rows:
            r['_oh_unsatisfied'] = 999  # default for missing PDB
            r['_oh_flipped'] = False
            pdb_path = find_pdb_for_name(r['name'], boltz_dirs)
            if pdb_path:
                n_oh, n_unsat, flipped = compute_oh_unsatisfied(str(pdb_path))
                if n_unsat is not None:
                    r['_oh_unsatisfied'] = n_unsat
                    # Flipped only meaningful for OH mode (hydroxyl at gate).
                    # In COO mode the carboxylate is at the gate, not a hydroxyl.
                    mode = classify_binding_mode(r)
                    r['_oh_flipped'] = bool(flipped) and mode == 'OH'
                    if r['_oh_flipped']:
                        n_flipped += 1

        # Filter out flipped ligands (7α-OH at water gate = wrong orientation)
        before_flip = len(rows)
        rows = [r for r in rows if not r['_oh_flipped']]
        if n_flipped:
            print(f"  Removed {n_flipped} flipped-ligand designs "
                  f"(7α-OH at gate): {before_flip} -> {len(rows)}")

        # Re-sort: primary = fewer unsatisfied, secondary = higher score
        rows.sort(key=lambda r: (-r['_oh_unsatisfied'], -r['_sort_score']))
        rows.sort(key=lambda r: r['_oh_unsatisfied'])

        # Break down OH satisfaction by binding mode
        from collections import Counter
        mode_sat = {'COO': Counter(), 'OH': Counter(), 'unknown': Counter()}
        for r in rows:
            mode = classify_binding_mode(r)
            n_unsat = r['_oh_unsatisfied']
            if n_unsat == 999:
                mode_sat[mode]['no_pdb'] += 1
            elif n_unsat == 0:
                mode_sat[mode]['all_sat'] += 1
            else:
                mode_sat[mode][f'{n_unsat}_unsat'] += 1

        print(f"\n  OH satisfaction by binding mode:")
        for mode in ['COO', 'OH']:
            counts = mode_sat[mode]
            total = sum(counts.values())
            if total == 0:
                continue
            all_sat = counts.get('all_sat', 0)
            no_pdb = counts.get('no_pdb', 0)
            parts = [f"all satisfied: {all_sat}"]
            for key in sorted(counts):
                if key in ('all_sat', 'no_pdb'):
                    continue
                n = int(key.split('_')[0])
                parts.append(f"{n} OH unsat: {counts[key]}")
            if no_pdb:
                parts.append(f"no PDB: {no_pdb}")
            print(f"    {mode:>3s} ({total:3d}): {', '.join(parts)}")

    # Select designs
    if args.binding_mode_stratify:
        if not args.mode_quotas:
            parser.error("--binding-mode-stratify requires --mode-quotas")
        quotas = parse_mode_quotas(args.mode_quotas)
        print(f"\nBinding-mode stratified selection: {quotas}")
        top_rows = select_stratified(
            rows, boltz_dirs, quotas, args.diverse, args.diverse_fraction,
            POCKET_POSITIONS)
    elif args.diverse:
        top_rows = select_diverse(
            rows, boltz_dirs, args.top_n, args.diverse_fraction,
            POCKET_POSITIONS)
    else:
        top_rows = rows[:args.top_n]

    print(f"\nSelecting {len(top_rows)} designs")
    if top_rows:
        print(f"  Score range: {top_rows[0]['_sort_score']:.4f} - "
              f"{top_rows[-1]['_sort_score']:.4f}")

    # Copy PDBs
    copied = 0
    missing = 0
    missing_names = []
    manifest_lines = []

    for row in top_rows:
        name = row['name']
        pdb_path = find_pdb_for_name(name, boltz_dirs)
        if pdb_path is None:
            missing += 1
            missing_names.append(name)
            continue

        dest = out_dir / pdb_path.name
        shutil.copy2(pdb_path, dest)
        manifest_lines.append(str(dest))
        copied += 1

    # Write manifest
    manifest_path = out_dir.parent / "selected_manifest.txt"
    with open(manifest_path, 'w') as f:
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nCopied {copied} PDBs to {out_dir}")
    if missing:
        print(f"  ({missing} PDBs not found)", file=sys.stderr)
        # Show first few missing names for debugging
        show = missing_names[:5]
        print(f"  First missing: {', '.join(show)}" +
              (f" (+ {missing - 5} more)" if missing > 5 else ""),
              file=sys.stderr)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
