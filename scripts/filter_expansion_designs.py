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
import shutil
import sys
from collections import Counter
from pathlib import Path

# PYR1 16 mutable pocket positions (1-indexed)
POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122,
                    141, 159, 160, 163, 164, 167]
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


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


def get_residue_from_pdb(pdb_path, resnum, chain='A'):
    """Read a single residue identity from a PDB file (fast)."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain:
                continue
            try:
                rn = int(line[22:26].strip())
            except ValueError:
                continue
            if rn == resnum:
                return three_to_one.get(line[17:20].strip())
    return None


def compute_oh_satisfaction(pdb_path, protein_chain='A', ligand_chain='B',
                            hbond_cutoff=3.5):
    """Check whether ligand hydroxyl O atoms have H-bond partners on protein.

    For bile acids, the key polar groups are:
      - Carboxylate (COO-): usually satisfied by water network / R116
      - Hydroxyl groups: 3-OH (all), 7-OH (CA, CDCA), 12-OH (CA, DCA)
    The steroid core OHs are the differentiating feature and must be
    satisfied by designed pocket residues.

    Returns dict with:
      n_oh: number of non-carboxylate hydroxyl O atoms
      n_oh_satisfied: OHs with protein O/N within hbond_cutoff
      n_oh_unsatisfied: OHs without partner
      oh_min_dists: list of min distances per OH (for debugging)
      n_coo: number of carboxylate O atoms
      n_coo_satisfied: COO Os with protein partner
    """
    import numpy as np

    result = {
        'n_oh': None, 'n_oh_satisfied': None, 'n_oh_unsatisfied': None,
        'oh_min_dists': [], 'n_coo': None, 'n_coo_satisfied': None,
    }

    # Parse PDB manually (avoids Biopython dependency for this function)
    protein_acceptors = []  # (x, y, z) for all protein O and N atoms
    ligand_oxygens = []     # (x, y, z, atom_name) for all ligand O atoms
    ligand_carbons = []     # (x, y, z, atom_name) for all ligand C atoms

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if ch == protein_chain:
                # Collect all protein O and N as potential H-bond partners
                elem = atom_name[0]
                if elem in ('O', 'N'):
                    protein_acceptors.append(np.array([x, y, z]))
            elif ch == ligand_chain:
                coord = np.array([x, y, z])
                elem = atom_name[0]
                if elem == 'O':
                    ligand_oxygens.append((coord, atom_name))
                elif elem == 'C':
                    ligand_carbons.append((coord, atom_name))

    if not ligand_oxygens or not protein_acceptors:
        return result

    protein_coords = np.array(protein_acceptors)

    # Identify carboxylate oxygens: C with exactly 2 O neighbors within 1.65 A
    coo_oxygen_indices = set()
    for ci, (c_coord, c_name) in enumerate(ligand_carbons):
        bonded_o_idx = []
        for oi, (o_coord, o_name) in enumerate(ligand_oxygens):
            if np.linalg.norm(o_coord - c_coord) < 1.65:
                bonded_o_idx.append(oi)
        if len(bonded_o_idx) == 2:
            for idx in bonded_o_idx:
                coo_oxygen_indices.add(idx)

    # Split into carboxylate vs hydroxyl oxygens
    hydroxyl_os = []
    carboxylate_os = []
    for i, (coord, name) in enumerate(ligand_oxygens):
        if i in coo_oxygen_indices:
            carboxylate_os.append(coord)
        else:
            hydroxyl_os.append(coord)

    # Check satisfaction for hydroxyl OHs
    oh_min_dists = []
    n_sat = 0
    for oh_coord in hydroxyl_os:
        dists = np.linalg.norm(protein_coords - oh_coord, axis=1)
        min_d = float(dists.min())
        oh_min_dists.append(round(min_d, 2))
        if min_d <= hbond_cutoff:
            n_sat += 1

    # Check satisfaction for carboxylate Os
    n_coo_sat = 0
    for coo_coord in carboxylate_os:
        dists = np.linalg.norm(protein_coords - coo_coord, axis=1)
        if float(dists.min()) <= hbond_cutoff:
            n_coo_sat += 1

    result['n_oh'] = len(hydroxyl_os)
    result['n_oh_satisfied'] = n_sat
    result['n_oh_unsatisfied'] = len(hydroxyl_os) - n_sat
    result['oh_min_dists'] = oh_min_dists
    result['n_coo'] = len(carboxylate_os)
    result['n_coo_satisfied'] = n_coo_sat

    return result


def compute_latch_rmsd(pdb_path, ref_pdb_path, protein_chain='A',
                       latch_residues=None):
    """Compute backbone RMSD of latch loop after CA-alignment to reference.

    The latch loop (residues 114-118, centered on R116) must maintain its
    conformation for HAB1 Trp385 insertion. If Boltz2 deforms this loop
    to let sidechains interact with ligand, the design is incompatible
    with ternary complex formation.

    Returns dict with:
      latch_rmsd: backbone RMSD (CA, C, N, O) of latch residues after
                  full-chain CA alignment (Angstroms)
      latch_ca_rmsd: CA-only RMSD of latch residues (simpler metric)
    """
    import numpy as np

    if latch_residues is None:
        latch_residues = [114, 115, 116, 117, 118]

    result = {'latch_rmsd': None, 'latch_ca_rmsd': None}

    bb_atoms = {'CA', 'C', 'N', 'O'}

    def _parse_chain_a(pdb_path):
        """Extract CA coords (for alignment) and latch backbone coords."""
        ca_coords = {}       # {resnum: (x,y,z)}
        latch_bb = {}        # {(resnum, atom_name): (x,y,z)}
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                ch = line[21]
                if ch != protein_chain:
                    continue
                atom_name = line[12:16].strip()
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z])

                if atom_name == 'CA' and resnum not in ca_coords:
                    ca_coords[resnum] = coord
                if resnum in latch_residues and atom_name in bb_atoms:
                    key = (resnum, atom_name)
                    if key not in latch_bb:
                        latch_bb[key] = coord

        return ca_coords, latch_bb

    ref_ca, ref_latch = _parse_chain_a(ref_pdb_path)
    pred_ca, pred_latch = _parse_chain_a(pdb_path)

    if not ref_ca or not pred_ca:
        return result

    # Find common CA residues for alignment
    common_res = sorted(set(ref_ca) & set(pred_ca))
    if len(common_res) < 10:
        return result

    ref_ca_arr = np.array([ref_ca[r] for r in common_res])
    pred_ca_arr = np.array([pred_ca[r] for r in common_res])

    # Kabsch alignment (center, rotate, translate)
    ref_center = ref_ca_arr.mean(axis=0)
    pred_center = pred_ca_arr.mean(axis=0)
    ref_centered = ref_ca_arr - ref_center
    pred_centered = pred_ca_arr - pred_center

    H = pred_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    # Find common latch backbone atoms
    common_latch = sorted(set(ref_latch) & set(pred_latch))
    if not common_latch:
        return result

    # Apply rotation to prediction latch atoms
    ref_latch_arr = np.array([ref_latch[k] for k in common_latch])
    pred_latch_arr = np.array([pred_latch[k] for k in common_latch])

    pred_latch_aligned = ((pred_latch_arr - pred_center) @ R.T) + ref_center

    # Full backbone RMSD
    diff = ref_latch_arr - pred_latch_aligned
    result['latch_rmsd'] = round(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))), 3)

    # CA-only RMSD
    ca_keys = [k for k in common_latch if k[1] == 'CA']
    if ca_keys:
        ref_ca_latch = np.array([ref_latch[k] for k in ca_keys])
        pred_ca_latch = np.array([pred_latch[k] for k in ca_keys])
        pred_ca_aligned = ((pred_ca_latch - pred_center) @ R.T) + ref_center
        diff_ca = ref_ca_latch - pred_ca_aligned
        result['latch_ca_rmsd'] = round(
            float(np.sqrt(np.mean(np.sum(diff_ca**2, axis=1)))), 3)

    return result


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


def plot_pocket_logo(sequences, ligand_name, out_png, positions=None):
    """Generate a sequence logo of pocket positions from full-length sequences.

    Args:
        sequences: list of full-length PYR1 sequences (181 AA)
        ligand_name: e.g. "CA" for the title
        out_png: output path
        positions: list of 1-indexed pocket positions (default: 16 mutable)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import logomaker
        import pandas as pd
    except ImportError as e:
        print(f"  WARNING: Cannot plot logo (missing {e.name}), skipping")
        return

    if positions is None:
        positions = POCKET_POSITIONS

    # Build frequency table from sequences
    counts = {pos: Counter() for pos in positions}
    for seq in sequences:
        for pos in positions:
            idx = pos - 1  # 1-indexed to 0-indexed
            if idx < len(seq):
                aa = seq[idx]
                if aa in AA_ORDER:
                    counts[pos][aa] += 1

    df = pd.DataFrame.from_dict(counts, orient="index").fillna(0)
    df = df.reindex(columns=AA_ORDER, fill_value=0)
    df = df.reindex(positions)

    # Convert to probabilities
    row_sums = df.sum(axis=1)
    row_sums = row_sums.replace(0, 1)  # avoid div by zero
    prob_df = df.div(row_sums, axis=0)

    # Reset index for logomaker (needs 0..N-1)
    plot_df = prob_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(0.6 * len(positions), 4))
    logomaker.Logo(plot_df, ax=ax, color_scheme="chemistry",
                   stack_order="small_on_top")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], rotation=45)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AA Probability")
    ax.set_xlabel("Pocket Position")
    ax.set_title(f"{ligand_name} Top Designs — Pocket Logo (n={len(sequences)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Logo: {out_png}")


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
    parser.add_argument(
        "--copy-top-pdbs", type=int, default=20, metavar="N",
        help="Copy top N PDB files per ligand into filtered/top_pdbs_<lig>/ "
             "(default: 20, set to 0 to disable)")
    parser.add_argument(
        "--exclude-mutations", nargs='+', default=[], metavar="POS_AA",
        help="Exclude designs with specific mutations, e.g. 117F 59P "
             "(reads from PDB to check)")
    parser.add_argument(
        "--ref-pdb", default=None,
        help="Reference PDB (3QN1_H2O.pdb) for latch RMSD calculation. "
             "If not provided, latch RMSD is skipped.")
    parser.add_argument(
        "--gate-latch-rmsd", type=float, default=None, metavar="RMSD",
        help="Maximum latch loop RMSD (res 114-118) in Angstroms. "
             "Designs above this are removed before ranking. "
             "Requires --ref-pdb. (e.g., 2.0)")
    parser.add_argument(
        "--gate-all-oh-satisfied", action="store_true",
        help="Require all ligand hydroxyl O atoms to have a protein H-bond "
             "partner within 3.5 A (n_oh_unsatisfied == 0). "
             "Superseded by --gate-max-unsatisfied-oh if both given.")
    parser.add_argument(
        "--gate-max-unsatisfied-oh", type=int, default=None, metavar="N",
        help="Allow at most N unsatisfied hydroxyl O atoms. Use 1 to "
             "permit the 3-OH (water-coordinating) to be unsatisfied "
             "while requiring all other OHs to have protein partners.")
    parser.add_argument(
        "--gate-coo-satisfied", action="store_true",
        help="Require at least 1 carboxylate O atom to have a protein "
             "H-bond partner within 3.5 A")

    args = parser.parse_args()

    if args.gate_latch_rmsd is not None and args.ref_pdb is None:
        parser.error("--gate-latch-rmsd requires --ref-pdb")

    root = Path(args.expansion_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse exclusion mutations (e.g., "117F" -> {117: 'F'})
    exclude_muts = {}
    for mut in args.exclude_mutations:
        pos = int(mut[:-1])
        aa = mut[-1].upper()
        exclude_muts[pos] = aa

    # Resolve OH gate: --gate-max-unsatisfied-oh supersedes --gate-all-oh-satisfied
    if args.gate_max_unsatisfied_oh is not None:
        max_unsatisfied_oh = args.gate_max_unsatisfied_oh
    elif args.gate_all_oh_satisfied:
        max_unsatisfied_oh = 0
    else:
        max_unsatisfied_oh = None

    print(f"Strategy H (relaxed) gates:")
    print(f"  pLDDT_ligand >= {args.gate_plddt}")
    print(f"  H-bond dist  <= {args.gate_hbond} A")
    if max_unsatisfied_oh is not None:
        if max_unsatisfied_oh == 0:
            print(f"  All ligand OH must be satisfied (H-bond <= 3.5 A)")
        else:
            print(f"  Max {max_unsatisfied_oh} unsatisfied OH allowed "
                  f"(H-bond <= 3.5 A)")
    if args.gate_coo_satisfied:
        print(f"  At least 1 COO O must be satisfied (H-bond <= 3.5 A)")
    if args.gate_latch_rmsd is not None:
        print(f"  Latch RMSD   <= {args.gate_latch_rmsd} A")
    print(f"  Rank by: binary_plddt_pocket (descending)")
    print(f"  Top N: {args.top_n} per ligand")
    print(f"  R116 flip threshold: < {args.r116_flip_threshold} A")
    if exclude_muts:
        excl_str = ", ".join(f"{p}{a}" for p, a in sorted(exclude_muts.items()))
        print(f"  Excluding: {excl_str}")
    print()

    # ── Clean previous outputs ──
    for old_file in out_dir.glob(f"top{args.top_n}_*.csv"):
        old_file.unlink()
    for old_file in out_dir.glob("combined_top*.csv"):
        old_file.unlink()
    for old_file in out_dir.glob("logo_*.png"):
        old_file.unlink()
    for old_file in out_dir.glob("top*.fasta"):
        old_file.unlink()
    for pdb_dir in out_dir.glob("top_pdbs_*"):
        if pdb_dir.is_dir():
            shutil.rmtree(pdb_dir)

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

        # ── Exclude specific mutations ──
        if exclude_muts:
            pre_excl = len(gated)
            filtered = []
            for row in gated:
                pdb_path = find_pdb_path(str(root), lig, row['name'])
                excluded = False
                if pdb_path:
                    for pos, bad_aa in exclude_muts.items():
                        actual_aa = get_residue_from_pdb(str(pdb_path), pos)
                        if actual_aa == bad_aa:
                            excluded = True
                            break
                if not excluded:
                    filtered.append(row)
            n_excl = pre_excl - len(filtered)
            gated = filtered
            excl_str = ", ".join(f"{p}{a}" for p, a in sorted(exclude_muts.items()))
            print(f"  Excluded {n_excl} designs with {excl_str} "
                  f"({len(gated)} remaining)")

        # ── Structural gates (OH satisfaction, latch RMSD) ──
        # These require PDB file access, so computed here before ranking.
        needs_oh_gate = max_unsatisfied_oh is not None or args.gate_coo_satisfied
        needs_latch_gate = args.gate_latch_rmsd is not None

        if needs_oh_gate or needs_latch_gate:
            pre_struct = len(gated)
            struct_filtered = []
            n_fail_oh = 0
            n_fail_coo = 0
            n_fail_latch = 0
            n_no_pdb = 0

            for row in gated:
                pdb_path = find_pdb_path(str(root), lig, row['name'])
                if not pdb_path:
                    n_no_pdb += 1
                    continue

                # OH / COO satisfaction
                if needs_oh_gate:
                    oh = compute_oh_satisfaction(str(pdb_path))
                    row['n_oh'] = oh['n_oh']
                    row['n_oh_satisfied'] = oh['n_oh_satisfied']
                    row['n_oh_unsatisfied'] = oh['n_oh_unsatisfied']
                    row['oh_min_dists'] = ';'.join(
                        str(d) for d in oh['oh_min_dists'])
                    row['n_coo_satisfied'] = oh['n_coo_satisfied']

                    if max_unsatisfied_oh is not None:
                        if oh['n_oh_unsatisfied'] is None or \
                                oh['n_oh_unsatisfied'] > max_unsatisfied_oh:
                            n_fail_oh += 1
                            continue
                    if args.gate_coo_satisfied:
                        if oh['n_coo_satisfied'] is None or oh['n_coo_satisfied'] < 1:
                            n_fail_coo += 1
                            continue

                # Latch RMSD
                if needs_latch_gate:
                    latch = compute_latch_rmsd(str(pdb_path), args.ref_pdb)
                    row['latch_rmsd'] = latch['latch_rmsd']
                    row['latch_ca_rmsd'] = latch['latch_ca_rmsd']

                    if latch['latch_rmsd'] is None or \
                            latch['latch_rmsd'] > args.gate_latch_rmsd:
                        n_fail_latch += 1
                        continue

                struct_filtered.append(row)

            print(f"\n  Structural gates ({pre_struct} → {len(struct_filtered)}):")
            if max_unsatisfied_oh is not None:
                print(f"    Fail OH satisfaction:  {n_fail_oh:>5}")
            if args.gate_coo_satisfied:
                print(f"    Fail COO satisfaction: {n_fail_coo:>5}")
            if needs_latch_gate:
                print(f"    Fail latch RMSD:      {n_fail_latch:>5}")
            if n_no_pdb:
                print(f"    No PDB found:         {n_no_pdb:>5}")
            print(f"    Pass structural:      {len(struct_filtered):>5}")

            gated = struct_filtered

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

        # ── OH satisfaction analysis ──
        # If structural gates were active, OH data is already on rows.
        # Otherwise, compute it now for reporting.
        if top:
            if not needs_oh_gate:
                print(f"\n  Checking ligand OH satisfaction (H-bond cutoff 3.5 A)...")
                for row in top:
                    if row.get('n_oh') is not None:
                        continue  # already computed
                    pdb_path = find_pdb_path(str(root), lig, row['name'])
                    if not pdb_path:
                        continue
                    oh = compute_oh_satisfaction(str(pdb_path))
                    if oh['n_oh'] is not None:
                        row['n_oh'] = oh['n_oh']
                        row['n_oh_satisfied'] = oh['n_oh_satisfied']
                        row['n_oh_unsatisfied'] = oh['n_oh_unsatisfied']
                        row['oh_min_dists'] = ';'.join(
                            str(d) for d in oh['oh_min_dists'])
                        row['n_coo_satisfied'] = oh['n_coo_satisfied']

            # Report OH stats
            all_unsat_counts = [r.get('n_oh_unsatisfied')
                                for r in top
                                if r.get('n_oh_unsatisfied') is not None]
            if all_unsat_counts:
                n_checked = len(all_unsat_counts)
                n_all_sat = sum(1 for u in all_unsat_counts if u == 0)
                n_any_unsat = sum(1 for u in all_unsat_counts if u > 0)
                mean_unsat = sum(all_unsat_counts) / n_checked
                oh_counts = [r.get('n_oh', 0) for r in top
                             if r.get('n_oh') is not None]
                typical_oh = (max(set(oh_counts), key=oh_counts.count)
                              if oh_counts else '?')
                label = "(gated)" if needs_oh_gate else "(analysis only)"
                print(f"\n  OH satisfaction {label} "
                      f"({typical_oh} OH groups per ligand):")
                print(f"    All OH satisfied:    {n_all_sat:>4} / {n_checked} "
                      f"({100 * n_all_sat / n_checked:.1f}%)")
                print(f"    Has unsatisfied OH:  {n_any_unsat:>4} / {n_checked} "
                      f"({100 * n_any_unsat / n_checked:.1f}%)")
                print(f"    Mean unsatisfied OH: {mean_unsat:.2f}")
                from collections import Counter as Ctr
                unsat_dist = Ctr(all_unsat_counts)
                for k in sorted(unsat_dist):
                    print(f"      {k} unsatisfied: {unsat_dist[k]} designs")

        # ── Latch loop RMSD ──
        # If latch gate was active, data is already on rows.
        # Otherwise, compute it now for reporting.
        if args.ref_pdb and top:
            if not needs_latch_gate:
                print(f"\n  Checking latch loop RMSD (res 114-118 vs 3QN1)...")
                for row in top:
                    if row.get('latch_rmsd') is not None:
                        continue  # already computed
                    pdb_path = find_pdb_path(str(root), lig, row['name'])
                    if not pdb_path:
                        continue
                    latch = compute_latch_rmsd(str(pdb_path), args.ref_pdb)
                    row['latch_rmsd'] = latch['latch_rmsd']
                    row['latch_ca_rmsd'] = latch['latch_ca_rmsd']

            latch_rmsds = [r.get('latch_rmsd') for r in top
                           if r.get('latch_rmsd') is not None]
            if latch_rmsds:
                mean_lr = sum(latch_rmsds) / len(latch_rmsds)
                n_low = sum(1 for r in latch_rmsds if r < 1.0)
                n_med = sum(1 for r in latch_rmsds if 1.0 <= r < 2.0)
                n_high = sum(1 for r in latch_rmsds if r >= 2.0)
                n_lr = len(latch_rmsds)
                label = "(gated)" if needs_latch_gate else "(analysis only)"
                print(f"\n  Latch RMSD {label}:")
                print(f"    < 1.0 A (good):       {n_low:>4} / {n_lr} "
                      f"({100 * n_low / n_lr:.1f}%)")
                print(f"    1.0-2.0 A (moderate):  {n_med:>4} / {n_lr} "
                      f"({100 * n_med / n_lr:.1f}%)")
                print(f"    > 2.0 A (deformed):    {n_high:>4} / {n_lr} "
                      f"({100 * n_high / n_lr:.1f}%)")
                print(f"    Range: {min(latch_rmsds):.2f} - "
                      f"{max(latch_rmsds):.2f} A (mean {mean_lr:.2f})")

        # ── Sequence logo ──
        if args.extract_sequences and top:
            seqs = [r['sequence'] for r in top if r.get('sequence')]
            if seqs:
                logo_path = out_dir / f"logo_{lig}.png"
                plot_pocket_logo(seqs, lig.upper(), str(logo_path))

        # ── Copy top N PDBs ──
        if args.copy_top_pdbs > 0 and top:
            pdb_dir = out_dir / f"top_pdbs_{lig}"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            n_copy = min(args.copy_top_pdbs, len(top))
            n_copied = 0
            for i, row in enumerate(top[:n_copy]):
                src = find_pdb_path(str(root), lig, row['name'])
                if src:
                    # Name with rank for easy sorting
                    dst = pdb_dir / f"rank{i + 1:03d}_{row['name']}.pdb"
                    shutil.copy2(str(src), str(dst))
                    n_copied += 1
            print(f"\n  Copied {n_copied} / {n_copy} PDBs to {pdb_dir}")

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
                'n_oh', 'n_oh_satisfied', 'n_oh_unsatisfied',
                'oh_min_dists', 'n_coo_satisfied',
                'latch_rmsd', 'latch_ca_rmsd',
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
