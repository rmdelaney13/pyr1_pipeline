#!/usr/bin/env python3
"""Expansion readiness analysis: convergence, pose clustering, and selection.

Determines whether more LigandMPNN expansion rounds are needed and selects
a maximally diverse pool of designs for Twist ordering.

Four analyses:
  1. Convergence diagnostics - score plateau, penetration, entropy
  2. Ligand pose clustering - RMSD-based grouping of binding modes
  3. OH-satisfaction fingerprinting - which pocket AAs satisfy each OH
  4. Two-level diversity selection - pose clusters x Hamming sub-clusters

Usage:
    python scripts/analyze_expansion_readiness.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --initial-csv-dir /scratch/alpine/ryde3462/boltz_bile_acids/csvs \
        --ref-pdb /path/to/3QN1_H2O.pdb \
        --ligands ca cdca dca \
        --top-n 300 \
        --select 155 \
        --pose-rmsd-cutoff 1.5 \
        --min-hamming 3 \
        --plot \
        --out-dir /scratch/alpine/ryde3462/expansion/selection

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Import from existing pipeline scripts
sys.path.insert(0, str(Path(__file__).parent))
from expansion_pocket_analysis import (
    WT_PYR1_SEQUENCE,
    POCKET_POSITIONS,
    build_signature_lookup,
    pocket_from_signature,
    wt_at_position,
)
from analyze_pocket_evolution import (
    compute_oh_contacts_detailed,
    find_pdb_for_design,
    greedy_hamming_cluster,
    hamming_distance,
    pocket_string,
    shannon_entropy,
)
from filter_expansion_designs import (
    load_scored_csv,
    extract_sequence_from_pdb,
    WATER_MEDIATED_OH,
)

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


# ---------------------------------------------------------------------------
# Section 1: Convergence Diagnostics
# ---------------------------------------------------------------------------

def parse_round_from_name(name):
    """Extract round number from design name."""
    import re
    m = re.search(r'_exp_r(\d+)_', str(name))
    if m:
        return int(m.group(1))
    return 0  # initial designs


def load_all_round_data(expansion_root, lig):
    """Load per-round CSVs and tag each row with round number."""
    lig_dir = Path(expansion_root) / lig
    all_rows = []

    # Round 0: initial scores
    round0_csv = lig_dir / "round_0" / "scores.csv"
    if round0_csv.exists():
        rows = load_scored_csv(str(round0_csv))
        for r in rows:
            r['_round'] = 0
        all_rows.extend(rows)

    # Subsequent rounds
    for round_dir in sorted(lig_dir.glob("round_*")):
        round_num = round_dir.name.replace("round_", "")
        try:
            round_num = int(round_num)
        except ValueError:
            continue
        if round_num == 0:
            continue

        new_csv = round_dir / "new_scores.csv"
        if new_csv.exists():
            rows = load_scored_csv(str(new_csv))
            for r in rows:
                r['_round'] = round_num
            all_rows.extend(rows)

    # Fallback: if no per-round CSVs, try boltz_scored.csv
    if not all_rows:
        scored = lig_dir / "boltz_scored.csv"
        if scored.exists():
            rows = load_scored_csv(str(scored))
            for r in rows:
                r['_round'] = parse_round_from_name(r.get('name', ''))
            all_rows.extend(rows)

    return all_rows


def get_score(row, score_col='binary_total_score'):
    """Safely extract numeric score."""
    v = row.get(score_col)
    if v is None:
        return -999.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return -999.0


def compute_convergence(all_rows, score_col='binary_total_score', top_n=200,
                        sig_lookup=None):
    """Compute convergence diagnostics across rounds.

    Args:
        all_rows: list of dicts with scores and _round
        score_col: column to rank by
        top_n: pool size for penetration/entropy analysis
        sig_lookup: dict {name: variant_signature} for pocket extraction.
                    If None, falls back to variant_signature in rows (often missing).

    Returns dict with:
      round_scores: {round -> {rank -> score}} for frontier tracking
      penetration: {round -> fraction of new designs in cumulative top-N}
      entropy: {round -> mean Shannon entropy across 16 pocket positions}
      unique_pockets: {round -> count of unique pocket seqs in top-N}
      verdict: CONVERGED | DIMINISHING RETURNS | STILL IMPROVING
    """
    if not all_rows:
        return None

    if sig_lookup is None:
        sig_lookup = {}

    rounds = sorted(set(r['_round'] for r in all_rows))

    # Build cumulative pool round by round
    cumulative = []
    round_scores = {}
    penetration = {}
    entropy_per_round = {}
    unique_pockets_per_round = {}

    rank_checkpoints = [1, 10, 50, 100, 165, top_n]

    for rnd in rounds:
        # Add this round's designs
        new_rows = [r for r in all_rows if r['_round'] == rnd]
        cumulative.extend(new_rows)

        # Sort cumulative by score
        cumulative.sort(key=lambda r: get_score(r, score_col), reverse=True)

        # Score at rank checkpoints
        scores_at_ranks = {}
        for rank in rank_checkpoints:
            if rank <= len(cumulative):
                scores_at_ranks[rank] = get_score(cumulative[rank - 1], score_col)
        round_scores[rnd] = scores_at_ranks

        # Penetration: fraction of new designs in cumulative top-N
        top_names = set(r.get('name', '') for r in cumulative[:top_n])
        new_names = set(r.get('name', '') for r in new_rows)
        n_new_in_top = len(top_names & new_names)
        penetration[rnd] = n_new_in_top / max(len(new_rows), 1)

        # Entropy and unique pockets in top-N
        top_pool = cumulative[:min(top_n, len(cumulative))]
        pocket_seqs = []
        for row in top_pool:
            name = row.get('name', '')
            # Try sig_lookup first, then row's variant_signature
            sig = sig_lookup.get(name, '')
            if not sig:
                sig = row.get('variant_signature', '')
            pocket = pocket_from_signature(sig)
            pocket_seqs.append(pocket_string(pocket))

        # Shannon entropy per position
        entropies = []
        for i, pos in enumerate(POCKET_POSITIONS):
            counter = Counter(ps[i] for ps in pocket_seqs if len(ps) > i)
            entropies.append(shannon_entropy(counter))
        entropy_per_round[rnd] = np.mean(entropies) if entropies else 0.0

        unique_pockets_per_round[rnd] = len(set(pocket_seqs))

    # Verdict logic
    # Three independent signals: score plateau, penetration, entropy
    # A signal is only meaningful if the data supports it
    verdict = "STILL IMPROVING"
    reasons = []

    if len(rounds) >= 3:
        # Signal 1: Score plateau at rank 165
        # Use absolute delta relative to score range, not percentage
        recent_scores = []
        for rnd in rounds[-3:]:
            s = round_scores[rnd].get(165, round_scores[rnd].get(100))
            if s is not None:
                recent_scores.append(s)

        score_plateau = False
        if len(recent_scores) >= 3:
            # Compare to full score range for context
            all_165 = [round_scores[r].get(165, round_scores[r].get(100))
                       for r in rounds]
            all_165 = [s for s in all_165 if s is not None]
            score_range = max(all_165) - min(all_165) if len(all_165) >= 2 else 1.0

            delta_1 = abs(recent_scores[-1] - recent_scores[-2])
            delta_2 = abs(recent_scores[-2] - recent_scores[-3])

            # Plateau if last 2 deltas are each <5% of total improvement range
            if score_range > 0 and delta_1 / score_range < 0.05 and delta_2 / score_range < 0.05:
                score_plateau = True
                reasons.append(
                    f"Score at rank 165 plateau: last 3 rounds "
                    f"{recent_scores[-3]:.3f} -> {recent_scores[-2]:.3f} -> {recent_scores[-1]:.3f} "
                    f"(deltas {delta_2:.3f}, {delta_1:.3f} vs range {score_range:.3f})")

        # Signal 2: Penetration
        recent_pen = [penetration[r] for r in rounds[-2:]]
        low_penetration = all(p < 0.05 for p in recent_pen)
        high_penetration = any(p > 0.15 for p in recent_pen)

        if low_penetration:
            reasons.append(f"<5% new designs entering top-{top_n} in last 2 rounds")
        elif high_penetration:
            reasons.append(
                f"Penetration still high ({recent_pen[-1]:.1%} in last round) "
                f"- expansion still productive")

        # Signal 3: Entropy (only flag if we have valid signatures)
        recent_ent = entropy_per_round[rounds[-1]]
        n_unique = unique_pockets_per_round[rounds[-1]]
        if n_unique <= 1 and not sig_lookup:
            reasons.append("Pocket entropy unavailable (no variant signatures in CSVs)")
        elif recent_ent < 0.5 and n_unique > 1:
            reasons.append(f"Mean pocket entropy = {recent_ent:.2f} bits (low diversity)")

        # Combined verdict
        # CONVERGED: score plateau AND low penetration
        # DIMINISHING RETURNS: score plateau OR low penetration (but not both)
        # Override: high penetration always means STILL IMPROVING
        if high_penetration:
            verdict = "STILL IMPROVING"
        elif score_plateau and low_penetration:
            verdict = "CONVERGED"
        elif score_plateau or low_penetration:
            verdict = "DIMINISHING RETURNS"
        else:
            verdict = "STILL IMPROVING"

    return {
        'round_scores': round_scores,
        'penetration': penetration,
        'entropy': entropy_per_round,
        'unique_pockets': unique_pockets_per_round,
        'rounds': rounds,
        'verdict': verdict,
        'reasons': reasons,
        'total_designs': len(cumulative),
    }


def print_convergence_report(conv, lig):
    """Print convergence diagnostics to console."""
    if conv is None:
        print(f"\n  {lig.upper()}: No data available for convergence analysis")
        return

    print(f"\n{'='*72}")
    print(f"  CONVERGENCE ANALYSIS: {lig.upper()}")
    print(f"  Total designs in pool: {conv['total_designs']}")
    print(f"{'='*72}")

    # Score frontier table
    print(f"\n  Score Frontier (binary_total_score at rank):")
    ranks = [1, 10, 50, 100, 165]
    header = f"  {'Round':>6}"
    for r in ranks:
        header += f"  {'Rank'+str(r):>8}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*8}" * len(ranks))

    for rnd in conv['rounds']:
        line = f"  {rnd:>6}"
        for r in ranks:
            s = conv['round_scores'][rnd].get(r)
            if s is not None:
                line += f"  {s:>8.3f}"
            else:
                line += f"  {'N/A':>8}"
        print(line)

    # Penetration
    print(f"\n  New Design Penetration (fraction entering top-200):")
    for rnd in conv['rounds']:
        pen = conv['penetration'][rnd]
        bar = '#' * int(pen * 50)
        print(f"    Round {rnd:>2}: {pen:>6.1%} {bar}")

    # Entropy
    print(f"\n  Mean Pocket Entropy (bits, top-200):")
    for rnd in conv['rounds']:
        ent = conv['entropy'][rnd]
        print(f"    Round {rnd:>2}: {ent:.2f} bits")

    # Unique pockets
    print(f"\n  Unique Pocket Sequences in Top-200:")
    for rnd in conv['rounds']:
        n = conv['unique_pockets'][rnd]
        print(f"    Round {rnd:>2}: {n}")

    # Verdict
    print(f"\n  {'='*40}")
    print(f"  VERDICT: {conv['verdict']}")
    for reason in conv['reasons']:
        print(f"    - {reason}")
    if not conv['reasons']:
        print(f"    - Scores still improving and design space not yet saturated")
    print(f"  {'='*40}")


def plot_convergence(conv, lig, out_dir):
    """Generate convergence diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Convergence Diagnostics: {lig.upper()}", fontsize=14)

    rounds = conv['rounds']

    # Panel 1: Score frontier
    ax = axes[0, 0]
    for rank in [1, 10, 50, 100, 165]:
        scores = [conv['round_scores'][r].get(rank) for r in rounds]
        valid_rounds = [r for r, s in zip(rounds, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            ax.plot(valid_rounds, valid_scores, 'o-', label=f'Rank {rank}', markersize=4)
    ax.set_xlabel('Round')
    ax.set_ylabel('binary_total_score')
    ax.set_title('Score Frontier')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Penetration
    ax = axes[0, 1]
    pens = [conv['penetration'][r] for r in rounds]
    ax.bar(rounds, pens, color='steelblue', alpha=0.7)
    ax.axhline(0.05, color='red', ls='--', alpha=0.5, label='5% threshold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Fraction in top-200')
    ax.set_title('New Design Penetration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Entropy
    ax = axes[1, 0]
    ents = [conv['entropy'][r] for r in rounds]
    ax.plot(rounds, ents, 'o-', color='darkgreen', markersize=5)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Low diversity threshold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean Shannon entropy (bits)')
    ax.set_title('Pocket Diversity (top-200)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Unique pockets
    ax = axes[1, 1]
    uniq = [conv['unique_pockets'][r] for r in rounds]
    ax.plot(rounds, uniq, 'o-', color='darkorange', markersize=5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Unique pocket sequences')
    ax.set_title('Pocket Uniqueness (top-200)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / f"convergence_{lig}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Section 2: Ligand Pose Clustering
# ---------------------------------------------------------------------------

def parse_pdb_coords(pdb_path, protein_chain='A', ligand_chain='B'):
    """Extract protein CA coords and ligand heavy-atom coords from PDB.

    Returns:
        ca_coords: dict {resnum: np.array([x,y,z])}
        ligand_atoms: list of (np.array([x,y,z]), element_symbol)
    """
    ca_coords = {}
    ligand_atoms = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resnum_str = line[22:26].strip()
            try:
                resnum = int(resnum_str)
            except ValueError:
                continue

            if ch == protein_chain:
                if atom_name == 'CA' and resnum not in ca_coords:
                    ca_coords[resnum] = np.array([x, y, z])
            elif ch == ligand_chain:
                # Heavy atoms only (skip H)
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
                if elem == 'H':
                    continue
                ligand_atoms.append((np.array([x, y, z]), elem))

    return ca_coords, ligand_atoms


def kabsch_rotation(P, Q):
    """Compute optimal rotation matrix (Kabsch algorithm).

    P, Q: Nx3 arrays (P is mobile, Q is fixed).
    Returns: 3x3 rotation matrix R such that R @ P.T ~ Q.T
    """
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def align_and_get_ligand_coords(ca_coords, ligand_atoms, ref_ca_coords):
    """Align structure to reference by CA atoms and return transformed ligand coords.

    Args:
        ca_coords: dict {resnum: xyz} for this structure
        ligand_atoms: list of (xyz, element) for this structure
        ref_ca_coords: dict {resnum: xyz} for reference

    Returns:
        aligned_ligand: list of (xyz_aligned, element) or None if alignment fails
    """
    common = sorted(set(ca_coords) & set(ref_ca_coords))
    if len(common) < 10:
        return None

    mobile = np.array([ca_coords[r] for r in common])
    fixed = np.array([ref_ca_coords[r] for r in common])

    # Center
    mobile_center = mobile.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    P = mobile - mobile_center
    Q = fixed - fixed_center

    R = kabsch_rotation(P, Q)

    # Transform ligand
    aligned = []
    for coord, elem in ligand_atoms:
        new_coord = R @ (coord - mobile_center) + fixed_center
        aligned.append((new_coord, elem))

    return aligned


def ligand_rmsd_hungarian(lig1, lig2):
    """Compute element-matched ligand RMSD using Hungarian algorithm.

    lig1, lig2: lists of (xyz, element)
    Returns: RMSD (float) or None if incompatible
    """
    from scipy.optimize import linear_sum_assignment

    if not lig1 or not lig2:
        return None

    # Group by element
    elems1 = defaultdict(list)
    elems2 = defaultdict(list)
    for coord, elem in lig1:
        elems1[elem].append(coord)
    for coord, elem in lig2:
        elems2[elem].append(coord)

    # Check element compatibility
    common_elems = set(elems1) & set(elems2)
    if not common_elems:
        return None

    total_sd = 0.0
    total_n = 0

    for elem in common_elems:
        coords1 = np.array(elems1[elem])
        coords2 = np.array(elems2[elem])
        n1, n2 = len(coords1), len(coords2)
        n = min(n1, n2)

        # Cost matrix: pairwise squared distances
        cost = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                cost[i, j] = np.sum((coords1[i] - coords2[j]) ** 2)

        row_idx, col_idx = linear_sum_assignment(cost)
        total_sd += sum(cost[r, c] for r, c in zip(row_idx[:n], col_idx[:n]))
        total_n += n

    if total_n == 0:
        return None

    return math.sqrt(total_sd / total_n)


def compute_pose_clusters(design_names, lig, expansion_root, ref_pdb,
                          rmsd_cutoff=1.5, score_col='binary_total_score',
                          rows_by_name=None):
    """Cluster designs by ligand pose RMSD after CA alignment to reference.

    Returns:
        cluster_ids: list of cluster assignments (same order as design_names)
        cluster_info: dict {cluster_id: {size, best_name, best_score, members}}
        pdb_found: list of booleans (which designs had PDBs)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Parse reference
    ref_ca, _ = parse_pdb_coords(str(ref_pdb))

    # Load and align all ligand coordinates
    print(f"    Loading and aligning {len(design_names)} PDBs...")
    aligned_ligands = []  # (index, aligned_coords)
    valid_indices = []
    pdb_found = [False] * len(design_names)

    for i, name in enumerate(design_names):
        pdb_path = find_pdb_for_design(name, lig, expansion_root)
        if pdb_path is None:
            continue
        pdb_found[i] = True

        ca, lig_atoms = parse_pdb_coords(str(pdb_path))
        aligned = align_and_get_ligand_coords(ca, lig_atoms, ref_ca)
        if aligned is None:
            continue

        aligned_ligands.append((i, aligned))
        valid_indices.append(i)

    n = len(aligned_ligands)
    print(f"    Successfully aligned {n}/{len(design_names)} structures")

    if n < 2:
        # Can't cluster with <2 structures
        cluster_ids = [0] * len(design_names)
        return cluster_ids, {0: {'size': n, 'members': valid_indices}}, pdb_found

    # Compute pairwise RMSD matrix
    print(f"    Computing {n*(n-1)//2} pairwise ligand RMSDs...")
    dist_matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            rmsd = ligand_rmsd_hungarian(aligned_ligands[a][1], aligned_ligands[b][1])
            if rmsd is None:
                rmsd = 10.0  # large default for incompatible
            dist_matrix[a, b] = rmsd
            dist_matrix[b, a] = rmsd

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=rmsd_cutoff, criterion='distance')

    # Map back to full design list
    cluster_ids = [-1] * len(design_names)
    for idx_in_valid, (orig_idx, _) in enumerate(aligned_ligands):
        cluster_ids[orig_idx] = int(labels[idx_in_valid]) - 1  # 0-indexed

    # Assign unaligned designs to cluster -1 (will be handled separately)

    # Build cluster info
    cluster_info = defaultdict(lambda: {'size': 0, 'members': [], 'best_score': -999,
                                         'best_name': ''})
    for i, cid in enumerate(cluster_ids):
        if cid < 0:
            continue
        cluster_info[cid]['size'] += 1
        cluster_info[cid]['members'].append(i)
        if rows_by_name:
            name = design_names[i]
            score = get_score(rows_by_name.get(name, {}), score_col)
            if score > cluster_info[cid]['best_score']:
                cluster_info[cid]['best_score'] = score
                cluster_info[cid]['best_name'] = name

    return cluster_ids, dict(cluster_info), pdb_found


def print_pose_cluster_report(cluster_info, lig):
    """Print pose cluster summary."""
    print(f"\n  POSE CLUSTERS: {lig.upper()}")
    print(f"  {'Cluster':>8}  {'Size':>6}  {'Best Score':>11}  Best Design")
    print(f"  {'-'*8}  {'-'*6}  {'-'*11}  {'-'*30}")
    for cid in sorted(cluster_info):
        info = cluster_info[cid]
        print(f"  {cid:>8}  {info['size']:>6}  {info['best_score']:>11.3f}  {info['best_name']}")


# ---------------------------------------------------------------------------
# Section 3: OH-Satisfaction Fingerprinting
# ---------------------------------------------------------------------------

def compute_oh_fingerprint(pdb_path, lig=None, protein_chain='A',
                            ligand_chain='B', hbond_cutoff=3.5):
    """Compute OH-satisfaction fingerprint for a design.

    Water-mediated OHs (looked up from WATER_MEDIATED_OH by ligand name) are
    labeled "->WATER" and do NOT count as unsatisfied.

    Returns a string like "OH2->S92|OH3->WATER|OH4->N120" or None if PDB not found.
    Also returns per-OH details list (with 'water_mediated' flag added).
    """
    contacts = compute_oh_contacts_detailed(str(pdb_path), protein_chain,
                                             ligand_chain, hbond_cutoff)
    if not contacts:
        return None, []

    water_set = WATER_MEDIATED_OH.get(lig.lower(), set()) if lig else set()

    parts = []
    for c in contacts:
        oh_idx = c['oh_index']
        is_water = oh_idx in water_set
        c['water_mediated'] = is_water

        if is_water:
            parts.append(f"OH{oh_idx}->WATER")
        elif c['satisfied']:
            resname = THREE_TO_ONE.get(c['nearest_resname'], c['nearest_resname'])
            parts.append(f"OH{oh_idx}->{resname}{c['nearest_resnum']}")
        else:
            parts.append(f"OH{oh_idx}->NONE")

    fingerprint = "|".join(parts)
    return fingerprint, contacts


def run_oh_fingerprinting(design_names, lig, expansion_root, rows_by_name=None):
    """Compute OH fingerprints for all designs.

    Returns:
        fingerprints: dict {name: fingerprint_string}
        fingerprint_groups: dict {fingerprint: [names]}
    """
    print(f"\n  OH FINGERPRINTING: {lig.upper()}")

    fingerprints = {}
    fingerprint_groups = defaultdict(list)
    n_found = 0
    n_total = len(design_names)

    for name in design_names:
        pdb_path = find_pdb_for_design(name, lig, expansion_root)
        if pdb_path is None:
            continue
        n_found += 1

        fp, details = compute_oh_fingerprint(str(pdb_path), lig=lig)
        if fp is None:
            continue

        fingerprints[name] = fp
        fingerprint_groups[fp].append(name)

    water_set = WATER_MEDIATED_OH.get(lig.lower(), set()) if lig else set()
    print(f"    PDBs found: {n_found}/{n_total}")
    if water_set:
        print(f"    Water-mediated OHs (excluded from satisfaction): "
              f"{', '.join(f'OH{i}' for i in sorted(water_set))}")
    print(f"    Unique OH fingerprints: {len(fingerprint_groups)}")

    # Print fingerprint table
    print(f"\n    {'Fingerprint':<50}  {'Count':>6}  {'Status'}")
    print(f"    {'-'*50}  {'-'*6}  {'-'*15}")
    for fp, names in sorted(fingerprint_groups.items(), key=lambda x: -len(x[1])):
        status = ""
        if len(names) < 3:
            status = "UNDERSAMPLED"
        elif len(names) > 30:
            status = "well-sampled"
        else:
            status = "adequate"
        print(f"    {fp:<50}  {len(names):>6}  {status}")

    return fingerprints, dict(fingerprint_groups)


# ---------------------------------------------------------------------------
# Section 3.5: Steroid Core Geometry Validation
# ---------------------------------------------------------------------------

def parse_ligand_atoms(pdb_path, ligand_chain='B'):
    """Extract ligand heavy atom coordinates, names, and elements."""
    atoms = []  # (coord, atom_name, element)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            if line[21] != ligand_chain:
                continue
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = atom_name[0]
            atoms.append((np.array([x, y, z]), atom_name, elem))
    return atoms


def find_ring_atoms(atoms, bond_cutoff_cc=1.65, bond_cutoff_co=1.55):
    """Identify ring atoms in a ligand using cycle detection on bond graph.

    Returns set of indices into atoms list that are part of ring systems.
    """
    n = len(atoms)
    # Build adjacency list from bond distances
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(atoms[i][0] - atoms[j][0])
            ei, ej = atoms[i][2], atoms[j][2]
            if ei == 'C' and ej == 'C' and d < bond_cutoff_cc:
                adj[i].add(j)
                adj[j].add(i)
            elif ((ei == 'C' and ej == 'O') or (ei == 'O' and ej == 'C')) \
                    and d < bond_cutoff_co:
                adj[i].add(j)
                adj[j].add(i)

    # Find all atoms in cycles using DFS
    ring_atoms = set()
    for start in range(n):
        if atoms[start][2] != 'C':
            continue
        # BFS to find cycles
        visited = {}
        queue = [(start, -1, [start])]
        while queue:
            node, parent, path = queue.pop(0)
            if node in visited:
                # Found a cycle — all atoms in the path are ring atoms
                if len(path) >= 4:
                    ring_atoms.update(path)
                continue
            visited[node] = True
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited and len(path) >= 3:
                    ring_atoms.update(path)
                    ring_atoms.add(neighbor)
                elif neighbor not in visited:
                    queue.append((neighbor, node, path + [neighbor]))

    return ring_atoms


def kabsch_align(coords_mobile, coords_target):
    """Kabsch alignment. Returns rotation matrix R and translation.

    Aligns mobile onto target. Both must be Nx3 arrays.
    """
    center_m = coords_mobile.mean(axis=0)
    center_t = coords_target.mean(axis=0)
    m = coords_mobile - center_m
    t = coords_target - center_t
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    return R, center_m, center_t


def compute_ring_planarity(coords):
    """Compute planarity of a set of 3D coordinates.

    Returns:
        max_deviation: maximum distance of any atom from best-fit plane (A)
        mean_deviation: mean distance from best-fit plane (A)
    """
    if len(coords) < 4:
        return 0.0, 0.0
    center = coords.mean(axis=0)
    centered = coords - center
    # SVD to find best-fit plane (normal = smallest singular vector)
    _, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    deviations = np.abs(centered @ normal)
    return float(deviations.max()), float(deviations.mean())


def _internal_dist_matrix(coords):
    """Compute NxN pairwise distance matrix for a set of 3D coordinates."""
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def check_ligand_geometry(pdb_path, ref_atoms, ref_ring_indices,
                           ligand_chain='B'):
    """Check steroid core geometry against reference using internal distances.

    Uses pose-independent internal distance matrix RMSD to detect ring
    distortion/collapse without being confused by different binding poses.
    Also computes ICP-aligned RMSD after iterative closest point matching.

    Returns dict with metrics, or None if failed.
    """
    from scipy.optimize import linear_sum_assignment

    query_atoms = parse_ligand_atoms(pdb_path, ligand_chain)
    if not query_atoms:
        return None

    # Get reference ring carbon coords
    ref_c_indices = [i for i in ref_ring_indices if ref_atoms[i][2] == 'C']
    ref_c_coords = np.array([ref_atoms[i][0] for i in ref_c_indices])
    n_ref = len(ref_c_coords)
    if n_ref < 4:
        return None

    # Get ALL query carbons (ring detection may differ between ref and query
    # due to Boltz2 bond length variation, so match by shape not topology)
    query_carbons = [(i, a) for i, a in enumerate(query_atoms) if a[2] == 'C']
    query_c_coords = np.array([query_atoms[i][0] for i, _ in query_carbons])
    n_query = len(query_c_coords)
    if n_query < n_ref:
        return None

    # --- Internal distance matrix RMSD (pose-independent) ---
    # Compute internal distance matrices
    ref_dist = _internal_dist_matrix(ref_c_coords)

    # Hungarian match query carbons to ref ring carbons using internal
    # distance similarity: cost[i,j] = how different query atom j's distance
    # profile is from ref atom i's distance profile.
    # First, find the best n_ref query atoms by matching distance profiles.
    # For each candidate assignment of n_ref query atoms, compute the RMSD of
    # internal distance matrices. Use Hungarian on a cost matrix where
    # cost[i,j] = sum of |d_ref(i,k) - d_query(j, best_match(k))| over k.
    # Approximate: use ICP on coordinates to get good correspondences first.

    # ICP: iterative closest point alignment
    # Start with Hungarian on raw coordinates (rough initial match)
    cost_init = np.zeros((n_ref, n_query))
    for i in range(n_ref):
        for j in range(n_query):
            cost_init[i, j] = np.linalg.norm(ref_c_coords[i] -
                                               query_c_coords[j])
    row_idx, col_idx = linear_sum_assignment(cost_init)

    # Iterate: align → rematch → realign (3 iterations sufficient)
    matched_query = query_c_coords[col_idx]
    for _icp_iter in range(3):
        R, cq, cr = kabsch_align(matched_query, ref_c_coords)
        aligned_all = (query_c_coords - cq) @ R.T + cr

        # Re-match on aligned coordinates
        cost_aligned = np.zeros((n_ref, len(aligned_all)))
        for i in range(n_ref):
            for j in range(len(aligned_all)):
                cost_aligned[i, j] = np.linalg.norm(
                    ref_c_coords[i] - aligned_all[j])
        row_idx, col_idx = linear_sum_assignment(cost_aligned)
        matched_query = query_c_coords[col_idx]

    # Final alignment with converged correspondences
    R, cq, cr = kabsch_align(matched_query, ref_c_coords)
    aligned_matched = (matched_query - cq) @ R.T + cr

    # ICP-aligned RMSD (the "core_rmsd")
    diffs = aligned_matched - ref_c_coords
    core_rmsd = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

    # Internal distance RMSD (pose-independent shape comparison)
    query_matched_dist = _internal_dist_matrix(matched_query)
    dist_diffs = ref_dist - query_matched_dist
    # Use upper triangle only
    triu_idx = np.triu_indices(n_ref, k=1)
    shape_rmsd = float(np.sqrt(np.mean(dist_diffs[triu_idx] ** 2)))

    # Ring planarity
    max_plan, mean_plan = compute_ring_planarity(aligned_matched)
    ref_max_plan, ref_mean_plan = compute_ring_planarity(ref_c_coords)

    return {
        'core_rmsd': round(core_rmsd, 3),
        'shape_rmsd': round(shape_rmsd, 3),
        'max_planarity': round(max_plan, 3),
        'mean_planarity': round(mean_plan, 3),
        'ref_max_planarity': round(ref_max_plan, 3),
        'ref_mean_planarity': round(ref_mean_plan, 3),
        'planarity_ratio': round(max_plan / ref_max_plan, 3)
                           if ref_max_plan > 0.01 else None,
        'n_ring_atoms': n_ref,
    }


def run_geometry_check(design_names, lig, expansion_root, ref_ligand_pdb,
                        core_rmsd_cutoff=1.5, planarity_ratio_cutoff=0.5):
    """Check steroid core geometry for all designs against reference.

    Uses two metrics:
    - shape_rmsd: internal distance matrix RMSD (pose-independent, primary gate)
    - core_rmsd: ICP-aligned coordinate RMSD (secondary, for reporting)

    The shape_rmsd compares the internal pairwise distances of ring carbons,
    so it detects ring collapse/distortion without being confused by different
    binding poses.

    Args:
        ref_ligand_pdb: path to reference PDB with correct ligand geometry
        core_rmsd_cutoff: max shape RMSD to reference (A) before flagging
        planarity_ratio_cutoff: if planarity ratio < this, ligand is too flat

    Returns:
        geometry: dict {name: geometry_dict}
        n_distorted: number of designs flagged as distorted
    """
    print(f"\n  LIGAND GEOMETRY CHECK: {lig.upper()}")
    print(f"    Reference: {ref_ligand_pdb}")

    # Parse reference ligand
    ref_atoms = parse_ligand_atoms(ref_ligand_pdb)
    if not ref_atoms:
        print(f"    ERROR: no ligand atoms in reference PDB")
        return {}, 0

    ref_ring_indices = find_ring_atoms(ref_atoms)
    ref_c_indices = [i for i in ref_ring_indices if ref_atoms[i][2] == 'C']
    n_ring = len(ref_c_indices)
    print(f"    Reference ring carbons: {n_ring}")

    ref_c_coords = np.array([ref_atoms[i][0] for i in ref_c_indices])
    ref_max_plan, ref_mean_plan = compute_ring_planarity(ref_c_coords)
    print(f"    Reference planarity: max={ref_max_plan:.3f} A, "
          f"mean={ref_mean_plan:.3f} A")

    geometry = {}
    n_checked = 0
    n_distorted = 0
    n_flat = 0
    n_high_shape_rmsd = 0

    for name in design_names:
        pdb_path = find_pdb_for_design(name, lig, expansion_root)
        if pdb_path is None:
            continue

        geo = check_ligand_geometry(str(pdb_path), ref_atoms,
                                     ref_ring_indices)
        if geo is None:
            continue

        n_checked += 1
        geometry[name] = geo

        distorted = False
        if geo['shape_rmsd'] > core_rmsd_cutoff:
            distorted = True
            n_high_shape_rmsd += 1
        if geo['planarity_ratio'] is not None and \
                geo['planarity_ratio'] < planarity_ratio_cutoff:
            distorted = True
            n_flat += 1

        geo['distorted'] = distorted
        if distorted:
            n_distorted += 1

    print(f"    Checked: {n_checked}/{len(design_names)}")
    print(f"    Distorted (shape RMSD > {core_rmsd_cutoff} A): "
          f"{n_high_shape_rmsd}")
    print(f"    Flattened (planarity ratio < {planarity_ratio_cutoff}): "
          f"{n_flat}")
    print(f"    Total flagged: {n_distorted}")

    if geometry:
        shape_rmsds = [g['shape_rmsd'] for g in geometry.values()]
        core_rmsds = [g['core_rmsd'] for g in geometry.values()]
        print(f"    Shape RMSD range: {min(shape_rmsds):.3f} - "
              f"{max(shape_rmsds):.3f} A "
              f"(median {sorted(shape_rmsds)[len(shape_rmsds)//2]:.3f})")
        print(f"    ICP-aligned RMSD range: {min(core_rmsds):.3f} - "
              f"{max(core_rmsds):.3f} A "
              f"(median {sorted(core_rmsds)[len(core_rmsds)//2]:.3f})")
        ratios = [g['planarity_ratio'] for g in geometry.values()
                  if g['planarity_ratio'] is not None]
        if ratios:
            print(f"    Planarity ratio range: {min(ratios):.3f} - "
                  f"{max(ratios):.3f} (ref=1.0, <{planarity_ratio_cutoff}=flat)")

    return geometry, n_distorted


# ---------------------------------------------------------------------------
# Section 4: Two-Level Diversity Selection
# ---------------------------------------------------------------------------

def two_level_select(design_names, cluster_ids, pocket_seqs, scores,
                     fingerprints, n_select, min_hamming=3, max_cluster_frac=0.40):
    """Two-level diversity selection: pose clusters x Hamming sub-clusters.

    Level 1: Allocate slots proportional to pose cluster size, min 3, cap at
             max_cluster_frac. Slots are capped at cluster size and surplus is
             redistributed to other clusters so all n_select slots are filled.
    Level 2: Within each pose cluster, sub-cluster by Hamming distance,
             round-robin select.

    Returns:
        selected: list of (index, tier) tuples
    """
    # Group by pose cluster
    pose_clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if cid < 0:
            continue  # skip unaligned designs
        pose_clusters[cid].append(i)

    # Also include unaligned designs in a special cluster
    unaligned = [i for i, cid in enumerate(cluster_ids) if cid < 0]
    if unaligned:
        max_cid = max(cid for cid in cluster_ids if cid >= 0) + 1 if any(
            cid >= 0 for cid in cluster_ids) else 0
        pose_clusters[max_cid] = unaligned

    n_clusters = len(pose_clusters)
    if n_clusters == 0:
        return []

    # Allocate slots proportionally, then cap at cluster size and redistribute
    total_valid = sum(len(members) for members in pose_clusters.values())
    allocation = {}
    for cid, members in pose_clusters.items():
        raw_alloc = max(3, int(n_select * len(members) / total_valid))
        allocation[cid] = min(raw_alloc, int(n_select * max_cluster_frac))

    # Cap each cluster's allocation at its actual size, redistribute surplus
    for _iteration in range(10):  # iterate until stable
        surplus = 0
        uncapped_total = 0
        for cid in allocation:
            cap = len(pose_clusters[cid])
            if allocation[cid] > cap:
                surplus += allocation[cid] - cap
                allocation[cid] = cap
            else:
                uncapped_total += len(pose_clusters[cid])

        if surplus == 0:
            break

        # Distribute surplus proportionally to uncapped clusters
        for cid in sorted(pose_clusters, key=lambda c: len(pose_clusters[c]),
                          reverse=True):
            if surplus <= 0:
                break
            cap = len(pose_clusters[cid])
            if allocation[cid] >= cap:
                continue  # already capped
            room = cap - allocation[cid]
            if uncapped_total > 0:
                share = max(1, int(surplus * len(pose_clusters[cid]) / uncapped_total))
            else:
                share = surplus
            add = min(share, room, surplus)
            allocation[cid] += add
            surplus -= add

    # Normalize to exactly n_select
    total_alloc = sum(allocation.values())
    if total_alloc > n_select:
        # Scale down proportionally, keep min 2
        scale = n_select / total_alloc
        for cid in allocation:
            allocation[cid] = max(2, int(allocation[cid] * scale))
    elif total_alloc < n_select:
        # Distribute remainder to largest clusters with room
        remainder = n_select - sum(allocation.values())
        sorted_cids = sorted(pose_clusters, key=lambda c: len(pose_clusters[c]),
                              reverse=True)
        for cid in sorted_cids:
            if remainder <= 0:
                break
            room = len(pose_clusters[cid]) - allocation[cid]
            if room > 0:
                add = min(room, remainder)
                allocation[cid] += add
                remainder -= add

    print(f"\n  Slot allocation across {n_clusters} pose clusters:")
    for cid in sorted(allocation):
        n_members = len(pose_clusters[cid])
        print(f"    Pose cluster {cid}: {allocation[cid]} slots "
              f"(from {n_members} designs)")

    # Level 2: Within each pose cluster, Hamming sub-cluster + round-robin
    selected = []
    for cid in sorted(pose_clusters):
        members = pose_clusters[cid]
        n_slots = allocation.get(cid, 0)
        if n_slots == 0 or not members:
            continue

        # Get pocket seqs and scores for this cluster
        member_pockets = [pocket_seqs[i] for i in members]
        member_scores = [scores[i] for i in members]

        if len(members) <= n_slots:
            # Take all
            for i in members:
                selected.append((i, 'score_pick'))
            continue

        # Hamming sub-cluster
        sub_cluster_ids, sub_centroids = greedy_hamming_cluster(
            member_pockets, member_scores, min_hamming
        )
        n_sub = len(sub_centroids)

        # Round-robin select from sub-clusters
        sub_members = defaultdict(list)
        for local_idx, sub_cid in enumerate(sub_cluster_ids):
            sub_members[sub_cid].append(
                (member_scores[local_idx], members[local_idx]))

        for sub_cid in sub_members:
            sub_members[sub_cid].sort(reverse=True)

        sub_pointers = {sc: 0 for sc in range(n_sub)}
        sub_selected = []
        sub_selected_set = set()

        while len(sub_selected) < n_slots:
            added = False
            for sc in range(n_sub):
                if len(sub_selected) >= n_slots:
                    break
                mems = sub_members.get(sc, [])
                ptr = sub_pointers[sc]
                while ptr < len(mems):
                    _, orig_idx = mems[ptr]
                    ptr += 1
                    if orig_idx not in sub_selected_set:
                        tier = ('score_pick' if ptr == 1
                                else 'diversity_pick')
                        sub_selected.append((orig_idx, tier))
                        sub_selected_set.add(orig_idx)
                        sub_pointers[sc] = ptr
                        added = True
                        break
                sub_pointers[sc] = ptr

            if not added:
                break

        selected.extend(sub_selected)

    return selected


def count_unsatisfied_oh(fingerprint, lig=None):
    """Count unsatisfied (non-water) OHs from a fingerprint string.

    Returns the number of OH->NONE entries (WATER entries are excluded).
    """
    if not fingerprint:
        return 999  # unknown = fail
    parts = fingerprint.split('|')
    return sum(1 for p in parts if p.endswith('->NONE'))


def run_selection(all_rows, lig, expansion_root, initial_csv_dir, ref_pdb,
                  n_select, top_n, rmsd_cutoff, min_hamming, out_dir,
                  score_col='binary_total_score', gate_plddt=0.65,
                  gate_hbond=4.5, gate_max_unsatisfied_oh=None,
                  ref_ligand_pdb=None, core_rmsd_cutoff=1.5,
                  planarity_ratio_cutoff=0.5,
                  do_plot=False, extract_poses=False):
    """Run full two-level selection for one ligand.

    Returns DataFrame with selected designs.
    """
    import pandas as pd

    if not all_rows:
        print(f"\n  {lig.upper()}: No designs available")
        return None

    # Sort by score, apply basic gates
    scored = [(get_score(r, score_col), r) for r in all_rows]
    scored.sort(key=lambda x: -x[0])

    gated = []
    for score, row in scored:
        plddt = row.get('binary_plddt_ligand')
        hbond = row.get('binary_hbond_distance')
        try:
            plddt = float(plddt) if plddt is not None else 0
            hbond = float(hbond) if hbond is not None else 999
        except (ValueError, TypeError):
            continue
        if plddt >= gate_plddt and hbond <= gate_hbond:
            gated.append(row)

    print(f"\n  {lig.upper()}: {len(gated)} designs pass gates "
          f"(pLDDT >= {gate_plddt}, hbond <= {gate_hbond})")

    if len(gated) < n_select:
        print(f"    Warning: only {len(gated)} gated designs, reducing top_n")
        top_n = len(gated)

    # Take top-N for clustering
    gated = gated[:top_n]
    design_names = [r.get('name', f'design_{i}') for i, r in enumerate(gated)]
    rows_by_name = {r.get('name', ''): r for r in gated}

    # Get pocket sequences
    lookup = build_signature_lookup(initial_csv_dir, expansion_root, lig)
    pocket_seqs = []
    for row in gated:
        name = row.get('name', '')
        sig = row.get('variant_signature', '')
        if not sig:
            sig = lookup.get(name, '')
        pocket = pocket_from_signature(sig)
        pocket_seqs.append(pocket_string(pocket))

    scores = [get_score(r, score_col) for r in gated]

    # Step 1: Pose clustering
    print(f"\n  POSE CLUSTERING: {lig.upper()}")
    cluster_ids, cluster_info, pdb_found = compute_pose_clusters(
        design_names, lig, expansion_root, ref_pdb,
        rmsd_cutoff=rmsd_cutoff, score_col=score_col,
        rows_by_name=rows_by_name
    )
    print_pose_cluster_report(cluster_info, lig)

    # Step 2: OH fingerprinting
    fingerprints, fp_groups = run_oh_fingerprinting(
        design_names, lig, expansion_root, rows_by_name
    )

    # Step 2.5: Ligand geometry check
    geometry = {}
    if ref_ligand_pdb:
        geometry, n_distorted = run_geometry_check(
            design_names, lig, expansion_root, ref_ligand_pdb,
            core_rmsd_cutoff=core_rmsd_cutoff,
            planarity_ratio_cutoff=planarity_ratio_cutoff)

        if n_distorted > 0:
            pre_geo = len(design_names)
            keep_mask_geo = []
            for i, name in enumerate(design_names):
                geo = geometry.get(name)
                keep_mask_geo.append(
                    geo is None or not geo.get('distorted', False))

            n_removed = sum(1 for k in keep_mask_geo if not k)
            print(f"\n  GEOMETRY GATE: removing {n_removed}/{pre_geo} "
                  f"designs with distorted steroid core")

            new_dn = []
            new_ci = []
            new_ps = []
            new_sc = []
            new_gt = []
            for i, keep in enumerate(keep_mask_geo):
                if keep:
                    new_dn.append(design_names[i])
                    new_ci.append(cluster_ids[i])
                    new_ps.append(pocket_seqs[i])
                    new_sc.append(scores[i])
                    new_gt.append(gated[i])

            design_names = new_dn
            cluster_ids = new_ci
            pocket_seqs = new_ps
            scores = new_sc
            gated = new_gt
            print(f"    {len(design_names)} designs remain after "
                  f"geometry gate")

    # OH satisfaction gate (applied after fingerprinting)
    if gate_max_unsatisfied_oh is not None:
        pre_oh = len(design_names)
        keep_mask = []
        for i, name in enumerate(design_names):
            fp = fingerprints.get(name, '')
            n_unsat = count_unsatisfied_oh(fp)
            keep_mask.append(n_unsat <= gate_max_unsatisfied_oh)

        n_removed = sum(1 for k in keep_mask if not k)
        print(f"\n  OH GATE: removing {n_removed}/{pre_oh} designs with "
              f">{gate_max_unsatisfied_oh} unsatisfied protein-contacting OHs")

        # Rebuild filtered arrays
        new_design_names = []
        new_cluster_ids = []
        new_pocket_seqs = []
        new_scores = []
        new_gated = []
        for i, keep in enumerate(keep_mask):
            if keep:
                new_design_names.append(design_names[i])
                new_cluster_ids.append(cluster_ids[i])
                new_pocket_seqs.append(pocket_seqs[i])
                new_scores.append(scores[i])
                new_gated.append(gated[i])

        design_names = new_design_names
        cluster_ids = new_cluster_ids
        pocket_seqs = new_pocket_seqs
        scores = new_scores
        gated = new_gated

        print(f"    {len(design_names)} designs remain after OH gate")

    # Step 3: Two-level selection
    actual_select = min(n_select, len(design_names))
    print(f"\n  DIVERSITY SELECTION: {lig.upper()} ({actual_select} designs"
          f"{' (reduced from ' + str(n_select) + ')' if actual_select < n_select else ''})")
    selected = two_level_select(
        design_names, cluster_ids, pocket_seqs, scores,
        fingerprints, actual_select, min_hamming
    )

    print(f"    Selected {len(selected)} designs")
    n_score = sum(1 for _, t in selected if t == 'score_pick')
    n_div = sum(1 for _, t in selected if t == 'diversity_pick')
    print(f"    Score picks: {n_score}, Diversity picks: {n_div}")

    # Build output DataFrame
    out_rows = []
    for rank, (idx, tier) in enumerate(selected, 1):
        row = gated[idx]
        name = design_names[idx]
        out_rows.append({
            'selection_rank': rank,
            'name': name,
            'ligand': lig,
            'selection_tier': tier,
            'pose_cluster': cluster_ids[idx],
            'pocket_seq': pocket_seqs[idx],
            'oh_fingerprint': fingerprints.get(name, ''),
            'composite_zscore': get_score(row, 'composite_zscore') if 'composite_zscore' in row else '',
            score_col: scores[idx],
            'binary_plddt_ligand': row.get('binary_plddt_ligand', ''),
            'binary_plddt_pocket': row.get('binary_plddt_pocket', ''),
            'binary_hbond_distance': row.get('binary_hbond_distance', ''),
            'binary_iptm': row.get('binary_iptm', ''),
            'binary_confidence_score': row.get('binary_confidence_score', ''),
            'variant_signature': row.get('variant_signature', ''),
            'shape_rmsd': geometry.get(name, {}).get('shape_rmsd', ''),
            'core_rmsd': geometry.get(name, {}).get('core_rmsd', ''),
            'planarity_ratio': geometry.get(name, {}).get(
                'planarity_ratio', ''),
        })

    df = pd.DataFrame(out_rows)

    # Write outputs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"selection_{lig}.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    # Summary report
    summary_path = out_dir / f"selection_summary_{lig}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"SELECTION SUMMARY: {lig.upper()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total selected: {len(selected)}\n")
        f.write(f"Score picks: {n_score}\n")
        f.write(f"Diversity picks: {n_div}\n\n")
        f.write(f"Pose clusters represented: "
                f"{len(set(cluster_ids[idx] for idx, _ in selected if cluster_ids[idx] >= 0))}\n")
        f.write(f"OH fingerprints represented: "
                f"{len(set(fingerprints.get(design_names[idx], '') for idx, _ in selected))}\n")
        f.write(f"Unique pocket sequences: "
                f"{len(set(pocket_seqs[idx] for idx, _ in selected))}\n\n")

        f.write("Pose Cluster Breakdown:\n")
        for cid in sorted(cluster_info):
            n_in_sel = sum(1 for idx, _ in selected if cluster_ids[idx] == cid)
            f.write(f"  Cluster {cid}: {n_in_sel} selected / {cluster_info[cid]['size']} total\n")

        f.write(f"\nOH Fingerprint Breakdown:\n")
        sel_fps = Counter(fingerprints.get(design_names[idx], 'unknown')
                          for idx, _ in selected)
        for fp, count in sel_fps.most_common():
            f.write(f"  {fp}: {count}\n")

    print(f"    Saved: {summary_path}")

    # Extract sequences and write FASTA
    fasta_path = out_dir / f"selection_{lig}.fasta"
    n_seqs = 0
    with open(fasta_path, 'w') as f:
        for rank, (idx, tier) in enumerate(selected, 1):
            name = design_names[idx]
            pdb_path = find_pdb_for_design(name, lig, expansion_root)
            if pdb_path:
                seq = extract_sequence_from_pdb(str(pdb_path))
                if seq:
                    f.write(f">{name} rank={rank} tier={tier} "
                            f"pose_cluster={cluster_ids[idx]} "
                            f"pocket={pocket_seqs[idx]}\n")
                    f.write(f"{seq}\n")
                    n_seqs += 1
    print(f"    FASTA: {n_seqs} sequences -> {fasta_path}")

    # Representative pose extraction
    if extract_poses:
        extract_representative_poses(
            design_names, cluster_ids, scores, fingerprints,
            lig, expansion_root, str(out_dir), rows_by_name)

    # Plotting
    if do_plot:
        selected_indices = [idx for idx, _ in selected]
        plot_pose_clusters(cluster_ids, scores, design_names, lig,
                           str(out_dir), selected_indices)
        plot_oh_fingerprint_sunburst(fp_groups, lig, str(out_dir))
        plot_score_by_cluster(cluster_ids, scores, lig, str(out_dir),
                              selected_indices)
        plot_selection_summary(df, cluster_info, fp_groups, lig, str(out_dir))
        plot_metric_distributions(gated, cluster_ids, lig, str(out_dir))

    return df


# ---------------------------------------------------------------------------
# Representative PDB extraction
# ---------------------------------------------------------------------------

def extract_representative_poses(design_names, cluster_ids, scores,
                                  fingerprints, lig, expansion_root, out_dir,
                                  rows_by_name=None):
    """Extract best-scoring PDB from each pose cluster and OH strategy.

    Copies PDBs to out_dir/representative_poses_{lig}/ with descriptive names.
    Writes a PyMOL script to load, align, and color them.
    """
    import shutil

    pose_dir = Path(out_dir) / f"representative_poses_{lig}"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Simplify fingerprint to OH strategy: e.g., "92+120" or "83+92"
    def oh_strategy(fp):
        if not fp:
            return "unknown"
        parts = fp.split('|')
        contacts = []
        n_none = 0
        for p in parts:
            arrow = p.split('->')
            if len(arrow) != 2:
                continue
            oh_label, target = arrow
            if target == 'WATER':
                continue
            elif target == 'NONE':
                n_none += 1
                contacts.append('NONE')
            else:
                # Extract residue number from target like "Q120"
                import re
                m = re.match(r'[A-Z](\d+)', target)
                if m:
                    contacts.append(m.group(1))
                else:
                    contacts.append(target)
        if all(c == 'NONE' for c in contacts):
            return "no_contact"
        return "+".join(contacts)

    # Group by (pose_cluster, oh_strategy), pick best-scoring
    from collections import defaultdict as dd
    groups = dd(list)
    for i, name in enumerate(design_names):
        cid = cluster_ids[i]
        if cid < 0:
            cid = -1
        strat = oh_strategy(fingerprints.get(name, ''))
        groups[(cid, strat)].append((scores[i], i, name))

    # Sort each group by score, pick best
    representatives = []
    for (cid, strat), members in sorted(groups.items()):
        members.sort(reverse=True)
        best_score, best_idx, best_name = members[0]
        representatives.append({
            'cluster': cid,
            'strategy': strat,
            'name': best_name,
            'score': best_score,
            'n_in_group': len(members),
            'fingerprint': fingerprints.get(best_name, ''),
        })

    # Copy PDBs and build PyMOL script
    pymol_lines = [
        f"# PyMOL script: representative poses for {lig.upper()}",
        f"# Generated by analyze_expansion_readiness.py",
        "",
        "from pymol import cmd",
        "",
        "# Set nice visualization defaults",
        "cmd.set('cartoon_fancy_helices', 1)",
        "cmd.set('cartoon_side_chain_helper', 1)",
        "cmd.set('stick_radius', 0.15)",
        "",
    ]

    copied = 0
    rep_info = []
    for rep in representatives:
        pdb_path = find_pdb_for_design(rep['name'], lig, expansion_root)
        if pdb_path is None:
            continue

        # Descriptive filename
        safe_strat = rep['strategy'].replace('+', '_')
        fname = (f"cluster{rep['cluster']}_{safe_strat}_"
                 f"{rep['name']}.pdb")
        dest = pose_dir / fname
        shutil.copy2(str(pdb_path), str(dest))
        copied += 1

        obj_name = f"c{rep['cluster']}_{safe_strat}"
        pymol_lines.append(
            f"cmd.load(r'{dest.name}', '{obj_name}')  "
            f"# score={rep['score']:.3f} n={rep['n_in_group']} "
            f"fp={rep['fingerprint']}")
        rep_info.append({**rep, 'pdb_file': fname, 'obj_name': obj_name})

    # Add alignment and visualization commands
    if rep_info:
        ref_obj = rep_info[0]['obj_name']
        pymol_lines.append("")
        pymol_lines.append(f"# Align all to first structure")
        for ri in rep_info[1:]:
            pymol_lines.append(
                f"cmd.align('{ri['obj_name']}', '{ref_obj}')")

        pymol_lines.append("")
        pymol_lines.append("# Color by pose cluster")
        cluster_colors = [
            'marine', 'orange', 'forest', 'firebrick', 'purple',
            'chocolate', 'pink', 'gray60', 'olive', 'teal',
            'lightblue', 'lightorange', 'palegreen', 'salmon', 'violet',
        ]
        seen_clusters = {}
        for ri in rep_info:
            cid = ri['cluster']
            if cid not in seen_clusters:
                seen_clusters[cid] = cluster_colors[
                    len(seen_clusters) % len(cluster_colors)]
            pymol_lines.append(
                f"cmd.color('{seen_clusters[cid]}', '{ri['obj_name']}')")

        pymol_lines.append("")
        pymol_lines.append("# Show ligand as sticks")
        pymol_lines.append("cmd.show('sticks', 'organic')")
        pymol_lines.append("cmd.show('cartoon', 'polymer')")
        pymol_lines.append("")
        pymol_lines.append("# Show pocket residues as sticks")
        pocket_sel = " or ".join(f"resi {p}" for p in POCKET_POSITIONS)
        pymol_lines.append(
            f"cmd.show('sticks', '({pocket_sel}) and polymer')")
        pymol_lines.append("")
        pymol_lines.append("cmd.zoom('organic', 8)")
        pymol_lines.append(
            f"print('Loaded {len(rep_info)} representative poses for "
            f"{lig.upper()}')")

    # Write PyMOL script
    pymol_path = pose_dir / f"load_poses_{lig}.py"
    with open(pymol_path, 'w') as f:
        f.write('\n'.join(pymol_lines) + '\n')

    # Write summary table
    summary_path = pose_dir / f"representatives_{lig}.csv"
    if rep_info:
        import csv as csv_mod
        with open(summary_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=[
                'cluster', 'strategy', 'name', 'score', 'n_in_group',
                'fingerprint', 'pdb_file'])
            writer.writeheader()
            for ri in rep_info:
                writer.writerow({k: ri[k] for k in writer.fieldnames})

    print(f"\n  REPRESENTATIVE POSES: {lig.upper()}")
    print(f"    Extracted {copied} PDBs to {pose_dir}")
    print(f"    PyMOL script: {pymol_path}")
    print(f"    Summary CSV: {summary_path}")

    # Print table
    print(f"\n    {'Cluster':>8}  {'Strategy':<20}  {'Score':>7}  "
          f"{'N':>4}  Design")
    print(f"    {'-'*8}  {'-'*20}  {'-'*7}  {'-'*4}  {'-'*25}")
    for ri in rep_info:
        print(f"    {ri['cluster']:>8}  {ri['strategy']:<20}  "
              f"{ri['score']:>7.3f}  {ri['n_in_group']:>4}  {ri['name']}")

    return rep_info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_mpl():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })
    return plt


CLUSTER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


def plot_pose_clusters(cluster_ids, scores, design_names, lig, out_dir,
                       selected_indices=None):
    """Scatter plot of designs colored by pose cluster, selected highlighted."""
    plt = _setup_mpl()

    valid = [(i, cluster_ids[i], scores[i]) for i in range(len(cluster_ids))
             if cluster_ids[i] >= 0]
    if not valid:
        return

    clusters = sorted(set(cid for _, cid, _ in valid))

    fig, ax = plt.subplots(figsize=(10, 5))

    for ci, cid in enumerate(clusters):
        members = [(i, s) for i, c, s in valid if c == cid]
        x = [m[0] for m in members]
        y = [m[1] for m in members]
        color = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
        ax.scatter(x, y, c=color, label=f'Pose cluster {cid} (n={len(members)})',
                   alpha=0.35, s=18, edgecolors='none')

    if selected_indices:
        sel_x = [i for i in selected_indices if i < len(scores)]
        sel_y = [scores[i] for i in sel_x]
        ax.scatter(sel_x, sel_y, facecolors='none', edgecolors='#d62728',
                   s=50, linewidths=1.2, label='Selected', zorder=5)

    ax.set_xlabel('Design index (score-ranked)')
    ax.set_ylabel('binary_total_score')
    ax.set_title(f'Pose Clusters & Selection: {lig.upper()}')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    out_path = Path(out_dir) / f"pose_clusters_{lig}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_oh_fingerprint_sunburst(fp_groups, lig, out_dir):
    """Horizontal bar chart of OH fingerprint frequencies."""
    plt = _setup_mpl()

    fps = sorted(fp_groups.items(), key=lambda x: -len(x[1]))
    if not fps:
        return

    # Show top 15 + aggregate rest
    show = fps[:15]
    rest_count = sum(len(names) for _, names in fps[15:])

    labels = [fp for fp, _ in show]
    counts = [len(names) for _, names in show]
    if rest_count > 0:
        labels.append(f'({len(fps) - 15} other patterns)')
        counts.append(rest_count)

    colors = ['steelblue'] * len(show) + ['lightgray']

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.35)))
    bars = ax.barh(range(len(labels)), counts, color=colors[:len(labels)],
                   alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Number of designs')
    ax.set_title(f'OH Satisfaction Strategies: {lig.upper()}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')

    # Annotate counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=8)

    plt.tight_layout()
    out_path = Path(out_dir) / f"oh_fingerprints_{lig}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_score_by_cluster(cluster_ids, scores, lig, out_dir,
                          selected_indices=None):
    """Box + strip plot of scores per pose cluster, selected designs marked."""
    plt = _setup_mpl()

    # Group scores by cluster
    cluster_scores = defaultdict(list)
    for i, (cid, s) in enumerate(zip(cluster_ids, scores)):
        if cid >= 0:
            cluster_scores[cid].append((s, i))

    if not cluster_scores:
        return

    cids = sorted(cluster_scores)
    data = [[s for s, _ in cluster_scores[c]] for c in cids]
    labels = [f'C{c}\n(n={len(cluster_scores[c])})' for c in cids]

    fig, ax = plt.subplots(figsize=(max(6, len(cids) * 1.2), 5))

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for ci, patch in enumerate(bp['boxes']):
        color = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor(color)

    # Strip plot (jittered)
    for ci, cid in enumerate(cids):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data[ci]))
        x = ci + 1 + jitter
        y = data[ci]
        color = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
        ax.scatter(x, y, c=color, alpha=0.3, s=12, edgecolors='none', zorder=3)

    # Mark selected
    if selected_indices:
        selected_set = set(selected_indices)
        for ci, cid in enumerate(cids):
            sel_scores = [s for s, idx in cluster_scores[cid] if idx in selected_set]
            if sel_scores:
                jitter = np.random.default_rng(99).uniform(-0.15, 0.15, len(sel_scores))
                ax.scatter(ci + 1 + jitter, sel_scores, facecolors='none',
                           edgecolors='red', s=40, linewidths=1, zorder=4)

    ax.set_xticklabels(labels)
    ax.set_ylabel('binary_total_score')
    ax.set_title(f'Score Distribution by Pose Cluster: {lig.upper()}')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    out_path = Path(out_dir) / f"score_by_cluster_{lig}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_selection_summary(df, cluster_info, fp_groups, lig, out_dir):
    """Multi-panel summary figure for PI presentation.

    Panel A: Pie chart of pose cluster allocation in selection
    Panel B: Score distribution of selected (violin) vs all gated (histogram)
    Panel C: Selection tier breakdown (stacked bar per pose cluster)
    Panel D: OH fingerprint coverage (top fingerprints, with selected count overlay)
    """
    plt = _setup_mpl()

    if df is None or len(df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Selection Summary: {lig.upper()}', fontsize=15, fontweight='bold')

    # Panel A: Pose cluster allocation pie
    ax = axes[0, 0]
    if 'pose_cluster' in df.columns:
        cluster_counts = df['pose_cluster'].value_counts().sort_index()
        cids = cluster_counts.index.tolist()
        sizes = cluster_counts.values.tolist()
        colors = [CLUSTER_PALETTE[int(c) % len(CLUSTER_PALETTE)] if c >= 0
                  else '#cccccc' for c in cids]
        labels = [f'C{c} ({n})' for c, n in zip(cids, sizes)]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, pctdistance=0.8)
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title('Pose Cluster Allocation')

    # Panel B: Score histogram of selected
    ax = axes[0, 1]
    score_col = 'binary_total_score'
    if score_col in df.columns:
        sel_scores = df[score_col].dropna().astype(float)
        ax.hist(sel_scores, bins=20, color='steelblue', alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax.axvline(sel_scores.median(), color='red', ls='--', lw=1.5,
                   label=f'Median: {sel_scores.median():.3f}')
        ax.set_xlabel('binary_total_score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution of Selected Designs')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

    # Panel C: Selection tier stacked bar per pose cluster
    ax = axes[1, 0]
    if 'pose_cluster' in df.columns and 'selection_tier' in df.columns:
        cids_all = sorted(df['pose_cluster'].unique())
        score_counts = []
        div_counts = []
        for cid in cids_all:
            subset = df[df['pose_cluster'] == cid]
            score_counts.append(len(subset[subset['selection_tier'] == 'score_pick']))
            div_counts.append(len(subset[subset['selection_tier'] == 'diversity_pick']))

        x = np.arange(len(cids_all))
        width = 0.6
        ax.bar(x, score_counts, width, label='Score picks', color='#1f77b4', alpha=0.8)
        ax.bar(x, div_counts, width, bottom=score_counts, label='Diversity picks',
               color='#ff7f0e', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in cids_all])
        ax.set_xlabel('Pose Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Selection Tier per Pose Cluster')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

    # Panel D: OH fingerprint coverage
    ax = axes[1, 1]
    if 'oh_fingerprint' in df.columns and fp_groups:
        # Top 10 fingerprints from full pool
        top_fps = sorted(fp_groups.items(), key=lambda x: -len(x[1]))[:10]
        fp_labels = [fp for fp, _ in top_fps]
        fp_total = [len(names) for _, names in top_fps]

        # Count how many of each are in selection
        sel_fp_counter = Counter(df['oh_fingerprint'].dropna())
        fp_selected = [sel_fp_counter.get(fp, 0) for fp in fp_labels]

        y = np.arange(len(fp_labels))
        ax.barh(y, fp_total, height=0.4, color='lightgray', alpha=0.8,
                label='Total gated')
        ax.barh(y - 0.2, fp_selected, height=0.4, color='steelblue', alpha=0.8,
                label='Selected')
        ax.set_yticks(y)
        ax.set_yticklabels(fp_labels, fontsize=6, fontfamily='monospace')
        ax.set_xlabel('Count')
        ax.set_title('OH Fingerprint Coverage')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_dir) / f"selection_summary_{lig}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metric_distributions(gated_rows, cluster_ids, lig, out_dir):
    """Violin plots of key Boltz2 metrics, split by pose cluster.

    Shows pLDDT_ligand, pLDDT_pocket, hbond_distance, ipTM per cluster.
    """
    plt = _setup_mpl()

    metrics = [
        ('binary_plddt_ligand', 'pLDDT Ligand', True),
        ('binary_plddt_pocket', 'pLDDT Pocket', True),
        ('binary_hbond_distance', 'H-bond Distance (A)', False),
        ('binary_iptm', 'ipTM', True),
    ]

    # Collect data per cluster per metric
    cluster_data = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(gated_rows):
        cid = cluster_ids[i] if i < len(cluster_ids) else -1
        if cid < 0:
            continue
        for col, _, _ in metrics:
            val = row.get(col)
            if val is not None:
                try:
                    cluster_data[cid][col].append(float(val))
                except (ValueError, TypeError):
                    pass

    if not cluster_data:
        return

    cids = sorted(cluster_data)
    n_clusters = len(cids)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label, higher_better) in zip(axes, metrics):
        data = []
        tick_labels = []
        for ci, cid in enumerate(cids):
            vals = cluster_data[cid].get(col, [])
            if vals:
                data.append(vals)
                tick_labels.append(f'C{cid}\n(n={len(vals)})')

        if not data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for ci, body in enumerate(parts['bodies']):
            color = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
            body.set_facecolor(color)
            body.set_alpha(0.4)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(f'Metric Distributions by Pose Cluster: {lig.upper()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_dir) / f"metrics_by_cluster_{lig}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cross_ligand_summary(all_dfs, out_dir):
    """Cross-ligand comparison figure for PI presentation.

    Panel A: Score distributions across ligands (violin)
    Panel B: Number of pose clusters per ligand (bar)
    Panel C: Unique OH fingerprints per ligand (bar)
    Panel D: Selection tier breakdown per ligand (stacked bar)
    """
    plt = _setup_mpl()
    import pandas as pd

    ligands = sorted(all_dfs.keys())
    dfs = {lig: df for lig, df in all_dfs.items() if df is not None and len(df) > 0}
    if not dfs:
        return

    ligands = sorted(dfs.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Cross-Ligand Selection Comparison', fontsize=15, fontweight='bold')

    lig_colors = {lig: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                  for i, lig in enumerate(ligands)}

    # Panel A: Score violin per ligand
    ax = axes[0, 0]
    score_data = []
    lig_labels = []
    for lig in ligands:
        df = dfs[lig]
        if 'binary_total_score' in df.columns:
            vals = df['binary_total_score'].dropna().astype(float).tolist()
            if vals:
                score_data.append(vals)
                lig_labels.append(f'{lig.upper()}\n(n={len(vals)})')

    if score_data:
        parts = ax.violinplot(score_data, showmedians=True, showextrema=False)
        for ci, body in enumerate(parts['bodies']):
            body.set_facecolor(lig_colors[ligands[ci]])
            body.set_alpha(0.5)
        parts['cmedians'].set_color('black')
        ax.set_xticks(range(1, len(lig_labels) + 1))
        ax.set_xticklabels(lig_labels)
    ax.set_ylabel('binary_total_score')
    ax.set_title('Score Distribution of Selected Designs')
    ax.grid(True, alpha=0.2, axis='y')

    # Panel B: Pose clusters per ligand
    ax = axes[0, 1]
    n_clusters = []
    for lig in ligands:
        df = dfs[lig]
        if 'pose_cluster' in df.columns:
            n_clusters.append(df['pose_cluster'].nunique())
        else:
            n_clusters.append(0)
    bars = ax.bar(range(len(ligands)), n_clusters,
                  color=[lig_colors[l] for l in ligands], alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(ligands)))
    ax.set_xticklabels([l.upper() for l in ligands])
    ax.set_ylabel('Number of pose clusters')
    ax.set_title('Binding Mode Diversity')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, n in zip(bars, n_clusters):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(n), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel C: Unique OH fingerprints per ligand
    ax = axes[1, 0]
    n_fps = []
    for lig in ligands:
        df = dfs[lig]
        if 'oh_fingerprint' in df.columns:
            n_fps.append(df['oh_fingerprint'].nunique())
        else:
            n_fps.append(0)
    bars = ax.bar(range(len(ligands)), n_fps,
                  color=[lig_colors[l] for l in ligands], alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(ligands)))
    ax.set_xticklabels([l.upper() for l in ligands])
    ax.set_ylabel('Unique OH fingerprints')
    ax.set_title('OH Satisfaction Strategy Diversity')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, n in zip(bars, n_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(n), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel D: Selection tier breakdown per ligand
    ax = axes[1, 1]
    if all('selection_tier' in dfs[l].columns for l in ligands):
        score_picks = [len(dfs[l][dfs[l]['selection_tier'] == 'score_pick'])
                       for l in ligands]
        div_picks = [len(dfs[l][dfs[l]['selection_tier'] == 'diversity_pick'])
                     for l in ligands]
        x = np.arange(len(ligands))
        width = 0.5
        ax.bar(x, score_picks, width, label='Score picks', color='#1f77b4', alpha=0.8)
        ax.bar(x, div_picks, width, bottom=score_picks, label='Diversity picks',
               color='#ff7f0e', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in ligands])
        ax.set_ylabel('Count')
        ax.set_title('Selection Strategy Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_dir) / "cross_ligand_summary.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_convergence_comparison(all_convergence, out_dir):
    """Overlay convergence curves for all ligands on one figure.

    2x2 grid: score frontier at rank 165, penetration, entropy, unique pockets.
    """
    plt = _setup_mpl()

    ligands = sorted(all_convergence.keys())
    lig_colors = {lig: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                  for i, lig in enumerate(ligands)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Convergence Comparison Across Ligands', fontsize=15, fontweight='bold')

    # Panel 1: Score at rank 165
    ax = axes[0, 0]
    for lig in ligands:
        conv = all_convergence[lig]
        if conv is None:
            continue
        rounds = conv['rounds']
        scores_165 = [conv['round_scores'][r].get(165, conv['round_scores'][r].get(100))
                      for r in rounds]
        valid = [(r, s) for r, s in zip(rounds, scores_165) if s is not None]
        if valid:
            ax.plot([r for r, _ in valid], [s for _, s in valid], 'o-',
                    color=lig_colors[lig], label=lig.upper(), markersize=5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Score at rank 165')
    ax.set_title('Score Frontier (rank 165)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 2: Penetration
    ax = axes[0, 1]
    for lig in ligands:
        conv = all_convergence[lig]
        if conv is None:
            continue
        rounds = conv['rounds']
        pens = [conv['penetration'][r] for r in rounds]
        ax.plot(rounds, pens, 'o-', color=lig_colors[lig], label=lig.upper(),
                markersize=5)
    ax.axhline(0.05, color='red', ls='--', alpha=0.5, label='5% threshold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Penetration fraction')
    ax.set_title('New Design Penetration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Entropy
    ax = axes[1, 0]
    for lig in ligands:
        conv = all_convergence[lig]
        if conv is None:
            continue
        rounds = conv['rounds']
        ents = [conv['entropy'][r] for r in rounds]
        ax.plot(rounds, ents, 'o-', color=lig_colors[lig], label=lig.upper(),
                markersize=5)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Low diversity')
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean entropy (bits)')
    ax.set_title('Pocket Diversity (top-200)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 4: Unique pockets
    ax = axes[1, 1]
    for lig in ligands:
        conv = all_convergence[lig]
        if conv is None:
            continue
        rounds = conv['rounds']
        uniq = [conv['unique_pockets'][r] for r in rounds]
        ax.plot(rounds, uniq, 'o-', color=lig_colors[lig], label=lig.upper(),
                markersize=5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Unique pocket sequences')
    ax.set_title('Pocket Uniqueness (top-200)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_dir) / "convergence_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expansion readiness analysis and diversity selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--expansion-root", required=True,
                        help="Root of expansion directory")
    parser.add_argument("--initial-csv-dir", required=True,
                        help="Directory with initial bile acid CSVs")
    parser.add_argument("--ref-pdb", required=True,
                        help="Reference PDB for alignment (e.g., 3QN1_H2O.pdb)")
    parser.add_argument("--ligands", nargs="+", default=["ca", "cdca", "dca"],
                        help="Ligands to analyze")
    parser.add_argument("--score-col", default="binary_total_score",
                        help="Score column for ranking")
    parser.add_argument("--top-n", type=int, default=300,
                        help="Number of top designs to cluster")
    parser.add_argument("--select", type=int, default=155,
                        help="Number of designs to select per ligand")
    parser.add_argument("--n-controls", type=int, default=10,
                        help="Slots reserved for controls (not filled by this script)")
    parser.add_argument("--pose-rmsd-cutoff", type=float, default=1.5,
                        help="RMSD cutoff for pose clustering (Angstroms)")
    parser.add_argument("--min-hamming", type=int, default=3,
                        help="Minimum Hamming distance for pocket sub-clustering")
    parser.add_argument("--gate-plddt", type=float, default=0.65,
                        help="Minimum binary_plddt_ligand gate")
    parser.add_argument("--gate-hbond", type=float, default=4.5,
                        help="Maximum binary_hbond_distance gate (Angstroms)")
    parser.add_argument("--gate-max-unsatisfied-oh", type=int, default=None,
                        metavar="N",
                        help="Max unsatisfied protein-contacting OHs allowed. "
                             "Water-mediated OHs are excluded automatically. "
                             "Use 0 to require all OHs satisfied, "
                             "1 to allow one unsatisfied (e.g., for CA).")
    parser.add_argument("--ref-ligand-dir", default=None,
                        help="Directory with reference ligand PDBs for "
                             "steroid core geometry validation. Expected "
                             "files: <lig>_*.pdb (one per ligand).")
    parser.add_argument("--core-rmsd-cutoff", type=float, default=1.5,
                        help="Max steroid core RMSD to reference (A) before "
                             "flagging as distorted (default: 1.5)")
    parser.add_argument("--planarity-ratio-cutoff", type=float, default=0.5,
                        help="Min planarity ratio (query/ref). Below this "
                             "the steroid core is too flat (default: 0.5)")
    parser.add_argument("--convergence-only", action="store_true",
                        help="Run only convergence analysis (no PDB parsing)")
    parser.add_argument("--skip-convergence", action="store_true",
                        help="Skip convergence analysis")
    parser.add_argument("--extract-poses", action="store_true",
                        help="Extract representative PDBs per pose cluster "
                             "and OH strategy, with PyMOL loading script")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for CSVs, FASTAs, and plots")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selection_dfs = {}
    all_convergence = {}

    for lig in args.ligands:
        print(f"\n{'#'*72}")
        print(f"#  {lig.upper()}")
        print(f"{'#'*72}")

        # Load all round data
        all_rows = load_all_round_data(args.expansion_root, lig)
        if not all_rows:
            print(f"  No data found for {lig.upper()}")
            continue

        print(f"  Loaded {len(all_rows)} total designs across "
              f"{len(set(r['_round'] for r in all_rows))} rounds")

        # Build signature lookup for pocket sequence extraction
        sig_lookup = build_signature_lookup(args.initial_csv_dir,
                                            args.expansion_root, lig)
        print(f"  Signature lookup: {len(sig_lookup)} entries")

        # Section 1: Convergence
        if not args.skip_convergence:
            conv = compute_convergence(all_rows, args.score_col,
                                       sig_lookup=sig_lookup)
            all_convergence[lig] = conv
            print_convergence_report(conv, lig)
            if args.plot:
                plot_convergence(conv, lig, str(out_dir))

        if args.convergence_only:
            continue

        # Find reference ligand PDB for geometry check
        ref_ligand_pdb = None
        if args.ref_ligand_dir:
            ref_dir = Path(args.ref_ligand_dir)
            candidates = list(ref_dir.glob(f"{lig}_*.pdb"))
            if candidates:
                ref_ligand_pdb = str(candidates[0])
                print(f"  Reference ligand PDB: {ref_ligand_pdb}")
            else:
                print(f"  WARNING: no reference PDB matching {lig}_*.pdb "
                      f"in {ref_dir}")

        # Sections 2-4: Pose clustering, OH fingerprinting, Selection
        df = run_selection(
            all_rows, lig, args.expansion_root, args.initial_csv_dir,
            args.ref_pdb, n_select=args.select, top_n=args.top_n,
            rmsd_cutoff=args.pose_rmsd_cutoff, min_hamming=args.min_hamming,
            out_dir=str(out_dir), score_col=args.score_col,
            gate_plddt=args.gate_plddt, gate_hbond=args.gate_hbond,
            gate_max_unsatisfied_oh=args.gate_max_unsatisfied_oh,
            ref_ligand_pdb=ref_ligand_pdb,
            core_rmsd_cutoff=args.core_rmsd_cutoff,
            planarity_ratio_cutoff=args.planarity_ratio_cutoff,
            do_plot=args.plot, extract_poses=args.extract_poses,
        )
        all_selection_dfs[lig] = df

    # Cross-ligand summary figures
    if args.plot and len(all_selection_dfs) > 1:
        plot_cross_ligand_summary(all_selection_dfs, str(out_dir))
    if args.plot and len(all_convergence) > 1:
        plot_convergence_comparison(all_convergence, str(out_dir))

    # Final summary
    print(f"\n{'='*72}")
    print(f"  DONE. Outputs in: {out_dir}")
    print(f"  Per ligand: selection_<lig>.csv, selection_<lig>.fasta,")
    print(f"              selection_summary_<lig>.txt")
    if not args.convergence_only:
        print(f"\n  Remaining slots for controls: {args.n_controls} per ligand")
        print(f"  Total designs to order: {args.select + args.n_controls} per ligand")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
