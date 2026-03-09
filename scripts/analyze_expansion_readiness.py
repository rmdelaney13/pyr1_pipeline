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


def compute_convergence(all_rows, score_col='binary_total_score', top_n=200):
    """Compute convergence diagnostics across rounds.

    Returns dict with:
      round_scores: {round -> {rank -> score}} for frontier tracking
      penetration: {round -> fraction of new designs in cumulative top-N}
      entropy: {round -> mean Shannon entropy across 16 pocket positions}
      unique_pockets: {round -> count of unique pocket seqs in top-N}
      verdict: CONVERGED | DIMINISHING RETURNS | STILL IMPROVING
    """
    if not all_rows:
        return None

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

    # Verdict
    verdict = "STILL IMPROVING"
    reasons = []

    if len(rounds) >= 3:
        # Check score plateau at rank 165
        recent_scores = []
        for rnd in rounds[-3:]:
            s = round_scores[rnd].get(165, round_scores[rnd].get(100))
            if s is not None:
                recent_scores.append(s)

        if len(recent_scores) >= 3:
            if recent_scores[-2] > 0:
                delta_1 = (recent_scores[-1] - recent_scores[-2]) / abs(recent_scores[-2])
                delta_2 = (recent_scores[-2] - recent_scores[-3]) / abs(recent_scores[-3])
                if abs(delta_1) < 0.02 and abs(delta_2) < 0.02:
                    verdict = "CONVERGED"
                    reasons.append(f"Score at rank 165 stable for 2 rounds "
                                   f"({recent_scores[-3]:.3f} -> {recent_scores[-2]:.3f} -> {recent_scores[-1]:.3f})")

        # Check penetration
        recent_pen = [penetration[r] for r in rounds[-2:]]
        if all(p < 0.05 for p in recent_pen):
            if verdict != "CONVERGED":
                verdict = "DIMINISHING RETURNS"
            reasons.append(f"<5% new designs entering top-{top_n} in last 2 rounds")

        # Check entropy
        recent_ent = entropy_per_round[rounds[-1]]
        if recent_ent < 0.5:
            reasons.append(f"Mean pocket entropy = {recent_ent:.2f} bits (low diversity)")
            if verdict == "STILL IMPROVING":
                verdict = "DIMINISHING RETURNS"

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

def compute_oh_fingerprint(pdb_path, protein_chain='A', ligand_chain='B',
                            hbond_cutoff=3.5):
    """Compute OH-satisfaction fingerprint for a design.

    Returns a string like "OH0->S92|OH1->H83|OH2->N120" or None if PDB not found.
    Also returns per-OH details list.
    """
    contacts = compute_oh_contacts_detailed(str(pdb_path), protein_chain,
                                             ligand_chain, hbond_cutoff)
    if not contacts:
        return None, []

    parts = []
    for c in contacts:
        if c['satisfied']:
            resname = THREE_TO_ONE.get(c['nearest_resname'], c['nearest_resname'])
            parts.append(f"OH{c['oh_index']}->{resname}{c['nearest_resnum']}")
        else:
            parts.append(f"OH{c['oh_index']}->NONE")

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

        fp, details = compute_oh_fingerprint(str(pdb_path))
        if fp is None:
            continue

        fingerprints[name] = fp
        fingerprint_groups[fp].append(name)

    print(f"    PDBs found: {n_found}/{n_total}")
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
# Section 4: Two-Level Diversity Selection
# ---------------------------------------------------------------------------

def two_level_select(design_names, cluster_ids, pocket_seqs, scores,
                     fingerprints, n_select, min_hamming=3, max_cluster_frac=0.40):
    """Two-level diversity selection: pose clusters x Hamming sub-clusters.

    Level 1: Allocate slots proportional to pose cluster size, min 3, cap at max_cluster_frac.
    Level 2: Within each pose cluster, sub-cluster by Hamming distance, round-robin select.

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

    # Allocate slots
    total_valid = sum(len(members) for members in pose_clusters.values())
    allocation = {}
    for cid, members in pose_clusters.items():
        raw_alloc = max(3, int(n_select * len(members) / total_valid))
        allocation[cid] = min(raw_alloc, int(n_select * max_cluster_frac))

    # Normalize to exactly n_select
    total_alloc = sum(allocation.values())
    if total_alloc > n_select:
        # Scale down proportionally, keep min 2
        scale = n_select / total_alloc
        for cid in allocation:
            allocation[cid] = max(2, int(allocation[cid] * scale))
    elif total_alloc < n_select:
        # Distribute remainder to largest clusters
        remainder = n_select - sum(allocation.values())
        sorted_cids = sorted(pose_clusters, key=lambda c: len(pose_clusters[c]),
                              reverse=True)
        for cid in sorted_cids:
            if remainder <= 0:
                break
            allocation[cid] += 1
            remainder -= 1

    print(f"\n  Slot allocation across {n_clusters} pose clusters:")
    for cid in sorted(allocation):
        n_members = len(pose_clusters[cid])
        print(f"    Pose cluster {cid}: {allocation[cid]} slots (from {n_members} designs)")

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
            sub_members[sub_cid].append((member_scores[local_idx], members[local_idx]))

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
                        # First pick per sub-cluster = score_pick, rest = diversity_pick
                        tier = 'score_pick' if ptr == 1 else 'diversity_pick'
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


def run_selection(all_rows, lig, expansion_root, initial_csv_dir, ref_pdb,
                  n_select, top_n, rmsd_cutoff, min_hamming, out_dir,
                  score_col='binary_total_score', gate_plddt=0.65,
                  gate_hbond=4.5, do_plot=False):
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

    # Step 3: Two-level selection
    print(f"\n  DIVERSITY SELECTION: {lig.upper()} ({n_select} designs)")
    selected = two_level_select(
        design_names, cluster_ids, pocket_seqs, scores,
        fingerprints, n_select, min_hamming
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
    parser.add_argument("--convergence-only", action="store_true",
                        help="Run only convergence analysis (no PDB parsing)")
    parser.add_argument("--skip-convergence", action="store_true",
                        help="Skip convergence analysis")
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

        # Section 1: Convergence
        if not args.skip_convergence:
            conv = compute_convergence(all_rows, args.score_col)
            all_convergence[lig] = conv
            print_convergence_report(conv, lig)
            if args.plot:
                plot_convergence(conv, lig, str(out_dir))

        if args.convergence_only:
            continue

        # Sections 2-4: Pose clustering, OH fingerprinting, Selection
        df = run_selection(
            all_rows, lig, args.expansion_root, args.initial_csv_dir,
            args.ref_pdb, n_select=args.select, top_n=args.top_n,
            rmsd_cutoff=args.pose_rmsd_cutoff, min_hamming=args.min_hamming,
            out_dir=str(out_dir), score_col=args.score_col,
            gate_plddt=args.gate_plddt, gate_hbond=args.gate_hbond,
            do_plot=args.plot,
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
