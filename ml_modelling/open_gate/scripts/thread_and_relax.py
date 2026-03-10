#!/usr/bin/env python3
"""
Step 3: Thread designed pocket mutations onto 3K3K open-gate backbone and relax.

For each variant in the MD candidates CSV:
  1. Load cleaned 3K3K chain A (open-gate apo)
  2. Parse variant_signature (Boltz numbering) and map to 3K3K numbering
  3. Apply mutations using PyRosetta mutate_residue()
  4. FastRelax with coordinate constraints to preserve open-gate backbone
  5. Save 5 trajectories, select lowest energy

Usage:
    # Single variant
    python thread_and_relax.py \
        --template inputs/3K3K.pdb \
        --alignment alignment_map.json \
        --signature "59A;81L;83L;92M;94V;108V;141K;159I;160V;163G;167L" \
        --pair-id pair_3098 \
        --output-dir outputs/threaded_relaxed/

    # Batch from CSV
    python thread_and_relax.py \
        --template inputs/3K3K.pdb \
        --alignment alignment_map.json \
        --csv ../analysis/boltz_LCA/md_candidates_lca_top100.csv \
        --output-dir outputs/threaded_relaxed/

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Amino acid 3-to-1 mapping
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_variant_signature(signature):
    """Parse variant signature like '59A;81L;83L' into {position: target_aa}.

    Positions are in Boltz numbering. Supports multiple formats:
        "59A;81L;83L"  -> {59: 'A', 81: 'L', 83: 'L'}
        "K59A;L81D"    -> {59: 'A', 81: 'D'}
    """
    if not signature or pd.isna(signature) or signature.strip() == "":
        return {}

    mutations = {}
    normalized = signature.replace("_", ";").replace(" ", ";")
    normalized = re.sub(r";+", ";", normalized)

    for mut in normalized.split(";"):
        mut = mut.strip()
        if not mut:
            continue
        match = re.match(r"^([A-Z])?(\d+)([A-Z])$", mut)
        if match:
            _, pos, target_aa = match.groups()
            mutations[int(pos)] = target_aa
        else:
            logger.warning(f"Could not parse mutation: '{mut}'")

    return mutations


def clean_3k3k_chain_a(pdb_path, chain="A"):
    """Load 3K3K and extract only protein chain A atoms.

    Removes waters, ions, ligands, and other chains. Returns a clean PDB string
    that can be loaded by PyRosetta.
    """
    clean_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain_id == chain:
                    clean_lines.append(line)
            elif line.startswith("TER"):
                chain_id = line[21] if len(line) > 21 else ""
                if chain_id == chain:
                    clean_lines.append(line)
                    break  # Stop after chain A TER
    clean_lines.append("END\n")
    return "".join(clean_lines)


def prepare_template(pdb_path, chain="A", output_path=None):
    """Clean 3K3K to protein chain A only and save to a temp file.

    Returns path to the cleaned PDB file.
    """
    clean_pdb = clean_3k3k_chain_a(pdb_path, chain)

    if output_path is None:
        output_path = Path(pdb_path).parent / f"3K3K_chain{chain}_clean.pdb"

    with open(output_path, "w") as f:
        f.write(clean_pdb)

    logger.info(f"Cleaned template saved to {output_path}")
    return str(output_path)


def thread_mutations(pose, mutations_boltz, alignment_map, chain="A"):
    """Thread mutations onto the pose using the Boltz->3K3K residue mapping.

    Args:
        pose: PyRosetta Pose (3K3K chain A)
        mutations_boltz: dict {boltz_position: target_aa}
        alignment_map: dict with 'boltz_to_3k3k' mapping
        chain: chain ID in the template PDB

    Returns:
        list of (3k3k_pos, wt_aa, target_aa) tuples for mutations applied
    """
    from pyrosetta.toolbox import mutate_residue

    boltz_to_3k3k = {int(k): v for k, v in alignment_map["boltz_to_3k3k"].items()}
    pdb_info = pose.pdb_info()
    applied = []

    for boltz_pos, target_aa in sorted(mutations_boltz.items()):
        # Map Boltz position to 3K3K position
        k3k_pos = boltz_to_3k3k.get(boltz_pos)
        if k3k_pos is None:
            logger.warning(f"Boltz pos {boltz_pos} has no mapping to 3K3K — skipping")
            continue

        # Convert PDB numbering to Rosetta pose index
        pose_idx = pdb_info.pdb2pose(chain, k3k_pos)
        if pose_idx == 0:
            logger.warning(f"3K3K pos {k3k_pos} (Boltz {boltz_pos}) not found in pose — skipping")
            continue

        # Get current residue
        current_aa = pose.residue(pose_idx).name1()

        if current_aa == target_aa:
            logger.info(f"Boltz {boltz_pos} -> 3K3K {k3k_pos} (pose {pose_idx}): "
                       f"already {target_aa}")
        else:
            logger.info(f"Boltz {boltz_pos} -> 3K3K {k3k_pos} (pose {pose_idx}): "
                       f"{current_aa} -> {target_aa}")
            mutate_residue(pose, pose_idx, target_aa)
            applied.append((k3k_pos, current_aa, target_aa))

    return applied


def find_shell_residues(pose, mutation_3k3k_positions, shell_distance=10.0, chain="A"):
    """Find pose residue indices within shell_distance of mutated positions.

    Returns set of pose residue indices.
    """
    from pyrosetta.rosetta.core.select.residue_selector import (
        ResidueIndexSelector, NeighborhoodResidueSelector
    )

    pdb_info = pose.pdb_info()
    mut_pose_indices = []
    for pdb_num in mutation_3k3k_positions:
        pose_idx = pdb_info.pdb2pose(chain, pdb_num)
        if pose_idx > 0:
            mut_pose_indices.append(pose_idx)

    if not mut_pose_indices:
        # If no specific mutations, repack everything
        return set(range(1, pose.total_residue() + 1))

    index_str = ",".join(str(i) for i in mut_pose_indices)
    mut_sel = ResidueIndexSelector(index_str)
    shell_sel = NeighborhoodResidueSelector(mut_sel, shell_distance, True)

    shell_vec = shell_sel.apply(pose)
    return {i for i in range(1, pose.total_residue() + 1) if shell_vec[i]}


def fast_relax_constrained(pose, shell_residues, scorefxn, coord_sdev=0.5, seed=None):
    """Run FastRelax with coordinate constraints and shell-limited repacking.

    Args:
        pose: PyRosetta Pose (already mutated)
        shell_residues: set of pose indices to allow repacking
        scorefxn: ScoreFunction
        coord_sdev: standard deviation for coordinate constraints (Angstroms)
        seed: random seed (or None for random)

    Returns:
        relaxed_pose, total_score
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking, IncludeCurrent, PreventRepacking
    )

    if seed is not None:
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    # Clone the pose
    work_pose = pose.clone()

    # Setup FastRelax
    fr = FastRelax()
    fr.set_scorefxn(scorefxn)
    fr.constrain_relax_to_start_coords(True)
    fr.coord_sdev(coord_sdev)

    # MoveMap: allow sidechain movement in shell, freeze backbone everywhere
    mm = MoveMap()
    mm.set_bb(False)  # freeze all backbone
    mm.set_chi(False)  # default: freeze all sidechains
    for res_idx in shell_residues:
        if res_idx <= work_pose.total_residue():
            mm.set_chi(res_idx, True)  # allow sidechain in shell
    fr.set_movemap(mm)

    # TaskFactory: restrict to repacking in shell only
    tf = TaskFactory()
    tf.push_back(IncludeCurrent())
    tf.push_back(RestrictToRepacking())
    prevent = PreventRepacking()
    for i in range(1, work_pose.total_residue() + 1):
        if i not in shell_residues:
            prevent.include_residue(i)
    tf.push_back(prevent)
    fr.set_task_factory(tf)

    # Run relaxation
    fr.apply(work_pose)

    score = scorefxn(work_pose)
    return work_pose, score


def compute_backbone_rmsd(pose1, pose2):
    """Compute CA RMSD between two poses (must have same number of residues)."""
    n = min(pose1.total_residue(), pose2.total_residue())
    coords1, coords2 = [], []
    for i in range(1, n + 1):
        if pose1.residue(i).has("CA") and pose2.residue(i).has("CA"):
            ca1 = pose1.residue(i).xyz("CA")
            ca2 = pose2.residue(i).xyz("CA")
            coords1.append([ca1.x, ca1.y, ca1.z])
            coords2.append([ca2.x, ca2.y, ca2.z])

    if not coords1:
        return float("inf")

    c1 = np.array(coords1)
    c2 = np.array(coords2)
    diff = c1 - c2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def process_single_variant(
    template_pdb, alignment_map, signature, pair_id,
    output_dir, n_relax=5, coord_sdev=0.5, shell_distance=10.0, chain="A"
):
    """Process a single variant: thread mutations and relax.

    Returns dict with results.
    """
    import pyrosetta
    from pyrosetta import pose_from_pdb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    best_output = output_dir / f"{pair_id}_threaded_relaxed.pdb"
    if best_output.exists():
        logger.info(f"  {pair_id} already processed — skipping")
        return {"pair_id": pair_id, "status": "SKIPPED"}

    # Parse mutations
    mutations = parse_variant_signature(signature)
    if not mutations:
        logger.warning(f"  No mutations parsed from signature: {signature}")
        return {"pair_id": pair_id, "status": "NO_MUTATIONS"}

    logger.info(f"  {pair_id}: {len(mutations)} mutations to thread")

    # Load template
    pose = pose_from_pdb(template_pdb)
    template_pose = pose.clone()  # Save for RMSD

    # Thread mutations
    applied = thread_mutations(pose, mutations, alignment_map, chain)
    logger.info(f"  Applied {len(applied)} mutations")

    # Identify all mutated positions (in 3K3K numbering) for shell computation
    boltz_to_3k3k = {int(k): v for k, v in alignment_map["boltz_to_3k3k"].items()}
    mut_3k3k_positions = [boltz_to_3k3k[bp] for bp in mutations.keys()
                          if bp in boltz_to_3k3k]

    # Find shell residues
    shell_residues = find_shell_residues(pose, mut_3k3k_positions, shell_distance, chain)
    logger.info(f"  Repack shell: {len(shell_residues)} residues within {shell_distance} A")

    # Score function
    scorefxn = pyrosetta.create_score_function("ref2015")
    # Enable coordinate constraint scoring
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    # Run multiple FastRelax trajectories
    best_score = float("inf")
    best_pose = None
    scores = []

    for traj in range(1, n_relax + 1):
        seed = 1000 + traj
        logger.info(f"  Trajectory {traj}/{n_relax} (seed={seed})...")
        t0 = time.time()

        relaxed_pose, score = fast_relax_constrained(
            pose, shell_residues, scorefxn, coord_sdev, seed
        )

        rmsd = compute_backbone_rmsd(template_pose, relaxed_pose)
        elapsed = time.time() - t0
        logger.info(f"    Score: {score:.1f}, backbone RMSD: {rmsd:.3f} A, "
                    f"time: {elapsed:.1f}s")
        scores.append(score)

        # Save trajectory
        traj_path = output_dir / f"{pair_id}_threaded_relaxed_traj{traj}.pdb"
        relaxed_pose.dump_pdb(str(traj_path))

        if score < best_score:
            best_score = score
            best_pose = relaxed_pose.clone()

    # Save best trajectory as the primary output
    best_pose.dump_pdb(str(best_output))
    best_rmsd = compute_backbone_rmsd(template_pose, best_pose)

    logger.info(f"  Best: score={best_score:.1f}, RMSD={best_rmsd:.3f} A -> {best_output}")

    if best_rmsd > 0.5:
        logger.warning(f"  WARNING: backbone RMSD {best_rmsd:.3f} A > 0.5 A threshold!")

    return {
        "pair_id": pair_id,
        "n_mutations": len(applied),
        "mutations_applied": "; ".join(f"{wt}{pos}{tgt}" for pos, wt, tgt in applied),
        "best_score": best_score,
        "best_rmsd": best_rmsd,
        "all_scores": scores,
        "status": "OK" if best_rmsd < 1.0 else "WARN_RMSD",
        "output_pdb": str(best_output),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Thread mutations onto 3K3K open-gate backbone and relax"
    )
    parser.add_argument("--template", required=True,
                        help="Path to 3K3K.pdb (will extract chain A)")
    parser.add_argument("--alignment", required=True,
                        help="Path to alignment_map.json from align_sequences.py")
    parser.add_argument("--chain", default="A",
                        help="Chain ID in 3K3K template (default: A)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for relaxed structures")

    # Single variant mode
    parser.add_argument("--signature",
                        help="Variant signature (e.g., '59A;81L;83L;...')")
    parser.add_argument("--pair-id",
                        help="Design identifier (e.g., pair_3098)")

    # Batch mode
    parser.add_argument("--csv",
                        help="CSV file with pair_id and variant_signature columns")
    parser.add_argument("--single",
                        help="Process only this pair_id from the CSV")

    # Relaxation parameters
    parser.add_argument("--n-relax", type=int, default=5,
                        help="Number of FastRelax trajectories (default: 5)")
    parser.add_argument("--coord-sdev", type=float, default=0.5,
                        help="Coordinate constraint std dev in Angstroms (default: 0.5)")
    parser.add_argument("--shell", type=float, default=10.0,
                        help="Repack shell radius in Angstroms (default: 10.0)")

    args = parser.parse_args()

    # Load alignment map
    with open(args.alignment) as f:
        alignment_map = json.load(f)

    # Prepare clean 3K3K template (protein chain A only)
    template_dir = Path(args.template).parent
    clean_template = prepare_template(args.template, args.chain,
                                       template_dir / f"3K3K_chain{args.chain}_clean.pdb")

    # Initialize PyRosetta
    import pyrosetta
    pyrosetta.init(
        "-ignore_unrecognized_res true "
        "-ex1 -ex2 "
        "-use_input_sc "
        "-mute all"
    )

    # Collect variants to process
    variants = []
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.single:
            df = df[df["pair_id"] == args.single]
            if df.empty:
                logger.error(f"pair_id '{args.single}' not found in CSV")
                sys.exit(1)
        for _, row in df.iterrows():
            variants.append((row["pair_id"], row["variant_signature"]))
    elif args.signature and args.pair_id:
        variants.append((args.pair_id, args.signature))
    else:
        parser.error("Provide --csv or both --signature and --pair-id")

    logger.info(f"Processing {len(variants)} variant(s)...")
    results = []
    for i, (pair_id, signature) in enumerate(variants):
        logger.info(f"\n[{i+1}/{len(variants)}] {pair_id}")
        result = process_single_variant(
            template_pdb=clean_template,
            alignment_map=alignment_map,
            signature=signature,
            pair_id=pair_id,
            output_dir=args.output_dir,
            n_relax=args.n_relax,
            coord_sdev=args.coord_sdev,
            shell_distance=args.shell,
            chain=args.chain,
        )
        results.append(result)

    # Save batch summary
    if len(results) > 1:
        summary_df = pd.DataFrame(results)
        summary_path = Path(args.output_dir) / "threading_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to {summary_path}")

        ok = sum(1 for r in results if r["status"] == "OK")
        warn = sum(1 for r in results if r["status"] == "WARN_RMSD")
        skip = sum(1 for r in results if r["status"] == "SKIPPED")
        logger.info(f"Results: {ok} OK, {warn} WARN, {skip} SKIPPED out of {len(results)}")


if __name__ == "__main__":
    main()
