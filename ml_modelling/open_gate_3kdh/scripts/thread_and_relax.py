#!/usr/bin/env python3
"""
Step 3: Thread ALL mutations onto 3KDH open-gate backbone, place ligand, and relax.

Threads both PYL2->PYR1 species conversion mutations (Category A) and designed
pocket mutations (Category B/C) onto the 3KDH PYL2 open-gate backbone. Optionally
places the Boltz-predicted ligand into the pocket BEFORE relaxation, so that
Rosetta's packer sees the ligand and avoids steric clashes.

Uses a two-stage FastRelax approach to handle the large mutation load (~30-70 mutations):
  Stage A: tight constraints (sdev=0.3), sidechains only
  Stage B: moderate constraints (sdev=0.5), allow backbone in dense regions

Usage:
    # Single variant (with ligand — recommended)
    python thread_and_relax.py \
        --template inputs/3KDH.pdb \
        --alignment alignment_map.json \
        --signature "59A;81L;83L;92M;94V;108V;141K;159I;160V;163G;167L" \
        --pair-id pair_3098 \
        --output-dir outputs/threaded_relaxed/ \
        --boltz-pdb inputs/boltz_predictions/pair_3098.pdb \
        --params params/LCA.params

    # Batch from CSV (with ligand)
    python thread_and_relax.py \
        --template inputs/3KDH.pdb \
        --alignment alignment_map.json \
        --csv ../analysis/boltz_LCA/md_candidates_lca_top100.csv \
        --output-dir outputs/threaded_relaxed/ \
        --boltz-dir inputs/boltz_predictions/ \
        --params params/LCA.params

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

AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_variant_signature(signature):
    """Parse variant signature like '59A;81L;83L' into {position: target_aa}.

    Positions are in Boltz numbering.
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


def clean_template_chain_a(pdb_path, chain="A"):
    """Load template PDB and extract only protein chain A atoms.

    Removes waters, ions, ligands, and other chains.
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
                    break
    clean_lines.append("END\n")
    return "".join(clean_lines)


def prepare_template(pdb_path, chain="A", output_path=None):
    """Clean template to protein chain A only and save."""
    clean_pdb = clean_template_chain_a(pdb_path, chain)

    if output_path is None:
        output_path = Path(pdb_path).parent / f"3KDH_chain{chain}_clean.pdb"

    with open(output_path, "w") as f:
        f.write(clean_pdb)

    logger.info(f"Cleaned template saved to {output_path}")
    return str(output_path)


def thread_all_mutations(pose, variant_mutations_boltz, alignment_map, chain="A"):
    """Thread ALL mutations: Category A (PYL2->PYR1) + B/C (pocket design).

    Builds the full mutation list from the alignment_map mutations_table,
    then overlays/confirms with the variant signature pocket mutations.
    The target at every position is the Boltz designed sequence.

    Args:
        pose: PyRosetta Pose (3KDH chain A)
        variant_mutations_boltz: dict {boltz_pos: target_aa} from variant signature
        alignment_map: dict with mutations_table and boltz_to_3kdh mapping
        chain: chain ID in the template PDB

    Returns:
        applied: list of (3kdh_pos, wt_aa, target_aa, category) tuples
    """
    from pyrosetta.toolbox import mutate_residue

    boltz_to_3kdh = {int(k): v for k, v in alignment_map["boltz_to_3kdh"].items()}
    pdb_info = pose.pdb_info()

    # Build full mutation dict from alignment mutations_table
    # This includes ALL positions where 3KDH differs from Boltz
    full_mutations = {}
    for entry in alignment_map.get("mutations_table", []):
        boltz_pos = entry["boltz_pos"]
        target_aa = entry["aa_boltz"]
        category = entry["category"]
        full_mutations[boltz_pos] = (target_aa, category)

    # Overlay variant signature mutations (these are the designed pocket mutations)
    for boltz_pos, target_aa in variant_mutations_boltz.items():
        existing = full_mutations.get(boltz_pos)
        if existing:
            cat = existing[1]
            if cat == "A":
                cat = "C"  # Was species conversion, but also has a design mutation
        else:
            cat = "B"
        full_mutations[boltz_pos] = (target_aa, cat)

    applied = []
    n_cat = {"A": 0, "B": 0, "C": 0}

    for boltz_pos in sorted(full_mutations.keys()):
        target_aa, category = full_mutations[boltz_pos]
        kdh_pos = boltz_to_3kdh.get(boltz_pos)
        if kdh_pos is None:
            logger.warning(f"Boltz pos {boltz_pos} has no mapping to 3KDH — skipping")
            continue

        pose_idx = pdb_info.pdb2pose(chain, kdh_pos)
        if pose_idx == 0:
            logger.warning(f"3KDH pos {kdh_pos} (Boltz {boltz_pos}) not in pose — skipping")
            continue

        current_aa = pose.residue(pose_idx).name1()

        if current_aa == target_aa:
            continue  # Already correct

        logger.info(f"  [{category}] Boltz {boltz_pos} -> 3KDH {kdh_pos}: "
                    f"{current_aa} -> {target_aa}")
        mutate_residue(pose, pose_idx, target_aa)
        applied.append((kdh_pos, current_aa, target_aa, category))
        n_cat[category] = n_cat.get(category, 0) + 1

    logger.info(f"  Threading summary: {len(applied)} mutations applied "
                f"(A={n_cat.get('A',0)}, B={n_cat.get('B',0)}, C={n_cat.get('C',0)})")

    return applied


def place_ligand_into_pose(pose, boltz_pdb, alignment_map, output_tmp_pdb,
                           ligand_chain="B", threaded_chain="A", boltz_chain="A"):
    """Place Boltz-predicted ligand into the threaded pose via Kabsch superposition.

    Dumps the current pose to a temp PDB, uses pocket-floor CA anchors to compute
    the Kabsch transform, applies it to the ligand coordinates, writes a merged PDB,
    and returns the path so the caller can reload with params.

    Returns:
        merged_pdb_path (str): path to merged protein+ligand PDB
        sup_rmsd (float): superposition RMSD on anchor CAs
        n_severe (int): number of severe clashes (<1.5 A)
        n_mild (int): number of mild clashes (<2.0 A)
    """
    # Import from place_ligand.py (same directory)
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from place_ligand import (
        parse_pdb_atoms, get_ca_coords, get_ligand_atoms,
        kabsch_superpose, transform_coordinates,
        write_merged_pdb, get_conect_records, clash_check
    )

    # Dump current pose (protein only, no ligand yet)
    tmp_protein = str(output_tmp_pdb) + ".protein.pdb"
    pose.dump_pdb(tmp_protein)

    threaded_atoms = parse_pdb_atoms(tmp_protein)
    boltz_atoms = parse_pdb_atoms(boltz_pdb)

    # Anchor positions for Kabsch superposition (pocket floor, excluding gate)
    anchor_boltz = alignment_map["anchor_positions_boltz"]
    anchor_3kdh = alignment_map["anchor_positions_3kdh"]

    threaded_ca = get_ca_coords(threaded_atoms, anchor_3kdh, chain=threaded_chain)
    boltz_ca = get_ca_coords(boltz_atoms, anchor_boltz, chain=boltz_chain)

    common_anchors = []
    for ab, ak in zip(anchor_boltz, anchor_3kdh):
        if ab in boltz_ca and ak in threaded_ca:
            common_anchors.append((ab, ak))

    if len(common_anchors) < 4:
        logger.error(f"  Only {len(common_anchors)} common anchors — need >= 4")
        Path(tmp_protein).unlink(missing_ok=True)
        return None, float("inf"), 0, 0

    mobile_coords = np.array([boltz_ca[ab] for ab, _ in common_anchors])
    target_coords = np.array([threaded_ca[ak] for _, ak in common_anchors])

    R, t, sup_rmsd = kabsch_superpose(mobile_coords, target_coords)
    logger.info(f"  Ligand placement: superposition RMSD = {sup_rmsd:.3f} A "
                f"({len(common_anchors)} anchors)")

    # Extract and transform ligand atoms
    lig_atoms = get_ligand_atoms(boltz_atoms, chain=ligand_chain)
    if not lig_atoms:
        logger.error(f"  No ligand atoms found in Boltz PDB (chain {ligand_chain})")
        Path(tmp_protein).unlink(missing_ok=True)
        return None, sup_rmsd, 0, 0

    lig_coords = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms])
    transformed = transform_coordinates(lig_coords, R, t)
    for i, atom in enumerate(lig_atoms):
        atom["x"] = transformed[i, 0]
        atom["y"] = transformed[i, 1]
        atom["z"] = transformed[i, 2]

    # Clash check (pre-relax, just for logging)
    protein_atoms = [a for a in threaded_atoms if a["record"] == "ATOM"]
    clash = clash_check(protein_atoms, lig_atoms)
    logger.info(f"  Pre-relax clashes: {clash['n_severe']} severe, {clash['n_mild']} mild, "
                f"min dist = {clash['min_distance']:.2f} A")

    # Write merged PDB
    conect = get_conect_records(boltz_pdb)
    merged_pdb = str(output_tmp_pdb)
    write_merged_pdb(protein_atoms, lig_atoms, conect, merged_pdb)
    logger.info(f"  Merged protein+ligand written to {merged_pdb}")

    # Clean up temp protein-only PDB
    Path(tmp_protein).unlink(missing_ok=True)

    return merged_pdb, sup_rmsd, clash["n_severe"], clash["n_mild"]


def find_shell_residues(pose, mutation_pdb_positions, shell_distance=10.0, chain="A"):
    """Find pose residue indices within shell_distance of mutated positions."""
    from pyrosetta.rosetta.core.select.residue_selector import (
        ResidueIndexSelector, NeighborhoodResidueSelector
    )

    pdb_info = pose.pdb_info()
    mut_pose_indices = []
    for pdb_num in mutation_pdb_positions:
        pose_idx = pdb_info.pdb2pose(chain, pdb_num)
        if pose_idx > 0:
            mut_pose_indices.append(pose_idx)

    if not mut_pose_indices:
        return set(range(1, pose.total_residue() + 1))

    index_str = ",".join(str(i) for i in mut_pose_indices)
    mut_sel = ResidueIndexSelector(index_str)
    shell_sel = NeighborhoodResidueSelector(mut_sel, shell_distance, True)

    shell_vec = shell_sel.apply(pose)
    return {i for i in range(1, pose.total_residue() + 1) if shell_vec[i]}


def add_coordinate_constraints(pose, coord_sdev=0.5):
    """Add coordinate constraints to all backbone heavy atoms.

    Uses CoordinateConstraint with HarmonicFunc (PyRosetta 2025.13 compatible).
    """
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.numeric import xyzVector_double_t

    func = HarmonicFunc(0.0, coord_sdev)
    fixed_atom = AtomID(1, 1)

    bb_atoms = ["N", "CA", "C", "O"]
    n_constrained = 0
    for res_i in range(1, pose.total_residue() + 1):
        residue = pose.residue(res_i)
        if not residue.is_protein():
            continue  # Skip ligand/non-protein residues
        for atom_name in bb_atoms:
            if residue.has(atom_name):
                atom_idx = residue.atom_index(atom_name)
                atom_id = AtomID(atom_idx, res_i)
                xyz = residue.xyz(atom_name)
                target = xyzVector_double_t(xyz.x, xyz.y, xyz.z)
                cst = CoordinateConstraint(atom_id, fixed_atom, target, func)
                pose.add_constraint(cst)
                n_constrained += 1

    logger.info(f"    Added {n_constrained} coordinate constraints (sdev={coord_sdev} A)")


def fast_relax_constrained(pose, shell_residues, scorefxn, coord_sdev=0.5,
                           bb_residues=None, seed=None):
    """Run FastRelax with coordinate constraints and shell-limited repacking.

    Args:
        pose: PyRosetta Pose (already mutated)
        shell_residues: set of pose indices to allow repacking
        scorefxn: ScoreFunction
        coord_sdev: standard deviation for coordinate constraints
        bb_residues: optional set of pose indices where backbone movement is allowed
        seed: random seed

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
    from pyrosetta.rosetta.core.scoring import coordinate_constraint

    if seed is not None:
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    work_pose = pose.clone()

    # Add coordinate constraints
    add_coordinate_constraints(work_pose, coord_sdev)

    # Enable coordinate_constraint score term
    scorefxn_cst = scorefxn.clone()
    scorefxn_cst.set_weight(coordinate_constraint, 1.0)

    # Setup FastRelax
    fr = FastRelax()
    fr.set_scorefxn(scorefxn_cst)

    # MoveMap
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for res_idx in shell_residues:
        if res_idx <= work_pose.total_residue():
            mm.set_chi(res_idx, True)
    # Allow backbone movement in dense mutation regions (Stage B)
    if bb_residues:
        for res_idx in bb_residues:
            if res_idx <= work_pose.total_residue():
                mm.set_bb(res_idx, True)
    fr.set_movemap(mm)

    # TaskFactory
    tf = TaskFactory()
    tf.push_back(IncludeCurrent())
    tf.push_back(RestrictToRepacking())
    prevent = PreventRepacking()
    for i in range(1, work_pose.total_residue() + 1):
        if i not in shell_residues:
            prevent.include_residue(i)
    tf.push_back(prevent)
    fr.set_task_factory(tf)

    fr.apply(work_pose)

    score = scorefxn(work_pose)
    return work_pose, score


def find_mutation_dense_regions(pose, mutation_pdb_positions, chain="A",
                                radius=8.0, density_threshold=3):
    """Find pose residue indices near clusters of >= density_threshold mutations.

    These regions benefit from backbone breathing in Stage B.
    """
    pdb_info = pose.pdb_info()
    mut_pose_indices = []
    for pdb_num in mutation_pdb_positions:
        idx = pdb_info.pdb2pose(chain, pdb_num)
        if idx > 0:
            mut_pose_indices.append(idx)

    dense_set = set()
    for res_i in range(1, pose.total_residue() + 1):
        if not pose.residue(res_i).has("CA"):
            continue
        ca = pose.residue(res_i).xyz("CA")
        count = 0
        for mut_idx in mut_pose_indices:
            if not pose.residue(mut_idx).has("CA"):
                continue
            mut_ca = pose.residue(mut_idx).xyz("CA")
            dist = ((ca.x - mut_ca.x)**2 + (ca.y - mut_ca.y)**2 +
                    (ca.z - mut_ca.z)**2)**0.5
            if dist <= radius:
                count += 1
        if count >= density_threshold:
            dense_set.add(res_i)

    return dense_set


def two_stage_fast_relax(pose, all_mutated_3kdh_positions, scorefxn,
                         coord_sdev_a=0.3, coord_sdev_b=0.5,
                         shell_a=8.0, shell_b=10.0, chain="A", seed=None,
                         frozen_pdb_positions=None, frozen_pose_indices=None):
    """Two-stage FastRelax for heavy mutation load (PYL2->PYR1 + pocket).

    Stage A: tight constraints (sdev_a), sidechains only, 8A shell
    Stage B: moderate constraints (sdev_b), allow backbone in dense regions, 10A shell

    Args:
        frozen_pdb_positions: set of PDB residue numbers to exclude from repacking
            (e.g., latch histidine H119 that must keep its outward rotamer)
        frozen_pose_indices: set of pose indices to additionally exclude from
            repacking (e.g., ligand residue)

    Returns:
        (relaxed_pose, final_score, stage_a_score, stage_b_score)
    """
    import pyrosetta

    if seed is not None:
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    # Convert frozen PDB positions to pose indices
    all_frozen = set(frozen_pose_indices or set())
    if frozen_pdb_positions:
        pdb_info = pose.pdb_info()
        for pdb_num in frozen_pdb_positions:
            pose_idx = pdb_info.pdb2pose(chain, pdb_num)
            if pose_idx > 0:
                all_frozen.add(pose_idx)
    frozen_pose_indices = all_frozen
    if frozen_pose_indices:
        logger.info(f"    Freezing {len(frozen_pose_indices)} residue(s) "
                    f"(pose indices: {sorted(frozen_pose_indices)})")

    work_pose = pose.clone()

    # --- Stage A: tight constraints, sidechain-only ---
    logger.info(f"    Stage A: coord_sdev={coord_sdev_a}, {shell_a}A shell, sidechains only")
    shell_residues_a = find_shell_residues(work_pose, all_mutated_3kdh_positions,
                                           shell_a, chain)
    shell_residues_a -= frozen_pose_indices
    work_pose, score_a = fast_relax_constrained(
        work_pose, shell_residues_a, scorefxn, coord_sdev=coord_sdev_a, seed=None
    )
    logger.info(f"    Stage A score: {score_a:.1f}")

    # --- Stage B: moderate constraints, backbone breathing in dense regions ---
    logger.info(f"    Stage B: coord_sdev={coord_sdev_b}, {shell_b}A shell, bb in dense regions")

    # Clear Stage A constraints before adding Stage B
    work_pose.remove_constraints()

    shell_residues_b = find_shell_residues(work_pose, all_mutated_3kdh_positions,
                                           shell_b, chain)
    shell_residues_b -= frozen_pose_indices
    dense_regions = find_mutation_dense_regions(
        work_pose, all_mutated_3kdh_positions, chain,
        radius=8.0, density_threshold=3
    )
    dense_regions -= frozen_pose_indices
    logger.info(f"    Dense regions: {len(dense_regions)} residues with bb movement")

    work_pose, score_b = fast_relax_constrained(
        work_pose, shell_residues_b, scorefxn, coord_sdev=coord_sdev_b,
        bb_residues=dense_regions, seed=None
    )
    logger.info(f"    Stage B score: {score_b:.1f}")

    return work_pose, score_b, score_a, score_b


def compute_backbone_rmsd(pose1, pose2):
    """Compute CA RMSD between two poses."""
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
    output_dir, n_relax=5, coord_sdev_a=0.3, coord_sdev_b=0.5, chain="A",
    boltz_pdb=None, params_path=None
):
    """Process a single variant: thread mutations, place ligand, and relax.

    If boltz_pdb and params_path are provided, the ligand is placed into the
    pocket BEFORE relaxation so Rosetta's packer avoids steric clashes.
    """
    import pyrosetta
    from pyrosetta import pose_from_pdb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_output = output_dir / f"{pair_id}_threaded_relaxed.pdb"
    if best_output.exists():
        logger.info(f"  {pair_id} already processed — skipping")
        return {"pair_id": pair_id, "status": "SKIPPED"}

    # Parse variant signature (pocket mutations from CSV)
    variant_mutations = parse_variant_signature(signature)
    logger.info(f"  {pair_id}: {len(variant_mutations)} pocket mutations from signature")

    has_ligand = boltz_pdb is not None and params_path is not None
    if has_ligand:
        logger.info(f"  Ligand-aware mode: will place ligand before relax")

    # Load template
    pose = pose_from_pdb(template_pdb)
    template_pose = pose.clone()

    # Thread ALL mutations (Category A + B + C)
    applied = thread_all_mutations(pose, variant_mutations, alignment_map, chain)
    if not applied:
        logger.warning(f"  No mutations applied — check alignment")
        return {"pair_id": pair_id, "status": "NO_MUTATIONS"}

    # Collect all mutated 3KDH positions for shell computation
    all_mut_3kdh_positions = [kdh_pos for kdh_pos, _, _, _ in applied]

    # Place ligand BEFORE relaxation (if Boltz PDB + params provided)
    ligand_pose_idx = None
    sup_rmsd_lig = None
    if has_ligand:
        tmp_merged = output_dir / f"{pair_id}_tmp_merged.pdb"
        merged_pdb, sup_rmsd_lig, n_severe, n_mild = place_ligand_into_pose(
            pose, boltz_pdb, alignment_map, tmp_merged,
            threaded_chain=chain
        )
        if merged_pdb is None:
            logger.error(f"  Ligand placement failed — falling back to no-ligand relax")
            has_ligand = False
        else:
            # Reload the merged PDB so PyRosetta sees the ligand
            pose = pose_from_pdb(merged_pdb)
            # Find the ligand residue index (non-protein)
            for i in range(1, pose.total_residue() + 1):
                if not pose.residue(i).is_protein():
                    ligand_pose_idx = i
                    break
            if ligand_pose_idx:
                logger.info(f"  Ligand loaded as pose residue {ligand_pose_idx}")
            else:
                logger.warning(f"  Could not find ligand residue in merged pose")
            # Clean up temp file
            Path(merged_pdb).unlink(missing_ok=True)

    # Freeze latch histidine to preserve outward rotamer from 3KDH
    frozen_positions = set()
    latch_pos = alignment_map.get("3kdh_latch_pos")
    if latch_pos:
        frozen_positions.add(int(latch_pos))
        logger.info(f"  Latch histidine 3KDH H{latch_pos} will be frozen during relax")

    # Score function
    scorefxn = pyrosetta.create_score_function("ref2015")

    # Frozen pose indices (latch + ligand)
    frozen_pose_indices_extra = set()
    if ligand_pose_idx:
        frozen_pose_indices_extra.add(ligand_pose_idx)

    # Run multiple trajectories
    best_score = float("inf")
    best_pose = None
    scores = []

    for traj in range(1, n_relax + 1):
        seed = 1000 + traj
        logger.info(f"  Trajectory {traj}/{n_relax} (seed={seed})...")
        t0 = time.time()

        relaxed_pose, final_score, score_a, score_b = two_stage_fast_relax(
            pose, all_mut_3kdh_positions, scorefxn,
            coord_sdev_a=coord_sdev_a, coord_sdev_b=coord_sdev_b,
            chain=chain, seed=seed,
            frozen_pdb_positions=frozen_positions,
            frozen_pose_indices=frozen_pose_indices_extra
        )

        rmsd = compute_backbone_rmsd(template_pose, relaxed_pose)
        elapsed = time.time() - t0
        logger.info(f"    Final: score={final_score:.1f}, RMSD={rmsd:.3f} A, "
                    f"time={elapsed:.1f}s")
        scores.append(final_score)

        # Save trajectory
        traj_path = output_dir / f"{pair_id}_threaded_relaxed_traj{traj}.pdb"
        relaxed_pose.dump_pdb(str(traj_path))

        if final_score < best_score:
            best_score = final_score
            best_pose = relaxed_pose.clone()

    # Save best
    best_pose.dump_pdb(str(best_output))
    best_rmsd = compute_backbone_rmsd(template_pose, best_pose)

    logger.info(f"  Best: score={best_score:.1f}, RMSD={best_rmsd:.3f} A -> {best_output}")

    if best_rmsd > 1.0:
        logger.warning(f"  WARNING: backbone RMSD {best_rmsd:.3f} A > 1.0 A!")

    if best_score > -300.0:
        logger.warning(
            f"  WARNING: Best score {best_score:.1f} REU > -300 threshold. "
            f"Structure may need additional relaxation or manual inspection."
        )

    # Count categories
    n_a = sum(1 for _, _, _, cat in applied if cat == "A")
    n_b = sum(1 for _, _, _, cat in applied if cat == "B")
    n_c = sum(1 for _, _, _, cat in applied if cat == "C")

    result = {
        "pair_id": pair_id,
        "n_mutations_total": len(applied),
        "n_cat_a": n_a,
        "n_cat_b": n_b,
        "n_cat_c": n_c,
        "mutations_applied": "; ".join(
            f"[{cat}]{wt}{pos}{tgt}" for pos, wt, tgt, cat in applied
        ),
        "best_score": best_score,
        "best_rmsd": best_rmsd,
        "all_scores": scores,
        "has_ligand": has_ligand,
        "status": "OK" if best_rmsd < 2.0 else "WARN_RMSD",
        "output_pdb": str(best_output),
    }
    if sup_rmsd_lig is not None:
        result["ligand_sup_rmsd"] = sup_rmsd_lig

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Thread mutations onto 3KDH open-gate backbone and two-stage relax"
    )
    parser.add_argument("--template", required=True,
                        help="Path to 3KDH.pdb (will extract chain A)")
    parser.add_argument("--alignment", required=True,
                        help="Path to alignment_map.json from align_sequences.py")
    parser.add_argument("--chain", default="A",
                        help="Chain ID in template (default: A)")
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
    parser.add_argument("--coord-sdev-a", type=float, default=0.3,
                        help="Stage A coordinate constraint sdev (default: 0.3 A)")
    parser.add_argument("--coord-sdev-b", type=float, default=0.5,
                        help="Stage B coordinate constraint sdev (default: 0.5 A)")

    # Ligand placement (place ligand BEFORE relax to avoid pocket clashes)
    parser.add_argument("--boltz-pdb",
                        help="Boltz prediction PDB for single-variant mode (ligand source)")
    parser.add_argument("--boltz-dir",
                        help="Directory of Boltz PDBs named {pair_id}.pdb (batch mode)")
    parser.add_argument("--params",
                        help="Rosetta .params file for the ligand")

    args = parser.parse_args()

    # Load alignment map
    with open(args.alignment) as f:
        alignment_map = json.load(f)

    # Prepare clean template
    template_dir = Path(args.template).parent
    clean_template = prepare_template(args.template, args.chain,
                                       template_dir / f"3KDH_chain{args.chain}_clean.pdb")

    # Initialize PyRosetta (with ligand params if provided)
    import pyrosetta
    init_flags = (
        "-ignore_unrecognized_res true "
        "-ex1 -ex2 "
        "-use_input_sc "
        "-mute all"
    )
    if args.params:
        params_path = Path(args.params).resolve()
        if not params_path.exists():
            logger.error(f"Params file not found: {params_path}")
            sys.exit(1)
        init_flags = f"-extra_res_fa {params_path} " + init_flags
        logger.info(f"Loading ligand params: {params_path}")
    pyrosetta.init(init_flags)

    # Collect variants
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

    # Resolve Boltz PDB paths
    boltz_dir = Path(args.boltz_dir) if args.boltz_dir else None
    params_path = str(Path(args.params).resolve()) if args.params else None

    logger.info(f"Processing {len(variants)} variant(s)...")
    if boltz_dir or args.boltz_pdb:
        logger.info(f"Ligand-aware mode: ligand placed before relax")
    results = []
    for i, (pair_id, signature) in enumerate(variants):
        logger.info(f"\n[{i+1}/{len(variants)}] {pair_id}")

        # Determine Boltz PDB for this variant
        boltz_pdb = None
        if args.boltz_pdb:
            boltz_pdb = args.boltz_pdb
        elif boltz_dir:
            candidate = boltz_dir / f"{pair_id}.pdb"
            if candidate.exists():
                boltz_pdb = str(candidate)
            else:
                logger.warning(f"  Boltz PDB not found: {candidate} — relax without ligand")

        result = process_single_variant(
            template_pdb=clean_template,
            alignment_map=alignment_map,
            signature=signature,
            pair_id=pair_id,
            output_dir=args.output_dir,
            n_relax=args.n_relax,
            coord_sdev_a=args.coord_sdev_a,
            coord_sdev_b=args.coord_sdev_b,
            chain=args.chain,
            boltz_pdb=boltz_pdb,
            params_path=params_path,
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
