# -*- coding: utf-8 -*-
"""
Relax and score a protein-ligand-water system, preserving water constraints
and relax protocol, then calculate interface scores within PyRosetta using
RosettaScriptsParser. Prints total scores for comparison and assesses
the quality of constrained water hydrogen bonds.
"""
import sys
import os
import math
import pyrosetta
from pyrosetta import init, pose_from_file, create_score_function
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, IncludeCurrent
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector, NeighborhoodResidueSelector, AndResidueSelector, NotResidueSelector, ResidueIndexSelector, OrResidueSelector
)
from pyrosetta.rosetta.core.scoring import atom_pair_constraint
from pyrosetta.rosetta.core.scoring import angle_constraint
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint, AngleConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set
from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
from pyrosetta.rosetta.protocols import jd2
from pyrosetta.rosetta.protocols.backrub import BackrubMover
from pyrosetta.rosetta.utility import vector1_unsigned_long


# ─────────────────────────────────────────────────────────────────────────────
def find_nearest_ligand_acceptor(pose, ligand_res, water_res,
                                 water_h_atom: str = "H2",
                                 elements=("O", "N"),
                                 max_dist: float = 3.5):
    """
    Scan all atoms in `ligand_res` whose element symbol is in `elements` (default: O or N),
    compute the distance to the water hydrogen named `water_h_atom` in `water_res`,
    and return the atom-name of the ligand atom whose distance is minimal
    (but not exceeding max_dist). If no candidate is found within max_dist, return None.

    Args:
        pose        : the Pose object
        ligand_res  : residue index of the ligand
        water_res   : residue index of the water
        water_h_atom: which atom of the water to use as the reference hydrogen (default "H2")
        elements    : tuple of element symbols to consider on the ligand (default ("O","N"))
        max_dist    : maximum H–A distance to consider (Å)

    Returns:
        A string (e.g., "O3" or "N1") for the ligand atom name that is closest to water:H2,
        or `None` if nothing is within `max_dist`.
    """
    # 1) Grab the xyz of the specified water hydrogen
    try:
        water_h_xyz = pose.residue(water_res).xyz(water_h_atom)
    except RuntimeError:
        # If the water does not have that atom, bail out
        return None

    best_atom = None
    best_dist = float("inf")
    ligand = pose.residue(ligand_res)

    # 2) Loop over all ligand atoms, check element type
    for idx in range(1, ligand.natoms() + 1):
        atom_name = ligand.atom_name(idx).strip()
        elem = ligand.atom_type(idx).element().upper()
        if elem not in elements:
            continue
        # compute distance to the water H
        dist = water_h_xyz.distance(ligand.xyz(idx))
        if dist < best_dist and dist <= max_dist:
            best_dist = dist
            best_atom = atom_name

    return best_atom



def constrain_water_network(pose, water_res, ligand_res):
    pdb_info = pose.pdb_info()

    def add_distance_constraint(r1, a1, r2, a2, dist=1.9, sd=0.2):
        aid1 = AtomID(pose.residue(r1).atom_index(a1), r1)
        aid2 = AtomID(pose.residue(r2).atom_index(a2), r2)
        pose.add_constraint(AtomPairConstraint(aid1, aid2, HarmonicFunc(dist, sd)))

    def add_angle_constraint(r1, a1, r2, a2, r3, a3, angle=180.0, sd=20.0):
        aid1 = AtomID(pose.residue(r1).atom_index(a1), r1)
        aid2 = AtomID(pose.residue(r2).atom_index(a2), r2)
        aid3 = AtomID(pose.residue(r3).atom_index(a3), r3)
        pose.add_constraint(AngleConstraint(aid1, aid2, aid3, HarmonicFunc(angle, sd)))

    # Find residue numbers based on chain and PDB number
    def find_residue(chain, number):
        for i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(i) == chain and pdb_info.number(i) == number:
                return i
        return None  # Handle case where residue is not found

    a86_res = find_residue("A", 86)
    a114_res = find_residue("A", 114)
    c360_res = find_residue("C", 360)

    # ─────────────────────────────────────────────────────────────────────────
    # ------------- NEW: find a ligand "acceptor" atom near the water H2 ------------
    ligand_acceptor = find_nearest_ligand_acceptor(pose, ligand_res, water_res, water_h_atom="H2")
    # ligand_acceptor will be something like "O3" or "O5", etc., or None if none found.

    if a86_res and a114_res and c360_res and ligand_acceptor is not None:
        # Distance constraints: all except the ligand–water constraint remain as before
        add_distance_constraint(water_res, "H1", a86_res, "O")
        add_distance_constraint(water_res, "O", a114_res, "H", dist=1.9, sd=0.1)
        add_distance_constraint(water_res, "O", c360_res, "HE1")
        # ─────────────────────────────────────────────────────────────────────────
        #    Replace the hard-coded ("O3") by ligand_acceptor
        add_distance_constraint(water_res, "H2", ligand_res, ligand_acceptor)
        # ─────────────────────────────────────────────────────────────────────────

        # Angle constraints to improve hydrogen bond geometry
        add_angle_constraint(a86_res, "O", water_res, "H1", water_res, "O")

        donor_atom_a114 = "N"  # still a placeholder; no change here
        if pose.residue(a114_res).has(donor_atom_a114) and pose.residue(a114_res).has("H"):
            add_angle_constraint(a114_res, donor_atom_a114,
                                 a114_res, "H",
                                 water_res, "O",
                                 angle=180,
                                 sd=10)

        donor_atom_c360 = "NE1"
        if pose.residue(c360_res).has(donor_atom_c360) and pose.residue(c360_res).has("HE1"):
            add_angle_constraint(c360_res, donor_atom_c360,
                                 c360_res, "HE1",
                                 water_res, "O")

        # ─────────────────────────────────────────────────────────────────────────
        #    Also replace the hard-coded ("O3") in the angle around ligand–water
        add_angle_constraint(ligand_res, ligand_acceptor, water_res, "H2", water_res, "O")
        # ─────────────────────────────────────────────────────────────────────────

        # Angle constraints around the water oxygen (H-O-H angle is typically ~104.5 degrees)
        add_angle_constraint(water_res, "H1", water_res, "O", water_res, "H2", angle=104.5, sd=10.0)

    else:
        if ligand_acceptor is None:
            print(f"Warning: No suitable ligand acceptor (N/O) within cutoff to water:H2 for ligand_res={ligand_res}.")
        print("Warning: One or more target residues for water constraints not found!")
    


def apply_interface_relax(pose, scorefxn):
    # 1) build your selectors
    ligand_sel = ChainSelector("B")
    water_sel  = ChainSelector("D")

    # residues within 10Å of ligand
    near10    = NeighborhoodResidueSelector(ligand_sel, 10.0)
    nearA10   = AndResidueSelector(near10, ChainSelector("A"))
    nearC10   = AndResidueSelector(near10, ChainSelector("C"))

    # residues within 3Å of water, excluding ligand
    near3water  = NeighborhoodResidueSelector(water_sel, 3.0)
    not_ligand  = NotResidueSelector(ligand_sel)
    backrub_sel = AndResidueSelector(near3water, not_ligand)

    # 2) build MoveMap for relax (A/C loops + all water backbone/chi, rest rigid)
    mm = MoveMap()
    total = pose.total_residue()
    for i in range(1, total+1):
        if nearA10.apply(pose)[i] or nearC10.apply(pose)[i] or water_sel.apply(pose)[i]:
            mm.set_bb(i, True)
            mm.set_chi(i, True)
        else:
            mm.set_bb(i, False)
            mm.set_chi(i, False)

    # enable rigid‐body on ligand & water
    ft = pose.fold_tree()
    for j in range(1, ft.num_jump()+1):
        down = ft.downstream_jump_residue(j)
        if ligand_sel.apply(pose)[down] or water_sel.apply(pose)[down]:
            mm.set_jump(j, True)

    # 4) repack-only
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)

    # 5) constrained FastRelax
    relax = FastRelax(scorefxn)
    relax.set_movemap(mm)
    relax.apply(pose)


def check_constrained_water_hbonds(pose, water_res, ligand_res):
    hb_set = HBondSet()
    fill_hbond_set(pose, False, hb_set)
    pdb_info = pose.pdb_info()

    def get_hbond(donor_res, donor_atom, acceptor_res, acceptor_atom):
        for i in range(1, hb_set.nhbonds() + 1):
            h = hb_set.hbond(i)
            don_r = h.don_res()
            don_a = pose.residue(don_r).atom_name(h.don_hatm()).strip()
            acc_r = h.acc_res()
            acc_a = pose.residue(acc_r).atom_name(h.acc_atm()).strip()

            match1 = (don_r == donor_res and don_a == donor_atom and acc_r == acceptor_res and acc_a == acceptor_atom)
            match2 = (don_r == acceptor_res and don_a == acceptor_atom and acc_r == donor_res and acc_a == donor_atom)
            if match1 or match2:
                return h
        return None

    def compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180, sd_d=0.2, sd_a=20):
        return math.exp(-((d - ideal_d) ** 2) / (2 * sd_d ** 2)) * math.exp(-((a - ideal_a) ** 2) / (2 * sd_a ** 2))

    a86_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="A" and pdb_info.number(i)==86)
    a114_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="A" and pdb_info.number(i)==114)
    c360_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="C" and pdb_info.number(i)==360)

    # ─────────────────────────────────────────────────────────────────────────────
    # ------------- NEW: re-find which ligand atom was used as acceptor ------------
    ligand_acceptor = find_nearest_ligand_acceptor(pose, ligand_res, water_res, water_h_atom="H2")
    # Use ligand_acceptor in the check for "H2(water) → ligand:acceptor"
    # ─────────────────────────────────────────────────────────────────────────────

    hbond_qualities = {}

    # Check H1(water) to O(A86)
    hb = get_hbond(water_res, "H1", a86_res, "O")
    if hb:
        d = hb.get_HAdist(pose)
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_H1_A86_O_quality"] = compute_hbond_quality(d, a)
    else:
        hbond_qualities["water_H1_A86_O_quality"] = "N/A"

    # Check O(water) to H(A114)
    hb = get_hbond(water_res, "O", a114_res, "H")
    if hb:
        d = hb.get_HAdist(pose)
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_O_A114_H_quality"] = compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180)
    else:
        hbond_qualities["water_O_A114_H_quality"] = "N/A"

    # Check O(water) to HE1(C360)
    hb = get_hbond(water_res, "O", c360_res, "HE1")
    if hb:
        d = hb.get_HAdist(pose)
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_O_C360_HE1_quality"] = compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180)
    else:
        hbond_qualities["water_O_C360_HE1_quality"] = "N/A"

    # Check H2(water) to ligand_acceptor
    if ligand_acceptor is not None:
        hb = get_hbond(water_res, "H2", ligand_res, ligand_acceptor)
        if hb:
            d = hb.get_HAdist(pose)
            a = hb.get_AHDangle(pose)
            hbond_qualities["water_H2_ligand_%s_quality" % ligand_acceptor] = compute_hbond_quality(d, a)
        else:
            hbond_qualities["water_H2_ligand_%s_quality" % ligand_acceptor] = "N/A"
    else:
        hbond_qualities["water_H2_ligand_quality"] = "N/A"

    return hbond_qualities


def compute_total_hb_to_lig(pose):
    hb_set = HBondSet()
    fill_hbond_set(pose, False, hb_set)
    pdb_info = pose.pdb_info()
    return sum(1 for i in range(1, hb_set.nhbonds() + 1)
               if "B" in {pdb_info.chain(hb_set.hbond(i).don_res()),
                           pdb_info.chain(hb_set.hbond(i).acc_res())})



def add_hbond_constraint(pose, res1, atom1, res2, atom2, sd=0.2):
    """
    Constrain the distance between (res1,atom1) and (res2,atom2)
    to its current value ± sd (Å), with given score weight.
    """
    a1 = pose.residue(res1).atom_index(atom1)
    a2 = pose.residue(res2).atom_index(atom2)
    xyz1 = pose.residue(res1).xyz(atom1)
    xyz2 = pose.residue(res2).xyz(atom2)
    dist0 = xyz1.distance(xyz2)

    func = HarmonicFunc(dist0, sd)
    cst = AtomPairConstraint(AtomID(a1, res1),
                             AtomID(a2, res2),
                             func)
    pose.add_constraint(cst)


def main():
    if len(sys.argv) != 4:
        print("Usage: python relax_and_score.py input.pdb output.pdb ligand.params")
        sys.exit(1)
    inp, outp, params = sys.argv[1:]
    init(f"-extra_res_fa {params} -ex1 -ex2aro -use_input_sc"
          f" -relax:fast -relax:default_repeats 5"
          f" -corrections::beta_nov16 true"
          f" -score:weights beta_nov16"
          f" -backrub:ntrials 50"
          f" -mute all")

    pose = pose_from_file(inp)
    pdb_info = pose.pdb_info()

    sf = create_score_function("beta_nov16")
    sf.set_weight(atom_pair_constraint, 1.0)
    sf.set_weight(angle_constraint,     1.0)

    # H-bond 1: VAL115/H ↔ HIS113/ND1
    add_hbond_constraint(pose, 115, "H", 113, "ND1", sd=0.2)

    # H-bond 2: HIS113/HE2 ↔ GLY110/O
    add_hbond_constraint(pose, 113, "HE2", 110, "O", sd=0.2)

    try:
        lig_res_index = next(i for i in range(1, pose.total_residue() + 1)
                             if pdb_info.chain(i) == "B" and pose.residue(i).has("O1"))
        wat_res_index = next(i for i in range(1, pose.total_residue() + 1)
                             if pdb_info.chain(i) == "D" and pose.residue(i).has("H1"))
        constrain_water_network(pose, wat_res_index, lig_res_index)
    except StopIteration:
        print(f"Warning: Could not find ligand or water residue with expected atom names in {inp}. Skipping water constraints.")
        lig_res_index = -1
        wat_res_index = -1

    pre_tot = sf(pose)
    pre_cst = pose.energies().total_energies()[atom_pair_constraint]

    apply_interface_relax(pose, sf)
    post_relax_total_score = sf(pose)
    post_relax_constraint_score = pose.energies().total_energies()[atom_pair_constraint]

    hbond_qualities = {}
    if wat_res_index != -1 and lig_res_index != -1:
        hbond_qualities = check_constrained_water_hbonds(pose, wat_res_index, lig_res_index)

    tot_hb = compute_total_hb_to_lig(pose)


    # … the rest of your script (XML scoring, printing out, writing the .sc file) …
    xml_path = "/projects/ryde3462/software/ligand_alignment_mpnn/interface_scoring.xml"
    rsp = RosettaScriptsParser()
    parsed_protocol = rsp.generate_mover(xml_path)
    parsed_protocol.apply(pose)
    scores = pose.scores

    print(scores)
    print("Pose Scores after XML:")
    for key in sorted(pose.scores.keys()):
        print(f"{key}: {pose.scores[key]}")

    score_dict = dict(jd2.get_string_real_pairs_from_current_job())
    interface_delta_b = score_dict.get("interface_delta_B", "N/A")
    print(f"interface_delta_b: {interface_delta_b}")
    print(score_dict)

    dG_sep = scores.get("dG_separated", "N/A")
    dsasa_int = scores.get("dSASA_int", "N/A")
    hbonds   = scores.get("hbonds_int", "N/A")
    delta_unsat = scores.get("delta_unsatHbonds", "N/A")
    shape_comp  = scores.get("sc_value", "N/A")
    cms_interface = scores.get("cms_interface", "N/A")
    total_score_xml = scores.get("total_score", "N/A")

    pose.dump_pdb(outp)

    print("\n--- Total Scores ---")
    print(f"Total Score After Relax: {post_relax_total_score:.2f}")
    print(f"Total Score from XML: {total_score_xml:.2f}")
    print("----------------------\n")

    print("\n--- Water Hydrogen Bond Qualities ---")
    for hb, quality in hbond_qualities.items():
        print(f"{hb}: {quality}")
    print("-------------------------------------\n")

    score_data = {
        "interface_delta_b": f"{interface_delta_b:.2f}" if isinstance(interface_delta_b, float) else interface_delta_b,
        "dG_sep": f"{dG_sep:.2f}" if isinstance(dG_sep, float) else dG_sep,
        "post_relax_total_score": f"{post_relax_total_score:.2f}",
        "post_relax_constraint_score": f"{post_relax_constraint_score:.2f}",
        "total_hbonds_to_ligand": tot_hb,
        "hbond_int": hbonds,
        "dsasa_int": f"{dsasa_int:.2f}" if isinstance(dsasa_int, float) else dsasa_int,
        "shape_complementarity": f"{shape_comp:.2f}" if isinstance(shape_comp, float) else shape_comp,
        "buried_unsatisfied_polars": f"{delta_unsat:.2f}" if isinstance(delta_unsat, float) else delta_unsat,
        "cms_interface": f"{cms_interface:.2f}" if isinstance(cms_interface, float) else cms_interface,
        "total_score_from_xml": f"{total_score_xml:.2f}" if isinstance(total_score_xml, float) else total_score_xml
    }

    score_data.update(hbond_qualities)  # Add the water HBond qualities

    output_score_path = f"{outp.replace('.pdb', '_score.sc')}"
    with open(output_score_path, "w") as f:
        f.write("SCORES:\n")
        for key, value in score_data.items():
            f.write(f"{key}: {value}\n")

    print(f"Scores written to: {output_score_path}")


if __name__ == "__main__":
    main()
