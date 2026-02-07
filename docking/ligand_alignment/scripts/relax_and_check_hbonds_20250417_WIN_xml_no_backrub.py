# -*- coding: utf-8 -*-
"""
Relax and score a protein-ligand-water system, preserving water constraints
and relax protocol, then calculate interface scores within PyRosetta using
RosettaScriptsParser. Prints total scores for comparison and assesses
the quality of constrained water hydrogen bonds.

Usage:
    python relax_and_score_xml_call.py input.pdb output.pdb ligand.params
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
    ChainSelector, NeighborhoodResidueSelector, AndResidueSelector, NotResidueSelector,ResidueIndexSelector,OrResidueSelector
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

    if a86_res and a114_res and c360_res:
        # Distance constraints
        add_distance_constraint(water_res, "H1", a86_res, "O")
        add_distance_constraint(water_res, "O", a114_res, "H", dist=1.9, sd=0.1) # Assuming protein H donates to water O
        add_distance_constraint(water_res, "O", c360_res, "HE1")
        add_distance_constraint(water_res, "H2", ligand_res, "O1")

        # Angle constraints to improve hydrogen bond geometry
        # Donor-H...Acceptor angle (ideally around 180 degrees)
        add_angle_constraint(a86_res, "O", water_res, "H1", water_res, "O")
        # Assuming A114-H donates to water Oxygen
        donor_atom_a114 = "N" # Placeholder - REPLACE with the correct donor atom name if different
        if pose.residue(a114_res).has(donor_atom_a114) and pose.residue(a114_res).has("H"):
            add_angle_constraint(a114_res, donor_atom_a114,
                                 a114_res, "H",
                                 water_res, "O", 
                                 angle=180,
                                 sd=10)

        # Assuming HE1 on C360 is bonded to Nitrogen
        donor_atom_c360 = "NE1" # Placeholder - REPLACE with the correct donor atom name if different
        if pose.residue(c360_res).has(donor_atom_c360) and pose.residue(c360_res).has("HE1"):
            add_angle_constraint(c360_res, donor_atom_c360,
                                 c360_res, "HE1",
                                 water_res, "O")

        add_angle_constraint(ligand_res, "O1", water_res, "H2", water_res, "O")

        # Angle constraints around the water oxygen (H-O-H angle is typically ~104.5 degrees)
        add_angle_constraint(water_res, "H1", water_res, "O", water_res, "H2", angle=104.5, sd=10.0)

    else:
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
            
            
    loop1 = ResidueIndexSelector("84-87")
    loop2 = ResidueIndexSelector("112-115")
    loops = OrResidueSelector(loop1, loop2)        
    
    
    mm_backrub = MoveMap()
    for i in range(1, total+1):
        if loops.apply(pose)[i]:
            mm_backrub.set_bb(i, True)
            mm_backrub.set_chi(i, True)
        else:
            mm_backrub.set_bb(i, False)
            mm_backrub.set_chi(i, False)

    #backrub = BackrubMover()
    #backrub.set_movemap(mm_backrub)     # builds segments from mm\
    #backrub.apply(pose)
    
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
            don_a = pose.residue(don_r).atom_name(h.don_hatm()).strip()  # use don_hatm() here
            acc_r = h.acc_res()
            acc_a = pose.residue(acc_r).atom_name(h.acc_atm()).strip()

            match1 = (don_r == donor_res and don_a == donor_atom and acc_r == acceptor_res and acc_a == acceptor_atom)
            match2 = (don_r == acceptor_res and don_a == acceptor_atom and acc_r == donor_res and acc_a == donor_atom)
            if match1 or match2:
                return h
        return None

    def compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180, sd_d=0.2, sd_a=20): # Wider SD for angle
        return math.exp(-((d - ideal_d) ** 2) / (2 * sd_d ** 2)) * math.exp(-((a - ideal_a) ** 2) / (2 * sd_a ** 2))

    a86_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="A" and pdb_info.number(i)==86)
    a114_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="A" and pdb_info.number(i)==114)
    c360_res = next(i for i in range(1, pose.total_residue()+1) if pdb_info.chain(i)=="C" and pdb_info.number(i)==360)

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
        d = hb.get_HAdist(pose)  # ✅ fixed
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_O_A114_H_quality"] = compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180)
    else:
        hbond_qualities["water_O_A114_H_quality"] = "N/A"

    # Check O(water) to HE1(C360)
    hb = get_hbond(water_res, "O", c360_res, "HE1")
    if hb:
        d = hb.get_HAdist(pose)  # ✅ fixed
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_O_C360_HE1_quality"] = compute_hbond_quality(d, a, ideal_d=1.9, ideal_a=180)
    else:
        hbond_qualities["water_O_C360_HE1_quality"] = "N/A"

    # Check H2(water) to O2(ligand)
    hb = get_hbond(water_res, "H2", ligand_res, "O1")
    if hb:
        d = hb.get_HAdist(pose)
        a = hb.get_AHDangle(pose)
        hbond_qualities["water_H2_ligand_O1_quality"] = compute_hbond_quality(d, a)
    else:
        hbond_qualities["water_H2_ligand_O1_quality"] = "N/A"

    return hbond_qualities

def compute_total_hb_to_lig(pose):
    hb_set = HBondSet()
    fill_hbond_set(pose, False, hb_set)
    pdb_info = pose.pdb_info()
    return sum(1 for i in range(1, hb_set.nhbonds() + 1)
               if "B" in {pdb_info.chain(hb_set.hbond(i).don_res()),
                           pdb_info.chain(hb_set.hbond(i).acc_res())})




  
       

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
    

    

    try:
        lig_res_index = next(i for i in range(1, pose.total_residue() + 1)
                                if pdb_info.chain(i) == "B" and pose.residue(i).has("O1"))
        wat_res_index = next(i for i in range(1, pose.total_residue() + 1)
                                if pdb_info.chain(i) == "D" and pose.residue(i).has("H1"))
        constrain_water_network(pose, wat_res_index, lig_res_index)
    except StopIteration:
        print(f"Warning: Could not find ligand or water residue with expected atom names in {inp}. Skipping water constraints.")
        lig_res_index = -1 # Set to an invalid value to avoid errors later
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
    #charge_sat = 1 if check_carboxylic_acid_satisfaction(pose).get("satisfied") else 0

    # Call XML here to score the interface using ParsedProtocol
    xml_path = "/projects/ryde3462/software/LigandMPNN/ligand_alignment_mpnn/rosetta/interface_scoring.xml"
    rsp = RosettaScriptsParser()
    parsed_protocol = rsp.generate_mover(xml_path)
    parsed_protocol.apply(pose)
    scores = pose.scores
    print(scores)
    print("Pose Scores after XML:")
    for key in sorted(pose.scores.keys()):
        print(f"{key}: {pose.scores[key]}")

    # Safely extract scores from the current job
    score_dict = dict(jd2.get_string_real_pairs_from_current_job())
    interface_delta_b = score_dict.get("interface_delta_B", "N/A")   
    
    dG_sep = scores.get("dG_separated", "N/A")
    dsasa_int = scores.get("dSASA_int", "N/A")
    hbonds = scores.get("hbonds_int", "N/A")
    delta_unsat = scores.get("delta_unsatHbonds", "N/A")
    shape_comp = scores.get("sc_value", "N/A")  
    cms_interface = scores.get("cms_interface", "N/A")
    total_score_xml = scores.get("total_score", "N/A")


    pose.dump_pdb(outp)


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
        "total_score_from_xml": f"{total_score_xml:.2f}" if isinstance(total_score_xml, float) else total_score_xml,
    }


    score_data.update(hbond_qualities) # Add the water HBond qualities

    output_score_path = f"{outp.replace('.pdb', '_score.sc')}"
    with open(output_score_path, "w") as f:
        f.write("SCORES:\n")
        for key, value in score_data.items():
            f.write(f"{key}: {value}\n")

    print(f"Scores written to: {output_score_path}")

if __name__ == "__main__":
    main()
