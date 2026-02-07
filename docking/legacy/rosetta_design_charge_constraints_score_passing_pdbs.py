import sys
from configparser import ConfigParser
import argparse
import os

import pandas as pd
import numpy as np

import alignment
import conformer_prep
import collision_check





def find_ligand_residue_index(pose, ligand_chain="B"):
    for res_index in range(1, pose.total_residue() + 1):
        if pose.residue(res_index).is_ligand() and pose.pdb_info().chain(res_index) == ligand_chain:
            return res_index
    raise ValueError(f"Ligand not found in chain {ligand_chain}")


def determine_mutations(pose, mutation_string):
    if mutation_string == "WT":
        pos_res = []
    else:
        mutations = mutation_string.split(" ")
        pos_res = [(int(mutation[1:-1]), mutation[-1:]) for mutation in mutations]

    return [(pose.pdb_info().pdb2pose("A", pos), res) for pos, res in pos_res]


def add_charge_constraints_safe(pose, ligand_residue_index, weight=1.0):
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint

    ligand_res = pose.residue(ligand_residue_index)

    for ligand_atom_index in range(1, ligand_res.natoms() + 1):
        ligand_charge = ligand_res.atomic_charge(ligand_atom_index)
        if abs(ligand_charge) > 0.01:
            ligand_atom_id = AtomID(ligand_atom_index, ligand_residue_index)
            for protein_res_index in range(1, pose.total_residue() + 1):
                if protein_res_index == ligand_residue_index:
                    continue
                protein_res = pose.residue(protein_res_index)
                for protein_atom_index in range(1, protein_res.natoms() + 1):
                    protein_atom_id = AtomID(protein_atom_index, protein_res_index)
                    try:
                        distance = ligand_res.xyz(ligand_atom_index).distance(protein_res.xyz(protein_atom_index))
                        harmonic_func = HarmonicFunc(distance, weight)
                        constraint = AtomPairConstraint(ligand_atom_id, protein_atom_id, harmonic_func)
                        pose.add_constraint(constraint)
                    except Exception as e:
                        print(f"Failed to add constraint: {e}")



def design(pose, resfile, ref_seq, fnr_bonus):

    # SET UP SCORE FUNCTION
    sf = pyrosetta.get_fa_scorefxn()
    sf.set_weight(pyrosetta.rosetta.core.scoring.fa_elec, 2.0)  # Increase electrostatics contribution
    sf.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 1.5)  # Favor hydrogen bonding
    sf.set_weight(pyrosetta.rosetta.core.scoring.res_type_constraint, 1.0)

    
    ligand_residue_index = pose.pdb_info().pdb2pose("B", 180)
    add_charge_constraints(pose, ligand_residue_index, weight=0.5)

    # SET UP SCORE BONUSES
    fnr_bonus = 4
    fnr = pyrosetta.rosetta.protocols.protein_interface_design.FavorNativeResidue(pose, fnr_bonus)
    print('FNR bonus: ' + str(fnr_bonus))

    # SET UP TASK FACTORY
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.ReadResfile(resfile))

    jump_num = pose.num_jump()
    jump_selector = pyrosetta.rosetta.core.select.jump_selector.JumpIndexSelector()
    jump_selector.jump(jump_num)
    print('number jumps', jump_num)

    # SET UP MOVEMAP
    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.add_jump_action(pyrosetta.rosetta.core.select.movemap.mm_enable, jump_selector)

    print('pre-design score: ' + str(sf(pose)))

    # SET UP FASTRELAX
    fr = pyrosetta.rosetta.protocols.relax.FastRelax(sf, standard_repeats = 3)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.ramp_down_constraints(False)
    fr.min_type('lbfgs_armijo_nonmonotone')
    fr.max_iter(300)

    print('finish setup')

    print(pose.sequence())
    print(pose.fold_tree())

    #Apply FastRelax
    fr.apply(pose)

    print('finish apply fastrelax')
    print('post-design score: ' + str(sf(pose)))


    # Determine mutations
    # this code is just giving you the printout of where the suggested mutations are
    new_seq = pose.sequence()
    pdb_num_mutations = []
    rosetta_mutations = []
    for i in range(len(ref_seq)): # ref_seq is the AA sequence from DesignPDBFile
        if new_seq[i] != ref_seq[i]:
            empty_string = " "
            pdb_num_mutations.append(f"{ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}({i+1}){new_seq[i]}")
            rosetta_mutations.append(f"{ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}{new_seq[i]}")

    rosetta_mutations = " ".join(rosetta_mutations)


    total_score = sf(pose)

    return (pose, rosetta_mutations, pdb_num_mutations, total_score)



def main(argv):
    # PARSING CONFIG FILE
    print("Parsing config file")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file",
                        help = "your config file",
                        default = "my_conf.txt")

    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)

    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["rosetta_design"]


    # Importing necessary dependencies
    print("Importing necessary dependencies")
    sys.path.append(default["PathToPyRosetta"])
    global pyrosetta, Pose
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    pyrosetta.init("-mute all -multithreading:total_threads 10")

    # Grabbing values from config file
    print("Grabbing info from config file")
    params_list = default["ParamsList"]
    path_to_conformers = default["PathToConformers"]
    path_to_complexes = spec["PathToComplexes"]
    passing_PDBs = str(spec["PassingPDBs"]).split()
 
    # convert a list of strings into a list of integers 
    PDBs_to_read = [int(x) for x in passing_PDBs]


    # make design output folder
    try:
        print(f"Attempting to make {path_to_complexes}")
        os.mkdir(path_to_complexes)
    except:
        print(f"{path_to_complexes} already made, continuing")

    resfile = spec["Resfile"]
    fnr_bonus = float(spec["FNRBonus"])

    new_df = [] 
    df = pd.DataFrame(PDBs_to_read)

    for i, structure in df.iterrows():
    # for i, molecule_df in df[subset_start:subset_end].iterrows():
        # can change range of row depending on what you want to design around. Future task will be selecting which conformers from list to design, particularly when there is no vdm score
        # note: design only happens on the first # of specified conformers, so if there are more conformers but the conformers have no score (because no mutations) then you will score a potentially unrepresentative subset
        
        repacked_number = structure[0]
        print(repacked_number)

        pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(pose, ["conformers/0/0.params"])
        pyrosetta.pose_from_file(pose, res_set, f"pass_score_repacked/repacked{repacked_number}.pdb")
        ref_seq = pose.sequence()

        out_pose, rosetta_mutations, pdb_num_mutations, total_score = design(pose, resfile, ref_seq, fnr_bonus = fnr_bonus)

        new_df.append([repacked_number, pdb_num_mutations, rosetta_mutations, total_score])
        out_pose.dump_pdb(f"{path_to_complexes}/design{repacked_number}.pdb")


    new_df = pd.DataFrame(new_df)

    new_df.columns = ["structure number", "Pose Numbered Mutations", "Rosetta Mutations", "total score"]
    new_df.to_pickle(f"{path_to_complexes}/complex_scores.pkl")




if __name__ == "__main__":
    main(sys.argv[1:])
