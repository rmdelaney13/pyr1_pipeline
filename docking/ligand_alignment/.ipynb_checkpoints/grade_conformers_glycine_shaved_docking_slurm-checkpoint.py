#!/usr/bin/env python
import sys
import os
import time
import random
import pandas as pd
import argparse
from configparser import ConfigParser

def csv_to_df(csv_file_name, auto, path_to_conformers, pose, target_res, lig_res_num):
    """
    Reads and processes the CSV file into a Pandas DataFrame.
    This function adds a 'Molecule File Stem' (for locating ligand files), 
    splits the 'Conformer Range', and converts the atom strings into tuples.
    """
    df = pd.read_csv(csv_file_name, index_col=False)

    # Generate file stem using the Molecule ID
    def create_file_stem(name):
        return f"{name}/{name}"
    df["Molecule File Stem"] = df["Molecule ID"].apply(create_file_stem)

    # Split the conformer range into a tuple (assumes a format like "1_2_")
    def split_crange(name):
        lst = name.split("_")
        return (lst[0], lst[1])
    df["Conformer Range"] = df["Conformer Range"].apply(split_crange)

    # If auto-alignment is enabled, process each row accordingly.
    if auto:
        print("Auto Generating Alignments")
        for i, row in df.iterrows():
            print(f"Processing row {i+1}/{len(df)}", end=" ")
            lig = Pose()
            mol_id = row["Molecule ID"]
            conf_num = 1
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, 
                         params_list=[f"{path_to_conformers}/{mol_id}/{mol_id}.params"])
            pyrosetta.pose_from_file(lig, res_set, f"{path_to_conformers}/{mol_id}/{mol_id}_{conf_num:04}.pdb")

            molecule_atoms, target_atoms = alignment.auto_align_residue_to_residue(
                lig, lig.residue(lig_res_num), target_res)
            df.loc[i, "Molecule Atoms"] = "-".join(molecule_atoms)
            df.loc[i, "Target Atoms"] = "-".join(target_atoms)

    # Convert atom strings into tuples
    def split_alabels(name):
        if name == "default":
            return ("CD2", "CZ2", "CZ3")
        return tuple(name.split("-"))
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(split_alabels)
    df["Target Atoms"] = df["Target Atoms"].apply(split_alabels)
    
    return df

def align_to_residue_and_check_collision(pose, res, path_to_conformers, df, jump_num, rotation, translation,
                                         upper_water_distance, lower_water_distance, backbone_clash_keep_sidechains,
                                         max_pass_score=-300, bin_width=1, vdw_modifier=0.7, include_sc=False,
                                         lig_res_num=1, task_index=0):
    """
    Runs the ligand alignment, perturbation, and collision checking.
    Output files (e.g., rotated and repacked PDBs) are written into directories 
    that include the task index, so that multiple independent runs do not conflict.
    """
    # Import PyRosetta protocols used here.
    import pyrosetta.rosetta.protocols.rigid as rigid_moves
    import pyrosetta.rosetta.protocols.grafting as graft
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.protocols import minimization_packing as pack_min

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

    # Setup scorefunction and packing
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    tf.push_back(operation.NoRepackDisulfides())
    tf.push_back(operation.RestrictToRepacking())
    packer = pack_min.PackRotamersMover()
    packer.task_factory(tf)

    sf_all = pyrosetta.get_fa_scorefxn()
    sf_rep = pyrosetta.get_fa_scorefxn()
    sf_rep.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 1.0)

    t0 = time.time()
    all_accepted = []
    accepted_conformations = []
    total_confs = 0

    alignto_pose = pose

    # Prepare a grid-check pose by mutating residues to Gly (except those in backbone_clash_keep_sidechains)
    loop_include_sidechain_pose = pose.clone()
    loop_include_sidechain = [int(x) for x in backbone_clash_keep_sidechains]
    for i in range(1, loop_include_sidechain_pose.total_residue() + 1):
        if i in loop_include_sidechain:
            continue
        else:
            pyrosetta.toolbox.mutate_residue(loop_include_sidechain_pose, i, 'G')

    backbone_grid = collision_check.CollisionGrid(loop_include_sidechain_pose, bin_width=bin_width,
                                                    vdw_modifier=vdw_modifier, include_sc=True)

    # Loop over all ligand conformers from the CSV
    for pose_info in conformer_prep.yield_ligand_poses(df=df, path_to_conformers=path_to_conformers,
                                                        post_accepted_conformers=False, ligand_residue=lig_res_num):
        if not pose_info:
            print("Conformer info missing")
            all_accepted.append(accepted_conformations)
            accepted_conformations = []
            continue

        conf = pose_info
        conf.align_to_target(res)

        location = len(alignto_pose.chain_sequence(1))
        new_pose = graft.insert_pose_into_pose(alignto_pose, conf.pose, location)
        ligand_res_index = location + 1

        keep_list = []
        count = 0
        # Try perturbations until one meets the collision and water-distance criteria
        while len(keep_list) == 0 and count < 10:
            copy_pose = new_pose.clone()
            pert_mover = rigid_moves.RigidBodyPerturbMover(jump_num, rotation, translation)
            pert_mover.apply(copy_pose)
            for atom_id in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                atom_coords = copy_pose.residue(ligand_res_index).xyz(atom_id)
                conf.pose.residue(1).set_xyz(atom_id, atom_coords)
            does_collide = conf.check_collision(backbone_grid)
            if not does_collide:
                water_resid = []
                for i in range(1, copy_pose.total_residue() + 1):
                    residue = copy_pose.residue(i)
                    if residue.is_water():
                        water_resid.append(i)
                for j in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                    atom = copy_pose.residue(ligand_res_index).atom_type(j)
                    if atom.element() in ["N", "O"]:
                        atom_coords = copy_pose.residue(ligand_res_index).xyz(j)
                        for idx in water_resid:
                            for w in range(1, copy_pose.residue(idx).natoms() + 1):
                                atom_w = copy_pose.residue(idx).atom_type(w)
                                if atom_w.element() == "O":
                                    water_coord = copy_pose.residue(idx).xyz(w)
                            if lower_water_distance < atom_coords.distance(water_coord) < upper_water_distance:
                                keep_list.append(idx)
            count += 1

        if len(keep_list) == 0:
            print("Conformer failed backbone clash and water distance check")
        else:
            # Write rotated PDB output to a task-specific directory
            out_dir = f"rotated_{task_index}"
            os.makedirs(out_dir, exist_ok=True)
            copy_pose.dump_pdb(f"{out_dir}/rotated_{task_index}_{total_confs}.pdb")

            packer.apply(copy_pose)
            ligand_res_index = location + 1
            for atom_id in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                atom_coords = copy_pose.residue(ligand_res_index).xyz(atom_id)
                conf.pose.residue(1).set_xyz(atom_id, atom_coords)

            remove_ligand_pose = copy_pose.clone()
            remove_ligand_pose.delete_residue_slow(ligand_res_index)
            grid = collision_check.CollisionGrid(remove_ligand_pose, bin_width=bin_width,
                                                   vdw_modifier=vdw_modifier, include_sc=include_sc)
            does_collide = conf.check_collision(grid)
            if not does_collide:
                accepted_conformations.append(conf.conf_num)
                score = sf_all(copy_pose)
                print("Score:", score)
                if score < max_pass_score:
                    conf.pose.pdb_info().name("pass_score")
                    pmm.apply(conf.pose)
                    out_dir = f"pass_score_repacked_{task_index}"
                    os.makedirs(out_dir, exist_ok=True)
                    copy_pose.dump_pdb(f"{out_dir}/repacked_{task_index}_{total_confs}.pdb")
                else:
                    conf.pose.pdb_info().name("fail_score")
                    pmm.apply(conf.pose)
                    out_dir = f"pass_fail-score_repacked_{task_index}"
                    os.makedirs(out_dir, exist_ok=True)
                    copy_pose.dump_pdb(f"{out_dir}/repacked_{task_index}_{total_confs}.pdb")
            else:
                out_dir = f"pass_initial-only_repacked_{task_index}"
                os.makedirs(out_dir, exist_ok=True)
                copy_pose.dump_pdb(f"{out_dir}/repacked_{task_index}_{total_confs}.pdb")
        total_confs += 1

    t1 = time.time()
    print(f"Total Conformers Processed: {total_confs}")
    print(f"Time Taken: {(t1-t0)/60:.2f} minutes")

def main(argv):
    parser = argparse.ArgumentParser(description="Ligand perturbation search using PyRosetta")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument("task_index", nargs="?", default=0, type=int, help="Job array index (default=0)")
    args = parser.parse_args(argv)

    # Set a random seed based on the task index for independent, random perturbations
    random.seed(args.task_index)

    # Read the configuration file
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["grade_conformers"]

    # Ensure PyRosetta is in our Python path
    sys.path.append(default["PathToPyRosetta"])

    global pyrosetta, Pose, alignment, conformer_prep, collision_check
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    # Initialize PyRosetta (adjust flags as needed)
    pyrosetta.init("-mute all")

    params_list = default["ParamsList"]

    # Load Pre and Post PDBs
    print("Loading Pre and Post PDBs")
    if params_list.strip():
        pre_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(pre_pose, params_list=params_list.split(" "))
        pyrosetta.pose_from_file(pre_pose, res_set, default["PrePDBFileName"])

        post_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(post_pose, params_list=params_list.split(" "))
        pyrosetta.pose_from_file(post_pose, res_set, default["PostPDBFileName"])
    else:
        pre_pose = pyrosetta.pose_from_pdb(default["PrePDBFileName"])
        post_pose = pyrosetta.pose_from_pdb(default["PostPDBFileName"])

    # Mutate selected positions in the Post PDB (Glycine shave)
    positions = str(spec["GlycineShavePositions"]).split()
    pocket_shave_positions = [int(x) for x in positions]
    for i in range(1, post_pose.total_residue() + 1):
        if i in pocket_shave_positions:
            pyrosetta.toolbox.mutate_residue(post_pose, i, 'G')
    post_pose.dump_pdb('post_mutate.pdb')

    # Identify the target residue for alignment
    res = pre_pose.residue(pre_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))
    path_to_conformers = default["PathToConformers"]

    # Process the CSV file directly (without caching to a pickle)
    df = csv_to_df(default["CSVFileName"],
                   auto=(default["AutoGenerateAlignment"] == "True"),
                   path_to_conformers=path_to_conformers,
                   pose=pre_pose,
                   target_res=res,
                   lig_res_num=int(default["LigandResidueNumber"]))

    # Get other parameters from the configuration
    jump_num = int(spec["JumpNum"])
    rotation = float(spec["Rotation"])
    translation = float(spec["Translation"])
    upper_water_distance = float(spec["UpperWaterDistance"])
    lower_water_distance = float(spec["LowerWaterDistance"])
    backbone_clash_keep_sidechains = str(spec["BackboneClashKeepSidechains"]).split()
    max_pass_score = float(spec["MaxScore"])
    bin_width = float(spec["BinWidth"])
    vdw_modifier = float(spec["VDW_Modifier"])
    include_sc = (spec["IncludeSC"] == "True")
    lig_res_num = int(default["LigandResidueNumber"])

    print("Starting ligand perturbation search with task index:", args.task_index)
    align_to_residue_and_check_collision(post_pose, res, path_to_conformers, df,
                                         jump_num, rotation, translation,
                                         upper_water_distance, lower_water_distance,
                                         backbone_clash_keep_sidechains, max_pass_score,
                                         bin_width, vdw_modifier, include_sc,
                                         lig_res_num, args.task_index)

if __name__ == "__main__":
    main(sys.argv[1:])

