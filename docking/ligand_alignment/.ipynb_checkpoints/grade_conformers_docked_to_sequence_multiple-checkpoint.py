import sys
import os
import time
import shutil
import pandas as pd
from configparser import ConfigParser
import argparse

def csv_to_df_pkl(csv_file_name, pkl_file_name, auto, path_to_conformers, pose, target_res, lig_res_num):
    """
    Automatically converts the CSV (made by create_table) into a DataFrame with all the necessary information.
    """
    df = pd.read_csv(csv_file_name, index_col=False)
    if pkl_file_name is None:
        pkl_file_name = f"{csv_file_name[:-4]}.pkl"

    # Creating the "file stem"
    def create_file_stem(name):
        return f"{name}/{name}"
    df["Molecule File Stem"] = df["Molecule ID"].apply(create_file_stem)

    # Turning the conformer range into a tuple
    def split_crange(name):
        lst = name.split("_")
        return (lst[0], lst[1])
    df["Conformer Range"] = df["Conformer Range"].apply(split_crange)

    # Auto-generating alignments if specified
    if auto:
        print("Auto Generating Alignments")
        for i, row in df.iterrows():
            print(f"{i} ({i+1}/{len(df)})", end=" ")
            lig = Pose()
            mol_id = row["Molecule ID"]
            conf_num = 1
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list=[f"{path_to_conformers}/{mol_id}/{mol_id}.params"])
            pyrosetta.pose_from_file(lig, res_set, f"{path_to_conformers}/{mol_id}/{mol_id}_{conf_num:04}.pdb")

            molecule_atoms, target_atoms = alignment.auto_align_residue_to_residue(lig, lig.residue(lig_res_num), target_res)
            df.loc[i, "Molecule Atoms"] = "-".join(molecule_atoms)
            df.loc[i, "Target Atoms"] = "-".join(target_atoms)

    # Turning molecule and target atoms into tuples
    def split_alabels(name):
        if name == "default":
            return ("CD2", "CZ2", "CZ3")
        lst = name.split("-")
        return tuple(lst)
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(split_alabels)
    df["Target Atoms"] = df["Target Atoms"].apply(split_alabels)

    # Serializing the resultant DataFrame
    df.to_pickle(pkl_file_name, protocol=4)

def align_to_residue_and_check_collision(pose, res, path_to_conformers, df, pkl_file,
                                         jump_num, rotation, translation,
                                         upper_water_distance, lower_water_distance,
                                         backbone_clash_keep_sidechains, max_pass_score=-300,
                                         bin_width=1, vdw_modifier=0.7, include_sc=False,
                                         lig_res_num=1, output_dirs=None):
    """
    Aligns and then checks for collisions.
    If an output_dirs dict is provided, file dumps are written to pass-specific directories.
    """
    import pyrosetta.rosetta.protocols.rigid as rigid_moves
    import pyrosetta.rosetta.protocols.grafting as graft
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.protocols import minimization_packing as pack_min

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

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

    # Use the provided pose as the base (merged pose)
    alignto_pose = pose

    # Prepare a backbone grid pose by mutating non-critical positions to glycine
    loop_include_sidechain_pose = pose.clone()
    loop_include_sidechain = [int(x) for x in backbone_clash_keep_sidechains]
    for i in range(1, loop_include_sidechain_pose.total_residue() + 1):
        if i in loop_include_sidechain:
            continue
        else:
            pyrosetta.toolbox.mutate_residue(loop_include_sidechain_pose, i, 'G')
    backbone_grid = collision_check.CollisionGrid(loop_include_sidechain_pose, bin_width=bin_width,
                                                    vdw_modifier=vdw_modifier, include_sc=True)

    # Iterate over all ligand conformers
    for pose_info in conformer_prep.yield_ligand_poses(df=df, path_to_conformers=path_to_conformers,
                                                        post_accepted_conformers=False, ligand_residue=lig_res_num):
        if not pose_info:
            print(f"{conf.name}, {conf.id}: {len(accepted_conformations)/conf.conf_num}")
            all_accepted.append(accepted_conformations)
            accepted_conformations = []
            continue

        conf = pose_info
        conf.align_to_target(res)

        # Graft the ligand onto the base pose
        location = len(alignto_pose.chain_sequence(1))
        new_pose = graft.insert_pose_into_pose(alignto_pose, conf.pose, location)
        ligand_res_index = location + 1

        # Try ligand perturbations up to 10 times
        keep_list = []
        count = 0
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
                for i in range(1, copy_pose.total_residue()+1):
                    residue = copy_pose.residue(i)
                    if residue.is_water():
                        water_resid.append(i)
                for j in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                    atom = copy_pose.residue(ligand_res_index).atom_type(j)
                    if atom.element() in ["N", "O"]:
                        atom_coords = copy_pose.residue(ligand_res_index).xyz(j)
                        for i in range(len(water_resid)):
                            for w in range(1, copy_pose.residue(water_resid[i]).natoms() + 1):
                                atom_w = copy_pose.residue(water_resid[i]).atom_type(w)
                                if atom_w.element() == "O":
                                    water_coord = copy_pose.residue(water_resid[i]).xyz(w)
                            distance_to_water = atom_coords.distance(water_coord)
                            if lower_water_distance < distance_to_water < upper_water_distance:
                                keep_list.append(i)
            count += 1

        if len(keep_list) == 0:
            print("Conformer failed backbone clash and water distance check")
        else:
            # Dump the rotated pose to the appropriate directory
            if output_dirs:
                pass_idx = output_dirs['current_pass']
                rotated_dir = output_dirs['rotated_dirs'][pass_idx]
                filename = os.path.join(rotated_dir, f"rotated{total_confs}.pdb")
            else:
                filename = os.path.join("rotated", f"rotated{total_confs}.pdb")
            copy_pose.dump_pdb(filename)

            packer.apply(copy_pose)

            # Update ligand coordinates
            ligand_res_index = location + 1
            for atom_id in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                atom_coords = copy_pose.residue(ligand_res_index).xyz(atom_id)
                conf.pose.residue(1).set_xyz(atom_id, atom_coords)

            # Collision check after repacking
            remove_ligand_pose = copy_pose.clone()
            remove_ligand_pose.delete_residue_slow(ligand_res_index)
            grid = collision_check.CollisionGrid(remove_ligand_pose, bin_width=bin_width,
                                                 vdw_modifier=vdw_modifier, include_sc=include_sc)
            does_collide = conf.check_collision(grid)
            if not does_collide:
                accepted_conformations.append(conf.conf_num)
                score = sf_all(copy_pose)
                print(score)
                if score < max_pass_score:
                    conf.pose.pdb_info().name("pass_score")
                    pmm.apply(conf.pose)
                    if output_dirs:
                        pass_idx = output_dirs['current_pass']
                        repacked_dir = output_dirs['pass_score_repacked_dirs'][pass_idx]
                        output_path = os.path.join(repacked_dir, f"repacked{total_confs}.pdb")
                    else:
                        output_path = os.path.join("pass_score_repacked", f"repacked{total_confs}.pdb")
                    copy_pose.dump_pdb(output_path)
                else:
                    conf.pose.pdb_info().name("fail_score")
                    pmm.apply(conf.pose)
                    if output_dirs:
                        pass_idx = output_dirs['current_pass']
                        repacked_dir = output_dirs['pass_fail_repacked_dirs'][pass_idx]
                        output_path = os.path.join(repacked_dir, f"repacked{total_confs}.pdb")
                    else:
                        output_path = os.path.join("pass_fail-score_repacked", f"repacked{total_confs}.pdb")
                    copy_pose.dump_pdb(output_path)
            else:
                if output_dirs:
                    pass_idx = output_dirs['current_pass']
                    repacked_dir = output_dirs['pass_initial_only_repacked_dirs'][pass_idx]
                    output_path = os.path.join(repacked_dir, f"repacked{total_confs}.pdb")
                else:
                    output_path = os.path.join("pass_initial-only_repacked", f"repacked{total_confs}.pdb")
                copy_pose.dump_pdb(output_path)
        total_confs += 1

    print(f"\n\n---Output, {pkl_file}---")
    print(f"\nNumber of Ligands: {len(all_accepted)}")
    ligands_accepted = len([e for e in all_accepted if e])
    print(f"Number of Ligands Accepted: {ligands_accepted}")
    print(f"Proportion of Ligands Accepted: {ligands_accepted/len(all_accepted) if all_accepted else 'N/A'}")
    total_accepted = sum([len(e) for e in all_accepted])
    print(f"\nNumber of Conformers: {total_confs}")
    print(f"Number of Conformers Accepted: {total_accepted}")
    print(f"Proportion of Conformers Accepted: {total_accepted/total_confs if total_confs else 'N/A'}")
    tf = time.time()
    print(f"\nTime taken: {(tf-t0)/60} minutes")
    print(f"Conformers per minute: {total_confs/(tf-t0)*60}")
    
    df["Accepted Conformers"] = all_accepted
    df.to_pickle(pkl_file, protocol=4)
    return total_confs

def aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes, final_dir_name='final_pass_score_repacked'):
    """
    Aggregates all pass_score_repacked directories into one final directory.
    """
    final_dir = os.path.join(os.getcwd(), final_dir_name)
    os.makedirs(final_dir, exist_ok=True)
    print(f"\nCreated final aggregation directory: {final_dir}")
    for pass_idx, source_dir in enumerate(pass_score_repacked_dirs, start=1):
        if not os.path.isdir(source_dir):
            print(f"Source directory {source_dir} does not exist. Skipping.")
            continue
        for file_name in os.listdir(source_dir):
            if file_name.startswith('repacked') and file_name.endswith('.pdb'):
                src_path = os.path.join(source_dir, file_name)
                new_file_name = f"pass{pass_idx}_{file_name}"
                dest_path = os.path.join(final_dir, new_file_name)
                try:
                    shutil.copyfile(src_path, dest_path)
                    print(f"Copied {src_path} to {dest_path}")
                except Exception as e:
                    print(f"Failed to copy {src_path} to {dest_path}: {e}")
    print(f"All pass_score_repacked files have been aggregated into {final_dir}")

def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="your config file", default="my_conf.txt")
    parser.add_argument("-c", "--passes_collision_check", help="Whether to only show ones that pass collision check (True/False)", default="False")
    if len(argv) == 0:
        parser.print_help()
        return
    args = parser.parse_args(argv)

    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["grade_conformers"]
    multiple = config["MULTIPLE_PASSES"] if "MULTIPLE_PASSES" in config else {}
    sys.path.append(default["PathToPyRosetta"])
    auto = default["AutoGenerateAlignment"] == "True"

    global pyrosetta, Pose, alignment, conformer_prep, collision_check, csv_to_df_pkl
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check
    from grade_conformers_docked_to_sequence import csv_to_df_pkl

    pyrosetta.init("-mute all")
    params_list = default["ParamsList"]

    print("Reading in Pre and Post PDBs")
    if len(params_list) > 0:
        pre_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(pre_pose, params_list=params_list.split(" "))
        pyrosetta.pose_from_file(pre_pose, res_set, default["PrePDBFileName"])
        post_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(post_pose, params_list=params_list.split(" "))
        pyrosetta.pose_from_file(post_pose, res_set, default["PostPDBFileName"])
    else:
        pre_pose = pyrosetta.pose_from_pdb(default["PrePDBFileName"])
        post_pose = pyrosetta.pose_from_pdb(default["PostPDBFileName"])

    positions = str(spec["MutatePositions"]).split()
    mutant_sidechain_positions = [int(x) for x in positions]
    AA_values = str(spec["MutateAA"]).split()
    mutant_sidechain_dict = dict(zip(mutant_sidechain_positions, AA_values))
    for i in range(1, post_pose.total_residue() + 1):
        if i in mutant_sidechain_positions:
            mutation = mutant_sidechain_dict[i]
            pyrosetta.toolbox.mutate_residue(post_pose, i, mutation)
    post_pose.dump_pdb('post_mutate.pdb')

    res = pre_pose.residue(pre_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))
    path_to_conformers = default["PathToConformers"]
    pkl_file_name = default["PKLFileName"]
    if pkl_file_name.strip() == "":
        pkl_file_name = None
    print(pkl_file_name)
    bin_width = float(spec["BinWidth"])
    vdw_modifier = float(spec["VDW_Modifier"])
    include_sc = spec["IncludeSC"] == "True"
    lig_res_num = int(default["LigandResidueNumber"])
    jump_num = int(spec["JumpNum"])
    rotation = float(spec["Rotation"])
    translation = float(spec["Translation"])
    upper_water_distance = float(spec["UpperWaterDistance"])
    lower_water_distance = float(spec["LowerWaterDistance"])
    backbone_clash_keep_sidechains = str(spec["BackboneClashKeepSidechains"]).split()
    max_pass_score = float(spec["MaxScore"])

    print("Attempting to read in .pkl file")
    try:
        df = pd.read_pickle(pkl_file_name)
    except:
        print(".pkl file not found, generating one instead (this is normal)")
        csv_to_df_pkl(default["CSVFileName"], pkl_file_name, auto, path_to_conformers, pre_pose, res, lig_res_num)
        df = pd.read_pickle(pkl_file_name)

    # Set up directories for each pass
    num_passes = int(multiple["NumPasses"]) if multiple and "NumPasses" in multiple else 1
    rotated_dirs = []
    pass_score_repacked_dirs = []
    pass_fail_repacked_dirs = []
    pass_initial_only_repacked_dirs = []
    for pass_idx in range(1, num_passes + 1):
        rotated_dir = f'rotated_{pass_idx}'
        pass_score_repacked_dir = f'pass_score_repacked_{pass_idx}'
        pass_fail_repacked_dir = f'pass_fail-score_repacked_{pass_idx}'
        pass_initial_only_repacked_dir = f'pass_initial-only_repacked_{pass_idx}'
        for directory in [rotated_dir, pass_score_repacked_dir, pass_fail_repacked_dir, pass_initial_only_repacked_dir]:
            try:
                os.mkdir(directory)
                print(f"Created directory: {directory}")
            except FileExistsError:
                print(f"Directory {directory} already exists.")
        rotated_dirs.append(rotated_dir)
        pass_score_repacked_dirs.append(pass_score_repacked_dir)
        pass_fail_repacked_dirs.append(pass_fail_repacked_dir)
        pass_initial_only_repacked_dirs.append(pass_initial_only_repacked_dir)

    combined_df = pd.DataFrame()
    for pass_idx in range(1, num_passes + 1):
        print(f"\n--- Starting Pass {pass_idx} ---")
        output_dirs = {
            'current_pass': pass_idx - 1,  # zero-indexed
            'rotated_dirs': rotated_dirs,
            'pass_score_repacked_dirs': pass_score_repacked_dirs,
            'pass_fail_repacked_dirs': pass_fail_repacked_dirs,
            'pass_initial_only_repacked_dirs': pass_initial_only_repacked_dirs
        }
        align_to_residue_and_check_collision(post_pose, res, path_to_conformers, df, pkl_file_name,
                                             jump_num, rotation, translation, upper_water_distance,
                                             lower_water_distance, backbone_clash_keep_sidechains,
                                             max_pass_score, bin_width, vdw_modifier, include_sc,
                                             lig_res_num, output_dirs)
        try:
            pass_df = pd.read_pickle(pkl_file_name)
            combined_df = pd.concat([combined_df, pass_df], ignore_index=True)
            print(f"Pass {pass_idx} DataFrame loaded and merged.")
        except Exception as e:
            print(f"Error loading pickle file for pass {pass_idx}: {e}")

    aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes)
    combined_pkl = "combined_" + pkl_file_name if pkl_file_name else "combined_results.pkl"
    combined_df.to_pickle(combined_pkl, protocol=4)
    print(f"\nAll passes completed. Combined results saved to {combined_pkl}")

if __name__ == "__main__":
    main(sys.argv[1:])
