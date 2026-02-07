#!/usr/bin/env python3
"""
Ligand Docking Pipeline with Grid Search

This script performs ligand docking and alignment using PyRosetta.
It reads a configuration file (e.g. config_multiple.txt) with sections such as:
  [DEFAULT], [MULTIPLE_PASSES], [grade_conformers], etc.
and a CSV file (e.g. nita_test_ligands.csv) containing ligand/conformer info.

This version removes pickle I/O and works solely with DataFrames and CSV output.
Repeated logic (file naming, ligand coordinate updates, water collision checking, etc.)
has been refactored into helper functions for clarity and efficiency.

A grid search routine is provided to explore perturbation parameters.
Use the flag "--grid_search" to run the grid search instead of the standard mode.
"""

import sys
import os
import time
import shutil
import pandas as pd
from configparser import ConfigParser
import argparse

# ---------------------- Helper Functions ---------------------- #
def update_ligand_coordinates(source_pose, conf, ligand_res_index):
    """
    Update ligand coordinates in conf.pose from the source_pose at the given residue.
    """
    try:
        natoms = source_pose.residue(ligand_res_index).natoms()
    except Exception as e:
        print(f"Error retrieving natoms for residue {ligand_res_index}: {e}")
        return
    for atom_id in range(1, natoms + 1):
        try:
            atom_coords = source_pose.residue(ligand_res_index).xyz(atom_id)
            conf.pose.residue(1).set_xyz(atom_id, atom_coords)
        except Exception as e:
            print(f"Error updating coordinates for residue {ligand_res_index} atom {atom_id}: {e}")
            continue

def apply_perturbation(pose, jump_num, rotation, translation):
    """
    Apply a rigid-body perturbation to the given pose.
    """
    from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
    mover = RigidBodyPerturbMover(jump_num, rotation, translation)
    mover.apply(pose)
    return pose

def get_water_oxygen_coords(pose):
    """
    Retrieve a list of coordinates for all water oxygen atoms in the pose.
    """
    coords = []
    for i in range(1, pose.total_residue() + 1):
        residue = pose.residue(i)
        if residue.is_water():
            try:
                natoms = residue.natoms()
            except Exception as e:
                print(f"Error getting natoms for water residue {i}: {e}")
                continue
            for atom_id in range(1, natoms + 1):
                try:
                    atom = residue.atom_type(atom_id)
                    if atom.element() == "O":
                        coords.append(residue.xyz(atom_id))
                except Exception as e:
                    print(f"Error retrieving atom type for water residue {i}, atom {atom_id}: {e}")
                    continue
    return coords

def get_output_filename(category, total_confs, output_dirs):
    """
    Return the output filename based on the file category and pass index.
    
    category: one of "rotated", "pass_score", "pass_fail", or "pass_initial".
    For "rotated", the file name is formatted as "rotated{n}.pdb"; for the others,
    it is "repacked{n}.pdb".
    """
    mapping = {
        "rotated": "rotated_dirs",
        "pass_score": "pass_score_repacked_dirs",
        "pass_fail": "pass_fail_repacked_dirs",
        "pass_initial": "pass_initial_only_repacked_dirs"
    }
    if output_dirs:
        pass_index = output_dirs['current_pass']
        directory = output_dirs[mapping[category]][pass_index]
    else:
        directory = category
    if category == "rotated":
        filename = f"rotated{total_confs}.pdb"
    else:
        filename = f"repacked{total_confs}.pdb"
    return os.path.join(directory, filename)

def load_csv_to_df(csv_file, auto, path_to_conformers, pose, target_res, lig_res_num):
    """
    Load and process the CSV file into a pandas DataFrame.
    
    Expected CSV columns:
      Molecule Name, Molecule ID, Conformer Range, Molecule Atoms, Target Atoms

    If auto is True, uses PyRosetta functions (via the alignment module)
    to generate alignment labels.
    """
    df = pd.read_csv(csv_file, index_col=False)
    df["Molecule File Stem"] = df["Molecule ID"].apply(lambda name: f"{name}/{name}")
    # Convert conformer range strings (e.g. "1_5_") into tuples (ignoring trailing underscore)
    df["Conformer Range"] = df["Conformer Range"].apply(lambda name: tuple(name.split("_")[:-1]))

    if auto:
        print("Auto Generating Alignments")
        for i, row in df.iterrows():
            print(f"Processing row {i+1}/{len(df)}", end=" ")
            from pyrosetta.rosetta.core.pose import Pose  # local import
            lig = Pose()
            mol_id = row["Molecule ID"]
            conf_num = 1
            params_file = f"{path_to_conformers}/{mol_id}/{mol_id}.params"
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list=[params_file])
            pdb_file = f"{path_to_conformers}/{mol_id}/{mol_id}_{conf_num:04}.pdb"
            pyrosetta.pose_from_file(lig, res_set, pdb_file)
            molecule_atoms, target_atoms = alignment.auto_align_residue_to_residue(
                lig, lig.residue(lig_res_num), target_res
            )
            df.loc[i, "Molecule Atoms"] = "-".join(molecule_atoms)
            df.loc[i, "Target Atoms"] = "-".join(target_atoms)
    # Convert atom labels from hyphen-separated strings to tuples.
    def parse_atoms(label):
        if label == "default":
            return ("CD2", "CZ2", "CZ3")
        return tuple(label.split("-"))
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(parse_atoms)
    df["Target Atoms"] = df["Target Atoms"].apply(parse_atoms)
    return df

# ---------------------- Core Function ---------------------- #
def align_to_residue_and_check_collision(
    pose,
    res,
    path_to_conformers,
    df,
    jump_num,
    rotation,
    translation,
    upper_water_distance,
    lower_water_distance,
    backbone_clash_keep_sidechains,
    max_pass_score=-300,
    bin_width=1,
    vdw_modifier=0.7,
    include_sc=False,
    lig_res_num=1,
    output_dirs=None
):
    """
    For each ligand conformer yielded from the DataFrame, perform:
      - Alignment to the target residue.
      - Insertion into the base pose.
      - Up to 10 perturbation attempts until the ligand satisfies backbone collision
        and water distance criteria.
      - Repacking and minimization.
      - Depending on score and collision, save the pose in a designated output folder.
    
    Returns the updated DataFrame with an added "Accepted Conformers" column.
    """
    import pyrosetta
    import conformer_prep
    import collision_check
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task import operation
    from pyrosetta.rosetta.protocols import minimization_packing as pack_min
    import pyrosetta.rosetta.protocols.grafting as graft

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

    # Setup packing.
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
    all_accepted = []           # List to store accepted conformers for each ligand.
    accepted_conformations = [] # Temporary list for current ligand.
    total_confs = 0

    # Set up base pose for merging ligand conformers.
    alignto_pose = pose

    # Prepare a modified pose for backbone collision grid.
    loop_include_pose = pose.clone()
    keep_indices = [int(x) for x in backbone_clash_keep_sidechains]
    for i in range(1, loop_include_pose.total_residue() + 1):
        if i not in keep_indices:
            pyrosetta.toolbox.mutate_residue(loop_include_pose, i, 'G')
    backbone_grid = collision_check.CollisionGrid(
        loop_include_pose, bin_width=bin_width, vdw_modifier=vdw_modifier, include_sc=True
    )

    # Process each ligand conformer.
    for pose_info in conformer_prep.yield_ligand_poses(
        df=df, path_to_conformers=path_to_conformers, post_accepted_conformers=False, ligand_residue=lig_res_num
    ):
        if not pose_info:
            # End of current ligand's conformers.
            all_accepted.append(accepted_conformations)
            accepted_conformations = []
            continue

        conf = pose_info
        conf.align_to_target(res)
        location = len(alignto_pose.chain_sequence(1))
        new_pose = graft.insert_pose_into_pose(alignto_pose, conf.pose, location)
        ligand_res_index = location + 1

        # Attempt perturbations (up to 10 tries).
        keep_list = []
        count = 0
        while not keep_list and count < 10:
            copy_pose = new_pose.clone()
            copy_pose = apply_perturbation(copy_pose, jump_num, rotation, translation)
            update_ligand_coordinates(copy_pose, conf, ligand_res_index)
            if not conf.check_collision(backbone_grid):
                water_coords = get_water_oxygen_coords(copy_pose)
                try:
                    natoms = copy_pose.residue(ligand_res_index).natoms()
                except Exception as e:
                    print(f"Error retrieving natoms for ligand residue {ligand_res_index}: {e}")
                    break
                for j in range(1, natoms + 1):
                    try:
                        atom_type = copy_pose.residue(ligand_res_index).atom_type(j)
                    except Exception as e:
                        print(f"Error retrieving atom type for ligand residue {ligand_res_index} atom {j}: {e}")
                        continue
                    if atom_type.element() in ["N", "O"]:
                        try:
                            atom_coords = copy_pose.residue(ligand_res_index).xyz(j)
                        except Exception as e:
                            print(f"Error retrieving coordinates for ligand residue {ligand_res_index} atom {j}: {e}")
                            continue
                        for w_coord in water_coords:
                            if lower_water_distance < atom_coords.distance(w_coord) < upper_water_distance:
                                keep_list.append(w_coord)
                                break
            count += 1

        if not keep_list:
            print("Conformer failed backbone clash and water distance check")
        else:
            rotated_file = get_output_filename("rotated", total_confs, output_dirs)
            copy_pose.dump_pdb(rotated_file)
            packer.apply(copy_pose)
            update_ligand_coordinates(copy_pose, conf, ligand_res_index)
            remove_ligand_pose = copy_pose.clone()
            remove_ligand_pose.delete_residue_slow(ligand_res_index)
            grid = collision_check.CollisionGrid(
                remove_ligand_pose, bin_width=bin_width, vdw_modifier=vdw_modifier, include_sc=include_sc
            )
            if not conf.check_collision(grid):
                accepted_conformations.append(conf.conf_num)
                score = sf_all(copy_pose)
                print(score)
                if score < max_pass_score:
                    conf.pose.pdb_info().name("pass_score")
                    pmm.apply(conf.pose)
                    out_file = get_output_filename("pass_score", total_confs, output_dirs)
                    copy_pose.dump_pdb(out_file)
                else:
                    conf.pose.pdb_info().name("fail_score")
                    pmm.apply(conf.pose)
                    out_file = get_output_filename("pass_fail", total_confs, output_dirs)
                    copy_pose.dump_pdb(out_file)
            else:
                out_file = get_output_filename("pass_initial", total_confs, output_dirs)
                copy_pose.dump_pdb(out_file)
        total_confs += 1

    # Ensure "Accepted Conformers" column length matches the DataFrame.
    if not all_accepted or len(all_accepted) != len(df):
        all_accepted = [[] for _ in range(len(df))]

    print("\n--- Output Summary ---")
    print(f"Number of Ligands: {len(all_accepted)}")
    ligands_accepted = len([e for e in all_accepted if e])
    print(f"Number of Ligands Accepted: {ligands_accepted}")
    print(f"Proportion of Ligands Accepted: {ligands_accepted/len(all_accepted) if all_accepted else 0}")
    total_accepted = sum(len(e) for e in all_accepted)
    print(f"Total Conformers: {total_confs}")
    print(f"Conformers Accepted: {total_accepted}")
    print(f"Proportion of Conformers Accepted: {total_accepted/total_confs if total_confs else 0}")
    tf_end = time.time()
    print(f"Time taken: {(tf_end - t0)/60:.2f} minutes")
    print(f"Conformers per minute: {total_confs/(tf_end - t0)*60:.2f}")

    df["Accepted Conformers"] = all_accepted
    return df

# ---------------------- Grid Search Function ---------------------- #
def grid_search_perturbation(
    base_pose,
    res,
    path_to_conformers,
    df,
    backbone_clash_keep_sidechains,
    lig_res_num,
    jump_num,
    rotation_list,
    translation_list,
    upper_water_distance,
    lower_water_distance,
    max_pass_score,
    bin_width,
    vdw_modifier,
    include_sc
):
    """
    Perform a grid search over given rotation angles and translation distances.
    
    For each combination, run the alignment and collision check procedure,
    then record metrics such as the total number of accepted conformers.
    
    Returns a list of dictionaries containing parameters and metric results.
    """
    import pyrosetta
    import conformer_prep
    import collision_check
    results = []
    for rotation in rotation_list:
        for translation in translation_list:
            print(f"Testing parameters: Rotation = {rotation}°, Translation = {translation}Å")
            test_pose = base_pose.clone()
            df_updated = align_to_residue_and_check_collision(
                pose=test_pose,
                res=res,
                path_to_conformers=path_to_conformers,
                df=df.copy(),
                jump_num=jump_num,
                rotation=rotation,
                translation=translation,
                upper_water_distance=upper_water_distance,
                lower_water_distance=lower_water_distance,
                backbone_clash_keep_sidechains=backbone_clash_keep_sidechains,
                max_pass_score=max_pass_score,
                bin_width=bin_width,
                vdw_modifier=vdw_modifier,
                include_sc=include_sc,
                lig_res_num=lig_res_num,
                output_dirs=None
            )
            # Compute metrics.
            accepted = df_updated.get("Accepted Conformers", [])
            total_accepted = sum(len(row) for row in accepted if isinstance(row, list))
            ligands_with_acceptance = sum(1 for row in accepted if row)
            results.append({
                "rotation": rotation,
                "translation": translation,
                "total_accepted": total_accepted,
                "ligands_accepted": ligands_with_acceptance
            })
    return results

# ---------------------- Main Function ---------------------- #
def main(argv):
    """
    Main execution pipeline:
      - Reads configuration and input files.
      - Initializes PyRosetta and loads pre/post PDBs.
      - Loads the CSV and processes ligand conformers.
      - Either runs multiple passes of alignment/collision checking or performs a grid search.
      - Aggregates output PDB files and saves a combined CSV.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="Path to the config file", default="config.ini", nargs="?")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search on perturbation parameters")
    parser.add_argument("-c", "--passes_collision_check", help="Whether to only show ones that pass collision check (True/False)", default="False")
    if not argv:
        parser.print_help()
        return
    args = parser.parse_args(argv)

    # Parse configuration file.
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["grade_conformers"]
    multiple = config["MULTIPLE_PASSES"]

    sys.path.append(default["PathToPyrosetta"])
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    auto = default.getboolean("AutoGenerateAlignment")
    pyrosetta.init("-mute all")

    print("Loading Pre and Post PDBs")
    params_list = default["ParamsList"]
    if params_list.strip():
        pre_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(pre_pose, params_list=params_list.split())
        pyrosetta.pose_from_file(pre_pose, res_set, default["PrePDBFileName"])
        post_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(post_pose, params_list=params_list.split())
        pyrosetta.pose_from_file(post_pose, res_set, default["PostPDBFileName"])
    else:
        pre_pose = pyrosetta.pose_from_pdb(default["PrePDBFileName"])
        post_pose = pyrosetta.pose_from_pdb(default["PostPDBFileName"])

    # Glycine shave: mutate designated residues to glycine.
    positions = spec["GlycineShavePositions"].split()
    pocket_shave_positions = [int(x) for x in positions]
    for i in range(1, post_pose.total_residue() + 1):
        if i in pocket_shave_positions:
            pyrosetta.toolbox.mutate_residue(post_pose, i, 'G')
    post_pose.dump_pdb('post_mutate.pdb')

    # Determine target residue using pre_pose.
    target_res_num = int(default["ResidueNumber"])
    chain_letter = default["ChainLetter"]
    pdb2pose = pre_pose.pdb_info().pdb2pose(chain_letter, target_res_num)
    res = pre_pose.residue(pdb2pose)

    path_to_conformers = default["PathToConformers"]
    lig_res_num = int(default["LigandResidueNumber"])
    print("Loading CSV and generating DataFrame")
    df = load_csv_to_df(default["CSVFileName"], auto, path_to_conformers, pre_pose, res, lig_res_num)
    print("CSV loaded and processed successfully.")

    if args.grid_search:
        # Define ranges for rotation and translation.
        rotation_range = [5, 15, 25, 35, 45, 65]      # degrees
        translation_range = [0.3, 0.5, 0.7,0.9,1.1]  # Ångstroms
        grid_results = grid_search_perturbation(
            base_pose=post_pose,
            res=res,
            path_to_conformers=path_to_conformers,
            df=df,
            backbone_clash_keep_sidechains=spec["BackboneClashKeepSidechains"].split(),
            lig_res_num=lig_res_num,
            jump_num=int(spec["JumpNum"]),
            rotation_list=rotation_range,
            translation_list=translation_range,
            upper_water_distance=float(spec["UpperWaterDistance"]),
            lower_water_distance=float(spec["LowerWaterDistance"]),
            max_pass_score=float(spec["MaxScore"]),
            bin_width=float(spec["BinWidth"]),
            vdw_modifier=float(spec["VDW_Modifier"]),
            include_sc=spec.getboolean("IncludeSC")
        )
        print("\n--- Grid Search Results ---")
        for res_item in grid_results:
            print(f"Rotation {res_item['rotation']}°, Translation {res_item['translation']}Å -> "
                  f"Total accepted: {res_item['total_accepted']}, "
                  f"Ligands accepted: {res_item['ligands_accepted']}")
    else:
        num_passes = multiple.getint("NumPasses", fallback=1)
        # Prepare output directories for each pass.
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
                os.makedirs(directory, exist_ok=True)
                print(f"Directory ensured: {directory}")
            rotated_dirs.append(rotated_dir)
            pass_score_repacked_dirs.append(pass_score_repacked_dir)
            pass_fail_repacked_dirs.append(pass_fail_repacked_dir)
            pass_initial_only_repacked_dirs.append(pass_initial_only_repacked_dir)

        combined_df = pd.DataFrame()
        # Run the docking routine for multiple passes.
        for pass_idx in range(1, num_passes + 1):
            print(f"\n--- Starting Pass {pass_idx} ---")
            output_dirs = {
                'current_pass': pass_idx - 1,  # zero-based index
                'rotated_dirs': rotated_dirs,
                'pass_score_repacked_dirs': pass_score_repacked_dirs,
                'pass_fail_repacked_dirs': pass_fail_repacked_dirs,
                'pass_initial_only_repacked_dirs': pass_initial_only_repacked_dirs
            }
            df = align_to_residue_and_check_collision(
                pose=post_pose,
                res=res,
                path_to_conformers=path_to_conformers,
                df=df,
                jump_num=int(spec["JumpNum"]),
                rotation=float(spec["Rotation"]),
                translation=float(spec["Translation"]),
                upper_water_distance=float(spec["UpperWaterDistance"]),
                lower_water_distance=float(spec["LowerWaterDistance"]),
                backbone_clash_keep_sidechains=spec["BackboneClashKeepSidechains"].split(),
                max_pass_score=float(spec["MaxScore"]),
                bin_width=float(spec["BinWidth"]),
                vdw_modifier=float(spec["VDW_Modifier"]),
                include_sc=spec.getboolean("IncludeSC"),
                lig_res_num=lig_res_num,
                output_dirs=output_dirs
            )
            combined_df = pd.concat([combined_df, df.copy()], ignore_index=True)
            print(f"Pass {pass_idx} completed and DataFrame updated.")

        # Aggregate repacked files.
        def aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes, final_dir_name='final_pass_score_repacked'):
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
            print(f"All repacked files have been aggregated into {final_dir}")

        aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes)
        combined_csv = "combined_" + os.path.basename(default["CSVFileName"])
        combined_df.to_csv(combined_csv, index=False)
        print(f"\nAll passes completed. Combined results saved to {combined_csv}")

if __name__ == "__main__":
    main(sys.argv[1:])
