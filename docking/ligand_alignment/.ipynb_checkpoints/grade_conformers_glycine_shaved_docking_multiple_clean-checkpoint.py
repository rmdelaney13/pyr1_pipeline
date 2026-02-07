#!/usr/bin/env python3
"""
Ligand Docking Pipeline

This script performs ligand docking and alignment using PyRosetta.
It reads a configuration file (e.g. config_multiple.txt) that should have
the following sections and keys:

[DEFAULT]
  - PathToRosetta: Path to your Rosetta installation.
  - PathToPyrosetta: Path to your PyRosetta installation.
  - PrePDBFileName: PDB file for the unmodified protein.
  - PostPDBFileName: PDB file after mutation/deletion.
  - ResidueNumber: Residue number of the target structural feature.
  - ChainLetter: Chain identifier for the target residue.
  - PathToConformers: Directory containing conformer files.
  - CSVFileName: Name of the CSV file containing ligand/conformer info.
  - PKLFileName: (Ignored in this version.)
  - ParamsList: (Optional) List of extra params files.
  - LigandResidueNumber: Ligand residue index.
  - AutoGenerateAlignment: Whether to automatically generate atom labels.

[MULTIPLE_PASSES]
  - NumPasses: Number of sampling passes to perform.
  - OutputDirBase: Base directory for pass-specific outputs.

[create_table]
  - MoleculeSDFs: SDF files for molecules (not used directly here).

[grade_conformers]
  - BinWidth: Width of the collision grid.
  - VDW_Modifier: Factor for van der Waals radii.
  - IncludeSC: Whether to include sidechain atoms for collision checking.
  - JumpNum: Jump number for ligand perturbation.
  - Rotation: Mean rotation (degrees) for ligand perturbation.
  - Translation: Mean translation (angstroms) for ligand perturbation.
  - UpperWaterDistance: Upper bound for water H-bond search.
  - LowerWaterDistance: Lower bound for water H-bond search.
  - BackboneClashKeepSidechains: Residue positions (space‐separated) to keep sidechains.
  - MaxScore: Score cutoff for accepting conformers.
  - GlycineShavePositions: Residue positions to “glycine shave” (for docking).

[rosetta_design]
  - (Design-related options; not used in this script)

CSV File Format:
  The CSV (e.g. nita_test_ligands.csv) should include the columns:
    Molecule Name, Molecule ID, Conformer Range, Molecule Atoms, Target Atoms

Example CSV content:
    Molecule Name,Molecule ID,Conformer Range,Molecule Atoms,Target Atoms
    0,0,1_5_,O2-N4-O1,O2-C10-C9
    0,0,1_5_,O1-N4-O2,O2-C10-C9

This version removes any pickle file I/O and works solely with DataFrames
in memory. Final results from multiple passes are aggregated and saved as a CSV.
"""

import sys
import os
import time
import shutil
import pandas as pd
from configparser import ConfigParser
import argparse

def load_csv_to_df(csv_file, auto, path_to_conformers, pose, target_res, lig_res_num):
    """
    Load and process the CSV file into a pandas DataFrame.

    This function reads the CSV file (which should contain columns like
    Molecule Name, Molecule ID, Conformer Range, Molecule Atoms, and Target Atoms)
    and then does the following:
      - Creates a file stem for each molecule.
      - Converts the conformer range from a string to a tuple.
      - If auto-alignment is enabled, it uses PyRosetta functions to generate
        alignment labels for the molecule and target atoms.
      - Finally, converts the atoms labels from hyphen-separated strings to tuples.

    Args:
        csv_file (str): Path to the CSV file.
        auto (bool): Whether to auto-generate atom alignments.
        path_to_conformers (str): Directory where conformer files are stored.
        pose (Pose): Pose used for alignment (e.g. from the pre-PDB file).
        target_res (Residue): Target residue for alignment.
        lig_res_num (int): Ligand residue number.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_csv(csv_file, index_col=False)

    # Create a "file stem" indicating where each ligand file is stored.
    df["Molecule File Stem"] = df["Molecule ID"].apply(lambda name: f"{name}/{name}")

    # Convert Conformer Range strings (e.g. "1_5_") to tuples.
    df["Conformer Range"] = df["Conformer Range"].apply(lambda name: tuple(name.split("_")[:-1]))

    # If auto-alignment is enabled, generate the atom labels automatically.
    if auto:
        print("Auto Generating Alignments")
        for i, row in df.iterrows():
            print(f"Processing row {i+1}/{len(df)}", end=" ")
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

    # Convert the atoms columns to tuples (using defaults if necessary).
    def parse_atoms(label):
        if label == "default":
            return ("CD2", "CZ2", "CZ3")
        return tuple(label.split("-"))
    
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(parse_atoms)
    df["Target Atoms"] = df["Target Atoms"].apply(parse_atoms)

    return df

def aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes, final_dir_name='final_pass_score_repacked'):
    """
    Aggregate repacked PDB files from different passes into one directory.

    Files are renamed to include the pass number to avoid conflicts.

    Args:
        pass_score_repacked_dirs (list): List of directories with repacked files.
        num_passes (int): Total number of passes executed.
        final_dir_name (str): Name of the final aggregation directory.
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

    print(f"All repacked files have been aggregated into {final_dir}")

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
    Align each ligand conformer to the target residue and perform collision checking.

    For each ligand conformer from the DataFrame, the following steps are performed:
      - Align the conformer to the target residue.
      - Insert the conformer into the base pose.
      - Attempt up to 10 perturbations (rigid-body moves) until a perturbation
        satisfies collision and water distance criteria.
      - Repack and minimize the pose.
      - Depending on the score (compared to max_pass_score), dump the pose
        into one of several output directories.

    Args:
        pose (Pose): Base pose for docking.
        res (Residue): Target residue for alignment.
        path_to_conformers (str): Directory with conformer files.
        df (pd.DataFrame): DataFrame containing ligand/conformer information.
        jump_num (int): Jump number for rigid body perturbation.
        rotation (float): Rotation angle for perturbation.
        translation (float): Translation distance for perturbation.
        upper_water_distance (float): Upper bound for water distance check.
        lower_water_distance (float): Lower bound for water distance check.
        backbone_clash_keep_sidechains (list): Residue positions to keep sidechains.
        max_pass_score (float): Score threshold for accepting conformers.
        bin_width (float): Grid width for collision checking.
        vdw_modifier (float): Modifier for van der Waals radii.
        include_sc (bool): Whether to include side chains in collision checking.
        lig_res_num (int): Ligand residue number.
        output_dirs (dict): Dictionary of output directory paths.

    Returns:
        pd.DataFrame: The input DataFrame updated with an "Accepted Conformers" column.
    """
    # Import PyRosetta modules used only within this function.
    import pyrosetta.rosetta.protocols.rigid as rigid_moves
    import pyrosetta.rosetta.protocols.grafting as graft
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task import operation
    from pyrosetta.rosetta.protocols import minimization_packing as pack_min

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

    # Setup score functions and packing tasks.
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

    all_accepted = []           # List of accepted conformers for each ligand.
    accepted_conformations = [] # Temporary list for current ligand.
    total_confs = 0

    # Use the provided pose as the base to merge ligand conformers.
    alignto_pose = pose

    # Create a copy of the pose for backbone collision grid calculation.
    loop_include_sidechain_pose = pose.clone()
    keep_list_indices = [int(x) for x in backbone_clash_keep_sidechains]
    for i in range(1, loop_include_sidechain_pose.total_residue() + 1):
        if i not in keep_list_indices:
            pyrosetta.toolbox.mutate_residue(loop_include_sidechain_pose, i, 'G')

    # Pre-calculate the collision grid for the protein backbone.
    backbone_grid = collision_check.CollisionGrid(
        loop_include_sidechain_pose,
        bin_width=bin_width,
        vdw_modifier=vdw_modifier,
        include_sc=True
    )

    # Iterate over each ligand conformer yielded by conformer_prep.
    for pose_info in conformer_prep.yield_ligand_poses(
        df=df,
        path_to_conformers=path_to_conformers,
        post_accepted_conformers=False,
        ligand_residue=lig_res_num
    ):
        if not pose_info:
            print(f"{conf.name}, {conf.id}: {len(accepted_conformations)}/{conf.conf_num}")
            all_accepted.append(accepted_conformations)
            accepted_conformations = []
            continue

        conf = pose_info  # Current ligand conformer.

        # Align the conformer to the target residue.
        conf.align_to_target(res)

        # Insert the ligand conformer into the base pose.
        location = len(alignto_pose.chain_sequence(1))
        new_pose = graft.insert_pose_into_pose(alignto_pose, conf.pose, location)
        ligand_res_index = location + 1

        # Attempt perturbations until criteria are met (up to 10 tries).
        keep_list = []
        count = 0
        while len(keep_list) == 0 and count < 10:
            copy_pose = new_pose.clone()
            pert_mover = rigid_moves.RigidBodyPerturbMover(jump_num, rotation, translation)
            pert_mover.apply(copy_pose)

            # Update ligand coordinates in the conformer.
            for atom_id in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                atom_coords = copy_pose.residue(ligand_res_index).xyz(atom_id)
                conf.pose.residue(1).set_xyz(atom_id, atom_coords)

            does_collide = conf.check_collision(backbone_grid)

            if not does_collide:
                # Check for acceptable water distances.
                water_resid = []
                for i in range(1, copy_pose.total_residue() + 1):
                    if copy_pose.residue(i).is_water():
                        water_resid.append(i)

                for j in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                    atom = copy_pose.residue(ligand_res_index).atom_type(j)
                    if atom.element() in ["N", "O"]:
                        atom_coords = copy_pose.residue(ligand_res_index).xyz(j)
                        for i in water_resid:
                            for w in range(1, copy_pose.residue(i).natoms() + 1):
                                water_atom = copy_pose.residue(i).atom_type(w)
                                if water_atom.element() == "O":
                                    water_coord = copy_pose.residue(i).xyz(w)
                            if lower_water_distance < atom_coords.distance(water_coord) < upper_water_distance:
                                keep_list.append(i)
            count += 1

        if len(keep_list) == 0:
            print("Conformer failed backbone clash and water distance check")
        else:
            # Save the rotated pose.
            if output_dirs:
                pass_index = output_dirs['current_pass']
                rotated_dir = output_dirs['rotated_dirs'][pass_index]
                filename = os.path.join(rotated_dir, f"rotated{total_confs}.pdb")
            else:
                filename = os.path.join('rotated', f"rotated{total_confs}.pdb")
            copy_pose.dump_pdb(filename)

            # Repack and minimize the pose.
            packer.apply(copy_pose)

            for atom_id in range(1, copy_pose.residue(ligand_res_index).natoms() + 1):
                atom_coords = copy_pose.residue(ligand_res_index).xyz(atom_id)
                conf.pose.residue(1).set_xyz(atom_id, atom_coords)

            # Recalculate collision grid after repacking.
            remove_ligand_pose = copy_pose.clone()
            remove_ligand_pose.delete_residue_slow(ligand_res_index)
            grid = collision_check.CollisionGrid(
                remove_ligand_pose,
                bin_width=bin_width,
                vdw_modifier=vdw_modifier,
                include_sc=include_sc
            )
            does_collide = conf.check_collision(grid)

            if not does_collide:
                accepted_conformations.append(conf.conf_num)
                score = sf_all(copy_pose)
                print(score)

                if score < max_pass_score:
                    conf.pose.pdb_info().name("pass_score")
                    pmm.apply(conf.pose)
                    if output_dirs:
                        pass_index = output_dirs['current_pass']
                        pass_score_dir = output_dirs['pass_score_repacked_dirs'][pass_index]
                        output_path = os.path.join(pass_score_dir, f"repacked{total_confs}.pdb")
                    else:
                        output_path = os.path.join('pass_score_repacked', f"repacked{total_confs}.pdb")
                    copy_pose.dump_pdb(output_path)
                else:
                    conf.pose.pdb_info().name("fail_score")
                    pmm.apply(conf.pose)
                    if output_dirs:
                        pass_index = output_dirs['current_pass']
                        pass_fail_dir = output_dirs['pass_fail_repacked_dirs'][pass_index]
                        output_path = os.path.join(pass_fail_dir, f"repacked{total_confs}.pdb")
                    else:
                        output_path = os.path.join('pass_fail-score_repacked', f"repacked{total_confs}.pdb")
                    copy_pose.dump_pdb(output_path)
            else:
                if output_dirs:
                    pass_index = output_dirs['current_pass']
                    initial_dir = output_dirs['pass_initial_only_repacked_dirs'][pass_index]
                    output_path = os.path.join(initial_dir, f"repacked{total_confs}.pdb")
                else:
                    output_path = os.path.join('pass_initial-only_repacked', f"repacked{total_confs}.pdb")
                copy_pose.dump_pdb(output_path)

        total_confs += 1

    # Print summary statistics.
    print(f"\n--- Output Summary ---")
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

    # Update the DataFrame with accepted conformers info.
    df["Accepted Conformers"] = all_accepted
    return df

def main(argv):
    """
    Main pipeline execution.
    
    Reads the configuration file, initializes PyRosetta, loads input PDBs and the CSV file,
    then performs multiple passes of ligand alignment and collision checking.
    Final outputs are saved as separate PDB files (in pass-specific directories) and
    a combined CSV with results.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="Path to the config file", default="config.ini", nargs="?")
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

    # Add PyRosetta path and initialize PyRosetta.
    sys.path.append(default["PathToPyrosetta"])
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    auto = default.getboolean("AutoGenerateAlignment")
    pyrosetta.init("-mute all")  

    # Load pre and post PDB files.
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

    # Glycine shave: mutate specified residues in post_pose.
    positions = spec["GlycineShavePositions"].split()
    pocket_shave_positions = [int(x) for x in positions]
    for i in range(1, post_pose.total_residue() + 1):
        if i in pocket_shave_positions:
            pyrosetta.toolbox.mutate_residue(post_pose, i, 'G')
    post_pose.dump_pdb('post_mutate.pdb')

    # Determine the target residue for alignment using pre_pose.
    target_res_num = int(default["ResidueNumber"])
    chain_letter = default["ChainLetter"]
    pdb2pose = pre_pose.pdb_info().pdb2pose(chain_letter, target_res_num)
    res = pre_pose.residue(pdb2pose)

    # Load CSV into a DataFrame.
    path_to_conformers = default["PathToConformers"]
    lig_res_num = int(default["LigandResidueNumber"])
    print("Loading CSV and generating DataFrame")
    df = load_csv_to_df(default["CSVFileName"], auto, path_to_conformers, pre_pose, res, lig_res_num)
    print("CSV loaded and processed successfully.")

    # Get multiple passes configuration.
    num_passes = multiple.getint("NumPasses", fallback=1)
    output_dir_base = multiple.get("OutputDirBase", fallback="pass_")

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

    # Loop through each pass, perform alignment and collision checking, and update the DataFrame.
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

    # Aggregate repacked files from all passes.
    aggregate_pass_score_repacked(pass_score_repacked_dirs, num_passes)

    # Save the combined results as a CSV file.
    combined_csv = "combined_" + os.path.basename(default["CSVFileName"])
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nAll passes completed. Combined results saved to {combined_csv}")

if __name__ == "__main__":
    main(sys.argv[1:])
