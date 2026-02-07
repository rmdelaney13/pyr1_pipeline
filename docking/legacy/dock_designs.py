import sys
import os
import time
import pandas as pd
from configparser import ConfigParser
import argparse

def parse_resfile(resfile_path):
    """Extract mutation positions and target amino acids from a resfile."""
    mutate_positions = []
    mutate_aa = []
    with open(resfile_path, "r") as f:
        for line in f:
            if "PIKAA" in line:
                parts = line.strip().split()
                chain = parts[0]
                position = int(parts[1])
                aa = parts[-1]
                mutate_positions.append((chain, position))
                mutate_aa.append(aa)
    return mutate_positions, mutate_aa

def apply_mutations(pose, mutations, aas):
    """Apply mutations to a pose."""
    for (chain, position), aa in zip(mutations, aas):
        pose_position = pose.pdb_info().pdb2pose(chain, position)
        if pose_position == 0:
            print(f"Warning: Position {position} on chain {chain} not found in pose.")
            continue
        pyrosetta.toolbox.mutate_residue(pose, pose_position, aa)

def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="your config file", default="my_conf.txt")
    args = parser.parse_args(argv)

    # Parse config file
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["grade_conformers"]

    # Import necessary PyRosetta dependencies
    global pyrosetta, Pose, alignment, conformer_prep, collision_check
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    pyrosetta.init("-mute all")

    resfiles_dir = default["ResfilesDir"]
    pdbs_dir = default["PathToPDBs"]
    path_to_docked_complexes = "/projects/ryde3462/software/kynurenic_acid1/pass_score_repacked"
    ligand_file = os.path.join(path_to_docked_complexes, f"repacked{identifier}.pdb")


    resfiles = sorted([f for f in os.listdir(resfiles_dir) if f.endswith(".resfile")])

    for resfile in resfiles:
        resfile_path = os.path.join(resfiles_dir, resfile)
        identifier = resfile.split('_')[0]
        protein_pdb = os.path.join(pdbs_dir, f"{identifier}.pdb")

        if not os.path.exists(protein_pdb):
            print(f"Missing PDB for resfile {resfile}. Skipping.")
            continue

        # Load protein pose
        protein_pose = Pose()
        pyrosetta.pose_from_file(protein_pose, protein_pdb)

        # Parse and apply mutations
        mutations, aas = parse_resfile(resfile_path)
        apply_mutations(protein_pose, mutations, aas)

        # Save mutated structure
        mutated_pdb = os.path.join("./mutated_pdbs", f"{identifier}_mutated.pdb")
        os.makedirs("./mutated_pdbs", exist_ok=True)
        protein_pose.dump_pdb(mutated_pdb)

        # Define the residue to align to (for ligand docking)
        target_res = protein_pose.residue(protein_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))

        # Prepare ligand docking
        ligand_pose = Pose()
        ligand_file = os.path.join(path_to_conformers, f"{identifier}.pdb")
        pyrosetta.pose_from_file(ligand_pose, ligand_file)

        print(f"Mutated and docking: {identifier}")

        # Dock and check collisions (from original align_to_residue_and_check_collision)
        align_to_residue_and_check_collision(
            protein_pose, target_res, path_to_conformers, None, None,
            int(spec["JumpNum"]), float(spec["Rotation"]), float(spec["Translation"]),
            float(spec["UpperWaterDistance"]), float(spec["LowerWaterDistance"]),
            str(spec["BackboneClashKeepSidechains"]).split(), float(spec["MaxScore"]),
            float(spec["BinWidth"]), float(spec["VDW_Modifier"]), spec["IncludeSC"] == "True", int(default["LigandResidueNumber"])
        )


if __name__ == "__main__":
    main(sys.argv[1:])

