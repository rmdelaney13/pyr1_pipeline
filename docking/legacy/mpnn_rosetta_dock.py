import sys
from configparser import ConfigParser
import argparse
import os
import pandas as pd
import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import ReadResfile, IncludeCurrent, RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover


def apply_resfile(pose, resfile_path):
    """Apply mutations specified in a resfile to the pose."""
    print(f"Applying resfile: {resfile_path}")
    
    task_factory = TaskFactory()
    task_factory.push_back(ReadResfile(resfile_path))
    task_factory.push_back(IncludeCurrent())
    task_factory.push_back(RestrictToRepacking())
    
    packer = PackRotamersMover()
    packer.task_factory(task_factory)
    packer.apply(pose)
    return pose


def score_pose(pose):
    """Score a pose using Rosetta's scoring function."""
    sf = pyrosetta.get_fa_scorefxn()
    total_score = sf(pose)
    interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover("A_B")  # Assuming chains A and B
    interface_analyzer.apply(pose)
    interface_dG = interface_analyzer.get_interface_dG()
    return total_score, interface_dG


def main(argv):
    parser = argparse.ArgumentParser(description="Grade docking conformers using mutations from resfiles.")
    parser.add_argument("config_file", help="Path to the configuration file.")
    args = parser.parse_args(argv)
    
    # Read configuration
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    resfiles_dir = default["ResfilesDir"]
    docked_dir = default["DockedDir"]
    output_dir = default["OutputDir"]
    params_file = default["ParamsFile"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PyRosetta
    pyrosetta.init(f"-mute all -extra_res_fa {params_file}")
    
    results = []
    
    for resfile in sorted(os.listdir(resfiles_dir)):
        if not resfile.endswith(".resfile"):
            continue
        
        identifier = resfile.split("_output")[0]
        docked_file = os.path.join(docked_dir, f"{identifier}.pdb")
        if not os.path.exists(docked_file):
            print(f"Warning: Docked complex {docked_file} not found. Skipping.")
            continue
        
        pose = Pose()
        pyrosetta.pose_from_file(pose, docked_file)
        
        # Apply mutations
        resfile_path = os.path.join(resfiles_dir, resfile)
        pose = apply_resfile(pose, resfile_path)
        
        # Score the mutated pose
        total_score, interface_dG = score_pose(pose)
        print(f"Scores for {identifier}: Total Score = {total_score}, Interface ΔG = {interface_dG}")
        
        # Save mutated pose
        output_pdb = os.path.join(output_dir, f"{identifier}_mutated.pdb")
        pose.dump_pdb(output_pdb)
        print(f"Saved mutated pose to {output_pdb}")
        
        # Record results
        results.append({"identifier": identifier, "total_score": total_score, "interface_dG": interface_dG})
    
    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "scoring_results.csv"), index=False)
    print(f"Saved scoring results to {output_dir}/scoring_results.csv")


if __name__ == "__main__":
    main(sys.argv[1:])

