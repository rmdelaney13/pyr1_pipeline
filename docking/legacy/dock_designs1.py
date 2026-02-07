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
    path_to_conformers = default["PathToConformers"]

    resfiles = sorted([f for f in os.listdir(resfiles_dir) if f.endswith(".resfile")])

    for resfile in resfiles:
        resfile_path = os.path.join(resfiles_dir, resfile)
        identifier = resfile.split('_')[0]
        protein_pdb = os.path.join(pdbs_dir, f"repacked{identifier}.pdb")

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
        mutated_pdb = os.path.join("./mutated_pdbs", f"repacked{identifier}_mutated.pdb")
        os.makedirs("./mutated_pdbs", exist_ok=True)
        protein_pose.dump_pdb(mutated_pdb)

        # Define the residue to align to (for ligand docking)
        target_res = protein_pose.residue(protein_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))

        # Prepare ligand docking
        ligand_pose = Pose()
        ligand_file = os.path.join(path_to_conformers, f"ligand_{identifier}.pdb")
        if not os.path.exists(ligand_file):
            print(f"Missing ligand file {ligand_file}. Skipping.")
            continue
        pyrosetta.pose_from_file(ligand_pose, ligand_file)

        print(f"Mutated and docking: repacked{identifier}")

        # Dock and check collisions (from original align_to_residue_and_check_collision)
        align_to_residue_and_check_collision(
            protein_pose, target_res, path_to_conformers, None, None,
            int(spec["JumpNum"]), float(spec["Rotation"]), float(spec["Translation"]),
            float(spec["UpperWaterDistance"]), float(spec["LowerWaterDistance"]),
            str(spec["BackboneClashKeepSidechains"]).split(), float(spec["MaxScore"]),
            float(spec["BinWidth"]), float(spec["VDW_Modifier"]), spec["IncludeSC"] == "True", int(default["LigandResidueNumber"])
        )

