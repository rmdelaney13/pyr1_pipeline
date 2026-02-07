import argparse
import csv
import fnmatch
import os
import re
import sys
from configparser import ConfigParser
from itertools import combinations


def _safe_get(config, section, key, fallback=None):
    if section in config and key in config[section]:
        return config[section][key]
    if key in config["DEFAULT"]:
        return config["DEFAULT"][key]
    return fallback


def _safe_getbool(config, section, key, fallback=False):
    raw = _safe_get(config, section, key, None)
    if raw is None:
        return fallback
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _split_patterns(raw_patterns):
    return [item for item in raw_patterns.split(" ") if item.strip()]


def _parse_target_atom_triplets(raw):
    if raw is None:
        return []
    items = re.split(r"[;\n]+", raw.strip())
    out = []
    for item in items:
        token = item.strip()
        if not token:
            continue
        atoms = tuple([x.strip() for x in token.split("-") if x.strip()])
        if len(atoms) != 3:
            raise ValueError(f"Target atom triplet must have 3 atoms, got: '{token}'")
        out.append(atoms)
    return out


def _read_params_atom_names(params_path):
    atom_names = []
    with open(params_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM "):
                fields = line.split()
                if len(fields) >= 2:
                    atom_names.append(fields[1].strip())
    return atom_names


def _get_rdkit_acceptor_triplets(molecule_sdf, include_reverse=True):
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    supplier = Chem.SDMolSupplier(molecule_sdf, removeHs=False)
    mol = None
    for entry in supplier:
        if entry is not None:
            mol = entry
            break
    if mol is None:
        return []

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    acceptor_idxs = sorted(
        {
            atom_id
            for feat in factory.GetFeaturesForMol(mol)
            if feat.GetFamily() == "Acceptor"
            for atom_id in feat.GetAtomIds()
        }
    )

    def _get_carbonyl_triplets(rdkit_mol):
        out = []

        def _append_triplets(c_idx, o_idx):
            c_atom = rdkit_mol.GetAtomWithIdx(c_idx)
            neighbors = [
                nbr.GetIdx()
                for nbr in c_atom.GetNeighbors()
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != o_idx
            ]
            for nbr_idx in neighbors:
                out.append((o_idx, c_idx, nbr_idx))
                if include_reverse:
                    out.append((o_idx, nbr_idx, c_idx))

        # Generic carbonyls: O=C-X
        carbonyl = Chem.MolFromSmarts("[CX3]=[OX1,OX2]")
        for c_idx, o_idx in rdkit_mol.GetSubstructMatches(carbonyl):
            _append_triplets(c_idx, o_idx)

        # Carboxyl/carboxylate groups: O=C-O
        carboxyl = Chem.MolFromSmarts("[CX3](=[OX1,OX2])[OX1,OX2H1,OX2-]")
        for c_idx, o1_idx, o2_idx in rdkit_mol.GetSubstructMatches(carboxyl):
            _append_triplets(c_idx, o1_idx)
            _append_triplets(c_idx, o2_idx)

        return out

    def _get_hydroxyl_triplets(rdkit_mol):
        # Explicitly include protonated hydroxyl oxygens: X-OH
        pattern = Chem.MolFromSmarts("[OX2H]-[*]")
        out = []
        for o_idx, x_idx in rdkit_mol.GetSubstructMatches(pattern):
            x_atom = rdkit_mol.GetAtomWithIdx(x_idx)
            second_shell = [
                nbr.GetIdx()
                for nbr in x_atom.GetNeighbors()
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != o_idx
            ]
            for n2 in second_shell:
                out.append((o_idx, x_idx, n2))
                if include_reverse:
                    out.append((o_idx, n2, x_idx))
        return out

    triplets = []
    for acc_idx in acceptor_idxs:
        acc = mol.GetAtomWithIdx(acc_idx)
        heavy_neighbors = [nbr.GetIdx() for nbr in acc.GetNeighbors() if nbr.GetAtomicNum() > 1]

        if len(heavy_neighbors) >= 2:
            for n1, n2 in combinations(heavy_neighbors, 2):
                triplets.append((acc_idx, n1, n2))
                if include_reverse:
                    triplets.append((acc_idx, n2, n1))
            continue

        if len(heavy_neighbors) == 1:
            n1 = heavy_neighbors[0]
            n1_atom = mol.GetAtomWithIdx(n1)
            second_shell = [
                nbr.GetIdx()
                for nbr in n1_atom.GetNeighbors()
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != acc_idx
            ]
            for n2 in second_shell:
                triplets.append((acc_idx, n1, n2))
                if include_reverse:
                    triplets.append((acc_idx, n2, n1))

    triplets.extend(_get_carbonyl_triplets(mol))
    triplets.extend(_get_hydroxyl_triplets(mol))

    seen = set()
    unique = []
    for triple in triplets:
        if triple not in seen:
            seen.add(triple)
            unique.append(triple)
    return unique


def _map_idx_triplets_to_atom_names(idx_triplets, atom_names):
    mapped = []
    for a, b, c in idx_triplets:
        if max(a, b, c) >= len(atom_names):
            continue
        mapped.append((atom_names[a], atom_names[b], atom_names[c]))
    return mapped


def generate_params_pdb_and_table(
    mtp,
    csv_file_name,
    path_to_conformers,
    molecule_sdfs,
    use_mol_id=False,
    no_name=False,
    dynamic_acceptor_alignment=False,
    target_atom_triplets=None,
    max_dynamic_alignments=0,
    include_reverse_neighbors=True,
):
    """
    Used internally to generate csv files for reading in by yield_ligand_pose, params files,
    and pdb objects for conformers, will create directories inside of path_to_conformers 
    which are either mol_id/ or mol_name/

    Arguments:
    csv_file: File to create or append new entries to
    path_to_confomers: directory where ligand conformer and params file directories should be held
    molecule_sdfs: list of strings of molec Moule sdfs to read in and generate params file/pdbs for
    use_mol_id: boolean, if a mol id is present on the third line of each molecule in the sdf
        uses that to name directories and files
    no_name: boolean, if the sdf file has no discernable name then just name them sequentially

    Creates:
    csv_file with the following columns:
        Molecule Name, Molecule ID, Conformer Range, Molecule Atoms, Residue Atoms
        Molecule name: Name of the molecule as determined by first line of SDF
        Molecule ID: ID of the molecule as determiend by the third line of SDF (optional)
        Conformer Range: How many conformers were used in generating the params file
            e.g. 1-1 for 1 conformer, 1-100 for 100 conformers etc
            (Note this does assume that your conformers are in separate files)
        Molecule Atoms: MANUAL ENTRY atom labels for atoms on the conformer pdbs that correspond to target atoms
            e.g. C1-C4-C6 denotes three atoms C1, C4, C6 in that order
        Target Atoms: MANUAL ENTRY atom labels for target atoms on a Residue Object
            e.g. CD2-CZ2-CZ3 denote three atoms CD2, CZ2, CZ3 in that order on the target residue

    """
    try:
        os.mkdir(path_to_conformers)
    except:
        print(f"Directory {path_to_conformers} already made")
        pass


    if target_atom_triplets is None:
        target_atom_triplets = []

    with open(csv_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Molecule Name", "Molecule ID", "Conformer Range", "Molecule Atoms", "Target Atoms"])
        for i, molecule_sdf in enumerate(molecule_sdfs):
            with open(molecule_sdf, "r", encoding='utf-8-sig') as sdf:
                lines = list(sdf)
                if len(lines) == 0:
                    continue
                mol_name = lines[0].strip()
                

                if use_mol_id:
                    mol_id = lines[2].split(" ")[0].strip()
                    file_stem = f"{mol_id}/{mol_id}"
                    dir_name = f"{path_to_conformers}/{mol_id}"
                elif not no_name:
                    mol_id = mol_name
                    file_stem =f"{mol_name}/{mol_name}"
                    dir_name = f"{path_to_conformers}/{mol_name}"

                else:
                    mol_name = i
                    mol_id = i
                    file_stem =f"{i}/{i}"
                    dir_name = f"{path_to_conformers}/{i}"

                try:
                    os.mkdir(dir_name)
                except:
                    print(f"Directory {dir_name} already made")
                    pass
                mtp.main([f"{molecule_sdf}", "-n", f"{path_to_conformers}/{file_stem}"])
                count = str(lines.count("$$$$\n"))
                conformer_range = f"1_{count}_"

                if dynamic_acceptor_alignment and target_atom_triplets:
                    params_file = f"{path_to_conformers}/{file_stem}.params"
                    atom_names = _read_params_atom_names(params_file)
                    idx_triplets = _get_rdkit_acceptor_triplets(
                        molecule_sdf,
                        include_reverse=include_reverse_neighbors,
                    )
                    mol_triplets = _map_idx_triplets_to_atom_names(idx_triplets, atom_names)

                    if max_dynamic_alignments > 0:
                        mol_triplets = mol_triplets[:max_dynamic_alignments]

                    if len(mol_triplets) == 0:
                        writer.writerow([mol_name, mol_id, conformer_range, "", ""])
                        continue

                    for mol_atoms in mol_triplets:
                        for tgt_atoms in target_atom_triplets:
                            writer.writerow(
                                [
                                    mol_name,
                                    mol_id,
                                    conformer_range,
                                    "-".join(mol_atoms),
                                    "-".join(tgt_atoms),
                                ]
                            )
                else:
                    writer.writerow([mol_name, mol_id, conformer_range, "", ""])


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file",
                        help = "your config file",
                        default = "my_conf.txt")
    
    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)

    config = ConfigParser()
    with open(args.config_file, "r", encoding="utf-8-sig") as handle:
        config.read_file(handle)
    spec = config["create_table"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_dir = os.path.normpath(os.path.join(script_dir, "..", "legacy"))
    if legacy_dir not in sys.path:
        sys.path.append(legacy_dir)
    import molfile_to_params as mtp

    csv_file_name = _safe_get(config, "create_table", "CSVFileName")
    path_to_conformers = _safe_get(config, "create_table", "PathToConformers")
    if not csv_file_name:
        raise KeyError("Missing CSVFileName in [create_table] or [DEFAULT]")
    if not path_to_conformers:
        raise KeyError("Missing PathToConformers in [create_table] or [DEFAULT]")
    input_molecule_sdfs = spec["MoleculeSDFs"].split(" ")


    # Grabbing all of the Molecule SDFs
    molecule_sdfs = []
    for inp in input_molecule_sdfs:
        if "*" in inp:
            if "/" in inp:
                directory = "/".join([e for e in inp.split("/")[:-1]])
            else:
                directory = "."

            for file in os.listdir(directory):
                if fnmatch.fnmatch(file.lower(), inp.split("/")[-1]):
                    molecule_sdfs.append(f"{directory}/{file}")
        else:
            molecule_sdfs.append(inp)

    use_mol_id = _safe_getbool(config, "create_table", "UseMoleculeID", fallback=False)
    no_name = _safe_getbool(config, "create_table", "NoName", fallback=False)
    dynamic_acceptor_alignment = _safe_getbool(
        config, "create_table", "DynamicAcceptorAlignment", fallback=False
    )
    include_reverse_neighbors = _safe_getbool(
        config, "create_table", "IncludeReverseNeighborOrder", fallback=True
    )
    max_dynamic_alignments = int(
        _safe_get(config, "create_table", "MaxDynamicAlignments", fallback="0")
    )
    target_atom_triplets = _parse_target_atom_triplets(
        _safe_get(config, "create_table", "TargetAtomTriplets", fallback="")
    )

    if dynamic_acceptor_alignment and len(target_atom_triplets) == 0:
        raise ValueError(
            "DynamicAcceptorAlignment=True but no TargetAtomTriplets were provided."
        )

    if dynamic_acceptor_alignment:
        try:
            import rdkit  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RDKit is required when DynamicAcceptorAlignment=True"
            ) from exc

    generate_params_pdb_and_table(
        mtp,
        csv_file_name,
        path_to_conformers,
        molecule_sdfs,
        use_mol_id,
        no_name,
        dynamic_acceptor_alignment=dynamic_acceptor_alignment,
        target_atom_triplets=target_atom_triplets,
        max_dynamic_alignments=max_dynamic_alignments,
        include_reverse_neighbors=include_reverse_neighbors,
    )
    print(f"Succesfully generated table at {csv_file_name} and conformers at {path_to_conformers}")
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
