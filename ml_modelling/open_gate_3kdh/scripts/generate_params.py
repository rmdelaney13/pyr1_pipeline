#!/usr/bin/env python3
"""
Step 2: Generate Rosetta .params file for LCA ligand from a Boltz prediction PDB.

Extracts the ligand from a Boltz PDB, assigns bond orders from SMILES using RDKit,
adds hydrogens, writes SDF, and generates a Rosetta .params file.

Usage:
    python generate_params.py \
        --boltz-pdb inputs/boltz_predictions/pair_3098.pdb \
        --smiles "C[C@H](CCC(=O)O)[C@@H]1CC[C@@H]2[C@]1(CC[C@H]3[C@H]2CC[C@@H]4[C@@]3(CC[C@@H](C4)O)C)C" \
        --output-dir params/ \
        --name LCA

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    print("ERROR: RDKit is required. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


def extract_ligand_pdb_block(pdb_path, ligand_chain="B"):
    """Extract HETATM and CONECT records for the ligand from a Boltz PDB file.

    Returns the ligand as a PDB block string.
    """
    hetatm_lines = []
    conect_lines = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                chain_id = line[21]
                if chain_id == ligand_chain:
                    hetatm_lines.append(line)
            elif line.startswith("CONECT"):
                conect_lines.append(line)

    if not hetatm_lines:
        print(f"ERROR: No HETATM records found for chain {ligand_chain} in {pdb_path}")
        return None

    pdb_block = "".join(hetatm_lines) + "".join(conect_lines) + "END\n"
    return pdb_block


def ligand_from_boltz(pdb_path, smiles, ligand_chain="B"):
    """Extract ligand from Boltz PDB, assign bond orders from SMILES, add hydrogens.

    Returns:
        mol: RDKit Mol object with hydrogens and 3D coordinates
        None on failure
    """
    pdb_block = extract_ligand_pdb_block(pdb_path, ligand_chain)
    if pdb_block is None:
        return None

    # Parse with RDKit
    pdb_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if pdb_mol is None:
        print("ERROR: RDKit could not parse the ligand PDB block")
        return None

    # Reference molecule from SMILES
    smi_mol = Chem.MolFromSmiles(smiles)
    if smi_mol is None:
        print(f"ERROR: RDKit could not parse SMILES: {smiles}")
        return None

    # Assign correct bond orders from SMILES template
    try:
        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    except Exception as e:
        print(f"ERROR: Bond order assignment failed: {e}")
        return None

    # Skip adding hydrogens — the ligand is frozen during relax so H-atoms
    # are unnecessary, and including them creates a mismatch with the Boltz PDB
    # (which has only heavy atoms) causing Rosetta fill_missing_atoms failures.

    print(f"  Ligand extracted: {pdb_mol.GetNumAtoms()} atoms "
          f"({pdb_mol.GetNumHeavyAtoms()} heavy)")
    return pdb_mol


def write_sdf(mol, output_path):
    """Write RDKit molecule to SDF file."""
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    print(f"  SDF written to {output_path}")


def write_mol2(mol, output_path):
    """Write ligand coordinates as a PDB file (for molfile_to_params input)."""
    pdb_block = Chem.MolToPDBBlock(mol)
    with open(output_path, "w") as f:
        f.write(pdb_block)
    print(f"  PDB written to {output_path}")


def try_molfile_to_params(sdf_path, name, output_dir):
    """Try to generate .params using Rosetta's molfile_to_params.py.

    Looks for the script in common locations. Returns True if successful.
    """
    # Search paths for the script
    script_candidates = [
        # This repo's copy
        Path(__file__).resolve().parents[3] / "docking" / "legacy" / "molfile_to_params.py",
        # Common Rosetta installation paths
        Path.home() / "rosetta" / "main" / "source" / "scripts" / "python" / "public" / "molfile_to_params.py",
        Path("/opt/rosetta/main/source/scripts/python/public/molfile_to_params.py"),
    ]

    # Also check if ROSETTA env var is set
    rosetta_path = os.environ.get("ROSETTA", os.environ.get("ROSETTA3"))
    if rosetta_path:
        script_candidates.insert(0,
            Path(rosetta_path) / "main" / "source" / "scripts" / "python" / "public" / "molfile_to_params.py")

    script = None
    for candidate in script_candidates:
        if candidate.exists():
            script = candidate
            break

    if script is None:
        print("  molfile_to_params.py not found in standard locations")
        return False

    # Check for rosetta_py dependency
    rosetta_py_dir = script.parent.parent
    env = os.environ.copy()
    if rosetta_py_dir.exists():
        env["PYTHONPATH"] = str(rosetta_py_dir) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, str(script),
        "-n", name,
        "-p", name,
        "--keep-names",
        str(sdf_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=str(output_dir), env=env,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            params_path = output_dir / f"{name}.params"
            if params_path.exists():
                print(f"  Params generated: {params_path}")
                return True
            else:
                print(f"  Script ran but {name}.params not found")
                print(f"  stdout: {result.stdout[:500]}")
        else:
            print(f"  molfile_to_params.py failed (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("  molfile_to_params.py timed out")
    except Exception as e:
        print(f"  Error running molfile_to_params.py: {e}")

    return False


# --- Rosetta atom type mapping for common organic atoms ---
ROSETTA_ATOM_TYPES = {
    # (atomic_num, hybridization, num_hydrogens, aromatic) -> rosetta_type
    # Carbon
    (6, "SP3", True): "CH1",   # sp3 carbon (with H simplification)
    (6, "SP2", True): "aroC",  # sp2 carbon
    (6, "SP", True): "CH0",
    # Oxygen
    (8, "SP3", True): "OH",    # hydroxyl
    (8, "SP2", True): "OOC",   # carbonyl/carboxylate
    # Nitrogen
    (7, "SP3", True): "Nlys",
    (7, "SP2", True): "Nbb",
    # Sulfur
    (16, "SP3", True): "SH1",
    # Hydrogen
    (1, None, True): "Hpol",   # default H
}


def get_rosetta_atom_type(atom, mol):
    """Map an RDKit atom to a Rosetta atom type string."""
    num = atom.GetAtomicNum()

    if num == 1:
        # Check if bonded to polar atom
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() in (7, 8, 16):
                return "Hpol"
        return "Hapo"

    if num == 6:
        hyb = str(atom.GetHybridization())
        total_h = atom.GetTotalNumHs()
        if atom.GetIsAromatic():
            if total_h > 0:
                return "aroC"
            return "aroC"
        if "SP3" in hyb:
            if total_h == 0:
                return "CH0"
            elif total_h == 1:
                return "CH1"
            elif total_h == 2:
                return "CH2"
            else:
                return "CH3"
        elif "SP2" in hyb:
            return "COO" if any(n.GetAtomicNum() == 8 for n in atom.GetNeighbors()) else "aroC"
        return "CH1"

    if num == 8:
        hyb = str(atom.GetHybridization())
        total_h = atom.GetTotalNumHs()
        if total_h > 0:
            return "OH"
        # Check for carboxylate
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                o_count = sum(1 for n in neighbor.GetNeighbors() if n.GetAtomicNum() == 8)
                if o_count >= 2:
                    return "OOC"
        if "SP2" in hyb:
            return "OOC"
        return "OH"

    if num == 7:
        total_h = atom.GetTotalNumHs()
        if total_h >= 2:
            return "Nlys"
        return "Nbb"

    if num == 16:
        return "SH1"

    return "VIRT"


def compute_icoor(mol, conformer):
    """Compute Rosetta ICOOR_INTERNAL records from 3D coordinates.

    Uses a tree traversal starting from a central atom. Each atom is defined
    by (distance, angle, torsion) relative to three previously defined atoms.
    """
    n_atoms = mol.GetNumAtoms()
    conf = conformer
    names = [mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
             if mol.GetAtomWithIdx(i).GetPDBResidueInfo() else f"X{i+1}"
             for i in range(n_atoms)]

    # Assign names if PDB info not available
    from collections import defaultdict
    elem_count = defaultdict(int)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        elem_count[sym] += 1
        if not atom.GetPDBResidueInfo():
            names[i] = f"{sym}{elem_count[sym]}"

    # Build the atom tree using BFS from the "neighbor atom" (most central)
    # Find most central atom (highest connectivity)
    centrality = [(sum(1 for _ in mol.GetAtomWithIdx(i).GetNeighbors()), i)
                  for i in range(n_atoms)]
    centrality.sort(reverse=True)
    root = centrality[0][1]

    # BFS to build parent tree
    visited = set()
    parent = {}
    grandparent = {}
    order = []

    from collections import deque
    queue = deque([root])
    visited.add(root)
    parent[root] = root
    grandparent[root] = root

    while queue:
        node = queue.popleft()
        order.append(node)
        atom = mol.GetAtomWithIdx(node)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited:
                visited.add(nidx)
                parent[nidx] = node
                grandparent[nidx] = parent.get(node, root)
                queue.append(nidx)

    def get_pos(idx):
        p = conf.GetAtomPosition(idx)
        return np.array([p.x, p.y, p.z])

    def compute_distance(a, b):
        return np.linalg.norm(get_pos(a) - get_pos(b))

    def compute_angle(a, b, c):
        """Angle at b between a-b-c."""
        ba = get_pos(a) - get_pos(b)
        bc = get_pos(c) - get_pos(b)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    def compute_torsion(a, b, c, d):
        """Torsion angle a-b-c-d."""
        p0 = get_pos(a)
        p1 = get_pos(b)
        p2 = get_pos(c)
        p3 = get_pos(d)
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return np.degrees(np.arctan2(y, x))

    icoor_lines = []
    for i, idx in enumerate(order):
        name = names[idx]
        p_idx = parent[idx]
        gp_idx = grandparent[idx]

        if i == 0:
            # Root atom: defined at origin
            stub1 = names[p_idx]
            stub2 = names[p_idx]
            stub3 = names[gp_idx]
            icoor_lines.append(
                f"ICOOR_INTERNAL  {name:>4s}    0.000000    0.000000    0.000000 "
                f"  {stub1:>4s}  {stub2:>4s}  {stub3:>4s}")
        elif i == 1:
            # Second atom: defined by distance only
            dist = compute_distance(idx, p_idx)
            stub1 = names[p_idx]
            stub2 = names[root]
            stub3 = names[gp_idx]
            icoor_lines.append(
                f"ICOOR_INTERNAL  {name:>4s}    0.000000  180.000000   {dist:9.6f} "
                f"  {stub1:>4s}  {stub2:>4s}  {stub3:>4s}")
        elif i == 2:
            # Third atom: defined by distance and angle
            dist = compute_distance(idx, p_idx)
            angle = compute_angle(idx, p_idx, grandparent[idx])
            stub1 = names[p_idx]
            stub2 = names[grandparent[idx]]
            stub3 = names[root]
            icoor_lines.append(
                f"ICOOR_INTERNAL  {name:>4s}    0.000000  {angle:10.6f}   {dist:9.6f} "
                f"  {stub1:>4s}  {stub2:>4s}  {stub3:>4s}")
        else:
            # All other atoms: distance, angle, torsion
            dist = compute_distance(idx, p_idx)
            angle = compute_angle(idx, p_idx, gp_idx)
            # Find great-grandparent for torsion
            ggp_idx = grandparent.get(gp_idx, root)
            torsion = compute_torsion(idx, p_idx, gp_idx, ggp_idx)
            stub1 = names[p_idx]
            stub2 = names[gp_idx]
            stub3 = names[ggp_idx]
            icoor_lines.append(
                f"ICOOR_INTERNAL  {name:>4s} {torsion:12.6f}  {angle:10.6f}   {dist:9.6f} "
                f"  {stub1:>4s}  {stub2:>4s}  {stub3:>4s}")

    return icoor_lines, names, root


def generate_params_from_rdkit(mol, name, output_path):
    """Generate a Rosetta .params file directly from an RDKit Mol object.

    This is a fallback when molfile_to_params.py is not available.
    Produces a functional (if not perfect) .params file.
    """
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Preserve existing PDB atom names from the Boltz PDB where available.
    # Only assign new names for atoms without PDB info (e.g., hydrogens from AddHs).
    from collections import defaultdict
    elem_count = defaultdict(int)
    atom_names = []

    # First pass: collect existing names to avoid collisions
    existing_names = set()
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        info = atom.GetPDBResidueInfo()
        if info is not None:
            aname = info.GetName().strip()
            if aname:
                existing_names.add(aname)

    # Second pass: assign names
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        info = atom.GetPDBResidueInfo()
        if info is not None:
            aname = info.GetName().strip()
            if aname:
                atom_names.append(aname)
                # Update residue info to match params name
                info.SetResidueName(name)
                info.SetResidueNumber(1)
                info.SetChainId("B")
                continue

        # No existing name — generate one (typically for added hydrogens)
        sym = atom.GetSymbol()
        elem_count[sym] += 1
        aname = f"{sym}{elem_count[sym]}"
        while aname in existing_names:
            elem_count[sym] += 1
            aname = f"{sym}{elem_count[sym]}"
        existing_names.add(aname)
        atom_names.append(aname)
        new_info = Chem.AtomPDBResidueInfo()
        new_info.SetName(f" {aname:<3s}")
        new_info.SetResidueName(name)
        new_info.SetResidueNumber(1)
        new_info.SetChainId("B")
        atom.SetPDBResidueInfo(new_info)

    lines = []
    lines.append(f"NAME {name}")
    lines.append(f"IO_STRING {name} Z")
    lines.append("TYPE LIGAND")
    lines.append("AA UNK")

    # ATOM lines
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        aname = atom_names[i]
        rtype = get_rosetta_atom_type(atom, mol)
        charge = float(atom.GetDoubleProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
        if np.isnan(charge):
            charge = 0.0
        lines.append(f"ATOM {aname:>4s} {rtype:<5s} X   {charge:6.2f}")

    # BOND_TYPE lines
    for bond in mol.GetBonds():
        a1 = atom_names[bond.GetBeginAtomIdx()]
        a2 = atom_names[bond.GetEndAtomIdx()]
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.DOUBLE:
            order = 2
        elif bt == Chem.rdchem.BondType.TRIPLE:
            order = 3
        elif bt == Chem.rdchem.BondType.AROMATIC:
            order = 4
        else:
            order = 1
        lines.append(f"BOND_TYPE {a1:>4s} {a2:>4s} {order}   ")

    # Skip CHI definitions — ligand is frozen during relax (no chi sampling),
    # and the SMARTS-based rotatable bond detection produces invalid atom
    # connectivity chains that fail Rosetta residue type validation.

    # NBR_ATOM — most central heavy atom
    heavy_indices = [i for i in range(n_atoms) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if heavy_indices:
        # Centroid of heavy atoms
        positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z] for i in heavy_indices])
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        nbr_idx = heavy_indices[np.argmin(dists)]
        nbr_name = atom_names[nbr_idx]

        # NBR_RADIUS — max distance from NBR_ATOM to any other atom
        nbr_pos = np.array([conf.GetAtomPosition(nbr_idx).x, conf.GetAtomPosition(nbr_idx).y,
                            conf.GetAtomPosition(nbr_idx).z])
        all_positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                                   conf.GetAtomPosition(i).z] for i in range(n_atoms)])
        nbr_radius = np.max(np.linalg.norm(all_positions - nbr_pos, axis=1))

        lines.append(f"NBR_ATOM {nbr_name:>4s}")
        lines.append(f"NBR_RADIUS {nbr_radius:.6f}")

    # ICOOR_INTERNAL
    icoor_lines, _, _ = compute_icoor(mol, conf)
    lines.extend(icoor_lines)

    # Write
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Params file generated: {output_path}")
    print(f"  {n_atoms} atoms, {mol.GetNumBonds()} bonds, 0 CHI angles (skipped)")


def validate_params(params_path, name):
    """Try loading the params file into PyRosetta to validate it."""
    try:
        import pyrosetta
        pyrosetta.init(f"-extra_res_fa {params_path} -mute all")
        # Check if the residue type is available
        rts = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")
        if rts.has_name(name):
            rt = rts.name_map(name)
            print(f"  Validation OK: {name} has {rt.natoms()} atoms in Rosetta")
            return True
        else:
            print(f"  WARNING: Params loaded but residue type '{name}' not found")
            return False
    except ImportError:
        print("  PyRosetta not available — skipping validation")
        return True  # Can't validate, assume OK
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rosetta .params file for LCA ligand from Boltz PDB"
    )
    parser.add_argument("--boltz-pdb", required=True,
                        help="Path to a Boltz prediction PDB containing the ligand")
    parser.add_argument("--smiles", required=True,
                        help="SMILES string for the ligand (for bond order assignment)")
    parser.add_argument("--output-dir", default="params/",
                        help="Directory to write params files")
    parser.add_argument("--name", default="LCA",
                        help="3-letter residue name for the ligand (default: LCA)")
    parser.add_argument("--ligand-chain", default="B",
                        help="Chain ID of the ligand in the Boltz PDB (default: B)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip PyRosetta validation of the generated params")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract and process ligand
    print(f"Extracting ligand from {args.boltz_pdb}...")
    mol = ligand_from_boltz(args.boltz_pdb, args.smiles, args.ligand_chain)
    if mol is None:
        print("FAILED: Could not extract/process ligand")
        sys.exit(1)

    # Compute Gasteiger charges for params generation
    AllChem.ComputeGasteigerCharges(mol)

    # Step 2: Write SDF
    sdf_path = output_dir / f"{args.name}.sdf"
    write_sdf(mol, sdf_path)

    # Step 3: Try molfile_to_params.py first
    params_path = output_dir / f"{args.name}.params"
    print(f"\nAttempting params generation with molfile_to_params.py...")
    success = try_molfile_to_params(sdf_path, args.name, output_dir)

    if not success:
        # Step 4: Fallback — generate params directly from RDKit
        print(f"\nFalling back to direct params generation from RDKit...")
        generate_params_from_rdkit(mol, args.name, params_path)

    # Step 5: Validate
    if not args.skip_validation:
        print(f"\nValidating params file...")
        validate_params(params_path, args.name)

    # Step 6: Also save the ligand PDB for reference
    lig_pdb_path = output_dir / f"{args.name}_reference.pdb"
    pdb_block = Chem.MolToPDBBlock(mol)
    with open(lig_pdb_path, "w") as f:
        f.write(pdb_block)
    print(f"\nReference ligand PDB saved to {lig_pdb_path}")

    print(f"\nDone. Params file: {params_path}")
    print(f"Use with PyRosetta: init('-extra_res_fa {params_path}')")


if __name__ == "__main__":
    main()
