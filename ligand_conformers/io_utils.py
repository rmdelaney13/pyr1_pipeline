"""
io_utils.py — input resolution, SDF/PDB writers, and atom-name helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from rdkit import Chem

logger = logging.getLogger("ligand_conformers")


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_input(input_spec: dict) -> Chem.Mol:
    """Convert an input specification to a 3-D RDKit Mol with explicit H.

    Parameters
    ----------
    input_spec : dict
        ``{"type": "smiles"|"pubchem"|"sdf"|"mol2"|"pdb", "value": <str>}``

    Returns
    -------
    Chem.Mol  with at least one conformer (may be 0-D if from SMILES — the
    caller will embed properly).

    Raises
    ------
    ValueError / RuntimeError on parse failure.
    """
    itype = input_spec["type"].lower()
    value = str(input_spec["value"])

    if itype == "smiles":
        return _mol_from_smiles(value)
    elif itype == "pubchem":
        return _mol_from_pubchem(value)
    elif itype == "sdf":
        return _mol_from_sdf(value)
    elif itype == "mol2":
        return _mol_from_mol2(value)
    elif itype == "pdb":
        return _mol_from_pdb(value)
    else:
        raise ValueError(f"Unknown input type: {itype!r}.  "
                         f"Use smiles|pubchem|sdf|mol2|pdb.")


def _mol_from_smiles(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
    mol = Chem.AddHs(mol)
    return mol


def _mol_from_pubchem(identifier: str) -> Chem.Mol:
    """Resolve a PubChem CID or compound name to SMILES, then parse.

    Uses ``pubchempy`` if available; otherwise raises with clear guidance.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        raise ImportError(
            "pubchempy is required for PubChem lookups but is not installed.\n"
            "  pip install pubchempy\n"
            "Alternatively, provide the SMILES directly with --input-type smiles."
        )

    logger.info("Looking up PubChem identifier: %s", identifier)

    # Try as integer CID first
    compounds = []
    try:
        cid = int(identifier)
        compounds = pcp.get_compounds(cid, "cid")
    except ValueError:
        pass

    # Fall back to name search
    if not compounds:
        compounds = pcp.get_compounds(identifier, "name")

    if not compounds:
        raise RuntimeError(
            f"PubChem lookup returned no results for: {identifier!r}\n"
            f"Try providing the SMILES directly with --input-type smiles."
        )

    smi = compounds[0].canonical_smiles
    logger.info("PubChem resolved to SMILES: %s (CID %s)",
                smi, compounds[0].cid)
    return _mol_from_smiles(smi)


def _mol_from_sdf(path: str) -> Chem.Mol:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SDF file not found: {p}")
    suppl = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=True)
    mol = next(iter(suppl), None)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse any molecule from {p}")
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def _mol_from_mol2(path: str) -> Chem.Mol:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MOL2 file not found: {p}")
    mol = Chem.MolFromMol2File(str(p), removeHs=False, sanitize=True)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse MOL2: {p}")
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def _mol_from_pdb(path: str) -> Chem.Mol:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDB file not found: {p}")
    mol = Chem.MolFromPDBFile(str(p), removeHs=False, sanitize=True)
    if mol is None:
        raise RuntimeError(
            f"RDKit could not parse PDB: {p}. "
            f"Try converting to SDF with OpenBabel first."
        )
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


# ---------------------------------------------------------------------------
# SDF writer
# ---------------------------------------------------------------------------

def write_sdf(mol: Chem.Mol, conf_ids: List[int],
              energies: Dict[int, float], path: Path,
              name_prefix: str = "ligand") -> None:
    """Write conformers to an SDF, sorted by the order of *conf_ids*."""
    w = Chem.SDWriter(str(path))
    for idx, cid in enumerate(conf_ids):
        # RDKit SDWriter works on the Mol's own conformers directly
        mol.SetProp("_Name", f"{name_prefix}_conf{cid:04d}")
        mol.SetProp("rdkit_energy", f"{energies.get(cid, float('nan')):.6f}")
        mol.SetProp("conf_rank", str(idx))
        mol.SetProp("conf_id", str(cid))
        w.write(mol, confId=cid)
    w.close()
    logger.info("Wrote %d conformers → %s", len(conf_ids), path)


# ---------------------------------------------------------------------------
# PDB writer with deterministic atom names
# ---------------------------------------------------------------------------

def export_pdb_with_atom_names(
    mol: Chem.Mol,
    conf_id: int,
    path: Path,
    preserve_names: bool = True,
) -> None:
    """Write a single conformer to PDB with consistent atom names.

    Atom naming strategy:
        1. If the molecule already has ``_Name`` atom props or PDB-derived
           atom names, preserve them (``preserve_names=True``).
        2. Otherwise, assign deterministic names:  element symbol + 1-based
           index within that element type (e.g. C1, C2, …, O1, N1, …).

    The output PDB is suitable for downstream docking tools that expect
    consistent, non-overlapping atom names.
    """
    # Work on a copy so we don't mutate the original
    mol_copy = Chem.RWMol(mol)

    if not preserve_names or not _has_pdb_atom_names(mol_copy):
        _assign_deterministic_atom_names(mol_copy)

    pdb_block = Chem.MolToPDBBlock(mol_copy, confId=conf_id)

    with open(path, "w") as fh:
        fh.write(pdb_block)
    logger.debug("PDB written: %s", path)


def _has_pdb_atom_names(mol: Chem.Mol) -> bool:
    """Check whether the mol already carries PDB-style atom info."""
    info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
    return info is not None


def _assign_deterministic_atom_names(mol: Chem.RWMol) -> None:
    """Assign atom names as ``<Element><counter>`` (PDB-style, 4-char field)."""
    elem_counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        name = f"{elem}{elem_counts[elem]}"
        # PDB atom name field is 4 chars, left-justified for 2-char elements
        if len(elem) == 1:
            pdb_name = f" {name:<3s}"
        else:
            pdb_name = f"{name:<4s}"

        info = Chem.AtomPDBResidueInfo()
        info.SetName(pdb_name)
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("X")
        info.SetIsHeteroAtom(True)
        atom.SetPDBResidueInfo(info)
