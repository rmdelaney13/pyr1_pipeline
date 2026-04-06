#!/usr/bin/env python3
"""
Analyze Boltz prediction outputs: confidence metrics, ligand RMSD, H-bond geometry.

Scans Boltz output directories for prediction results and extracts:
  - ipTM, ligand_iptm, protein_iptm, pLDDT (from confidence JSON)
  - Per-chain pLDDT for protein vs ligand (from pLDDT NPZ)
  - Binary-to-ternary ligand RMSD (from PDB structures)
  - H-bond water geometry: distance and angle (from PDB structures)
  - Affinity predictions if available (from affinity JSON)

Usage:
    # Analyze binary predictions
    python analyze_boltz_output.py \
        --binary-dir /scratch/alpine/ryde3462/boltz_lca/output_tier1_binary \
        --ref-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb \
        --out results_tier1.csv

    # Analyze binary + ternary (computes RMSD between them)
    python analyze_boltz_output.py \
        --binary-dir /scratch/alpine/ryde3462/boltz_lca/output_tier1_binary \
        --ternary-dir /scratch/alpine/ryde3462/boltz_lca/output_tier1_ternary \
        --ref-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb \
        --out results_tier1.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ligand geometry validation, HAB1 clash check, latch RMSD
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ligand_geometry import (
        LigandGeometryChecker, HAB1ClashChecker, compute_latch_rmsd,
    )
    _HAS_LIGAND_GEOMETRY = True
except ImportError:
    _HAS_LIGAND_GEOMETRY = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# PYR1 binding pocket positions (Boltz/AF3 numbering, 1-indexed)
# 16 mutable positions: Rosetta numbering + 2 (except pos 59 which is unchanged)
POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]


# ═══════════════════════════════════════════════════════════════════
# BOLTZ CONFIDENCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def find_boltz_predictions(output_dir: str) -> List[Dict[str, Path]]:
    """Find all Boltz prediction outputs in a directory.

    Boltz organizes output as:
        output_dir/boltz_results_<name>/predictions/<name>/<name>_model_0.pdb
        output_dir/boltz_results_<name>/predictions/<name>/confidence_<name>_model_0.json

    Returns list of dicts with keys: name, pdb, confidence, plddt, pae, affinity
    """
    out = Path(output_dir)
    predictions = []

    for results_dir in sorted(out.glob("boltz_results_*")):
        pred_dir = results_dir / "predictions"
        if not pred_dir.exists():
            continue

        for name_dir in sorted(pred_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            name = name_dir.name

            # Find model_0 structure (PDB or CIF)
            struct_path = name_dir / f"{name}_model_0.pdb"
            if not struct_path.exists():
                struct_path = name_dir / f"{name}_model_0.cif"
            if not struct_path.exists():
                continue

            entry = {
                'name': name,
                'structure': struct_path,
                'confidence': name_dir / f"confidence_{name}_model_0.json",
                'plddt': name_dir / f"plddt_{name}_model_0.npz",
                'pae': name_dir / f"pae_{name}_model_0.npz",
                'affinity': name_dir / f"affinity_{name}.json",
            }
            predictions.append(entry)

    return predictions


def extract_confidence_metrics(conf_path: Path) -> Dict[str, Optional[float]]:
    """Extract metrics from Boltz confidence JSON.

    Boltz confidence JSON contains:
        confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
        complex_plddt, complex_iplddt, complex_pde, complex_ipde,
        chains_ptm, pair_chains_iptm
    """
    metrics = {
        'iptm': None,
        'ligand_iptm': None,
        'protein_iptm': None,
        'ptm': None,
        'confidence_score': None,
        'complex_plddt': None,
        'complex_iplddt': None,
        'complex_pde': None,
        'complex_ipde': None,
    }

    if not conf_path.exists():
        return metrics

    try:
        with open(conf_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse {conf_path}: {e}")
        return metrics

    for key in metrics:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                metrics[key] = round(float(val), 4)

    return metrics


def extract_plddt_per_chain(plddt_path: Path, structure_path: Path) -> Dict[str, Optional[float]]:
    """Extract per-chain mean pLDDT from NPZ + structure file.

    The pLDDT NPZ contains per-token scores. We need the structure to
    map tokens to chains (protein A vs ligand B vs HAB1 C).
    """
    result = {'plddt_protein': None, 'plddt_ligand': None, 'plddt_hab1': None, 'plddt_pocket': None}

    if not plddt_path.exists():
        return result

    try:
        plddt_data = np.load(plddt_path)
        # NPZ typically has a single array; try common keys
        plddt = None
        for key in plddt_data.files:
            plddt = plddt_data[key]
            break
        if plddt is None:
            return result
    except Exception as e:
        logger.warning(f"Failed to load pLDDT NPZ {plddt_path}: {e}")
        return result

    # Parse structure to get chain assignments per token
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ext = structure_path.suffix.lower()
            if ext == '.cif':
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            struct = parser.get_structure("s", str(structure_path))
    except Exception as e:
        logger.warning(f"Failed to parse structure for pLDDT chain mapping: {e}")
        return result

    # Count residues per chain (tokens = residues for protein, 1 for ligand)
    chain_residue_counts = {}
    for model in struct:
        for chain in model:
            chain_residue_counts[chain.id] = len(list(chain.get_residues()))
        break

    # Map pLDDT tokens to chains
    token_idx = 0
    chain_plddts = {}
    for chain_id, n_res in chain_residue_counts.items():
        end_idx = min(token_idx + n_res, len(plddt))
        if token_idx < len(plddt):
            chain_plddts[chain_id] = float(plddt[token_idx:end_idx].mean())
        token_idx = end_idx

    result['plddt_protein'] = round(chain_plddts.get('A', 0), 4) if 'A' in chain_plddts else None
    result['plddt_ligand'] = round(chain_plddts.get('B', 0), 4) if 'B' in chain_plddts else None
    result['plddt_hab1'] = round(chain_plddts.get('C', 0), 4) if 'C' in chain_plddts else None

    # Pocket pLDDT: mean over 16 binding pocket positions (chain A only)
    # Pocket positions are 1-indexed; pLDDT tokens for chain A start at 0
    n_chain_a = chain_residue_counts.get('A', 0)
    if n_chain_a > 0:
        pocket_indices = [p - 1 for p in POCKET_POSITIONS if p - 1 < n_chain_a and p - 1 < len(plddt)]
        if pocket_indices:
            result['plddt_pocket'] = round(float(plddt[pocket_indices].mean()), 4)

    return result


def extract_affinity(affinity_path: Path) -> Dict[str, Optional[float]]:
    """Extract affinity predictions from Boltz affinity JSON."""
    result = {
        'affinity_pred_value': None,
        'affinity_probability_binary': None,
    }

    if not affinity_path.exists():
        return result

    try:
        with open(affinity_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse affinity JSON {affinity_path}: {e}")
        return result

    for key in result:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                result[key] = round(float(val), 4)

    return result


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL ANALYSIS (RMSD, H-BOND GEOMETRY)
# ═══════════════════════════════════════════════════════════════════

def _resolve_element(atom) -> str:
    """Get element symbol from an atom."""
    elem = atom.element.strip().upper() if atom.element else ''
    if not elem:
        name = atom.get_name().strip()
        if name:
            elem = name[0].upper()
    return elem


def _get_ca_atoms(structure, chain_id: str):
    """Extract CA atoms sorted by residue ID."""
    ca_atoms = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    for atom in res:
                        if atom.get_name() == 'CA':
                            ca_atoms.append(atom)
        break
    return sorted(ca_atoms, key=lambda a: a.get_parent().get_id()[1])


def _get_ligand_heavy_atoms(structure, chain_id: str):
    """Extract all heavy atoms from a ligand chain."""
    atoms = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    for atom in res:
                        elem = _resolve_element(atom)
                        if elem not in ('H', ''):
                            atoms.append(atom)
        break
    return atoms


def compute_ligand_rmsd(
    struct1_path: str,
    struct2_path: str,
    protein_chain: str = 'A',
    ligand_chain: str = 'B',
) -> Optional[float]:
    """Compute ligand heavy-atom RMSD between two structures after CA alignment.

    Uses element-based matching with Hungarian algorithm.
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.error("Biopython and scipy required for RMSD calculation")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def _parse(path):
            p = Path(path)
            if p.suffix.lower() == '.cif':
                return MMCIFParser(QUIET=True).get_structure("s", str(p))
            return PDBParser(QUIET=True).get_structure("s", str(p))

        try:
            s1 = _parse(struct1_path)
            s2 = _parse(struct2_path)
        except Exception as e:
            logger.error(f"Failed to parse structures for RMSD: {e}")
            return None

    ca1 = _get_ca_atoms(s1, protein_chain)
    ca2 = _get_ca_atoms(s2, protein_chain)

    if not ca1 or not ca2:
        logger.warning(f"No CA atoms for chain {protein_chain}")
        return None

    n_ca = min(len(ca1), len(ca2))
    if n_ca < 10:
        return None

    # Superimpose s1 onto s2
    sup = Superimposer()
    try:
        sup.set_atoms(ca2[:n_ca], ca1[:n_ca])  # s2 is fixed, s1 is mobile
        sup.apply(list(s1.get_atoms()))
    except Exception as e:
        logger.error(f"Superposition failed: {e}")
        return None

    lig1 = _get_ligand_heavy_atoms(s1, ligand_chain)
    lig2 = _get_ligand_heavy_atoms(s2, ligand_chain)

    if not lig1 or not lig2:
        return None

    # Element-based matching with Hungarian algorithm
    lig1_by_elem = {}
    for atom in lig1:
        elem = _resolve_element(atom)
        lig1_by_elem.setdefault(elem, []).append(atom)

    lig2_by_elem = {}
    for atom in lig2:
        elem = _resolve_element(atom)
        lig2_by_elem.setdefault(elem, []).append(atom)

    all_sq_dists = []
    for elem in set(lig1_by_elem) & set(lig2_by_elem):
        atoms1 = lig1_by_elem[elem]
        atoms2 = lig2_by_elem[elem]

        coords1 = np.array([a.get_coord() for a in atoms1])
        coords2 = np.array([a.get_coord() for a in atoms2])

        n1, n2 = len(coords1), len(coords2)
        cost = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                cost[i, j] = np.sum((coords1[i] - coords2[j])**2)

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            all_sq_dists.append(cost[r, c])

    if not all_sq_dists:
        return None

    return round(float(np.sqrt(np.mean(all_sq_dists))), 3)


# PYR1 pocket residues (1-indexed) — residues within 5Å of ligand in 3QN1
POCKET_RESIDUES_PYR1 = [
    59, 62, 81, 83, 88, 92, 94, 108, 110,
    115, 116, 117, 118, 120, 122, 141, 159, 160, 161, 163, 164, 167,
]


def compute_pocket_rmsd(
    struct1_path: str,
    struct2_path: str,
    protein_chain: str = 'A',
    pocket_residues: List[int] = None,
) -> Optional[float]:
    """Compute CA RMSD of pocket residues between two structures after full-chain superposition.

    Superimposes struct1 onto struct2 by all PYR1 CA atoms, then measures
    RMSD of only the pocket residue CA atoms. Detects whether the binding
    pocket conformation changes between binary and ternary predictions —
    e.g. gate loop movement or pocket collapse when HAB1 docks.

    High pocket RMSD (>1.5Å) with high ternary_iptm suggests induced fit
    or pocket disruption; low pocket RMSD confirms consistent pocket geometry.
    """
    if pocket_residues is None:
        pocket_residues = POCKET_RESIDUES_PYR1

    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    except ImportError:
        logger.error("Biopython required for pocket RMSD")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def _parse(path):
            p = Path(path)
            if p.suffix.lower() == '.cif':
                return MMCIFParser(QUIET=True).get_structure("s", str(p))
            return PDBParser(QUIET=True).get_structure("s", str(p))

        try:
            s1 = _parse(struct1_path)
            s2 = _parse(struct2_path)
        except Exception as e:
            logger.error(f"Failed to parse structures for pocket RMSD: {e}")
            return None

    ca1 = _get_ca_atoms(s1, protein_chain)
    ca2 = _get_ca_atoms(s2, protein_chain)

    if not ca1 or not ca2:
        return None

    n_ca = min(len(ca1), len(ca2))
    if n_ca < 10:
        return None

    # Superimpose s1 onto s2 by all CA atoms
    sup = Superimposer()
    try:
        sup.set_atoms(ca2[:n_ca], ca1[:n_ca])
        sup.apply(list(s1.get_atoms()))
    except Exception as e:
        logger.error(f"Pocket RMSD superposition failed: {e}")
        return None

    # Extract pocket CA coordinates from both structures
    pocket_set = set(pocket_residues)

    def _get_pocket_ca(struct):
        coords = {}
        for model in struct:
            for chain in model:
                if chain.id != protein_chain:
                    continue
                for res in chain:
                    resnum = res.get_id()[1]
                    if resnum in pocket_set:
                        for atom in res:
                            if atom.get_name() == 'CA':
                                coords[resnum] = atom.get_coord()
            break
        return coords

    coords1 = _get_pocket_ca(s1)
    coords2 = _get_pocket_ca(s2)

    # Only compare residues present in both
    common = sorted(set(coords1) & set(coords2))
    if len(common) < 3:
        logger.warning(f"Too few shared pocket residues ({len(common)}) for RMSD")
        return None

    sq_dists = [
        np.sum((coords1[r] - coords2[r]) ** 2)
        for r in common
    ]
    return round(float(np.sqrt(np.mean(sq_dists))), 3)


def compute_hbond_water_geometry(
    struct_path: str,
    ref_model_path: str,
    protein_chain: str = 'A',
    ligand_chain: str = 'B',
) -> Dict[str, Optional[float]]:
    """Compute H-bond water geometry after CA-aligning onto 3QN1 reference.

    1. Distance: conserved water O → closest ligand H-bond acceptor (O or N)
    2. Angle: Pro88:O — water:O — closest_ligand_acceptor (vertex at water O)
    """
    result = {'hbond_distance': None, 'hbond_angle': None}

    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    except ImportError:
        logger.error("Biopython required for H-bond geometry")
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def _parse(path):
            p = Path(path)
            if p.suffix.lower() == '.cif':
                return MMCIFParser(QUIET=True).get_structure("s", str(p))
            return PDBParser(QUIET=True).get_structure("s", str(p))

        try:
            pred_struct = _parse(struct_path)
            ref_struct = _parse(ref_model_path)
        except Exception as e:
            logger.error(f"Failed to parse structures for H-bond: {e}")
            return result

    ref_ca = _get_ca_atoms(ref_struct, 'A')
    pred_ca = _get_ca_atoms(pred_struct, protein_chain)

    if not ref_ca or not pred_ca:
        return result

    n_ca = min(len(ref_ca), len(pred_ca))
    if n_ca < 10:
        return result

    # Find reference water O (chain D:1:O) and predicted Pro88:O
    def _find_atom(struct, chain_id, res_id, atom_name):
        for model in struct:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.get_id()[1] != res_id:
                        continue
                    for atom in residue:
                        if atom.get_name() == atom_name:
                            return atom
            break
        return None

    water_O = _find_atom(ref_struct, 'D', 1, 'O')
    pro88_O = _find_atom(pred_struct, protein_chain, 88, 'O')

    if water_O is None or pro88_O is None:
        if water_O is None:
            logger.warning("Reference water D:1:O not found")
        if pro88_O is None:
            logger.warning("Pro88:O not found in prediction")
        return result

    # Get ligand atoms
    ligand_atoms = _get_ligand_heavy_atoms(pred_struct, ligand_chain)
    if not ligand_atoms:
        return result

    # Superimpose prediction onto reference
    sup = Superimposer()
    try:
        sup.set_atoms(ref_ca[:n_ca], pred_ca[:n_ca])
        sup.apply(ligand_atoms)
        sup.apply([pro88_O])
    except Exception as e:
        logger.error(f"H-bond superposition failed: {e}")
        return result

    # Find closest O or N in ligand to water position
    water_coord = water_O.get_coord()
    pro88_coord = pro88_O.get_coord()

    min_dist = float('inf')
    closest_coord = None

    for atom in ligand_atoms:
        elem = _resolve_element(atom)
        if elem in ('O', 'N'):
            coord = atom.get_coord()
            dist = float(np.linalg.norm(coord - water_coord))
            if dist < min_dist:
                min_dist = dist
                closest_coord = coord

    if closest_coord is None:
        return result

    result['hbond_distance'] = round(min_dist, 3)

    # Angle: Pro88:O — water:O — ligand_acceptor
    vec_to_pro = pro88_coord - water_coord
    vec_to_lig = closest_coord - water_coord

    norm_pro = np.linalg.norm(vec_to_pro)
    norm_lig = np.linalg.norm(vec_to_lig)

    if norm_pro < 1e-6 or norm_lig < 1e-6:
        return result

    cos_angle = np.clip(np.dot(vec_to_pro, vec_to_lig) / (norm_pro * norm_lig), -1.0, 1.0)
    result['hbond_angle'] = round(float(np.degrees(np.arccos(cos_angle))), 1)

    return result


# ═══════════════════════════════════════════════════════════════════
# LIGAND ORIENTATION / FLIP DETECTION
# ═══════════════════════════════════════════════════════════════════

def _find_carboxylate_atoms(ligand_atoms):
    """Identify carboxylate C and O atoms using distance-based bonding.

    A carboxylate carbon has exactly 2 oxygen neighbors within 1.65 Å.
    Works for LCA (C24-COO-), GLCA (glycine terminal COO-), LCA3S (C24-COO-).
    The sulfonate S in LCA3S is not a carbon, so it is never misidentified.
    The amide C=O in GLCA has only 1 O neighbor, so it is not misidentified.

    Returns (coo_carbon, [coo_oxygen1, coo_oxygen2]) or (None, []) if not found.
    """
    oxygens = [a for a in ligand_atoms if _resolve_element(a) == 'O']
    carbons = [a for a in ligand_atoms if _resolve_element(a) == 'C']

    for c in carbons:
        c_coord = c.get_coord()
        bonded_o = [
            o for o in oxygens
            if np.linalg.norm(o.get_coord() - c_coord) < 1.65
        ]
        if len(bonded_o) == 2:
            return c, bonded_o

    return None, []


def compute_ligand_flip_metrics(
    struct_path: str,
    ref_model_path: str,
    protein_chain: str = 'A',
    ligand_chain: str = 'B',
) -> Dict[str, Optional[float]]:
    """Detect the COO-to-R116 artifact in binary Boltz2 predictions.

    In binary (PYR1+ligand only), the carboxylate (COO-) can form an electrostatic
    salt bridge with R116 (latch), boosting binary confidence even for non-binders.
    This artifact is impossible in ternary because HAB1 Trp385 (the "lock") blocks
    R116. This function measures the carboxylate-to-R116 distance to quantify the
    artifact — it is orientation-agnostic and does NOT assume that OH-near-water is
    the only correct binding mode (COO-near-water is also a valid mode).

    After CA-superimposing onto the 3QN1 reference:
      coo_to_r116_dist  : closest carboxylate O to R116 guanidinium (NE/NH1/NH2)
                          (lower = carboxylate in the R116 artifact position)
                          (higher = carboxylate away from R116 = no artifact)
      coo_to_water_dist : closest carboxylate O to reference water (informational)
      oh_to_water_dist  : closest non-carboxylate O to reference water (informational;
                          NOTE: use with caution — COO-near-water is also a valid mode)
      flip_score        : coo_to_water_dist - oh_to_water_dist (informational only;
                          NOT a reliable filter since both orientations can be correct)

    For LCA3S: the carboxylate at C24 is correctly identified (2 O neighbors within
    1.65 Å). The sulfonate S and its 4 O atoms are never confused for a carboxylate.
    """
    result: Dict[str, Optional[float]] = {
        'oh_to_water_dist': None,
        'coo_to_water_dist': None,
        'coo_to_r116_dist': None,
        'flip_score': None,
    }

    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    except ImportError:
        logger.error("Biopython required for flip metric calculation")
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def _parse(path):
            p = Path(path)
            if p.suffix.lower() == '.cif':
                return MMCIFParser(QUIET=True).get_structure("s", str(p))
            return PDBParser(QUIET=True).get_structure("s", str(p))

        try:
            pred_struct = _parse(struct_path)
            ref_struct = _parse(ref_model_path)
        except Exception as e:
            logger.error(f"Failed to parse structures for flip metrics: {e}")
            return result

    ref_ca = _get_ca_atoms(ref_struct, 'A')
    pred_ca = _get_ca_atoms(pred_struct, protein_chain)

    if not ref_ca or not pred_ca:
        return result

    n_ca = min(len(ref_ca), len(pred_ca))
    if n_ca < 10:
        return result

    # Reference water O (D:1:O from 3QN1)
    def _find_atom(struct, chain_id, res_id, atom_name):
        for model in struct:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.get_id()[1] != res_id:
                        continue
                    for atom in residue:
                        if atom.get_name() == atom_name:
                            return atom
            break
        return None

    water_O = _find_atom(ref_struct, 'D', 1, 'O')
    if water_O is None:
        logger.warning("Reference water D:1:O not found — skipping flip metrics")
        return result

    # R116 guanidinium atoms on the predicted structure (will be moved by superimposer)
    r116_atoms = []
    for model in pred_struct:
        for chain in model:
            if chain.id != protein_chain:
                continue
            for residue in chain:
                if residue.get_id()[1] == 116:
                    for atom in residue:
                        if atom.get_name() in ('NE', 'NH1', 'NH2', 'CZ'):
                            r116_atoms.append(atom)
            break
        break

    ligand_atoms = _get_ligand_heavy_atoms(pred_struct, ligand_chain)
    if not ligand_atoms:
        return result

    # Identify carboxylate C and its 2 oxygens
    _, coo_oxygens = _find_carboxylate_atoms(ligand_atoms)
    if not coo_oxygens:
        logger.warning(f"No carboxylate carbon found in ligand at {struct_path} — skipping flip metrics")
        return result

    coo_oxygen_set = set(id(o) for o in coo_oxygens)
    oxygens = [a for a in ligand_atoms if _resolve_element(a) == 'O']
    hydroxyl_oxygens = [o for o in oxygens if id(o) not in coo_oxygen_set]

    # Superimpose prediction onto reference (moves ligand + R116 atoms to ref frame)
    atoms_to_move = ligand_atoms + r116_atoms
    sup = Superimposer()
    try:
        sup.set_atoms(ref_ca[:n_ca], pred_ca[:n_ca])
        sup.apply(atoms_to_move)
    except Exception as e:
        logger.error(f"Flip metrics superposition failed: {e}")
        return result

    water_coord = water_O.get_coord()

    # oh_to_water_dist
    if hydroxyl_oxygens:
        oh_dists = [float(np.linalg.norm(o.get_coord() - water_coord))
                    for o in hydroxyl_oxygens]
        result['oh_to_water_dist'] = round(min(oh_dists), 3)

    # coo_to_water_dist
    coo_water_dists = [float(np.linalg.norm(o.get_coord() - water_coord))
                       for o in coo_oxygens]
    result['coo_to_water_dist'] = round(min(coo_water_dists), 3)

    # coo_to_r116_dist
    if r116_atoms:
        r116_coords = np.array([a.get_coord() for a in r116_atoms])
        coo_coords = np.array([o.get_coord() for o in coo_oxygens])
        diff = coo_coords[:, None, :] - r116_coords[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        result['coo_to_r116_dist'] = round(float(dists.min()), 3)
    else:
        logger.warning("R116 not found in predicted structure — coo_to_r116_dist unavailable")

    # flip_score = COO-to-water minus OH-to-water
    if result['oh_to_water_dist'] is not None and result['coo_to_water_dist'] is not None:
        result['flip_score'] = round(
            result['coo_to_water_dist'] - result['oh_to_water_dist'], 3
        )

    return result


# ═══════════════════════════════════════════════════════════════════
# HAB1 Trp211 – LIGAND DISTANCE (ternary "lock" diagnostic)
# ═══════════════════════════════════════════════════════════════════

def compute_hab1_trp_ligand_distance(
    struct_path: str,
    hab1_chain: str = 'C',
    ligand_chain: str = 'B',
    trp_resid: int = 211,
) -> Optional[float]:
    """Compute minimum distance from HAB1 Trp211 sidechain to ligand.

    Trp211 in the 337-aa HAB1 construct = Trp385 in full-length HAB1,
    the "lock" residue (IQWQGA motif) that inserts into PYR1's pocket
    upon gate closure. A short distance (<5 A) indicates proper ternary
    assembly; a large distance (>10 A) means HAB1 is misoriented.

    No superposition needed — measured directly on predicted coordinates.
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        logger.error("Biopython required for Trp211 distance")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Path(struct_path)
        try:
            if p.suffix.lower() == '.cif':
                struct = MMCIFParser(QUIET=True).get_structure("s", str(p))
            else:
                struct = PDBParser(QUIET=True).get_structure("s", str(p))
        except Exception as e:
            logger.error(f"Failed to parse structure for Trp211 distance: {e}")
            return None

    # Trp sidechain atom names (indole ring + CB)
    trp_sc_names = {'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'}

    # Find Trp211 sidechain atoms on HAB1 chain
    trp_atoms = []
    for model in struct:
        for chain in model:
            if chain.id != hab1_chain:
                continue
            for residue in chain:
                if residue.get_id()[1] == trp_resid:
                    for atom in residue:
                        if atom.get_name() in trp_sc_names:
                            trp_atoms.append(atom)
                    break
        break

    if not trp_atoms:
        logger.warning(f"Trp{trp_resid} sidechain not found on chain {hab1_chain}")
        return None

    # Get ligand heavy atoms
    ligand_atoms = _get_ligand_heavy_atoms(struct, ligand_chain)
    if not ligand_atoms:
        logger.warning(f"No ligand atoms on chain {ligand_chain}")
        return None

    # Compute minimum distance
    trp_coords = np.array([a.get_coord() for a in trp_atoms])
    lig_coords = np.array([a.get_coord() for a in ligand_atoms])

    # Pairwise distances: (n_trp, n_lig)
    diff = trp_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_dist = float(dists.min())

    return round(min_dist, 3)


# ═══════════════════════════════════════════════════════════════════
# BURIED UNSATISFIED POLAR ATOMS (BUNs) — via bunsalyze
# ═══════════════════════════════════════════════════════════════════

def compute_buns(
    struct_path: str,
    ligand_smiles: str,
    sasa_threshold: float = 1.0,
) -> Dict[str, Optional[float]]:
    """Compute buried unsatisfied polar atoms using bunsalyze (Polizzi lab).

    Uses SASA + alpha hull convexity for burial detection and proper H-bond
    geometry with clash penalties.

    BUNs Score = 2 * ligand_BUNs + 1 * protein_BUNs
    (per Luo et al. exatecan reranking formula)

    Requires: pip install bunsalyze
    """
    result = {
        'protein_buns': None, 'ligand_buns': None, 'buns_score': None,
        'ligand_fraction_unsat': None, 'protein_fraction_unsat': None,
    }

    try:
        import prody as pr
        from bunsalyze.bunsalyze import main as bunsalyze_main
    except ImportError:
        logger.error("bunsalyze and prody required for BUNs calculation. "
                      "Install with: pip install bunsalyze")
        return result

    try:
        pr.confProDy(verbosity='none')
        complex_ = pr.parsePDB(str(struct_path))
        buns_result = bunsalyze_main(
            input_path=str(struct_path),
            protein_complex=complex_,
            smiles=ligand_smiles,
            sasa_threshold=sasa_threshold,
            silent=True,
        )
    except Exception as e:
        logger.warning(f"bunsalyze failed for {struct_path}: {e}")
        return result

    protein_buns = len(buns_result.get('protein_buns', []))
    ligand_buns = len(buns_result.get('ligand_buns', []))

    result['protein_buns'] = protein_buns
    result['ligand_buns'] = ligand_buns
    result['buns_score'] = 2 * ligand_buns + protein_buns
    result['ligand_fraction_unsat'] = buns_result.get('ligand_buried_fraction_unsat')
    result['protein_fraction_unsat'] = buns_result.get('protein_buried_fraction_unsat')

    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def extract_pocket_sequence(pdb_path, pocket_positions=None):
    """Extract 16-residue pocket sequence from a PDB.

    Returns string like 'HHFVKSTIKEVFSYIA' or None if not enough residues found.
    """
    if pocket_positions is None:
        pocket_positions = POCKET_POSITIONS
    residues = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[21] != 'A':
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resnum in pocket_positions:
                    residues[resnum] = THREE_TO_ONE.get(resname, 'X')
    except Exception:
        return None
    if len(residues) < len(pocket_positions):
        return None
    return ''.join(residues.get(p, 'X') for p in sorted(pocket_positions))


def classify_binding_mode(row):
    """Classify a design's binding orientation.

    Uses binary_coo_to_water_dist and binary_oh_to_water_dist from row dict.
    normal:  OH at top of pocket, mediating conserved water (most common).
    flipped: COO at top of pocket, mediating conserved water.
    Returns 'normal', 'flipped', or 'unknown'.
    """
    try:
        coo_dist = float(row.get('binary_coo_to_water_dist', 999))
        oh_dist = float(row.get('binary_oh_to_water_dist', 999))
    except (ValueError, TypeError):
        return 'unknown'

    if coo_dist >= 99 and oh_dist >= 99:
        return 'unknown'
    if coo_dist < oh_dist and coo_dist < 4.0:
        return 'flipped'
    if oh_dist <= coo_dist and oh_dist < 4.0:
        return 'normal'
    return 'unknown'


def compute_polar_unsatisfied(pdb_path, gate_residue=88, sat_cutoff=3.5):
    """Count unsatisfied ligand polar oxygens (both OH and COO).

    Checks ALL ligand oxygens for protein contacts, with one exemption:
    the oxygen closest to the gate residue (water-mediated, not modeled
    by Boltz) is skipped.

    Returns dict with:
        n_polar_checked: number of ligand O atoms checked (total - 1 water)
        n_polar_unsatisfied: how many lack a protein O/N within sat_cutoff
        n_coo_unsatisfied: unsatisfied carboxylate oxygens specifically
        n_oh_unsatisfied: unsatisfied hydroxyl oxygens specifically
        water_mediated_type: 'OH' or 'COO' (which type is at the gate)
        flipped: True if water-mediated OH is closer to COO than the other OH
    Returns None if PDB cannot be parsed.
    """
    protein_acceptors = []
    ligand_oxygens = []
    ligand_carbons = []
    gate_ca = None

    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = atom_name[0]

                if ch == 'A':
                    if elem in ('O', 'N'):
                        protein_acceptors.append(np.array([x, y, z]))
                    resnum = int(line[22:26].strip())
                    if resnum == gate_residue and atom_name == 'CA':
                        gate_ca = np.array([x, y, z])
                elif ch == 'B':
                    coord = np.array([x, y, z])
                    if elem == 'O':
                        ligand_oxygens.append((coord, atom_name))
                    elif elem == 'C':
                        ligand_carbons.append((coord, atom_name))
    except Exception:
        return None

    if not ligand_oxygens or not protein_acceptors:
        return None

    protein_coords = np.array(protein_acceptors)

    # Identify carboxylate oxygens
    coo_indices = set()
    for c_coord, _ in ligand_carbons:
        bonded = [i for i, (o_coord, _) in enumerate(ligand_oxygens)
                  if np.linalg.norm(o_coord - c_coord) < 1.65]
        if len(bonded) == 2:
            coo_indices.update(bonded)

    # Identify water-mediated oxygen: closest to gate CA (any type)
    water_idx = None
    if gate_ca is not None and len(ligand_oxygens) > 1:
        gate_dists = [(np.linalg.norm(coord - gate_ca), i)
                      for i, (coord, _) in enumerate(ligand_oxygens)]
        gate_dists.sort()
        water_idx = gate_dists[0][1]

    water_mediated_type = None
    if water_idx is not None:
        water_mediated_type = 'COO' if water_idx in coo_indices else 'OH'

    # Check flipped orientation (only relevant when 2+ hydroxyl OHs)
    hydroxyl_os = [(i, coord) for i, (coord, _) in enumerate(ligand_oxygens)
                   if i not in coo_indices]
    flipped = False
    if (water_idx is not None and water_idx not in coo_indices
            and len(hydroxyl_os) == 2 and coo_indices):
        coo_coords = [ligand_oxygens[i][0] for i in coo_indices]
        coo_centroid = np.mean(coo_coords, axis=0)
        water_coord = ligand_oxygens[water_idx][0]
        other_idx = [i for i, _ in hydroxyl_os if i != water_idx][0]
        other_coord = ligand_oxygens[other_idx][0]
        if float(np.linalg.norm(water_coord - coo_centroid)) < \
           float(np.linalg.norm(other_coord - coo_centroid)):
            flipped = True

    # Count unsatisfied oxygens (skip water-mediated)
    n_polar_checked = 0
    n_polar_unsatisfied = 0
    n_coo_unsatisfied = 0
    n_oh_unsatisfied = 0
    for i, (coord, _) in enumerate(ligand_oxygens):
        if i == water_idx:
            continue
        n_polar_checked += 1
        dists = np.linalg.norm(protein_coords - coord, axis=1)
        if float(dists.min()) > sat_cutoff:
            n_polar_unsatisfied += 1
            if i in coo_indices:
                n_coo_unsatisfied += 1
            else:
                n_oh_unsatisfied += 1

    return {
        'n_polar_checked': n_polar_checked,
        'n_polar_unsatisfied': n_polar_unsatisfied,
        'n_coo_unsatisfied': n_coo_unsatisfied,
        'n_oh_unsatisfied': n_oh_unsatisfied,
        'water_mediated_type': water_mediated_type,
        'flipped': flipped,
    }


BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}

def compute_interface_buried_unsats(pdb_path, interface_cutoff=5.0,
                                     hbond_cutoff=3.5,
                                     designed_positions=None):
    """Count buried unsatisfied polar atoms at the protein-ligand interface.

    Only checks sidechain polar atoms at designed_positions (defaults to
    POCKET_POSITIONS). For each designed residue within interface_cutoff of
    any ligand heavy atom, checks if its sidechain O/N atoms have an H-bond
    partner (any O/N on protein or ligand) within hbond_cutoff.

    Also counts ligand polar atoms lacking protein/ligand polar contacts.

    Returns dict or None if parsing fails.
    """
    if designed_positions is None:
        designed_positions = set(POCKET_POSITIONS)
    else:
        designed_positions = set(designed_positions)

    # Parse PDB
    protein_atoms = []   # (coord, resnum, resname, atom_name, elem)
    ligand_atoms = []    # (coord, atom_name, elem)

    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) >= 78 else atom_name[0]

                if ch == 'A':
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    protein_atoms.append(
                        (np.array([x, y, z]), resnum, resname, atom_name, elem))
                elif ch == 'B':
                    ligand_atoms.append(
                        (np.array([x, y, z]), atom_name, elem))
    except Exception:
        return None

    if not protein_atoms or not ligand_atoms:
        return None

    ligand_heavy = [(c, an, el) for c, an, el in ligand_atoms
                    if el not in ('H', 'D')]
    ligand_heavy_coords = np.array([a[0] for a in ligand_heavy])

    # Collect all polar (O/N) coordinates for satisfaction checks
    all_polar = []
    for coord, _, _, atom_name, elem in protein_atoms:
        if elem in ('O', 'N'):
            all_polar.append(coord)
    for coord, _, elem in ligand_atoms:
        if elem in ('O', 'N'):
            all_polar.append(coord)
    if not all_polar:
        return None
    all_polar_coords = np.array(all_polar)

    # Find which designed positions are at the interface
    interface_designed = set()
    for coord, resnum, _, _, elem in protein_atoms:
        if resnum not in designed_positions:
            continue
        if elem in ('H', 'D'):
            continue
        dists = np.linalg.norm(ligand_heavy_coords - coord, axis=1)
        if float(dists.min()) <= interface_cutoff:
            interface_designed.add(resnum)

    # Check sidechain polar atoms of designed interface residues
    sc_checked = 0
    sc_unsatisfied = 0
    unsatisfied_details = []
    for coord, resnum, resname, atom_name, elem in protein_atoms:
        if resnum not in interface_designed:
            continue
        if atom_name in BACKBONE_ATOMS:
            continue
        if elem not in ('O', 'N'):
            continue

        sc_checked += 1
        # Check for any polar partner within hbond_cutoff (excluding self)
        dists = np.linalg.norm(all_polar_coords - coord, axis=1)
        partners = np.sum((dists > 0.1) & (dists <= hbond_cutoff))
        if partners == 0:
            sc_unsatisfied += 1
            unsatisfied_details.append(
                '%s_%d_%s' % (resname, resnum, atom_name))

    # Check ligand polar atoms (all O/N)
    lig_checked = 0
    lig_unsatisfied = 0
    for coord, atom_name, elem in ligand_atoms:
        if elem not in ('O', 'N'):
            continue
        lig_checked += 1
        dists = np.linalg.norm(all_polar_coords - coord, axis=1)
        partners = np.sum((dists > 0.1) & (dists <= hbond_cutoff))
        if partners == 0:
            lig_unsatisfied += 1

    return {
        'n_designed_interface_res': len(interface_designed),
        'n_sc_polar_checked': sc_checked,
        'n_sc_polar_unsatisfied': sc_unsatisfied,
        'n_lig_polar_checked': lig_checked,
        'n_lig_polar_unsatisfied': lig_unsatisfied,
        'n_interface_unsatisfied_total': sc_unsatisfied + lig_unsatisfied,
        'interface_unsatisfied_details': ';'.join(unsatisfied_details) if unsatisfied_details else '',
    }


def _parse_ligand_polar_env(pdb_path, gate_residue=88):
    """Parse PDB for ligand oxygens and protein polar atoms.

    Shared helper for OH and COO contact analysis.
    Returns (protein_polar, ligand_oxygens, ligand_carbons, coo_indices,
             water_idx, gate_ca) or None if parsing fails.
    """
    protein_polar = []  # (coord, resnum, resname, atom_name)
    ligand_oxygens = []
    ligand_carbons = []
    gate_ca = None

    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = atom_name[0]

                if ch == 'A':
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    if elem in ('O', 'N'):
                        protein_polar.append(
                            (np.array([x, y, z]), resnum, resname, atom_name))
                    if resnum == gate_residue and atom_name == 'CA':
                        gate_ca = np.array([x, y, z])
                elif ch == 'B':
                    coord = np.array([x, y, z])
                    if elem == 'O':
                        ligand_oxygens.append((coord, atom_name))
                    elif elem == 'C':
                        ligand_carbons.append((coord, atom_name))
    except Exception:
        return None

    if not ligand_oxygens or not protein_polar:
        return None

    # Identify carboxylate oxygens
    coo_indices = set()
    for c_coord, _ in ligand_carbons:
        bonded = [i for i, (o_coord, _) in enumerate(ligand_oxygens)
                  if np.linalg.norm(o_coord - c_coord) < 1.65]
        if len(bonded) == 2:
            coo_indices.update(bonded)

    # Identify water-mediated oxygen (closest to gate CA)
    water_idx = None
    if gate_ca is not None and len(ligand_oxygens) > 1:
        gate_dists = [(np.linalg.norm(coord - gate_ca), i)
                      for i, (coord, _) in enumerate(ligand_oxygens)]
        gate_dists.sort()
        water_idx = gate_dists[0][1]

    return (protein_polar, ligand_oxygens, ligand_carbons,
            coo_indices, water_idx, gate_ca)


def find_oh_polar_contacts(pdb_path, gate_residue=88, cutoff=3.5):
    """Find closest protein polar atom to each non-water ligand hydroxyl O.

    Returns list of dicts (one per non-water OH):
        {oh_atom, closest_res, closest_resname, closest_atom, closest_dist}
    Returns None if PDB cannot be parsed.
    """
    parsed = _parse_ligand_polar_env(pdb_path, gate_residue)
    if parsed is None:
        return None
    protein_polar, ligand_oxygens, _, coo_indices, water_idx, _ = parsed

    # Non-water hydroxyl oxygens
    hydroxyl_indices = [i for i in range(len(ligand_oxygens))
                        if i not in coo_indices and i != water_idx]

    protein_coords = np.array([p[0] for p in protein_polar])
    results = []
    for i in hydroxyl_indices:
        coord, oh_name = ligand_oxygens[i]
        dists = np.linalg.norm(protein_coords - coord, axis=1)
        # Find closest polar atom within cutoff
        min_idx = np.argmin(dists)
        min_dist = float(dists[min_idx])
        if min_dist <= cutoff:
            _, resnum, resname, atom_name = protein_polar[min_idx]
            results.append({
                'oh_atom': oh_name,
                'closest_res': resnum,
                'closest_resname': resname,
                'closest_atom': atom_name,
                'closest_dist': round(min_dist, 2),
            })
        else:
            results.append({
                'oh_atom': oh_name,
                'closest_res': None,
                'closest_resname': None,
                'closest_atom': None,
                'closest_dist': None,
            })

    return results


def find_coo_polar_contacts(pdb_path, gate_residue=88, cutoff=3.5):
    """Find closest protein polar atom to each non-water carboxylate oxygen.

    Returns list of dicts (one per non-water COO oxygen):
        {coo_atom, closest_res, closest_resname, closest_atom, closest_dist}
    Returns None if PDB cannot be parsed.
    """
    parsed = _parse_ligand_polar_env(pdb_path, gate_residue)
    if parsed is None:
        return None
    protein_polar, ligand_oxygens, _, coo_indices, water_idx, _ = parsed

    # Non-water carboxylate oxygens
    coo_contact_indices = [i for i in range(len(ligand_oxygens))
                           if i in coo_indices and i != water_idx]

    if not coo_contact_indices:
        return []

    protein_coords = np.array([p[0] for p in protein_polar])
    results = []
    for i in coo_contact_indices:
        coord, coo_name = ligand_oxygens[i]
        dists = np.linalg.norm(protein_coords - coord, axis=1)
        min_idx = np.argmin(dists)
        min_dist = float(dists[min_idx])
        if min_dist <= cutoff:
            _, resnum, resname, atom_name = protein_polar[min_idx]
            results.append({
                'coo_atom': coo_name,
                'closest_res': resnum,
                'closest_resname': resname,
                'closest_atom': atom_name,
                'closest_dist': round(min_dist, 2),
            })
        else:
            results.append({
                'coo_atom': coo_name,
                'closest_res': None,
                'closest_resname': None,
                'closest_atom': None,
                'closest_dist': None,
            })

    return results


def analyze_predictions(
    binary_dirs: List[str] = None,
    ternary_dirs: List[str] = None,
    ref_pdb: str = None,
    ligand_smiles: str = None,
    ref_ligand_pdb: str = None,
    ref_ligand_chain: str = 'B',
    ref_ternary_pdb: str = None,
    geometry_weight: float = 2.0,
    unsat_penalty: float = 0.0,
    gate_hbond_max: float = 4.0,
    gate_hbond_min: float = 1.8,
    gate_plddt_ligand: float = 0.65,
    gate_coo_to_r116: float = 4.0,
    gate_hab1_clash: float = 2.0,
    gate_sidechain_hab1trp: float = 2.0,
    gate_latch_rmsd: float = 1.0,
    oh_contact_exclude_res: List[int] = None,
) -> List[Dict]:
    """Analyze all Boltz predictions and return list of metric dicts."""

    results = []

    # Initialize ligand geometry checker if reference provided
    geom_checker = None
    if ref_ligand_pdb:
        if _HAS_LIGAND_GEOMETRY:
            try:
                geom_checker = LigandGeometryChecker(
                    ref_ligand_pdb, ligand_chain=ref_ligand_chain)
            except (ValueError, Exception) as e:
                logger.warning(f"Could not initialize ligand geometry checker: {e}")
        else:
            logger.warning("ligand_geometry module not available — skipping geometry check")

    # Initialize HAB1 clash checker if ternary reference provided
    clash_checker = None
    if ref_ternary_pdb:
        if _HAS_LIGAND_GEOMETRY:
            try:
                clash_checker = HAB1ClashChecker(
                    ref_ternary_pdb,
                    sidechain_clash_residues=[117, 159])
            except (ValueError, Exception) as e:
                logger.warning(f"Could not initialize HAB1 clash checker: {e}")
        else:
            logger.warning("ligand_geometry module not available — skipping HAB1 clash check")

    # Collect predictions from all directories
    binary_preds = {}
    ternary_preds = {}

    for d in (binary_dirs or []):
        preds = find_boltz_predictions(d)
        for p in preds:
            binary_preds[p['name']] = p
        logger.info(f"Found {len(preds)} binary predictions in {d}")

    for d in (ternary_dirs or []):
        preds = find_boltz_predictions(d)
        for p in preds:
            ternary_preds[p['name']] = p
        logger.info(f"Found {len(preds)} ternary predictions in {d}")

    all_names = sorted(set(list(binary_preds.keys()) + list(ternary_preds.keys())))
    logger.info(f"Total unique names: {len(all_names)}")

    for name in all_names:
        row = {'name': name}

        # ── Binary metrics ──
        if name in binary_preds:
            bp = binary_preds[name]

            conf = extract_confidence_metrics(bp['confidence'])
            for k, v in conf.items():
                row[f'binary_{k}'] = v

            plddt = extract_plddt_per_chain(bp['plddt'], bp['structure'])
            row['binary_plddt_protein'] = plddt['plddt_protein']
            row['binary_plddt_ligand'] = plddt['plddt_ligand']
            row['binary_plddt_pocket'] = plddt['plddt_pocket']

            aff = extract_affinity(bp['affinity'])
            for k, v in aff.items():
                row[f'binary_{k}'] = v

            if ref_pdb:
                geom = compute_hbond_water_geometry(
                    str(bp['structure']), ref_pdb,
                    protein_chain='A', ligand_chain='B',
                )
                row['binary_hbond_distance'] = geom['hbond_distance']
                row['binary_hbond_angle'] = geom['hbond_angle']

                flip = compute_ligand_flip_metrics(
                    str(bp['structure']), ref_pdb,
                    protein_chain='A', ligand_chain='B',
                )
                row['binary_oh_to_water_dist'] = flip['oh_to_water_dist']
                row['binary_coo_to_water_dist'] = flip['coo_to_water_dist']
                row['binary_coo_to_r116_dist'] = flip['coo_to_r116_dist']
                row['binary_flip_score'] = flip['flip_score']

            # Ligand geometry / stereochemistry check
            if geom_checker:
                lg = geom_checker.check(str(bp['structure']), ligand_chain='B')
                if lg:
                    row['binary_core_rmsd'] = lg['core_rmsd']
                    row['binary_max_dev'] = lg['max_dev']
                    row['binary_planarity_ratio'] = lg['planarity_ratio']
                    row['binary_ligand_distorted'] = 1 if lg['distorted'] else 0

            # HAB1 clash check (ligand vs Trp lock + sidechain clashes)
            if clash_checker:
                clash = clash_checker.check(str(bp['structure']))
                if clash:
                    row['binary_hab1_clash_dist'] = clash['min_dist']
                    row['binary_hab1_clash_res'] = clash['closest_hab1_res']
                    for resnum, sc_clash in clash.get('sidechain_clashes', {}).items():
                        if sc_clash is not None:
                            row[f'binary_res{resnum}_hab1trp_dist'] = sc_clash['min_dist']
                        else:
                            row[f'binary_res{resnum}_hab1trp_dist'] = None

            # Latch loop RMSD (res 114-118)
            if ref_pdb and _HAS_LIGAND_GEOMETRY:
                latch = compute_latch_rmsd(str(bp['structure']), ref_pdb)
                row['binary_latch_rmsd'] = latch['latch_rmsd']
                row['binary_latch_ca_rmsd'] = latch['latch_ca_rmsd']

            if ligand_smiles:
                buns = compute_buns(str(bp['structure']), ligand_smiles)
                row['binary_protein_buns'] = buns['protein_buns']
                row['binary_ligand_buns'] = buns['ligand_buns']
                row['binary_buns_score'] = buns['buns_score']
                row['binary_ligand_fraction_unsat'] = buns['ligand_fraction_unsat']
                row['binary_protein_fraction_unsat'] = buns['protein_fraction_unsat']

        # ── Ternary metrics ──
        if name in ternary_preds:
            tp = ternary_preds[name]

            conf = extract_confidence_metrics(tp['confidence'])
            for k, v in conf.items():
                row[f'ternary_{k}'] = v

            plddt = extract_plddt_per_chain(tp['plddt'], tp['structure'])
            row['ternary_plddt_protein'] = plddt['plddt_protein']
            row['ternary_plddt_ligand'] = plddt['plddt_ligand']
            row['ternary_plddt_hab1'] = plddt['plddt_hab1']
            row['ternary_plddt_pocket'] = plddt['plddt_pocket']

            aff = extract_affinity(tp['affinity'])
            for k, v in aff.items():
                row[f'ternary_{k}'] = v

            if ref_pdb:
                geom = compute_hbond_water_geometry(
                    str(tp['structure']), ref_pdb,
                    protein_chain='A', ligand_chain='B',
                )
                row['ternary_hbond_distance'] = geom['hbond_distance']
                row['ternary_hbond_angle'] = geom['hbond_angle']

                flip = compute_ligand_flip_metrics(
                    str(tp['structure']), ref_pdb,
                    protein_chain='A', ligand_chain='B',
                )
                row['ternary_oh_to_water_dist'] = flip['oh_to_water_dist']
                row['ternary_coo_to_water_dist'] = flip['coo_to_water_dist']
                row['ternary_coo_to_r116_dist'] = flip['coo_to_r116_dist']
                row['ternary_flip_score'] = flip['flip_score']

            # Ligand geometry / stereochemistry check
            if geom_checker:
                lg = geom_checker.check(str(tp['structure']), ligand_chain='B')
                if lg:
                    row['ternary_core_rmsd'] = lg['core_rmsd']
                    row['ternary_max_dev'] = lg['max_dev']
                    row['ternary_planarity_ratio'] = lg['planarity_ratio']
                    row['ternary_ligand_distorted'] = 1 if lg['distorted'] else 0

            # Latch loop RMSD (res 114-118)
            if ref_pdb and _HAS_LIGAND_GEOMETRY:
                latch = compute_latch_rmsd(str(tp['structure']), ref_pdb)
                row['ternary_latch_rmsd'] = latch['latch_rmsd']
                row['ternary_latch_ca_rmsd'] = latch['latch_ca_rmsd']

            # HAB1 Trp211 ("lock") distance to ligand
            trp_dist = compute_hab1_trp_ligand_distance(
                str(tp['structure']),
                hab1_chain='C', ligand_chain='B', trp_resid=211,
            )
            row['ternary_trp211_ligand_distance'] = trp_dist

            if ligand_smiles:
                buns = compute_buns(str(tp['structure']), ligand_smiles)
                row['ternary_protein_buns'] = buns['protein_buns']
                row['ternary_ligand_buns'] = buns['ligand_buns']
                row['ternary_buns_score'] = buns['buns_score']
                row['ternary_ligand_fraction_unsat'] = buns['ligand_fraction_unsat']
                row['ternary_protein_fraction_unsat'] = buns['protein_fraction_unsat']

        # ── Binary-to-ternary structural comparisons ──
        if name in binary_preds and name in ternary_preds:
            rmsd = compute_ligand_rmsd(
                str(binary_preds[name]['structure']),
                str(ternary_preds[name]['structure']),
                protein_chain='A', ligand_chain='B',
            )
            row['ligand_rmsd_binary_vs_ternary'] = rmsd

            pocket_rmsd = compute_pocket_rmsd(
                str(binary_preds[name]['structure']),
                str(ternary_preds[name]['structure']),
                protein_chain='A',
            )
            row['pocket_rmsd_binary_vs_ternary'] = pocket_rmsd

        # ── Composite scores ──
        for prefix in ('binary', 'ternary'):
            plddt_lig = row.get(f'{prefix}_plddt_ligand')
            pbind = row.get(f'{prefix}_affinity_probability_binary')
            hbond_dist = row.get(f'{prefix}_hbond_distance')
            hbond_ang = row.get(f'{prefix}_hbond_angle')

            # Boltz score (NISE-style): ligand pLDDT + P(binder), range 0-2
            if plddt_lig is not None and pbind is not None:
                row[f'{prefix}_boltz_score'] = round(plddt_lig + pbind, 4)
            else:
                row[f'{prefix}_boltz_score'] = None

            # Geometry scores: water network quality (Gaussian proximity to reference)
            # NOTE: raw hbond_distance and hbond_angle are better discriminators
            # than these Gaussian scores in combination analysis (z-score approach).
            # Distance component: Gaussian centered at 2.7A (ideal O-H...O heavy-atom distance)
            #   sigma=0.8A, so score ~1.0 at 2.7A, ~0.5 at 1.9/3.5A, ~0 beyond 5A
            # Angle component: Gaussian centered at 90.5° (3QN1 crystal reference)
            #   Actual ideal may range from ~90° (crystal) to ~109.5° (tetrahedral)
            #   sigma=25° to accommodate this uncertainty
            import math
            dist_score = None
            ang_score = None
            if hbond_dist is not None:
                ideal_dist = 2.7  # Angstroms
                dist_sigma = 0.8
                dist_score = math.exp(-0.5 * ((hbond_dist - ideal_dist) / dist_sigma) ** 2)
            if hbond_ang is not None:
                ideal_ang = 90.5  # degrees (from 3QN1 crystal: Pro88:O—water:O—ABA:O2)
                ang_sigma = 25.0
                ang_score = math.exp(-0.5 * ((hbond_ang - ideal_ang) / ang_sigma) ** 2)

            row[f'{prefix}_geometry_dist_score'] = round(dist_score, 4) if dist_score is not None else None
            row[f'{prefix}_geometry_ang_score'] = round(ang_score, 4) if ang_score is not None else None
            if dist_score is not None:
                combined_ang = ang_score if ang_score is not None else 1.0
                row[f'{prefix}_geometry_score'] = round(
                    0.5 * dist_score + 0.5 * combined_ang, 4)
            else:
                row[f'{prefix}_geometry_score'] = None

            # Total score: plddt_ligand + geometry_weight * geometry_score
            # Weight=2.0 emphasizes water network H-bond geometry (range 0-3)
            geom_s = row.get(f'{prefix}_geometry_score')
            if plddt_lig is not None and geom_s is not None:
                row[f'{prefix}_total_score'] = round(
                    plddt_lig + geometry_weight * geom_s, 4)
            else:
                row[f'{prefix}_total_score'] = None

        # ── Gate pass/fail columns (binary only) ──
        if name in binary_preds:
            hbd = row.get('binary_hbond_distance')
            row['pass_hbond_dist_max'] = int(hbd is not None and hbd < gate_hbond_max) if hbd is not None else 0
            row['pass_hbond_dist_min'] = int(hbd is not None and hbd > gate_hbond_min) if hbd is not None else 0

            pll = row.get('binary_plddt_ligand')
            row['pass_plddt_ligand'] = int(pll is not None and pll > gate_plddt_ligand) if pll is not None else 0

            coo_r116 = row.get('binary_coo_to_r116_dist')
            row['pass_coo_to_r116'] = int(coo_r116 is not None and coo_r116 > gate_coo_to_r116) if coo_r116 is not None else 1

            lig_dist = row.get('binary_ligand_distorted')
            row['pass_ligand_geometry'] = int(lig_dist is None or lig_dist < 1)

            clash_d = row.get('binary_hab1_clash_dist')
            row['pass_hab1_clash'] = int(clash_d is None or clash_d > gate_hab1_clash)

            # Sidechain vs HAB1 Trp clash gate (residues 117, 159)
            sc_clash_pass = True
            for sc_res in [117, 159]:
                sc_d = row.get(f'binary_res{sc_res}_hab1trp_dist')
                if sc_d is not None and sc_d < gate_sidechain_hab1trp:
                    sc_clash_pass = False
                    break
            row['pass_sidechain_hab1trp'] = int(sc_clash_pass)

            latch = row.get('binary_latch_rmsd')
            row['pass_latch_rmsd'] = int(latch is not None and latch < gate_latch_rmsd) if latch is not None else 1

            # ── Binding mode ──
            row['binary_binding_mode'] = classify_binding_mode(row)

            # ── Pocket sequence ──
            bp = binary_preds[name]
            pocket_seq = extract_pocket_sequence(str(bp['structure']))
            row['binary_pocket_sequence'] = pocket_seq if pocket_seq else ''

            # ── Polar satisfaction + ring OH flipped check ──
            polar = compute_polar_unsatisfied(str(bp['structure']))
            if polar is not None:
                row['binary_n_polar_checked'] = polar['n_polar_checked']
                row['binary_n_polar_unsatisfied'] = polar['n_polar_unsatisfied']
                row['binary_n_coo_unsatisfied'] = polar['n_coo_unsatisfied']
                row['binary_n_oh_unsatisfied'] = polar['n_oh_unsatisfied']
                row['binary_water_mediated_type'] = polar['water_mediated_type']
                row['binary_ring_oh_flipped'] = int(bool(polar['flipped']))
            else:
                row['binary_n_polar_checked'] = None
                row['binary_n_polar_unsatisfied'] = None
                row['binary_n_coo_unsatisfied'] = None
                row['binary_n_oh_unsatisfied'] = None
                row['binary_water_mediated_type'] = None
                row['binary_ring_oh_flipped'] = 0

            # ── OH polar contact identification ──
            oh_contacts = find_oh_polar_contacts(str(bp['structure']))
            if oh_contacts:
                # Report the first non-water OH (typically O44 for CDCA)
                oc = oh_contacts[0]
                row['binary_oh_contact_atom'] = oc['oh_atom']
                row['binary_oh_contact_res'] = oc['closest_res']
                row['binary_oh_contact_resname'] = oc['closest_resname']
                row['binary_oh_contact_protein_atom'] = oc['closest_atom']
                row['binary_oh_contact_dist'] = oc['closest_dist']
            else:
                row['binary_oh_contact_atom'] = None
                row['binary_oh_contact_res'] = None
                row['binary_oh_contact_resname'] = None
                row['binary_oh_contact_protein_atom'] = None
                row['binary_oh_contact_dist'] = None

            # ── COO polar contact identification ──
            coo_contacts = find_coo_polar_contacts(str(bp['structure']))
            if coo_contacts:
                oc = coo_contacts[0]
                row['binary_coo_contact_atom'] = oc['coo_atom']
                row['binary_coo_contact_res'] = oc['closest_res']
                row['binary_coo_contact_resname'] = oc['closest_resname']
                row['binary_coo_contact_protein_atom'] = oc['closest_atom']
                row['binary_coo_contact_dist'] = oc['closest_dist']
            else:
                row['binary_coo_contact_atom'] = None
                row['binary_coo_contact_res'] = None
                row['binary_coo_contact_resname'] = None
                row['binary_coo_contact_protein_atom'] = None
                row['binary_coo_contact_dist'] = None

            # ── Interface buried unsatisfied polars ──
            iface = compute_interface_buried_unsats(str(bp['structure']))
            if iface is not None:
                row['binary_n_interface_res'] = iface['n_designed_interface_res']
                row['binary_n_sc_polar_checked'] = iface['n_sc_polar_checked']
                row['binary_n_sc_polar_unsatisfied'] = iface['n_sc_polar_unsatisfied']
                row['binary_n_lig_polar_checked'] = iface['n_lig_polar_checked']
                row['binary_n_lig_polar_unsatisfied'] = iface['n_lig_polar_unsatisfied']
                row['binary_n_interface_unsatisfied'] = iface['n_interface_unsatisfied_total']
                row['binary_interface_unsat_details'] = iface['interface_unsatisfied_details']
            else:
                row['binary_n_interface_res'] = None
                row['binary_n_sc_polar_checked'] = None
                row['binary_n_sc_polar_unsatisfied'] = None
                row['binary_n_lig_polar_checked'] = None
                row['binary_n_lig_polar_unsatisfied'] = None
                row['binary_n_interface_unsatisfied'] = None
                row['binary_interface_unsat_details'] = None

            # Ring OH flipped only matters for normal-mode designs
            # (bad geometry where the ring OH closest to water is also
            # closest to the COO, indicating incorrect steroid orientation)
            mode = row['binary_binding_mode']
            is_ring_flipped = row['binary_ring_oh_flipped'] == 1 and mode == 'normal'
            row['pass_ring_oh_flipped'] = int(not is_ring_flipped)

            # OH contact at excluded residue (e.g. latch H115) → count as
            # unsatisfied buried polar so that unsat_penalty penalizes it
            row['binary_oh_contact_excluded'] = 0
            if oh_contact_exclude_res and mode == 'normal':
                oh_res = row.get('binary_oh_contact_res')
                if oh_res is not None and oh_res in oh_contact_exclude_res:
                    row['binary_oh_contact_excluded'] = 1
                    if row.get('binary_n_polar_unsatisfied') is not None:
                        row['binary_n_polar_unsatisfied'] += 1
                        row['binary_n_oh_unsatisfied'] += 1

            # ── Unsatisfied polar penalty (applied to binary_total_score) ──
            if unsat_penalty > 0 and row.get('binary_total_score') is not None:
                n_unsat = row.get('binary_n_polar_unsatisfied')
                if n_unsat is not None:
                    row['binary_total_score'] = round(
                        row['binary_total_score'] - unsat_penalty * n_unsat, 4)

            # ── pass_all: AND of all gates ──
            row['pass_all'] = int(all([
                row['pass_hbond_dist_max'],
                row['pass_hbond_dist_min'],
                row['pass_plddt_ligand'],
                row['pass_coo_to_r116'],
                row['pass_ligand_geometry'],
                row['pass_hab1_clash'],
                row['pass_sidechain_hab1trp'],
                row['pass_latch_rmsd'],
                row['pass_ring_oh_flipped'],
            ]))

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Boltz prediction outputs")
    parser.add_argument("--binary-dir", nargs='+', default=None,
                        help="One or more directories with binary Boltz predictions")
    parser.add_argument("--ternary-dir", nargs='+', default=None,
                        help="One or more directories with ternary Boltz predictions")
    parser.add_argument("--ref-pdb", default=None,
                        help="Reference PDB with conserved water (e.g., 3QN1_H2O.pdb)")
    parser.add_argument("--ligand-smiles", default=None,
                        help="Ligand SMILES string for BUNs computation (requires bunsalyze). "
                             "Omit to skip BUNs scoring.")
    parser.add_argument("--ref-ligand-pdb", default=None,
                        help="Reference PDB with correct ligand geometry for stereochemistry "
                             "validation (e.g., a docked PDB with correct steroid ring chirality). "
                             "Omit to skip ligand geometry check.")
    parser.add_argument("--ref-ternary-pdb", default=None,
                        help="Ternary reference PDB with HAB1 (chain C) for ligand-HAB1 "
                             "clash detection. Omit to skip clash check.")
    parser.add_argument("--geometry-weight", type=float, default=2.0,
                        help="Weight for geometry score in total_score "
                             "(default: 2.0, range becomes 0 to 2+weight)")
    parser.add_argument("--unsat-penalty", type=float, default=0.0,
                        help="Penalty per unsatisfied polar contact subtracted from "
                             "total_score (default: 0.0). E.g. 0.5 means each "
                             "unsatisfied OH/COO costs 0.5 score points.")
    parser.add_argument("--gate-hbond-max", type=float, default=4.0,
                        help="Max H-bond distance gate (default: 4.0)")
    parser.add_argument("--gate-hbond-min", type=float, default=1.8,
                        help="Min H-bond distance gate (default: 1.8)")
    parser.add_argument("--gate-plddt-ligand", type=float, default=0.65,
                        help="Min ligand pLDDT gate (default: 0.65)")
    parser.add_argument("--gate-coo-to-r116", type=float, default=4.0,
                        help="Min COO-to-R116 distance gate (default: 4.0)")
    parser.add_argument("--gate-hab1-clash", type=float, default=2.0,
                        help="Min HAB1 clash distance gate (default: 2.0)")
    parser.add_argument("--gate-sidechain-hab1trp", type=float, default=2.0,
                        help="Min sidechain-to-HAB1 Trp distance gate for "
                             "residues 117/159 (default: 2.0)")
    parser.add_argument("--gate-latch-rmsd", type=float, default=1.0,
                        help="Max latch RMSD gate (default: 1.0)")
    parser.add_argument("--oh-contact-exclude-res", type=int, nargs='*', default=None,
                        help="Boltz residue numbers to exclude as OH polar anchors "
                             "(e.g. 115 for latch His). OH-mode designs anchored by "
                             "these residues will fail pass_all.")
    parser.add_argument("--ref-ligand-chain", default='B',
                        help="Chain ID of the ligand in --ref-ligand-pdb (default: B). "
                             "Use when the ref PDB has the ligand on a non-B chain "
                             "(e.g. X for docked PDBs from systematic_dock).")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")

    args = parser.parse_args()

    if not args.binary_dir and not args.ternary_dir:
        print("ERROR: Must provide at least one of --binary-dir or --ternary-dir", file=sys.stderr)
        sys.exit(1)

    if args.ligand_smiles:
        print(f"BUNs scoring enabled (bunsalyze) with SMILES: {args.ligand_smiles[:40]}...")

    results = analyze_predictions(
        binary_dirs=args.binary_dir,
        ternary_dirs=args.ternary_dir,
        ref_pdb=args.ref_pdb,
        ligand_smiles=args.ligand_smiles,
        ref_ligand_pdb=args.ref_ligand_pdb,
        ref_ligand_chain=args.ref_ligand_chain,
        ref_ternary_pdb=args.ref_ternary_pdb,
        geometry_weight=args.geometry_weight,
        unsat_penalty=args.unsat_penalty,
        gate_hbond_max=args.gate_hbond_max,
        gate_hbond_min=args.gate_hbond_min,
        gate_plddt_ligand=args.gate_plddt_ligand,
        gate_coo_to_r116=args.gate_coo_to_r116,
        gate_hab1_clash=args.gate_hab1_clash,
        gate_sidechain_hab1trp=args.gate_sidechain_hab1trp,
        gate_latch_rmsd=args.gate_latch_rmsd,
        oh_contact_exclude_res=args.oh_contact_exclude_res,
    )

    if not results:
        print("No predictions found to analyze", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    fieldnames = list(results[0].keys())
    # Ensure all keys from all rows are captured
    for row in results:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {out_path}")

    # Print summary stats
    for prefix in ('binary', 'ternary'):
        iptm_key = f'{prefix}_iptm'
        vals = [r[iptm_key] for r in results if r.get(iptm_key) is not None]
        if vals:
            print(f"  {prefix} ipTM: mean={np.mean(vals):.3f}, "
                  f"median={np.median(vals):.3f}, "
                  f"range=[{min(vals):.3f}, {max(vals):.3f}]")

    rmsd_vals = [r['ligand_rmsd_binary_vs_ternary'] for r in results
                 if r.get('ligand_rmsd_binary_vs_ternary') is not None]
    if rmsd_vals:
        print(f"  Binary-ternary ligand RMSD: mean={np.mean(rmsd_vals):.3f}, "
              f"median={np.median(rmsd_vals):.3f}")

    # Ligand geometry summary
    for prefix in ('binary', 'ternary'):
        dist_key = f'{prefix}_ligand_distorted'
        distorted = [r[dist_key] for r in results if r.get(dist_key) is not None]
        if distorted:
            n_dist = sum(distorted)
            n_total = len(distorted)
            print(f"  {prefix} ligand geometry: {n_dist}/{n_total} distorted "
                  f"({100*n_dist/n_total:.1f}%)")
            max_devs = [r[f'{prefix}_max_dev'] for r in results
                        if r.get(f'{prefix}_max_dev') is not None]
            if max_devs:
                print(f"    max_dev: mean={np.mean(max_devs):.3f}, "
                      f"median={np.median(max_devs):.3f}, "
                      f"range=[{min(max_devs):.3f}, {max(max_devs):.3f}]")

    # HAB1 clash summary
    clash_vals = [r.get('binary_hab1_clash_dist') for r in results
                  if r.get('binary_hab1_clash_dist') is not None]
    if clash_vals:
        n_hard = sum(1 for v in clash_vals if v < 2.0)
        n_soft = sum(1 for v in clash_vals if 2.0 <= v < 3.0)
        print(f"  HAB1 clash: {n_hard} hard (<2A), {n_soft} soft (2-3A), "
              f"{len(clash_vals)-n_hard-n_soft} clear (>3A) of {len(clash_vals)}")

    # Sidechain vs HAB1 Trp clash summary
    for sc_res in [117, 159]:
        col = f'binary_res{sc_res}_hab1trp_dist'
        sc_vals = [r.get(col) for r in results if r.get(col) is not None]
        if sc_vals:
            n_hard = sum(1 for v in sc_vals if v < 2.0)
            n_soft = sum(1 for v in sc_vals if 2.0 <= v < 3.0)
            print(f"  Res {sc_res} vs HAB1 Trp: {n_hard} hard (<2A), "
                  f"{n_soft} soft (2-3A), "
                  f"{len(sc_vals)-n_hard-n_soft} clear (>3A) of {len(sc_vals)}")

    # OH contact exclusion summary
    n_excluded = sum(1 for r in results if r.get('binary_oh_contact_excluded') == 1)
    if n_excluded:
        from collections import Counter
        exc_res = Counter()
        for r in results:
            if r.get('binary_oh_contact_excluded') == 1:
                res = r.get('binary_oh_contact_res')
                rname = r.get('binary_oh_contact_resname', '?')
                exc_res[f'{rname}{res}'] += 1
        exc_str = ', '.join(f'{k}={v}' for k, v in exc_res.most_common())
        print(f"  OH contact excluded (counted as unsat): {n_excluded} designs ({exc_str})")

    # Latch RMSD summary
    latch_vals = [r.get('binary_latch_rmsd') for r in results
                  if r.get('binary_latch_rmsd') is not None]
    if latch_vals:
        n_high = sum(1 for v in latch_vals if v >= 1.0)
        print(f"  Latch RMSD: mean={np.mean(latch_vals):.3f}, "
              f"median={np.median(latch_vals):.3f}, "
              f"{n_high}/{len(latch_vals)} above 1.0A")

    # Gate summary
    gate_cols = [
        'pass_hbond_dist_max', 'pass_hbond_dist_min', 'pass_plddt_ligand',
        'pass_coo_to_r116', 'pass_ligand_geometry', 'pass_hab1_clash',
        'pass_sidechain_hab1trp', 'pass_latch_rmsd',
        'pass_ring_oh_flipped', 'pass_all',
    ]
    has_gates = any(r.get('pass_all') is not None for r in results)
    if has_gates:
        n_total = len(results)
        print(f"\nScored: {n_total} designs")
        n_pass = sum(1 for r in results if r.get('pass_all') == 1)
        print(f"Pass all gates: {n_pass} ({100*n_pass/n_total:.1f}%)")
        for col in gate_cols:
            if col == 'pass_all':
                continue
            n_col = sum(1 for r in results if r.get(col) == 1)
            print(f"  {col}: {n_col} ({100*n_col/n_total:.1f}%)")

        # Binding modes (pass_all only)
        from collections import Counter
        passing = [r for r in results if r.get('pass_all') == 1]
        mode_counts = Counter(r.get('binary_binding_mode', 'unknown') for r in passing)
        mode_str = ', '.join(f"{m}={mode_counts[m]}" for m in ['normal', 'flipped', 'unknown'] if mode_counts[m])
        print(f"Binding modes (pass_all): {mode_str}")

        # Unique pocket sequences (pass_all only)
        pocket_seqs = set(r.get('binary_pocket_sequence', '') for r in passing
                          if r.get('binary_pocket_sequence'))
        print(f"Unique pocket sequences (pass_all): {len(pocket_seqs)}")

        # Polar satisfaction breakdown
        polar_data = [r for r in results if r.get('binary_n_polar_checked') is not None]
        if polar_data:
            n_checked_vals = [r['binary_n_polar_checked'] for r in polar_data]
            n_unsat_vals = [r['binary_n_polar_unsatisfied'] for r in polar_data]
            n_zero = sum(1 for v in n_unsat_vals if v == 0)
            print(f"\nPolar satisfaction ({len(polar_data)} designs, "
                  f"checking {n_checked_vals[0]} oxygens each excl. water-mediated):")
            print(f"  All satisfied (0 unsat): {n_zero} ({100*n_zero/len(polar_data):.1f}%)")
            for u in sorted(set(n_unsat_vals)):
                if u == 0:
                    continue
                ct = sum(1 for v in n_unsat_vals if v == u)
                print(f"  {u} unsatisfied: {ct} ({100*ct/len(polar_data):.1f}%)")
            # COO vs OH breakdown for unsatisfied
            n_coo_unsat = sum(1 for r in polar_data if r.get('binary_n_coo_unsatisfied', 0) > 0)
            n_oh_unsat = sum(1 for r in polar_data if r.get('binary_n_oh_unsatisfied', 0) > 0)
            print(f"  Designs with unsatisfied COO: {n_coo_unsat}, OH: {n_oh_unsat}")


if __name__ == "__main__":
    main()
