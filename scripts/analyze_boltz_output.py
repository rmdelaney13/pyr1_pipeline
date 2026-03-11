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

# Ligand geometry validation (steroid ring distortion / stereochemistry)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ligand_geometry import LigandGeometryChecker
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

def analyze_predictions(
    binary_dirs: List[str] = None,
    ternary_dirs: List[str] = None,
    ref_pdb: str = None,
    ligand_smiles: str = None,
    ref_ligand_pdb: str = None,
) -> List[Dict]:
    """Analyze all Boltz predictions and return list of metric dicts."""

    results = []

    # Initialize ligand geometry checker if reference provided
    geom_checker = None
    if ref_ligand_pdb:
        if _HAS_LIGAND_GEOMETRY:
            try:
                geom_checker = LigandGeometryChecker(ref_ligand_pdb)
            except (ValueError, Exception) as e:
                logger.warning(f"Could not initialize ligand geometry checker: {e}")
        else:
            logger.warning("ligand_geometry module not available — skipping geometry check")

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
                    0.7 * dist_score + 0.3 * combined_ang, 4)
            else:
                row[f'{prefix}_geometry_score'] = None

            # Total score: boltz_score + geometry_score, range 0-3
            boltz_s = row.get(f'{prefix}_boltz_score')
            geom_s = row.get(f'{prefix}_geometry_score')
            if boltz_s is not None and geom_s is not None:
                row[f'{prefix}_total_score'] = round(boltz_s + geom_s, 4)
            else:
                row[f'{prefix}_total_score'] = None

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


if __name__ == "__main__":
    main()
