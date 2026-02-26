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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    result = {'plddt_protein': None, 'plddt_ligand': None, 'plddt_hab1': None}

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
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════

def analyze_predictions(
    binary_dirs: List[str] = None,
    ternary_dirs: List[str] = None,
    ref_pdb: str = None,
) -> List[Dict]:
    """Analyze all Boltz predictions and return list of metric dicts."""

    results = []

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

        # ── Binary-to-ternary ligand RMSD ──
        if name in binary_preds and name in ternary_preds:
            rmsd = compute_ligand_rmsd(
                str(binary_preds[name]['structure']),
                str(ternary_preds[name]['structure']),
                protein_chain='A', ligand_chain='B',
            )
            row['ligand_rmsd_binary_vs_ternary'] = rmsd

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
    parser.add_argument("--out", required=True,
                        help="Output CSV path")

    args = parser.parse_args()

    if not args.binary_dir and not args.ternary_dir:
        print("ERROR: Must provide at least one of --binary-dir or --ternary-dir", file=sys.stderr)
        sys.exit(1)

    results = analyze_predictions(
        binary_dirs=args.binary_dir,
        ternary_dirs=args.ternary_dir,
        ref_pdb=args.ref_pdb,
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


if __name__ == "__main__":
    main()
