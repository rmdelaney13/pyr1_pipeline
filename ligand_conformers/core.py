"""
core.py — conformer-generation funnel.

Pipeline:
    1. Resolve input → RDKit Mol (with 3-D coords via ETKDGv3 embedding)
    2. MMFF94s minimise every conformer (UFF fallback)
    3. Energy pre-filter (keep top N)
    4. Butina RMSD clustering on heavy atoms
    5. Select lowest-energy rep per cluster
    6. (Optional) OpenMM implicit-solvent minimise + MD anneal → re-rank
    7. Final diverse K-selection
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit import __version__ as _rdkit_version
from rdkit.Chem import AllChem, Descriptors, rdMolAlign, rdMolDescriptors
from rdkit.ML.Cluster import Butina

from ligand_conformers.config import ConformerConfig
from ligand_conformers.io_utils import (
    export_pdb_with_atom_names,
    write_sdf,
    resolve_input,
)

logger = logging.getLogger("ligand_conformers")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """Returned by :func:`generate_conformer_set`."""
    success: bool
    ligand_id: str
    input_type: str
    input_value: str
    seed: int
    num_generated: int = 0
    num_minimized: int = 0
    num_clusters: int = 0
    selected_ids: List[int] = field(default_factory=list)
    cluster_map: Dict[int, int] = field(default_factory=dict)   # confId → clusterId
    energies_mmff: Dict[int, float] = field(default_factory=dict)
    energies_openmm: Dict[int, float] = field(default_factory=dict)
    outdir: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _heavy_atom_indices(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _pairwise_rmsd_matrix(mol: Chem.Mol, conf_ids: List[int],
                          align: bool = True) -> list:
    """Compute NxN heavy-atom RMSD matrix.  Returns condensed 1-D list
    (lower triangle, row-major) suitable for Butina.

    Uses a *copy* of the molecule for alignment so the original
    coordinates are not modified.
    """
    ha = _heavy_atom_indices(mol)
    atom_map = list(zip(ha, ha))
    n = len(conf_ids)
    # Work on a copy so AlignMol doesn't corrupt the original coords
    mol_copy = Chem.RWMol(mol)
    dists = []
    for i in range(1, n):
        for j in range(i):
            if align:
                rmsd = rdMolAlign.AlignMol(mol_copy, mol_copy,
                                           prbCid=conf_ids[i],
                                           refCid=conf_ids[j],
                                           atomMap=atom_map)
            else:
                rmsd = AllChem.GetConformerRMS(mol_copy, conf_ids[i],
                                              conf_ids[j],
                                              prealigned=True)
            dists.append(rmsd)
    return dists


def _butina_cluster(dist_list: list, n: int, cutoff: float) -> List[List[int]]:
    """Butina clustering returning list-of-lists of indices."""
    cs = Butina.ClusterData(dist_list, n, cutoff, isDistData=True)
    return [list(c) for c in cs]


def _minimise_single(mol_block: str, conf_id: int, variant: str,
                     max_iters: int) -> Tuple[int, int, float]:
    """Worker for parallel MMFF/UFF minimisation.

    Returns ``(conf_id, converged_flag, energy)``.
    Works on a copy to be picklable across processes.
    """
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is None:
        return (conf_id, -1, float("inf"))

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        if ff is not None:
            status = ff.Minimize(maxIts=max_iters)
            energy = ff.CalcEnergy()
            return (conf_id, status, energy)

    # UFF fallback
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
    if ff is None:
        return (conf_id, -1, float("inf"))
    status = ff.Minimize(maxIts=max_iters)
    energy = ff.CalcEnergy()
    return (conf_id, status, energy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conformer_set(
    input_spec: dict,
    outdir: Path,
    cfg: ConformerConfig | None = None,
) -> ConformerResult:
    """End-to-end conformer funnel.

    Parameters
    ----------
    input_spec : dict
        ``{"type": "smiles"|"pubchem"|"sdf"|"mol2"|"pdb", "value": <str>}``
        *value* is a SMILES string, PubChem CID/name, or file path.
    outdir : Path
        Output directory (created if absent).
    cfg : ConformerConfig, optional
        Tuneable parameters; uses defaults when *None*.

    Returns
    -------
    ConformerResult
    """
    if cfg is None:
        cfg = ConformerConfig()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── logging setup ──────────────────────────────────────────────────
    _setup_logging(outdir)

    result = ConformerResult(
        success=False,
        ligand_id=cfg.ligand_id,
        input_type=input_spec["type"],
        input_value=str(input_spec["value"]),
        seed=cfg.seed,
    )

    t0_total = time.time()

    # ── 1. resolve input to RDKit Mol ──────────────────────────────────
    logger.info("=== Conformer generation: %s ===", cfg.ligand_id)
    logger.info("Input: type=%s  value=%s", input_spec["type"], input_spec["value"])

    try:
        mol = resolve_input(input_spec)
    except Exception as exc:
        msg = f"Input resolution failed: {exc}"
        logger.error(msg)
        result.errors.append(msg)
        _write_outputs(result, outdir, cfg, mol=None)
        return result

    logger.info("Mol parsed: %d atoms, %d heavy atoms, formula %s",
                mol.GetNumAtoms(),
                Descriptors.HeavyAtomCount(mol),
                rdMolDescriptors.CalcMolFormula(mol))

    # Tautomer / protonation hook (placeholder)
    _warn_tautomer_placeholder(mol)

    # ── 2. embed conformers ────────────────────────────────────────────
    t0 = time.time()
    params = AllChem.ETKDGv3()
    params.randomSeed = cfg.seed
    params.pruneRmsThresh = cfg.prune_rms_thresh
    params.useRandomCoords = cfg.use_random_coords
    params.numThreads = cfg.nprocs

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=cfg.num_confs,
                                               params=params))
    result.timings["embed_s"] = time.time() - t0

    if not conf_ids:
        msg = ("No conformers generated.  Try increasing num_confs or "
               "reducing prune_rms_thresh.")
        logger.error(msg)
        result.errors.append(msg)
        _write_outputs(result, outdir, cfg, mol=None)
        return result

    result.num_generated = len(conf_ids)
    logger.info("Embedded %d conformers (%.1f s)",
                len(conf_ids), result.timings["embed_s"])

    # ── 3. MMFF94s minimise ────────────────────────────────────────────
    t0 = time.time()
    energies: Dict[int, float] = {}

    if cfg.nprocs > 1:
        energies = _minimise_parallel(mol, conf_ids, cfg)
    else:
        energies = _minimise_serial(mol, conf_ids, cfg)

    result.timings["minimise_s"] = time.time() - t0
    result.num_minimized = len(energies)
    logger.info("Minimised %d conformers (%.1f s)",
                len(energies), result.timings["minimise_s"])

    # write raw SDF (all minimised, sorted by energy)
    sorted_ids = sorted(energies, key=lambda c: energies[c])
    write_sdf(mol, sorted_ids, energies, outdir / "conformers_raw.sdf",
              name_prefix=cfg.ligand_id)

    # ── 4. energy pre-filter ───────────────────────────────────────────
    keep_n = min(cfg.energy_pre_filter_n, len(sorted_ids))
    pre_ids = sorted_ids[:keep_n]
    logger.info("Energy pre-filter: kept %d / %d", len(pre_ids), len(sorted_ids))

    # ── 5. Butina RMSD cluster ─────────────────────────────────────────
    t0 = time.time()
    dist_list = _pairwise_rmsd_matrix(mol, pre_ids, align=cfg.align_before_cluster)
    clusters = _butina_cluster(dist_list, len(pre_ids), cfg.cluster_rmsd_cutoff)
    result.timings["cluster_s"] = time.time() - t0
    result.num_clusters = len(clusters)
    logger.info("Clustered into %d clusters (cutoff %.2f Å, %.1f s)",
                len(clusters), cfg.cluster_rmsd_cutoff,
                result.timings["cluster_s"])

    # map cluster-local indices back to conf_ids
    cluster_reps: List[Tuple[int, int, float]] = []  # (confId, clusterId, energy)
    cluster_id_map: Dict[int, int] = {}
    for ci, members in enumerate(clusters):
        best_idx = min(members, key=lambda m: energies[pre_ids[m]])
        best_cid = pre_ids[best_idx]
        cluster_reps.append((best_cid, ci, energies[best_cid]))
        for m in members:
            cluster_id_map[pre_ids[m]] = ci

    result.cluster_map = cluster_id_map

    # write clustered SDF
    rep_ids = [cid for cid, _, _ in cluster_reps]
    write_sdf(mol, rep_ids, energies, outdir / "conformers_clustered.sdf",
              name_prefix=cfg.ligand_id)

    # ── 6. (optional) OpenMM anneal ────────────────────────────────────
    openmm_energies: Dict[int, float] = {}
    if cfg.openmm_refine:
        t0 = time.time()
        openmm_energies = _openmm_refine(mol, rep_ids, cfg, outdir)
        result.timings["openmm_s"] = time.time() - t0
        result.energies_openmm = openmm_energies
        if openmm_energies:
            # re-sort reps by OpenMM energy
            cluster_reps = sorted(cluster_reps,
                                  key=lambda x: openmm_energies.get(x[0], float("inf")))
            logger.info("OpenMM refinement done (%.1f s)",
                        result.timings["openmm_s"])
        else:
            logger.warning("OpenMM refinement returned no results; "
                           "falling back to MMFF ranking.")

    # ── 7. final K-selection ───────────────────────────────────────────
    selected = _select_final(cluster_reps, clusters, pre_ids, energies,
                             openmm_energies, cfg)
    result.selected_ids = selected
    result.energies_mmff = {cid: energies[cid] for cid in selected}

    logger.info("Selected %d final conformers", len(selected))

    # ── 8. write outputs ───────────────────────────────────────────────
    _write_final_outputs(mol, selected, energies, openmm_energies,
                         cluster_id_map, outdir, cfg)
    _write_report_csv(result, energies, openmm_energies, cluster_id_map,
                      mol, selected, outdir, cfg)
    _write_metadata(result, cfg, outdir)

    result.outdir = outdir
    result.success = True
    result.timings["total_s"] = time.time() - t0_total
    logger.info("=== Done (%.1f s total) ===", result.timings["total_s"])
    return result


# ---------------------------------------------------------------------------
# Minimisation helpers
# ---------------------------------------------------------------------------

def _minimise_serial(mol, conf_ids, cfg):
    energies = {}
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=cfg.ff_variant)
    for cid in conf_ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=cfg.ff_max_iters)
                energies[cid] = ff.CalcEnergy()
                continue
        # UFF fallback
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is not None:
            ff.Minimize(maxIts=cfg.ff_max_iters)
            energies[cid] = ff.CalcEnergy()
        else:
            logger.warning("No force field available for conf %d; skipping", cid)
    return energies


def _minimise_parallel(mol, conf_ids, cfg):
    """Parallel MMFF minimisation across processes.

    Each worker receives a MolBlock for a *single* conformer so that
    RDKit objects don't need to be pickled across the process boundary.
    """
    energies = {}
    futures = {}
    with ProcessPoolExecutor(max_workers=cfg.nprocs) as pool:
        for cid in conf_ids:
            block = Chem.MolToMolBlock(mol, confId=cid)
            fut = pool.submit(_minimise_single, block, cid,
                              cfg.ff_variant, cfg.ff_max_iters)
            futures[fut] = cid
        for fut in as_completed(futures):
            cid_out, status, energy = fut.result()
            if energy < float("inf"):
                # Write minimised coords back into mol
                # (parallel path: we cannot write back easily, so we
                #  accept the pre-minimisation coords for geometry.
                #  Energies are still from post-minimisation.)
                energies[cid_out] = energy
            else:
                logger.warning("Minimisation failed for conf %d", cid_out)

    # For the parallel path, do a quick in-process serial re-minimise
    # on the top candidates only (to get correct coords).
    # This is fast because we only redo the best ones.
    sorted_cids = sorted(energies, key=lambda c: energies[c])
    keep = sorted_cids[:cfg.energy_pre_filter_n]
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=cfg.ff_variant)
    final_energies = {}
    for cid in keep:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=cfg.ff_max_iters)
                final_energies[cid] = ff.CalcEnergy()
                continue
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is not None:
            ff.Minimize(maxIts=cfg.ff_max_iters)
            final_energies[cid] = ff.CalcEnergy()

    return final_energies if final_energies else energies


# ---------------------------------------------------------------------------
# OpenMM refinement
# ---------------------------------------------------------------------------

def _openmm_refine(mol, conf_ids, cfg, outdir):
    """Optional OpenMM implicit-solvent minimise + short MD anneal.

    Returns dict of confId → potential energy (kJ/mol), or empty dict
    if OpenMM is unavailable.
    """
    try:
        from ligand_conformers.openmm_refine import refine_conformers
    except ImportError as exc:
        logger.warning("OpenMM refinement requested but import failed: %s", exc)
        logger.warning("Install OpenMM + openmmforcefields to enable refinement.\n"
                       "  conda install -c conda-forge openmm openmmforcefields")
        return {}

    return refine_conformers(mol, conf_ids, cfg, outdir)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_final(cluster_reps, clusters, pre_ids, energies,
                  openmm_energies, cfg):
    """Diversity-first selection: best-per-cluster, then backfill."""
    k = cfg.k_final

    # Sort reps by energy (openmm if available, else mmff)
    def _sort_key(rep_tuple):
        cid, ci, mmff_e = rep_tuple
        return openmm_energies.get(cid, mmff_e)

    reps_sorted = sorted(cluster_reps, key=_sort_key)

    selected = []
    used_clusters = set()

    if cfg.selection_policy == "diverse":
        # Phase 1: one per cluster (best energy)
        for cid, ci, _ in reps_sorted:
            if len(selected) >= k:
                break
            if ci not in used_clusters:
                selected.append(cid)
                used_clusters.add(ci)

        # Phase 2: backfill from largest clusters
        if len(selected) < k:
            # Build a pool of next-best conformers from each cluster
            backfill = []
            for ci, members in enumerate(clusters):
                member_cids = [pre_ids[m] for m in members]
                member_cids_sorted = sorted(member_cids,
                                            key=lambda c: energies.get(c, float("inf")))
                # Skip the rep we already took
                for mc in member_cids_sorted:
                    if mc not in selected:
                        backfill.append((mc, ci, energies.get(mc, float("inf"))))

            # Sort backfill by energy, preferring larger clusters (tie-break)
            cluster_sizes = {ci: len(members) for ci, members in enumerate(clusters)}
            backfill.sort(key=lambda x: (x[2], -cluster_sizes.get(x[1], 0)))

            for cid, ci, _ in backfill:
                if len(selected) >= k:
                    break
                if cid not in selected:
                    selected.append(cid)

    else:
        # Pure energy ranking
        all_cids = sorted(energies, key=lambda c: openmm_energies.get(c, energies[c]))
        selected = all_cids[:k]

    return selected


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_final_outputs(mol, selected, energies, openmm_energies,
                         cluster_map, outdir, cfg):
    """Write conformers_final.sdf and per-conformer files."""
    write_sdf(mol, selected, energies, outdir / "conformers_final.sdf",
              name_prefix=cfg.ligand_id)

    per_dir = outdir / "conformers_final"
    per_dir.mkdir(exist_ok=True)

    for rank, cid in enumerate(selected):
        tag = f"conf_{rank:03d}"
        sdf_path = per_dir / f"{tag}.sdf"
        pdb_path = per_dir / f"{tag}.pdb"

        write_sdf(mol, [cid], energies, sdf_path, name_prefix=cfg.ligand_id)
        export_pdb_with_atom_names(mol, cid, pdb_path)


def _write_report_csv(result, energies, openmm_energies, cluster_map,
                      mol, selected, outdir, cfg):
    """Write conformer_report.csv."""
    # Compute RMSD to the best (lowest energy) conformer
    best_cid = selected[0] if selected else None
    ha = _heavy_atom_indices(mol)

    rows = []
    for rank, cid in enumerate(selected):
        if best_cid is not None and cid != best_cid:
            rmsd_to_best = rdMolAlign.AlignMol(
                mol, mol, prbCid=cid, refCid=best_cid,
                atomMap=list(zip(ha, ha)))
        else:
            rmsd_to_best = 0.0

        rows.append({
            "ligand_id": cfg.ligand_id,
            "input_type": result.input_type,
            "input_value": result.input_value,
            "seed": cfg.seed,
            "num_generated": result.num_generated,
            "num_minimized": result.num_minimized,
            "num_clusters": result.num_clusters,
            "rmsd_cutoff_A": cfg.cluster_rmsd_cutoff,
            "prune_rms_A": cfg.prune_rms_thresh,
            "selected_rank": rank,
            "selected_conf_id": cid,
            "selected_cluster_id": cluster_map.get(cid, -1),
            "mmff_energy": f"{energies.get(cid, float('nan')):.4f}",
            "openmm_energy": (f"{openmm_energies[cid]:.4f}"
                              if cid in openmm_energies else ""),
            "rmsd_to_best": f"{rmsd_to_best:.4f}",
            "notes": "; ".join(result.errors) if result.errors else "",
        })

    csv_path = outdir / "conformer_report.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Report: %s", csv_path)


def _write_metadata(result, cfg, outdir):
    """Write metadata.json with full provenance."""
    cfg._versions = _collect_versions()
    meta = {
        "parameters": cfg.to_dict(),
        "result_summary": {
            "success": result.success,
            "num_generated": result.num_generated,
            "num_minimized": result.num_minimized,
            "num_clusters": result.num_clusters,
            "num_selected": len(result.selected_ids),
            "selected_conf_ids": result.selected_ids,
            "timings": result.timings,
            "errors": result.errors,
        },
        "input": {
            "type": result.input_type,
            "value": result.input_value,
        },
        "versions": cfg._versions,
    }
    meta_path = outdir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Metadata: %s", meta_path)


def _collect_versions():
    versions = {
        "rdkit": _rdkit_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import openmm
        versions["openmm"] = openmm.__version__
    except ImportError:
        pass
    try:
        import openmmforcefields
        versions["openmmforcefields"] = openmmforcefields.__version__
    except ImportError:
        pass
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    return versions


# ---------------------------------------------------------------------------
# Logging / misc
# ---------------------------------------------------------------------------

def _setup_logging(outdir: Path):
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Also log to file
    fh = logging.FileHandler(outdir / "conformer_generation.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(fh)


def _warn_tautomer_placeholder(mol):
    """Placeholder for future tautomer/protonation enumeration."""
    logger.info("NOTE: Tautomer/protonation enumeration is not yet implemented. "
                "The input protonation state is used as-is.  To enumerate "
                "tautomers, preprocess with Dimorphite-DL or RDKit "
                "TautomerEnumerator before calling this module.")


def _write_outputs(result, outdir, cfg, mol):
    """Minimal output write for early-failure paths."""
    _write_metadata(result, cfg, outdir)
