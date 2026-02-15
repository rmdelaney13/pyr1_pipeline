"""
openmm_refine.py — optional OpenMM implicit-solvent minimisation + MD anneal.

This module is imported lazily by core.py only when ``--openmm-refine`` is
requested.  If OpenMM or openmmforcefields are missing, the import will
fail and core.py will fall back to MMFF-only ranking.

Protocol per conformer:
    1. Parameterise with OpenFF (via openmmforcefields SMIRNOFFTemplateGenerator).
    2. Energy minimise (L-BFGS, implicit GBSA or vacuum).
    3. Short NVT MD anneal (default 50 ps, 300 K, Langevin).
    4. Final energy minimise.
    5. Record potential energy (kJ/mol).
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("ligand_conformers")


def refine_conformers(
    mol: Chem.Mol,
    conf_ids: List[int],
    cfg,  # ConformerConfig (avoid circular import)
    outdir: Path,
) -> Dict[int, float]:
    """Run OpenMM minimise + MD anneal on each conformer.

    Returns
    -------
    dict mapping confId → potential energy in kJ/mol after final minimisation.
    Empty dict if anything fails catastrophically.
    """
    # Lazy imports so the rest of the package works without OpenMM
    try:
        import openmm
        from openmm import app, unit, LangevinMiddleIntegrator
        from openmm.app import Simulation, Modeller
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    except ImportError as exc:
        logger.error("OpenMM import failed: %s", exc)
        logger.error(
            "Install required packages:\n"
            "  conda install -c conda-forge openmm openmmforcefields openff-toolkit"
        )
        return {}

    energies: Dict[int, float] = {}

    # Build the SMIRNOFF generator once (expensive)
    try:
        smirnoff = SMIRNOFFTemplateGenerator(
            molecules=_rdmol_to_openff(mol),
            forcefield=cfg.openmm_forcefield,
        )
    except Exception as exc:
        logger.error("SMIRNOFF parameterisation failed: %s", exc)
        return {}

    for cid in conf_ids:
        try:
            energy = _refine_single(
                mol, cid, smirnoff, cfg, openmm, app, unit,
                LangevinMiddleIntegrator,
            )
            energies[cid] = energy
            logger.info("  conf %d → OpenMM energy: %.2f kJ/mol", cid, energy)
        except Exception as exc:
            logger.warning("  conf %d OpenMM refinement failed: %s", cid, exc)

    return energies


def _refine_single(mol, cid, smirnoff, cfg, openmm, app, unit,
                   LangevinMiddleIntegrator):
    """Minimise + anneal one conformer, return potential energy (kJ/mol)."""

    # Write conformer to PDB in memory
    pdb_block = Chem.MolToPDBBlock(mol, confId=cid)
    pdb_io = io.StringIO(pdb_block)
    pdb = app.PDBFile(pdb_io)

    # Build force field with SMIRNOFF for the ligand
    if cfg.openmm_solvent == "implicit":
        ff = app.ForceField("implicit/gbn2.xml")
    else:
        ff = app.ForceField()

    ff.registerTemplateGenerator(smirnoff.generator)

    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    step_size = cfg.openmm_md_step_size_fs * unit.femtoseconds
    integrator = LangevinMiddleIntegrator(
        cfg.openmm_md_temperature_K * unit.kelvin,
        1.0 / unit.picoseconds,
        step_size,
    )

    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Phase 1: minimise
    simulation.minimizeEnergy(maxIterations=cfg.openmm_minimize_maxiter)

    # Phase 2: MD anneal
    simulation.context.setVelocitiesToTemperature(
        cfg.openmm_md_temperature_K * unit.kelvin
    )
    simulation.step(cfg.openmm_md_steps)

    # Phase 3: final minimise
    simulation.minimizeEnergy(maxIterations=cfg.openmm_minimize_maxiter)

    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return pe


def _rdmol_to_openff(mol):
    """Convert an RDKit Mol to an OpenFF Molecule for parameterisation."""
    from openff.toolkit import Molecule as OFFMolecule

    # Remove Hs for the topology (OpenFF re-adds them)
    mol_noH = Chem.RemoveHs(mol)
    return [OFFMolecule.from_rdkit(mol_noH)]
