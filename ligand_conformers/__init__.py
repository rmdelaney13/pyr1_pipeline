"""
ligand_conformers — expedited conformer funnel for the pyr1 docking/design pipeline.

Workflow:
    RDKit ETKDGv3 embed → MMFF94s minimise → energy pre-filter →
    Butina RMSD cluster → best-per-cluster → (optional) OpenMM anneal →
    final diverse selection (K conformers).

Public API:
    generate_conformer_set(input_spec, outdir, cfg) -> ConformerResult
"""

from ligand_conformers.config import ConformerConfig
from ligand_conformers.core import generate_conformer_set, ConformerResult

__all__ = ["ConformerConfig", "generate_conformer_set", "ConformerResult"]
