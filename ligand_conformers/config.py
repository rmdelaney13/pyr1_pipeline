"""
ConformerConfig — all tuneable parameters for the conformer-generation funnel.

Values can be set programmatically, via CLI flags, or loaded from a unified
pipeline config file (INI format, ``[conformer_generation]`` section).
"""

from __future__ import annotations

import json
from configparser import ConfigParser
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_SECTION = "conformer_generation"


def _strip_inline_comment(value: str) -> str:
    """Strip ``  # ...`` or ``\\t# ...`` inline comments (pipeline convention)."""
    if value is None:
        return value
    idx = value.find("  #")
    if idx == -1:
        idx = value.find("\t#")
    if idx != -1:
        value = value[:idx]
    return value.strip()


def _cfg_get(config: ConfigParser, key: str, fallback=None) -> Optional[str]:
    if _SECTION in config and key in config[_SECTION]:
        return _strip_inline_comment(config[_SECTION][key])
    if key in config["DEFAULT"]:
        return _strip_inline_comment(config["DEFAULT"][key])
    return fallback


@dataclass
class ConformerConfig:
    """All parameters for the conformer-generation stage."""

    # ── embedding ───────────────────────────────────────────────────────
    num_confs: int = 500
    seed: int = 42
    prune_rms_thresh: float = 0.5          # Å, ETKDGv3 pruneRmsThresh
    use_random_coords: bool = False

    # ── force-field minimisation ────────────────────────────────────────
    ff_variant: str = "MMFF94s"            # MMFF94s | UFF (auto-fallback)
    ff_max_iters: int = 500
    energy_pre_filter_n: int = 100         # keep top-N by energy before clustering

    # ── clustering ──────────────────────────────────────────────────────
    cluster_rmsd_cutoff: float = 1.25      # Å, Butina centroid cutoff
    align_before_cluster: bool = True      # align heavy atoms before RMSD

    # ── selection ───────────────────────────────────────────────────────
    k_final: int = 10                      # number of conformers to output
    selection_policy: str = "diverse"       # diverse | energy

    # ── OpenMM refinement (optional) ────────────────────────────────────
    openmm_refine: bool = False
    openmm_forcefield: str = "openff-2.1.0"
    openmm_solvent: str = "implicit"       # implicit | vacuum
    openmm_minimize_maxiter: int = 500
    openmm_md_steps: int = 25000           # 50 ps at 2 fs step
    openmm_md_temperature_K: float = 300.0
    openmm_md_step_size_fs: float = 2.0

    # ── performance ─────────────────────────────────────────────────────
    nprocs: int = 1

    # ── I/O ─────────────────────────────────────────────────────────────
    ligand_id: str = "ligand"

    # ── version info (filled at run-time) ───────────────────────────────
    _versions: dict = field(default_factory=dict, repr=False)

    # ── helpers ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_versions"] = self._versions
        return d

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_config_file(cls, path: str | Path) -> "ConformerConfig":
        """Load from a unified pipeline INI config that has a
        ``[conformer_generation]`` section."""
        cp = ConfigParser()
        with open(path, "r", encoding="utf-8-sig") as fh:
            cp.read_file(fh)
        if _SECTION not in cp:
            return cls()

        def _int(key, fb):
            v = _cfg_get(cp, key, None)
            return int(v) if v is not None else fb

        def _float(key, fb):
            v = _cfg_get(cp, key, None)
            return float(v) if v is not None else fb

        def _bool(key, fb):
            v = _cfg_get(cp, key, None)
            if v is None:
                return fb
            return v.lower() in {"1", "true", "yes", "on"}

        def _str(key, fb):
            v = _cfg_get(cp, key, None)
            return v if v is not None else fb

        return cls(
            num_confs=_int("NumConfs", cls.num_confs),
            seed=_int("Seed", cls.seed),
            prune_rms_thresh=_float("PruneRmsThresh", cls.prune_rms_thresh),
            use_random_coords=_bool("UseRandomCoords", cls.use_random_coords),
            ff_variant=_str("FFVariant", cls.ff_variant),
            ff_max_iters=_int("FFMaxIters", cls.ff_max_iters),
            energy_pre_filter_n=_int("EnergyPreFilterN", cls.energy_pre_filter_n),
            cluster_rmsd_cutoff=_float("ClusterRMSDCutoff", cls.cluster_rmsd_cutoff),
            align_before_cluster=_bool("AlignBeforeCluster", cls.align_before_cluster),
            k_final=_int("KFinal", cls.k_final),
            selection_policy=_str("SelectionPolicy", cls.selection_policy),
            openmm_refine=_bool("OpenMMRefine", cls.openmm_refine),
            openmm_forcefield=_str("OpenMMForceField", cls.openmm_forcefield),
            openmm_solvent=_str("OpenMMSolvent", cls.openmm_solvent),
            openmm_minimize_maxiter=_int("OpenMMMinimizeMaxIter", cls.openmm_minimize_maxiter),
            openmm_md_steps=_int("OpenMMMDSteps", cls.openmm_md_steps),
            openmm_md_temperature_K=_float("OpenMMMDTemperatureK", cls.openmm_md_temperature_K),
            openmm_md_step_size_fs=_float("OpenMMMDStepSizeFs", cls.openmm_md_step_size_fs),
            nprocs=_int("NProcs", cls.nprocs),
            ligand_id=_str("LigandID", cls.ligand_id),
        )
