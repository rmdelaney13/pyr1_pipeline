#!/usr/bin/env python3
"""
Tests for the ligand_conformers module.

Run from the repo root:
    python -m pytest tests/test_ligand_conformers.py -v

These are integration-style tests that exercise the full pipeline on small
molecules.  They should complete in under a minute on a single core.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ligand_conformers.config import ConformerConfig
from ligand_conformers.core import generate_conformer_set


# ---------------------------------------------------------------------------
# Test 1: SMILES input → kynurenine
# ---------------------------------------------------------------------------

class TestSmilesKynurenine:
    """End-to-end: kynurenine SMILES → K conformers in conformers_final.sdf."""

    SMILES = "NC(CC(=O)c1ccccc1N)C(O)=O"  # L-Kynurenine

    def test_generates_conformers(self, tmp_path):
        cfg = ConformerConfig(
            num_confs=50,        # small for speed
            seed=42,
            prune_rms_thresh=0.3,
            energy_pre_filter_n=30,
            cluster_rmsd_cutoff=1.0,
            k_final=5,
            nprocs=1,
            ligand_id="kynurenine",
        )
        input_spec = {"type": "smiles", "value": self.SMILES}
        result = generate_conformer_set(input_spec, tmp_path, cfg)

        # Basic success checks
        assert result.success, f"Generation failed: {result.errors}"
        assert result.num_generated > 0
        assert result.num_minimized > 0
        assert result.num_clusters > 0
        assert len(result.selected_ids) > 0
        assert len(result.selected_ids) <= cfg.k_final

        # Output files exist
        assert (tmp_path / "conformers_raw.sdf").exists()
        assert (tmp_path / "conformers_clustered.sdf").exists()
        assert (tmp_path / "conformers_final.sdf").exists()
        assert (tmp_path / "conformer_report.csv").exists()
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "conformer_generation.log").exists()

        # Per-conformer directory has the right number of files
        per_dir = tmp_path / "conformers_final"
        assert per_dir.is_dir()
        sdf_files = list(per_dir.glob("conf_*.sdf"))
        pdb_files = list(per_dir.glob("conf_*.pdb"))
        assert len(sdf_files) == len(result.selected_ids)
        assert len(pdb_files) == len(result.selected_ids)

        # Metadata is valid JSON with expected keys
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert "parameters" in meta
        assert "result_summary" in meta
        assert "versions" in meta
        assert meta["result_summary"]["success"] is True

        # CSV has header + data rows
        csv_lines = (tmp_path / "conformer_report.csv").read_text().splitlines()
        assert len(csv_lines) == len(result.selected_ids) + 1  # header + data

    def test_deterministic_with_same_seed(self, tmp_path):
        """Same seed should produce identical conformer sets."""
        cfg = ConformerConfig(
            num_confs=30,
            seed=123,
            prune_rms_thresh=0.5,
            energy_pre_filter_n=20,
            cluster_rmsd_cutoff=1.0,
            k_final=3,
            ligand_id="kyn_repro",
        )
        input_spec = {"type": "smiles", "value": self.SMILES}

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        r1 = generate_conformer_set(input_spec, dir1, cfg)
        r2 = generate_conformer_set(input_spec, dir2, cfg)

        assert r1.success and r2.success
        assert r1.num_generated == r2.num_generated
        assert r1.selected_ids == r2.selected_ids


# ---------------------------------------------------------------------------
# Test 2: PubChem lookup (graceful skip if pubchempy is not installed)
# ---------------------------------------------------------------------------

class TestPubChem:
    """PubChem CID → conformers (or graceful skip)."""

    def test_pubchem_lookup(self, tmp_path):
        try:
            import pubchempy  # noqa: F401
        except ImportError:
            pytest.skip("pubchempy not installed; skipping PubChem test.")

        cfg = ConformerConfig(
            num_confs=30,
            seed=42,
            prune_rms_thresh=0.5,
            energy_pre_filter_n=20,
            cluster_rmsd_cutoff=1.0,
            k_final=3,
            ligand_id="kynurenine_cid",
        )
        # CID 846 = L-Kynurenine
        input_spec = {"type": "pubchem", "value": "846"}
        result = generate_conformer_set(input_spec, tmp_path, cfg)

        assert result.success, f"PubChem generation failed: {result.errors}"
        assert len(result.selected_ids) > 0
        assert (tmp_path / "conformers_final.sdf").exists()

    def test_pubchem_missing_graceful(self, tmp_path, monkeypatch):
        """When pubchempy is absent, we get a clear ImportError in result."""
        # Temporarily hide pubchempy
        monkeypatch.setitem(sys.modules, "pubchempy", None)

        cfg = ConformerConfig(num_confs=10, k_final=2)
        input_spec = {"type": "pubchem", "value": "846"}
        result = generate_conformer_set(input_spec, tmp_path, cfg)

        assert not result.success
        assert any("pubchempy" in e.lower() or "import" in e.lower()
                    for e in result.errors)


# ---------------------------------------------------------------------------
# Test 3: CLI entrypoint smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    """Verify the CLI can be invoked programmatically."""

    def test_cli_smiles(self, tmp_path):
        from ligand_conformers.__main__ import main
        main([
            "--input", "c1ccccc1",
            "--input-type", "smiles",
            "--outdir", str(tmp_path),
            "--num-confs", "20",
            "--k-final", "3",
            "--seed", "7",
            "--ligand-id", "benzene",
        ])
        assert (tmp_path / "conformers_final.sdf").exists()
