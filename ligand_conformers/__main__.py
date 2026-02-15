#!/usr/bin/env python3
"""
CLI entrypoint for the ligand conformer-generation stage.

Usage examples:

    # From SMILES
    python -m ligand_conformers --input "OC(=O)C(N)Cc1ccc(O)cc1" \\
           --input-type smiles --outdir /scratch/conformers/tyr

    # From PubChem name
    python -m ligand_conformers --input kynurenine \\
           --input-type pubchem --outdir /scratch/conformers/kyn

    # From PubChem CID
    python -m ligand_conformers --input 846 \\
           --input-type pubchem --outdir /scratch/conformers/kyn

    # From existing SDF file
    python -m ligand_conformers --input ligand.sdf \\
           --input-type sdf --outdir /scratch/conformers/lig

    # With OpenMM refinement
    python -m ligand_conformers --input "C1=CC=CC=C1" \\
           --input-type smiles --outdir /tmp/out --openmm-refine

    # Using a pipeline config file for defaults
    python -m ligand_conformers --config config.txt \\
           --input "OC(=O)C(N)CC1=CC(O)=CC=C1" --input-type smiles \\
           --outdir /scratch/conformers/kyn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ligand_conformers",
        description="Generate a diverse set of low-strain ligand conformers.",
    )

    # ── required I/O ────────────────────────────────────────────────────
    p.add_argument("--input", required=True,
                   help="SMILES string, PubChem CID/name, or path to a "
                        "structure file (SDF/MOL2/PDB).")
    p.add_argument("--input-type", required=True,
                   choices=["smiles", "pubchem", "sdf", "mol2", "pdb"],
                   help="How to interpret --input.")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory (created if absent).")

    # ── optional config file ────────────────────────────────────────────
    p.add_argument("--config", default=None, type=Path,
                   help="Pipeline INI config with [conformer_generation] section. "
                        "CLI flags override config values.")

    # ── embedding ───────────────────────────────────────────────────────
    p.add_argument("--num-confs", type=int, default=None,
                   help="Number of ETKDGv3 conformers to embed (default 500).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default 42).")
    p.add_argument("--prune-rms", type=float, default=None,
                   help="RMSD prune threshold during embedding, Å (default 0.5).")

    # ── force field ─────────────────────────────────────────────────────
    p.add_argument("--ff-variant", default=None,
                   choices=["MMFF94s", "UFF"],
                   help="Force field for minimisation (default MMFF94s).")
    p.add_argument("--ff-max-iters", type=int, default=None,
                   help="Max FF minimisation iterations (default 500).")
    p.add_argument("--energy-pre-filter-n", type=int, default=None,
                   help="Keep top-N by energy before clustering (default 100).")

    # ── clustering ──────────────────────────────────────────────────────
    p.add_argument("--cluster-rmsd", type=float, default=None,
                   help="Butina RMSD cutoff, Å (default 1.25).")
    p.add_argument("--no-align", action="store_true",
                   help="Skip alignment before RMSD clustering.")

    # ── selection ───────────────────────────────────────────────────────
    p.add_argument("-k", "--k-final", type=int, default=None,
                   help="Number of final conformers (default 10).")
    p.add_argument("--selection-policy", default=None,
                   choices=["diverse", "energy"],
                   help="Selection policy (default diverse).")

    # ── OpenMM ──────────────────────────────────────────────────────────
    p.add_argument("--openmm-refine", action="store_true", default=None,
                   help="Enable OpenMM implicit-solvent refinement + MD anneal.")
    p.add_argument("--openmm-steps", type=int, default=None,
                   help="MD anneal steps (default 25000 = 50 ps at 2 fs).")

    # ── performance ─────────────────────────────────────────────────────
    p.add_argument("--nprocs", type=int, default=None,
                   help="Parallel workers for minimisation (default 1).")

    # ── naming ──────────────────────────────────────────────────────────
    p.add_argument("--ligand-id", default=None,
                   help="Ligand identifier for output naming (default 'ligand').")

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Late import so --help is fast even without rdkit
    from ligand_conformers.config import ConformerConfig
    from ligand_conformers.core import generate_conformer_set

    # Build config: start from config file if given, then overlay CLI flags
    if args.config is not None:
        cfg = ConformerConfig.from_config_file(args.config)
    else:
        cfg = ConformerConfig()

    # CLI overrides
    if args.num_confs is not None:
        cfg.num_confs = args.num_confs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.prune_rms is not None:
        cfg.prune_rms_thresh = args.prune_rms
    if args.ff_variant is not None:
        cfg.ff_variant = args.ff_variant
    if args.ff_max_iters is not None:
        cfg.ff_max_iters = args.ff_max_iters
    if args.energy_pre_filter_n is not None:
        cfg.energy_pre_filter_n = args.energy_pre_filter_n
    if args.cluster_rmsd is not None:
        cfg.cluster_rmsd_cutoff = args.cluster_rmsd
    if args.no_align:
        cfg.align_before_cluster = False
    if args.k_final is not None:
        cfg.k_final = args.k_final
    if args.selection_policy is not None:
        cfg.selection_policy = args.selection_policy
    if args.openmm_refine is not None:
        cfg.openmm_refine = args.openmm_refine
    if args.openmm_steps is not None:
        cfg.openmm_md_steps = args.openmm_steps
    if args.nprocs is not None:
        cfg.nprocs = args.nprocs
    if args.ligand_id is not None:
        cfg.ligand_id = args.ligand_id

    input_spec = {"type": args.input_type, "value": args.input}

    result = generate_conformer_set(input_spec, args.outdir, cfg)

    if result.success:
        print(f"\n[OK] {len(result.selected_ids)} conformers written to {args.outdir}")
        print(f"     Final SDF:   {args.outdir}/conformers_final.sdf")
        print(f"     Per-conf:    {args.outdir}/conformers_final/")
        print(f"     Report:      {args.outdir}/conformer_report.csv")
        print(f"     Metadata:    {args.outdir}/metadata.json")
    else:
        print(f"\n[FAIL] Conformer generation failed.", file=sys.stderr)
        for err in result.errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
