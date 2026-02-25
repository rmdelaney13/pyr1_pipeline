#!/usr/bin/env python3
"""
Generate a manifest for pairs that need H-bond geometry (re)computation.

Scans tier cache directories for pairs where summary.json exists but
min_dist_to_ligand_O is null. Outputs a TSV manifest compatible with
compute_af3_rmsd.py --recompute.

Usage:
    python generate_geometry_manifest.py \
        --cache-dir /scratch/alpine/ryde3462/ml_dataset/tier3_pnas_cutler \
        --cache-dir /scratch/alpine/ryde3462/ml_dataset/tier5_artificial \
        --output /scratch/alpine/ryde3462/ml_dataset/geometry_recompute_manifest.tsv
"""

import argparse
import json
import sys
from pathlib import Path


def find_af3_cif(pair_dir: Path, mode: str) -> Path:
    """Locate the AF3 model CIF for a pair+mode.

    CIF files live in the batch-level af3_staging directory, not inside
    the pair directory. Structure:
        {batch_dir}/af3_staging/{mode}_output/{pair_id}_{mode}_model.cif
    or nested:
        {batch_dir}/af3_staging/{mode}_output/{pair_id}_{mode}/{pair_id}_{mode}_model.cif
    """
    pair_id = pair_dir.name
    batch_dir = pair_dir.parent  # e.g., tier3_pnas_cutler_batch01/
    name = f"{pair_id}_{mode}"

    # Primary: batch-level af3_staging directory
    af3_staging = batch_dir / "af3_staging"
    af3_output_dir = af3_staging / f"{mode}_output"

    candidates = [
        # Flat layout
        af3_output_dir / f"{name}_model.cif",
        # Nested layout
        af3_output_dir / name / f"{name}_model.cif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: glob af3_staging for this pair's CIF
    if af3_output_dir.exists():
        cifs = list(af3_output_dir.glob(f"{name}*_model.cif"))
        if cifs:
            return cifs[0]
        cifs = list(af3_output_dir.glob(f"{name}/*_model.cif"))
        if cifs:
            return cifs[0]

    # Last resort: check inside the pair's af3_{mode} directory
    af3_dir = pair_dir / f"af3_{mode}"
    if af3_dir.exists():
        cifs = list(af3_dir.glob("*_model.cif"))
        if cifs:
            return cifs[0]
        cifs = list(af3_dir.glob("*/*_model.cif"))
        if cifs:
            return cifs[0]

    return None


def scan_tier(cache_dir: Path, modes=("binary", "ternary")):
    """Find pairs needing geometry computation in a tier directory."""
    entries = []
    pair_dirs = sorted(cache_dir.glob("*/pair_*"))

    for pair_dir in pair_dirs:
        for mode in modes:
            summary_path = pair_dir / f"af3_{mode}" / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Check if geometry is missing
            if data.get("min_dist_to_ligand_O") is not None:
                continue

            # Find the CIF file
            cif_path = find_af3_cif(pair_dir, mode)
            if cif_path is None:
                print(f"  WARNING: no CIF found for {pair_dir.name}/af3_{mode}",
                      file=sys.stderr)
                continue

            entries.append((str(pair_dir), str(cif_path), mode))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest for H-bond geometry recomputation")
    parser.add_argument("--cache-dir", action="append", required=True,
                        help="Tier cache directory (can specify multiple)")
    parser.add_argument("--output", required=True,
                        help="Output manifest TSV path")
    parser.add_argument("--modes", nargs="+", default=["binary", "ternary"],
                        help="AF3 modes to check (default: binary ternary)")
    args = parser.parse_args()

    all_entries = []
    for cd in args.cache_dir:
        cache_path = Path(cd)
        if not cache_path.exists():
            print(f"WARNING: {cache_path} not found, skipping", file=sys.stderr)
            continue
        print(f"Scanning {cache_path.name}...", file=sys.stderr)
        entries = scan_tier(cache_path, tuple(args.modes))
        print(f"  Found {len(entries)} entries needing geometry", file=sys.stderr)
        all_entries.extend(entries)

    # Write manifest
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for pair_dir, cif_path, mode in all_entries:
            f.write(f"{pair_dir}\t{cif_path}\t{mode}\n")

    print(f"\nWrote {len(all_entries)} entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
