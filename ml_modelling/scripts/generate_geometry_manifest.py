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
    """Locate the AF3 model CIF for a pair+mode."""
    af3_dir = pair_dir / f"af3_{mode}"
    pair_id = pair_dir.name

    # Check common CIF naming patterns
    candidates = [
        # CIF directly in af3_binary/ or af3_ternary/
        af3_dir / f"{pair_id}_{mode}_model.cif",
        # Nested: af3_binary/{pair_id}_{mode}/{pair_id}_{mode}_model.cif
        af3_dir / f"{pair_id}_{mode}" / f"{pair_id}_{mode}_model.cif",
    ]

    # Also glob for any .cif in the af3 directory
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: find any *_model.cif in the directory
    if af3_dir.exists():
        cifs = list(af3_dir.glob("*_model.cif"))
        if cifs:
            return cifs[0]
        # Check one level deeper
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
