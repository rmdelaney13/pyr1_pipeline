#!/usr/bin/env python3
"""
Generate a standalone single-chain .a3m MSA for HAB1 (or any protein).

Boltz stores multi-chain MSAs as processed CSV files (e.g., name_2.csv),
NOT as per-chain .a3m files. The raw .a3m files only contain the first
protein chain. To get a standalone .a3m for HAB1 to use with
prepare_boltz_yamls.py --hab1-msa, we run a quick single-chain Boltz
prediction and extract the resulting MSA.

Usage:
    # Generate HAB1 MSA via Boltz (requires GPU + boltz_env):
    python scripts/extract_chain_msa.py \
        --sequence "GAMGRSVYEL...KFKTRT" \
        --out-dir /scratch/alpine/ryde3462/boltz_lca/hab1_msa \
        --name hab1 \
        --run-boltz

    # Or just create a single-sequence .a3m (no homologs, fast):
    python scripts/extract_chain_msa.py \
        --sequence "GAMGRSVYEL...KFKTRT" \
        --out-dir /scratch/alpine/ryde3462/boltz_lca/hab1_msa \
        --name hab1

    # Or from the WT ternary Boltz output, extract the .a3m from the
    # MSA cache directory (if available):
    python scripts/extract_chain_msa.py \
        --boltz-msa-dir /scratch/.../msa/ \
        --chain-index 2 \
        --out-dir /scratch/.../hab1_msa \
        --name hab1
"""

import argparse
import sys
from pathlib import Path


def create_single_sequence_a3m(sequence: str, name: str, out_path: Path):
    """Create a minimal .a3m with just the query sequence (no homologs)."""
    with open(out_path, 'w') as f:
        f.write(f">{name}\n")
        f.write(f"{sequence}\n")
    print(f"Created single-sequence .a3m: {out_path}")
    print(f"  Query length: {len(sequence)} aa")
    print(f"  NOTE: No homologs. For better quality, use --run-boltz to generate MSA.")


def create_boltz_yaml(sequence: str, name: str, out_dir: Path) -> Path:
    """Create a Boltz YAML for single-chain MSA generation."""
    yaml_path = out_dir / f"{name}.yaml"
    with open(yaml_path, 'w') as f:
        f.write("version: 1\n")
        f.write("sequences:\n")
        f.write("  - protein:\n")
        f.write("      id: A\n")
        f.write(f'      sequence: "{sequence}"\n')
    return yaml_path


def find_a3m_in_boltz_output(boltz_out_dir: Path, name: str) -> Path:
    """Find the generated .a3m file in Boltz output.

    Boltz saves MSAs in:
        boltz_results_<name>/msa/<name>_unpaired_*/uniref.a3m
        boltz_results_<name>/msa/<name>_unpaired_*/bfd.*.a3m
    """
    # Search recursively for any .a3m file (prefer uniref)
    for pattern in ["**/uniref.a3m", "**/*.a3m"]:
        matches = sorted(boltz_out_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate standalone per-chain .a3m MSA for Boltz predictions")
    parser.add_argument("--sequence", default=None,
                        help="Protein sequence (amino acid string)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for .a3m file")
    parser.add_argument("--name", default="hab1",
                        help="Name for the output files (default: hab1)")
    parser.add_argument("--run-boltz", action="store_true",
                        help="Run Boltz with --use_msa_server to generate proper MSA "
                             "(requires GPU + boltz_env). Without this flag, creates "
                             "a single-sequence .a3m (no homologs).")
    parser.add_argument("--boltz-cache", default="/projects/ryde3462/software/boltz_cache",
                        help="Boltz model cache directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.sequence:
        # Default to HAB1 sequence
        args.sequence = (
            "GAMGRSVYELDCIPLWGTVSIQGNRSEMEDAFAVSPHFLKLPIKMLMGDHEGMSPSLTHLTGHFFGVY"
            "DGHGGHKVADYCRDRLHFALAEEIERIKDELCKRNTGEGRQVQWDKVFTSCFLTVDGEIEGKIGRAVVG"
            "SSDKVLEAVASETVGSTAVVALVCSSHIVVSNCGDSRAVLFRGKEAMPLSVDHKPDREDEYARIENAGGK"
            "VIQWQGARVFGVLAMSRSIGDRYLKPYVIPEPEVTFMPRSREDECLILASDGLWDVMNNQEVCEIARRR"
            "ILMWHKKNGAPPLAERGKGIDPACQAAADYLSMLALQKGSKDNISIIVIDLKAQRKFKTRT"
        )
        print(f"Using default HAB1 sequence ({len(args.sequence)} aa)")

    out_a3m = out_dir / f"{args.name}.a3m"

    if not args.run_boltz:
        # Just create a single-sequence .a3m
        create_single_sequence_a3m(args.sequence, args.name, out_a3m)
        return

    # Run Boltz to generate proper MSA with homologs
    import subprocess

    yaml_path = create_boltz_yaml(args.sequence, args.name, out_dir)
    boltz_out = out_dir / "boltz_output"

    print(f"Running Boltz to generate MSA for {args.name}...")
    print(f"  YAML: {yaml_path}")
    print(f"  Output: {boltz_out}")

    cmd = [
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(boltz_out),
        "--cache", args.boltz_cache,
        "--recycling_steps", "1",
        "--diffusion_samples", "1",
        "--output_format", "pdb",
        "--use_msa_server",
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Boltz prediction failed (exit code {result.returncode})",
              file=sys.stderr)
        print("  This is OK — we only need the MSA, not the structure prediction.",
              file=sys.stderr)

    # Find the generated .a3m (MSA is generated before prediction, so it
    # should exist even if the prediction step crashed)
    src_a3m = find_a3m_in_boltz_output(boltz_out, args.name)
    if src_a3m is None:
        print("ERROR: Could not find generated .a3m in Boltz output", file=sys.stderr)
        print(f"  Searched: {boltz_out}", file=sys.stderr)
        # List what's actually there for debugging
        for p in sorted(boltz_out.rglob("*")):
            if p.is_file():
                print(f"    {p}", file=sys.stderr)
        sys.exit(1)

    # Copy to output location
    import shutil
    shutil.copy2(src_a3m, out_a3m)
    print(f"\nHAB1 MSA generated successfully!")
    print(f"  Source: {src_a3m}")
    print(f"  Output: {out_a3m}")

    # Count sequences
    n_seqs = sum(1 for line in open(out_a3m) if line.startswith('>'))
    print(f"  Sequences: {n_seqs}")


if __name__ == "__main__":
    main()
