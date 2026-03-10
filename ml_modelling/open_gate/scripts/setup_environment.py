#!/usr/bin/env python3
"""
Step 0: Environment setup for the open-gate PYR1 structure preparation pipeline.

Checks dependencies, creates directory structure, and downloads reference PDBs.

Usage:
    python setup_environment.py --project-dir ml_modelling/open_gate/
    python setup_environment.py --project-dir ml_modelling/open_gate/ --link-boltz-pdbs ml_modelling/analysis/boltz_LCA/md_candidate_pdbs

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


def check_dependency(module_name, install_hint):
    """Check if a Python module is importable. Print status and return True/False."""
    try:
        __import__(module_name)
        print(f"  [OK]   {module_name}")
        return True
    except ImportError:
        print(f"  [FAIL] {module_name} -- {install_hint}")
        return False


def check_all_dependencies():
    """Check all required Python dependencies. Returns True if all present."""
    print("Checking dependencies...")
    all_ok = True

    all_ok &= check_dependency("pyrosetta",
        "conda install -c https://USERNAME:PASSWORD@conda.graylab.jhu.edu pyrosetta")
    all_ok &= check_dependency("numpy", "pip install numpy")
    all_ok &= check_dependency("pandas", "pip install pandas")
    all_ok &= check_dependency("Bio", "pip install biopython")
    all_ok &= check_dependency("rdkit", "conda install -c conda-forge rdkit")
    all_ok &= check_dependency("prody", "pip install prody")

    if not all_ok:
        print("\nSome dependencies are missing. Install them before proceeding.")
    else:
        print("\nAll dependencies satisfied.")

    return all_ok


def create_directory_structure(project_dir):
    """Create the project directory tree."""
    print(f"\nCreating directory structure under {project_dir}/...")
    dirs = [
        "scripts",
        "inputs/boltz_predictions",
        "outputs/threaded_relaxed",
        "outputs/open_gate_structures",
        "params",
        "logs",
    ]
    for d in dirs:
        p = project_dir / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"  {p}")


def download_pdb(pdb_id, output_path):
    """Download a PDB file from RCSB if not already present."""
    if output_path.exists():
        print(f"  {pdb_id}.pdb already exists at {output_path}")
        return True

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from {url}...")
    try:
        urllib.request.urlretrieve(url, str(output_path))
        print(f"  Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        print(f"  Download manually: {url} -> {output_path}")
        return False


def link_boltz_pdbs(source_dir, target_dir):
    """Symlink or copy Boltz prediction PDBs into inputs/boltz_predictions/."""
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()

    if not source.exists():
        print(f"  WARNING: Source directory does not exist: {source}")
        return 0

    pdb_files = list(source.glob("*.pdb"))
    if not pdb_files:
        print(f"  WARNING: No PDB files found in {source}")
        return 0

    count = 0
    for pdb in pdb_files:
        dest = target / pdb.name
        if dest.exists():
            continue
        try:
            # Try symlink first (works on Unix, needs developer mode on Windows)
            os.symlink(str(pdb), str(dest))
            count += 1
        except OSError:
            # Fall back to copy on Windows
            shutil.copy2(str(pdb), str(dest))
            count += 1

    print(f"  Linked {count} new PDB files ({len(pdb_files)} total in source)")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Setup environment for open-gate PYR1 structure prep pipeline"
    )
    parser.add_argument("--project-dir", required=True,
                        help="Root directory for the open-gate pipeline (e.g., ml_modelling/open_gate)")
    parser.add_argument("--link-boltz-pdbs",
                        help="Path to directory containing Boltz prediction PDBs to link into inputs/")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Skip dependency checking")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()

    # Step 1: Check dependencies
    if not args.skip_deps:
        deps_ok = check_all_dependencies()
        if not deps_ok:
            print("\nContinuing setup despite missing dependencies...")

    # Step 2: Create directory structure
    create_directory_structure(project_dir)

    # Step 3: Download reference PDBs
    print("\nDownloading reference PDB files...")
    inputs_dir = project_dir / "inputs"
    download_pdb("3K3K", inputs_dir / "3K3K.pdb")

    # Step 4: Link Boltz prediction PDBs
    if args.link_boltz_pdbs:
        print(f"\nLinking Boltz prediction PDBs from {args.link_boltz_pdbs}...")
        link_boltz_pdbs(args.link_boltz_pdbs, inputs_dir / "boltz_predictions")
    else:
        print("\nNo --link-boltz-pdbs specified. Place Boltz PDBs in:")
        print(f"  {inputs_dir / 'boltz_predictions'}/")

    # Step 5: Write setup log
    log = {
        "timestamp": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "inputs_dir": str(inputs_dir),
        "pdb_3k3k": str(inputs_dir / "3K3K.pdb"),
        "boltz_dir": str(inputs_dir / "boltz_predictions"),
    }
    log_path = project_dir / "setup_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSetup log written to {log_path}")

    print("\nSetup complete. Next step: run align_sequences.py")


if __name__ == "__main__":
    main()
