#!/usr/bin/env python3
"""
aggregate_jsons_into_batches.py

Scan a root directory (e.g., "msas/") for all JSON files and
group them into new subdirectories named batch_01, batch_02, …,
each containing up to `limit` JSONs. By default, `limit=60`.

Usage:
    python aggregate_jsons_into_batches.py /path/to/msas [--limit 60]

Result:
    /path/to/msas/
    ├── pass13_repacked_.../
    │   └── pass13_repacked_..._data.json
    ├── pass22_repacked_.../
    │   └── pass22_repacked_..._data.json
    ├── batch_01/
    │   ├── pass13_repacked_..._data.json
    │   ├── pass22_repacked_..._data.json
    │   └── ...
    ├── batch_02/
    │   └── ...
    └── …    
"""

import os
import shutil
import argparse

def gather_all_jsons(root_dir):
    """
    Walk `root_dir` recursively and return a sorted list of full paths
    to every .json file found.
    """
    json_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                json_paths.append(os.path.join(dirpath, fname))
    json_paths.sort()
    return json_paths

def make_batches(json_paths, root_dir, limit):
    """
    Given a list of JSON paths, split them into chunks of size `limit`,
    create directories batch_01, batch_02, … under `root_dir`, and copy
    each JSON into the appropriate batch folder.
    """
    total = len(json_paths)
    if total == 0:
        print("No JSON files found. Nothing to do.")
        return

    num_batches = (total + limit - 1) // limit

    for batch_idx in range(num_batches):
        batch_name = f"batch_{batch_idx+1:02d}"
        batch_dir = os.path.join(root_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        start = batch_idx * limit
        end = min(start + limit, total)
        for json_path in json_paths[start:end]:
            fname = os.path.basename(json_path)
            dst = os.path.join(batch_dir, fname)
            try:
                shutil.copy(json_path, dst)
            except Exception as e:
                print(f"  [!] Failed to copy {json_path} → {dst}: {e}")
        print(f"Wrote JSONs {start+1}–{end} into {batch_name}/")

    print(f"\nTotal JSONs: {total}, split into {num_batches} batches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate JSON files into batch directories (max 60 per batch)."
    )
    parser.add_argument(
        "root_dir",
        help="Path to the directory containing all AF3‐generated subfolders (e.g. /scratch/alpine/ryde3462/msas)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of JSON files per batch directory (default: 60)."
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory.")
        exit(1)

    all_jsons = gather_all_jsons(root)
    make_batches(all_jsons, root, args.limit)

