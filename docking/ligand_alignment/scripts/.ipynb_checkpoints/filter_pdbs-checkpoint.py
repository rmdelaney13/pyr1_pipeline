#!/usr/bin/env python3
import os
import csv
import argparse
import shutil
import sys
import math

def move_top_dG_sep(csv_path, base_dir, dest_dir, top_fraction=0.10):
    """
    Read the CSV at csv_path, identify the top `top_fraction` (default 10%) of rows
    by lowest dG_sep (column name "dG_sep"). For each of those rows, derive the PDB
    filename from the "filename" column (strip "_score.sc" and append ".pdb"), then
    move that .pdb from base_dir into dest_dir (skipping any that end with "_tmp.pdb").
    """

    # 1. Verify base_dir exists
    if not os.path.isdir(base_dir):
        print(f"Error: base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    # 2. Create dest_dir if needed
    os.makedirs(dest_dir, exist_ok=True)

    # 3. Read all rows and collect (filename, dG_sep)
    rows = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required_cols = {"filename", "dG_sep"}
        if not required_cols.issubset(reader.fieldnames):
            missing = required_cols - set(reader.fieldnames)
            print(f"Error: Missing required column(s) in CSV: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            fname = row["filename"].strip()
            dg_val = row["dG_sep"].strip()
            try:
                dg = float(dg_val)
            except ValueError:
                # Skip rows where dG_sep is not a valid float
                continue
            rows.append((fname, dg))

    if not rows:
        print("Error: No valid rows with numeric dG_sep found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # 4. Sort rows by dG_sep (ascending)
    rows.sort(key=lambda x: x[1])

    # 5. Determine how many constitute the top fraction
    total = len(rows)
    n_top = max(1, math.ceil(total * top_fraction))

    top_rows = rows[:n_top]

    moved_count = 0
    missing_count = 0

    # 6. For each top row, move its corresponding .pdb
    for fname, dg in top_rows:
        if not fname.endswith("_score.sc"):
            # Skip if filename does not follow expected pattern
            continue

        prefix = fname[:-len("_score.sc")]
        pdb_name = prefix + ".pdb"

        # Skip any *_tmp.pdb
        if pdb_name.endswith("_tmp.pdb"):
            continue

        src_pdb_path = os.path.join(base_dir, pdb_name)
        if os.path.isfile(src_pdb_path):
            dest_pdb_path = os.path.join(dest_dir, pdb_name)
            try:
                shutil.move(src_pdb_path, dest_pdb_path)
                moved_count += 1
            except Exception as e:
                print(f"Warning: could not move '{src_pdb_path}' → '{dest_pdb_path}': {e}", file=sys.stderr)
        else:
            print(f"Warning: PDB not found: '{pdb_name}' (looking in {base_dir})", file=sys.stderr)
            missing_count += 1

    print(f"Done. {moved_count} PDB files moved into '{dest_dir}'. {missing_count} were not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move the top 10% (by lowest dG_sep) of PDBs based on the CSV."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the aggregated CSV (must have 'filename' and 'dG_sep' columns)."
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing all *.pdb files corresponding to the score filenames."
    )
    parser.add_argument(
        "dest_dir",
        help="Directory where you want to move the top 10% .pdb files (will be created if needed)."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of lowest dG_sep rows to select (default: 0.10 for top 10%)."
    )
    args = parser.parse_args()

    move_top_dG_sep(args.csv_file, args.base_dir, args.dest_dir, top_fraction=args.fraction)
