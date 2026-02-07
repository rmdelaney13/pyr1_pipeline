#!/usr/bin/env bash

# rename_wat_to_hoh.sh
# Usage: ./rename_wat_to_hoh.sh input.pdb output.pdb
# Replaces any HETATM record whose 3-letter residue is "WAT" with "HOH".

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input.pdb output.pdb"
  exit 1
fi

infile="$1"
outfile="$2"

# For any line beginning with "HETATM" where columns 18–20 == "WAT",
# replace those three characters with "HOH" and write to $outfile.
sed '/^HETATM/ s/^\(.\{17\}\)WAT/\1HOH/' "$infile" > "$outfile"

echo "Renamed WAT→HOH in $infile → $outfile"

