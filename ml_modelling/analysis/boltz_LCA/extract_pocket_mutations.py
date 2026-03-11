#!/usr/bin/env python3
"""Extract pocket mutations for all MD candidates into a CSV with one column per position."""

import csv
from pathlib import Path

WT_PYR1_SEQUENCE = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]


def wt_at(pos):
    return WT_PYR1_SEQUENCE[pos - 1]


def parse_signature(sig):
    muts = {}
    if not sig:
        return muts
    for tok in sig.split(';'):
        tok = tok.strip()
        if not tok:
            continue
        pos = int(tok[:-1])
        aa = tok[-1]
        muts[pos] = aa
    return muts


here = Path(__file__).parent
infile = here / "md_candidates_lca_top100.csv"
outfile = here / "md_candidates_pocket_mutations.csv"

with open(infile) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

header = ["pair_id", "rank", "category", "label", "label_tier"]
for pos in POCKET_POSITIONS:
    header.append(f"pos{pos}_{wt_at(pos)}")

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        muts = parse_signature(row["variant_signature"])
        out = [row["pair_id"], row["rank"], row["category"], row["label"], row["label_tier"]]
        for pos in POCKET_POSITIONS:
            aa = muts.get(pos, wt_at(pos))
            mut_flag = "*" if pos in muts else ""
            out.append(f"{aa}{mut_flag}")
        writer.writerow(out)

print(f"Wrote {len(rows)} rows to {outfile}")
