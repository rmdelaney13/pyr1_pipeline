#!/usr/bin/env python3
"""
Step 1: Three-way sequence alignment and mutation mapping.

Aligns PYL2 (3KDH), PYR1 WT (3QN1), and the designed Boltz sequence to:
  1. Map residue numbers between 3KDH and Boltz (for threading)
  2. Categorize each mutation as:
     - Category A: PYL2->PYR1 species conversion (aa_3kdh != aa_3qn1, aa_3qn1 == aa_boltz)
     - Category B: PYR1->designed pocket mutation (aa_3kdh == aa_3qn1, aa_3qn1 != aa_boltz)
     - Category C: Three-way difference (aa_3kdh != aa_3qn1 != aa_boltz)
  3. Verify H119 latch histidine orientation

Strategy: Two composed pairwise alignments (3KDH<->Boltz and 3QN1<->Boltz),
using Boltz as the common reference frame.

Usage:
    python align_sequences.py \
        --pdb-3kdh inputs/3KDH.pdb \
        --pdb-3qn1 inputs/3QN1.pdb \
        --boltz-pdb inputs/boltz_predictions/pair_3098.pdb \
        --output-json alignment_map.json

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import sys
from pathlib import Path

from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1

# 16 designable pocket positions (Boltz numbering, 1-181 construct)
POCKET_POSITIONS_BOLTZ = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122,
                          141, 159, 160, 163, 164, 167]

# Gate loop residues (Boltz numbering) — alpha3-beta4 loop
GATE_LOOP_BOLTZ = list(range(87, 97))  # 87-96 inclusive

# Anchor positions for ligand superposition (Boltz numbering)
# Pocket-floor residues on rigid secondary structure, excluding gate loop and pos 59
ANCHOR_POSITIONS_BOLTZ = [81, 83, 108, 110, 117, 120, 122, 141,
                          159, 160, 163, 164, 167]

# Known PYL2 latch histidine position (PYL2/3KDH numbering)
PYL2_LATCH_HIS = 119


def extract_sequence_from_pdb(pdb_path, chain="A"):
    """Extract amino acid sequence and residue numbers from PDB ATOM records.

    Returns:
        residues: list of (resnum, resname_1letter) tuples, in PDB order
        sequence: one-letter amino acid string
    """
    seen = set()
    residues = []

    three_to_one = {}
    for k, v in protein_letters_3to1.items():
        three_to_one[k.upper()] = v.upper()
    three_to_one["MSE"] = "M"

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21]
            if chain_id != chain:
                continue
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            key = (chain_id, resnum)
            if key in seen:
                continue
            seen.add(key)

            aa = three_to_one.get(resname, "X")
            residues.append((resnum, aa))

    residues.sort(key=lambda x: x[0])
    sequence = "".join(aa for _, aa in residues)

    return residues, sequence


def build_residue_mapping(residues_a, seq_a, residues_b, seq_b, label_a="A", label_b="B"):
    """Align two sequences and build a residue number mapping.

    Returns:
        a_to_b: dict mapping A resnums -> B resnums
        b_to_a: dict mapping B resnums -> A resnums
        alignment_details: list of (resnum_a, aa_a, resnum_b, aa_b) for aligned positions
    """
    alignments = pairwise2.align.globalms(
        seq_a, seq_b,
        match=2, mismatch=-1, open=-5, extend=-0.5,
        one_alignment_only=True
    )

    if not alignments:
        print(f"ERROR: Could not align {label_a} and {label_b} sequences")
        sys.exit(1)

    aligned_a, aligned_b, score, begin, end = alignments[0]

    a_to_b = {}
    b_to_a = {}
    alignment_details = []

    idx_a = 0
    idx_b = 0

    for i in range(len(aligned_a)):
        char_a = aligned_a[i]
        char_b = aligned_b[i]

        has_a = char_a != "-"
        has_b = char_b != "-"

        if has_a and has_b:
            resnum_a = residues_a[idx_a][0]
            resnum_b = residues_b[idx_b][0]
            a_to_b[resnum_a] = resnum_b
            b_to_a[resnum_b] = resnum_a
            alignment_details.append((resnum_a, char_a, resnum_b, char_b))
            idx_a += 1
            idx_b += 1
        elif has_a:
            idx_a += 1
        elif has_b:
            idx_b += 1

    return a_to_b, b_to_a, alignment_details


def classify_region(boltz_pos):
    """Classify a Boltz position as pocket, gate_loop, or distal."""
    if boltz_pos in POCKET_POSITIONS_BOLTZ:
        return "pocket"
    if boltz_pos in GATE_LOOP_BOLTZ:
        return "gate_loop"
    return "distal"


def build_three_way_alignment(res_3kdh, seq_3kdh, res_3qn1, seq_3qn1, res_boltz, seq_boltz):
    """Run two pairwise alignments and compose them into a three-way mapping.

    Alignment 1: 3KDH <-> Boltz (primary threading map)
    Alignment 2: 3QN1 <-> Boltz (for mutation categorization)

    Returns:
        result dict with all mappings, mutations table, and metadata
    """
    # Alignment 1: 3KDH (PYL2) <-> Boltz (designed PYR1)
    print("\n--- Alignment 1: 3KDH (PYL2) <-> Boltz (designed PYR1) ---")
    kdh_to_boltz, boltz_to_3kdh, details_kdh_boltz = build_residue_mapping(
        res_3kdh, seq_3kdh, res_boltz, seq_boltz,
        label_a="3KDH", label_b="Boltz"
    )
    n_match_1 = sum(1 for _, aa, _, ab in details_kdh_boltz if aa == ab)
    pct_id_1 = 100 * n_match_1 / len(details_kdh_boltz) if details_kdh_boltz else 0
    print(f"  Aligned: {len(details_kdh_boltz)} positions, "
          f"identity: {n_match_1}/{len(details_kdh_boltz)} ({pct_id_1:.1f}%)")

    # Alignment 2: 3QN1 (PYR1 WT) <-> Boltz (designed PYR1)
    print("\n--- Alignment 2: 3QN1 (PYR1 WT) <-> Boltz (designed PYR1) ---")
    qn1_to_boltz, boltz_to_3qn1, details_qn1_boltz = build_residue_mapping(
        res_3qn1, seq_3qn1, res_boltz, seq_boltz,
        label_a="3QN1", label_b="Boltz"
    )
    n_match_2 = sum(1 for _, aa, _, ab in details_qn1_boltz if aa == ab)
    pct_id_2 = 100 * n_match_2 / len(details_qn1_boltz) if details_qn1_boltz else 0
    print(f"  Aligned: {len(details_qn1_boltz)} positions, "
          f"identity: {n_match_2}/{len(details_qn1_boltz)} ({pct_id_2:.1f}%)")

    # Build lookup dicts for quick access
    kdh_aa_by_resnum = {r: aa for r, aa in res_3kdh}
    qn1_aa_by_resnum = {r: aa for r, aa in res_3qn1}
    boltz_aa_by_resnum = {r: aa for r, aa in res_boltz}

    # Compose: for each Boltz position, look up 3KDH and 3QN1 identities
    mutations_table = []
    n_cat_a, n_cat_b, n_cat_c = 0, 0, 0

    # Iterate over all Boltz positions that have a 3KDH mapping
    for boltz_pos in sorted(boltz_to_3kdh.keys()):
        kdh_pos = boltz_to_3kdh[boltz_pos]
        qn1_pos = boltz_to_3qn1.get(boltz_pos)

        aa_kdh = kdh_aa_by_resnum.get(kdh_pos, "?")
        aa_qn1 = qn1_aa_by_resnum.get(qn1_pos, "?") if qn1_pos else "?"
        aa_boltz = boltz_aa_by_resnum.get(boltz_pos, "?")

        # Only record positions that need mutation
        if aa_kdh == aa_boltz:
            continue

        # Categorize
        if aa_qn1 != "?" and aa_qn1 == aa_boltz and aa_kdh != aa_qn1:
            category = "A"  # PYL2->PYR1 species conversion
            n_cat_a += 1
        elif aa_qn1 != "?" and aa_kdh == aa_qn1 and aa_qn1 != aa_boltz:
            category = "B"  # PYR1->designed pocket mutation
            n_cat_b += 1
        elif aa_qn1 != "?" and aa_kdh != aa_qn1 and aa_qn1 != aa_boltz:
            category = "C"  # Three-way difference
            n_cat_c += 1
        else:
            # Edge case: no 3QN1 mapping or other
            category = "A"  # Default to species conversion
            n_cat_a += 1

        region = classify_region(boltz_pos)

        mutations_table.append({
            "boltz_pos": boltz_pos,
            "3kdh_pos": kdh_pos,
            "3qn1_pos": qn1_pos,
            "aa_3kdh": aa_kdh,
            "aa_3qn1": aa_qn1,
            "aa_boltz": aa_boltz,
            "category": category,
            "region": region,
        })

    return {
        "kdh_to_boltz": kdh_to_boltz,
        "boltz_to_3kdh": boltz_to_3kdh,
        "qn1_to_boltz": qn1_to_boltz,
        "boltz_to_3qn1": boltz_to_3qn1,
        "details_kdh_boltz": details_kdh_boltz,
        "details_qn1_boltz": details_qn1_boltz,
        "mutations_table": mutations_table,
        "n_category_a": n_cat_a,
        "n_category_b": n_cat_b,
        "n_category_c": n_cat_c,
        "pct_identity_kdh_boltz": pct_id_1,
        "pct_identity_qn1_boltz": pct_id_2,
    }


def verify_latch_histidine(boltz_to_3kdh, kdh_aa_by_resnum, boltz_aa_by_resnum):
    """Verify that 3KDH H119 (latch) maps correctly and report."""
    print(f"\n--- Latch Histidine Verification ---")

    # Find which Boltz position maps to 3KDH residue 119
    kdh_to_boltz = {v: k for k, v in boltz_to_3kdh.items()}
    latch_boltz = kdh_to_boltz.get(PYL2_LATCH_HIS)

    aa_at_latch_kdh = kdh_aa_by_resnum.get(PYL2_LATCH_HIS, "?")

    if latch_boltz is None:
        print(f"  WARNING: 3KDH residue {PYL2_LATCH_HIS} not mapped to Boltz")
        return PYL2_LATCH_HIS, aa_at_latch_kdh, None

    aa_at_latch_boltz = boltz_aa_by_resnum.get(latch_boltz, "?")
    region = classify_region(latch_boltz)

    print(f"  3KDH H{PYL2_LATCH_HIS} -> Boltz pos {latch_boltz}: "
          f"3KDH={aa_at_latch_kdh}, Boltz={aa_at_latch_boltz}")
    print(f"  Region: {region}")

    if aa_at_latch_kdh != "H":
        print(f"  WARNING: Expected H at 3KDH {PYL2_LATCH_HIS}, got {aa_at_latch_kdh}")

    if latch_boltz in POCKET_POSITIONS_BOLTZ:
        print(f"  WARNING: Latch histidine maps to a POCKET position!")

    if aa_at_latch_kdh == "H":
        print(f"  OK: Latch histidine confirmed at 3KDH pos {PYL2_LATCH_HIS}")
        print(f"  In 3KDH crystal, H{PYL2_LATCH_HIS} sidechain points OUT "
              f"(away from pocket, toward solvent)")

    return PYL2_LATCH_HIS, aa_at_latch_kdh, latch_boltz


def print_mutations_summary(mutations_table, n_cat_a, n_cat_b, n_cat_c):
    """Print formatted summary of all mutations to thread."""
    total = len(mutations_table)
    print(f"\n--- Mutations to Thread: {total} total ---")
    print(f"  Category A (PYL2->PYR1): {n_cat_a}")
    print(f"  Category B (PYR1->designed): {n_cat_b}")
    print(f"  Category C (three-way diff): {n_cat_c}")

    if total > 90:
        print(f"\n  WARNING: {total} mutations = {total/180*100:.0f}% of residues!")
        print(f"  Consider additional relaxation cycles.")

    print(f"\n{'3KDH_pos':>10} {'3KDH_aa':>8} {'3QN1_aa':>8} {'Boltz_aa':>9} "
          f"{'Target':>7} {'Cat':>4} {'Region':>12}")
    print("-" * 70)

    for entry in mutations_table:
        importance = ""
        if entry["region"] == "gate_loop":
            importance = " *GATE*"
        elif entry["region"] == "pocket":
            importance = " *POCKET*"

        print(f"{entry['3kdh_pos']:>10} {entry['aa_3kdh']:>8} {entry['aa_3qn1']:>8} "
              f"{entry['aa_boltz']:>9} {entry['aa_boltz']:>7} {entry['category']:>4} "
              f"{entry['region']:>12}{importance}")


def main():
    parser = argparse.ArgumentParser(
        description="Three-way alignment: 3KDH (PYL2) / 3QN1 (PYR1 WT) / Boltz (designed)"
    )
    parser.add_argument("--pdb-3kdh", required=True, help="Path to 3KDH.pdb (PYL2 apo)")
    parser.add_argument("--pdb-3qn1", required=True, help="Path to 3QN1.pdb (PYR1 WT)")
    parser.add_argument("--boltz-pdb", required=True,
                        help="Path to any Boltz prediction PDB (for designed sequence)")
    parser.add_argument("--chain-3kdh", default="A",
                        help="Chain ID in 3KDH (default: A)")
    parser.add_argument("--chain-3qn1", default="A",
                        help="Chain ID in 3QN1 (default: A)")
    parser.add_argument("--chain-boltz", default="A",
                        help="Chain ID in Boltz PDB (default: A)")
    parser.add_argument("--output-json", default="alignment_map.json",
                        help="Output JSON file for residue mapping")
    args = parser.parse_args()

    # Extract sequences
    print(f"Extracting sequence from 3KDH ({args.pdb_3kdh}, chain {args.chain_3kdh})...")
    res_3kdh, seq_3kdh = extract_sequence_from_pdb(args.pdb_3kdh, args.chain_3kdh)
    print(f"  {len(res_3kdh)} residues, range {res_3kdh[0][0]}-{res_3kdh[-1][0]}")

    print(f"Extracting sequence from 3QN1 ({args.pdb_3qn1}, chain {args.chain_3qn1})...")
    res_3qn1, seq_3qn1 = extract_sequence_from_pdb(args.pdb_3qn1, args.chain_3qn1)
    print(f"  {len(res_3qn1)} residues, range {res_3qn1[0][0]}-{res_3qn1[-1][0]}")

    print(f"Extracting sequence from Boltz PDB ({args.boltz_pdb}, chain {args.chain_boltz})...")
    res_boltz, seq_boltz = extract_sequence_from_pdb(args.boltz_pdb, args.chain_boltz)
    print(f"  {len(res_boltz)} residues, range {res_boltz[0][0]}-{res_boltz[-1][0]}")

    # Three-way alignment
    result = build_three_way_alignment(
        res_3kdh, seq_3kdh, res_3qn1, seq_3qn1, res_boltz, seq_boltz
    )

    # Print mutations summary
    print_mutations_summary(
        result["mutations_table"],
        result["n_category_a"], result["n_category_b"], result["n_category_c"]
    )

    # Verify latch histidine
    kdh_aa = {r: aa for r, aa in res_3kdh}
    boltz_aa = {r: aa for r, aa in res_boltz}
    latch_3kdh_pos, latch_aa, latch_boltz_pos = verify_latch_histidine(
        result["boltz_to_3kdh"], kdh_aa, boltz_aa
    )

    # Map pocket, gate, anchor positions to 3KDH numbering
    boltz_to_3kdh = result["boltz_to_3kdh"]
    pocket_3kdh = [boltz_to_3kdh[bp] for bp in POCKET_POSITIONS_BOLTZ
                   if bp in boltz_to_3kdh]
    anchor_3kdh = [boltz_to_3kdh[bp] for bp in ANCHOR_POSITIONS_BOLTZ
                   if bp in boltz_to_3kdh]
    gate_3kdh = [boltz_to_3kdh[bp] for bp in GATE_LOOP_BOLTZ
                 if bp in boltz_to_3kdh]

    # Print anchor mapping
    print(f"\nAnchor positions for ligand superposition (Boltz -> 3KDH):")
    missing_anchors = []
    for bp in ANCHOR_POSITIONS_BOLTZ:
        kp = boltz_to_3kdh.get(bp)
        if kp is not None:
            print(f"  Boltz {bp:>3} -> 3KDH {kp:>3}  OK")
        else:
            print(f"  Boltz {bp:>3} -> NOT FOUND  *** PROBLEM ***")
            missing_anchors.append(bp)

    if missing_anchors:
        print(f"\n  ERROR: {len(missing_anchors)} anchor positions missing!")

    # Build WT identity dicts at pocket positions
    wt_3qn1_pocket = {}
    wt_3kdh_pocket = {}
    for bp in POCKET_POSITIONS_BOLTZ:
        qn1_pos = result["boltz_to_3qn1"].get(bp)
        kdh_pos = boltz_to_3kdh.get(bp)
        if qn1_pos:
            qn1_aa = {r: aa for r, aa in res_3qn1}.get(qn1_pos, "?")
            wt_3qn1_pocket[str(bp)] = qn1_aa
        if kdh_pos:
            wt_3kdh_pocket[str(bp)] = kdh_aa.get(kdh_pos, "?")

    # Save JSON
    output = {
        "boltz_to_3kdh": {str(k): v for k, v in boltz_to_3kdh.items()},
        "3kdh_to_boltz": {str(k): v for k, v in result["kdh_to_boltz"].items()},
        "boltz_to_3qn1": {str(k): v for k, v in result["boltz_to_3qn1"].items()},
        "3qn1_to_boltz": {str(k): v for k, v in result["qn1_to_boltz"].items()},
        "pocket_positions_boltz": POCKET_POSITIONS_BOLTZ,
        "pocket_positions_3kdh": pocket_3kdh,
        "gate_loop_boltz": GATE_LOOP_BOLTZ,
        "gate_loop_3kdh": gate_3kdh,
        "anchor_positions_boltz": ANCHOR_POSITIONS_BOLTZ,
        "anchor_positions_3kdh": anchor_3kdh,
        "wt_3qn1_at_pocket": wt_3qn1_pocket,
        "wt_3kdh_at_pocket": wt_3kdh_pocket,
        "mutations_table": result["mutations_table"],
        "n_category_a": result["n_category_a"],
        "n_category_b": result["n_category_b"],
        "n_category_c": result["n_category_c"],
        "3kdh_latch_pos": latch_3kdh_pos,
        "3kdh_latch_aa": latch_aa,
        "latch_boltz_pos": latch_boltz_pos,
        "3kdh_sequence_range": [res_3kdh[0][0], res_3kdh[-1][0]],
        "3qn1_sequence_range": [res_3qn1[0][0], res_3qn1[-1][0]],
        "boltz_sequence_range": [res_boltz[0][0], res_boltz[-1][0]],
        "3kdh_chain": args.chain_3kdh,
        "3qn1_chain": args.chain_3qn1,
        "boltz_chain": args.chain_boltz,
        "n_aligned_3kdh_boltz": len(result["details_kdh_boltz"]),
        "n_aligned_3qn1_boltz": len(result["details_qn1_boltz"]),
        "pct_identity_kdh_boltz": round(result["pct_identity_kdh_boltz"], 1),
        "pct_identity_qn1_boltz": round(result["pct_identity_qn1_boltz"], 1),
    }

    out_path = Path(args.output_json)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAlignment mapping saved to {out_path}")
    print(f"  {len(boltz_to_3kdh)} Boltz->3KDH mappings")
    print(f"  {len(result['boltz_to_3qn1'])} Boltz->3QN1 mappings")
    print(f"  {len(pocket_3kdh)}/{len(POCKET_POSITIONS_BOLTZ)} pocket positions mapped")
    print(f"  {len(anchor_3kdh)}/{len(ANCHOR_POSITIONS_BOLTZ)} anchor positions mapped")
    print(f"  {len(result['mutations_table'])} total mutations to thread")


if __name__ == "__main__":
    main()
