#!/usr/bin/env python3
"""
Step 1: Sequence alignment and residue mapping between 3K3K and Boltz structures.

Builds an explicit residue number mapping by aligning the sequences extracted
from PDB ATOM records. No hardcoded offsets — the mapping is derived
computationally from the actual modeled residues.

Usage:
    python align_sequences.py \
        --pdb-3k3k inputs/3K3K.pdb \
        --boltz-pdb inputs/boltz_predictions/pair_3098.pdb \
        --output-json alignment_map.json

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import sys
from collections import OrderedDict
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


def extract_sequence_from_pdb(pdb_path, chain="A"):
    """Extract amino acid sequence and residue numbers from PDB ATOM records.

    Returns:
        residues: list of (resnum, resname_1letter) tuples, in PDB order
        sequence: one-letter amino acid string
    """
    seen = set()
    residues = []

    # Mapping for non-standard 3-letter codes
    three_to_one = {}
    for k, v in protein_letters_3to1.items():
        three_to_one[k.upper()] = v.upper()
    # Common extras
    three_to_one["MSE"] = "M"  # selenomethionine

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

    # Sort by residue number (should already be ordered but be safe)
    residues.sort(key=lambda x: x[0])
    sequence = "".join(aa for _, aa in residues)

    return residues, sequence


def build_residue_mapping(residues_a, seq_a, residues_b, seq_b, label_a="A", label_b="B"):
    """Align two sequences and build a residue number mapping.

    Args:
        residues_a: list of (resnum, aa) for structure A
        seq_a: one-letter sequence for structure A
        residues_b: list of (resnum, aa) for structure B
        seq_b: one-letter sequence for structure B

    Returns:
        a_to_b: dict mapping A resnums -> B resnums (only for aligned positions)
        b_to_a: dict mapping B resnums -> A resnums
        alignment_details: list of (resnum_a, aa_a, resnum_b, aa_b) for aligned positions
    """
    # Global alignment with standard BLOSUM-like scoring
    alignments = pairwise2.align.globalms(
        seq_a, seq_b,
        match=2, mismatch=-1, open=-5, extend=-0.5,
        one_alignment_only=True
    )

    if not alignments:
        print(f"ERROR: Could not align {label_a} and {label_b} sequences")
        sys.exit(1)

    aligned_a, aligned_b, score, begin, end = alignments[0]

    # Walk through the alignment and build the mapping
    a_to_b = {}
    b_to_a = {}
    alignment_details = []

    idx_a = 0  # index into residues_a
    idx_b = 0  # index into residues_b

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


def classify_position(boltz_pos):
    """Classify a Boltz position as pocket, gate_loop, or distal."""
    if boltz_pos in POCKET_POSITIONS_BOLTZ:
        return "pocket"
    if boltz_pos in GATE_LOOP_BOLTZ:
        return "gate_loop"
    return "distal"


def print_alignment_summary(alignment_details, boltz_to_3k3k):
    """Print a formatted summary of the alignment with difference highlighting."""
    # Find positions where sequences differ
    diffs = [(ra, aa, rb, ab) for ra, aa, rb, ab in alignment_details if aa != ab]

    print(f"\nAlignment summary: {len(alignment_details)} aligned positions, "
          f"{len(diffs)} differences")

    if diffs:
        print(f"\n{'3K3K_pos':>10} {'3K3K_aa':>8} {'Boltz_pos':>10} {'Boltz_aa':>9} "
              f"{'Category':>12} {'Concern':>10}")
        print("-" * 70)
        for resnum_3k3k, aa_3k3k, resnum_boltz, aa_boltz in diffs:
            cat = classify_position(resnum_boltz)
            if cat == "gate_loop":
                concern = "CRITICAL"
            elif cat == "pocket":
                concern = "Low*"  # will be mutated anyway
            else:
                concern = "Low"
            print(f"{resnum_3k3k:>10} {aa_3k3k:>8} {resnum_boltz:>10} {aa_boltz:>9} "
                  f"{cat:>12} {concern:>10}")

        gate_diffs = [d for d in diffs if classify_position(d[2]) == "gate_loop"]
        if gate_diffs:
            print(f"\n  WARNING: {len(gate_diffs)} difference(s) in the gate loop region!")
            print("  The gate loop has a completely different conformation in 3K3K (open)")
            print("  vs Boltz (closed), so sequence differences here are expected and OK")
            print("  as long as the backbone comes from 3K3K.")
    else:
        print("  Sequences are identical at all aligned positions.")

    # Report pocket positions mapping
    print(f"\nPocket position mapping (Boltz -> 3K3K):")
    for bp in POCKET_POSITIONS_BOLTZ:
        kp = boltz_to_3k3k.get(bp)
        if kp is not None:
            # Find the AAs at this position
            aa_3k3k = None
            aa_boltz = None
            for r3, a3, rb, ab in alignment_details:
                if rb == bp:
                    aa_3k3k = a3
                    aa_boltz = ab
                    break
            status = "OK" if aa_3k3k else "??"
            aa_info = f"(3K3K: {aa_3k3k}, Boltz: {aa_boltz})" if aa_3k3k else ""
            print(f"  Boltz {bp:>3} -> 3K3K {kp:>3}  {status}  {aa_info}")
        else:
            print(f"  Boltz {bp:>3} -> NOT FOUND IN 3K3K  *** PROBLEM ***")

    # Report anchor positions
    print(f"\nAnchor positions for ligand superposition (Boltz -> 3K3K):")
    missing_anchors = []
    for bp in ANCHOR_POSITIONS_BOLTZ:
        kp = boltz_to_3k3k.get(bp)
        if kp is not None:
            print(f"  Boltz {bp:>3} -> 3K3K {kp:>3}  OK")
        else:
            print(f"  Boltz {bp:>3} -> NOT FOUND  *** PROBLEM ***")
            missing_anchors.append(bp)

    if missing_anchors:
        print(f"\n  ERROR: {len(missing_anchors)} anchor positions missing from 3K3K!")
        print("  Ligand superposition may be unreliable.")


def main():
    parser = argparse.ArgumentParser(
        description="Align 3K3K and Boltz sequences, build residue mapping"
    )
    parser.add_argument("--pdb-3k3k", required=True, help="Path to 3K3K.pdb")
    parser.add_argument("--boltz-pdb", required=True,
                        help="Path to any Boltz prediction PDB (for reference sequence)")
    parser.add_argument("--chain-3k3k", default="A",
                        help="Chain ID in 3K3K (default: A = open-lid apo)")
    parser.add_argument("--chain-boltz", default="A",
                        help="Chain ID in Boltz PDB (default: A)")
    parser.add_argument("--output-json", default="alignment_map.json",
                        help="Output JSON file for residue mapping")
    args = parser.parse_args()

    # Extract sequences from PDB files
    print(f"Extracting sequence from 3K3K ({args.pdb_3k3k}, chain {args.chain_3k3k})...")
    res_3k3k, seq_3k3k = extract_sequence_from_pdb(args.pdb_3k3k, args.chain_3k3k)
    print(f"  {len(res_3k3k)} residues, range {res_3k3k[0][0]}-{res_3k3k[-1][0]}")

    print(f"Extracting sequence from Boltz PDB ({args.boltz_pdb}, chain {args.chain_boltz})...")
    res_boltz, seq_boltz = extract_sequence_from_pdb(args.boltz_pdb, args.chain_boltz)
    print(f"  {len(res_boltz)} residues, range {res_boltz[0][0]}-{res_boltz[-1][0]}")

    # Build the mapping
    print("\nAligning sequences...")
    k3k_to_boltz, boltz_to_3k3k, alignment_details = build_residue_mapping(
        res_3k3k, seq_3k3k, res_boltz, seq_boltz,
        label_a="3K3K", label_b="Boltz"
    )

    # Print summary
    print_alignment_summary(alignment_details, boltz_to_3k3k)

    # Convert pocket and anchor positions to 3K3K numbering
    pocket_3k3k = [boltz_to_3k3k[bp] for bp in POCKET_POSITIONS_BOLTZ
                   if bp in boltz_to_3k3k]
    anchor_3k3k = [boltz_to_3k3k[bp] for bp in ANCHOR_POSITIONS_BOLTZ
                   if bp in boltz_to_3k3k]
    gate_3k3k = [boltz_to_3k3k[bp] for bp in GATE_LOOP_BOLTZ
                 if bp in boltz_to_3k3k]

    # Build the 3K3K WT sequence at pocket positions (for comparison during threading)
    wt_3k3k_pocket = {}
    for bp in POCKET_POSITIONS_BOLTZ:
        kp = boltz_to_3k3k.get(bp)
        if kp is not None:
            for r3, a3, rb, ab in alignment_details:
                if rb == bp:
                    wt_3k3k_pocket[str(bp)] = a3
                    break

    # Save mapping as JSON (convert int keys to strings for JSON)
    output = {
        "boltz_to_3k3k": {str(k): v for k, v in boltz_to_3k3k.items()},
        "3k3k_to_boltz": {str(k): v for k, v in k3k_to_boltz.items()},
        "pocket_positions_boltz": POCKET_POSITIONS_BOLTZ,
        "pocket_positions_3k3k": pocket_3k3k,
        "gate_loop_boltz": GATE_LOOP_BOLTZ,
        "gate_loop_3k3k": gate_3k3k,
        "anchor_positions_boltz": ANCHOR_POSITIONS_BOLTZ,
        "anchor_positions_3k3k": anchor_3k3k,
        "wt_3k3k_at_pocket": wt_3k3k_pocket,
        "3k3k_sequence_range": [res_3k3k[0][0], res_3k3k[-1][0]],
        "boltz_sequence_range": [res_boltz[0][0], res_boltz[-1][0]],
        "3k3k_chain": args.chain_3k3k,
        "boltz_chain": args.chain_boltz,
        "n_aligned": len(alignment_details),
    }

    out_path = Path(args.output_json)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAlignment mapping saved to {out_path}")
    print(f"  {len(boltz_to_3k3k)} Boltz->3K3K mappings")
    print(f"  {len(pocket_3k3k)}/{len(POCKET_POSITIONS_BOLTZ)} pocket positions mapped")
    print(f"  {len(anchor_3k3k)}/{len(ANCHOR_POSITIONS_BOLTZ)} anchor positions mapped")


if __name__ == "__main__":
    main()
