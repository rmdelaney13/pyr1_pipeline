#!/usr/bin/env python3
"""
Create Boltz prediction CSVs for GlycoLCA and LCA-3-S using:
  - Binders from tier1_strong_binders.csv (experimentally confirmed)
  - Non-binders from tier4_LCA_screen.csv (same library screen → also non-binders for conjugates)

Cross-references variant_signatures to ensure no binder appears in the non-binder set.

Usage:
    python prepare_boltz_lca_conjugates.py \
        --tier1 ml_modelling/data/tiers/tier1_strong_binders.csv \
        --tier4 ml_modelling/data/tiers/tier4_LCA_screen.csv \
        --out-dir ml_modelling/data/boltz_lca_conjugates \
        --n-nonbinders 500 --seed 42
"""

import argparse
import csv
import random
import sys
from pathlib import Path

# SMILES for each ligand
SMILES = {
    "GlycoLithocholic Acid": "C[C@H](CCC(=O)NCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@H](C4)O)C)C",
    "Lithocholic Acid 3 -S": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4[C@@]3(CC[C@H](C4)OS(=O)(=O)O)C)C",
    "Lithocholic Acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CCC[C@H]4O)C)C",
}

FIELDNAMES = [
    "pair_id", "ligand_name", "ligand_smiles", "variant_name",
    "variant_signature", "label", "label_tier", "label_source",
    "label_confidence", "affinity_uM",
]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def normalize_sig(sig):
    """Normalize a variant signature for comparison."""
    if not sig or sig.strip() in ("", "nan", "None"):
        return ""
    parts = sorted(sig.replace("_", ";").split(";"))
    return ";".join(p.strip() for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(description="Create Boltz CSVs for LCA conjugates")
    parser.add_argument("--tier1", required=True, help="Path to tier1_strong_binders.csv")
    parser.add_argument("--tier4", required=True, help="Path to tier4_LCA_screen.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for new CSVs")
    parser.add_argument("--n-nonbinders", type=int, default=500,
                        help="Number of non-binders to sample (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read tier files ──
    tier1 = read_csv(args.tier1)
    tier4 = read_csv(args.tier4)

    print(f"Tier1: {len(tier1)} rows")
    print(f"Tier4: {len(tier4)} rows")

    # ── Extract binders from tier1 ──
    glca_binders = [r for r in tier1 if r["ligand_name"] == "GlycoLithocholic Acid"]
    lca3s_binders = [r for r in tier1 if r["ligand_name"] == "Lithocholic Acid 3 -S"]
    lca_binders = [r for r in tier1
                   if r["ligand_name"] == "Lithocholic Acid"
                   or (r["ligand_name"].startswith("Lithocholic Acid")
                       and "Glyco" not in r["ligand_name"]
                       and "3 -S" not in r["ligand_name"])]
    # Filter to only plain LCA
    lca_binders = [r for r in tier1
                   if r.get("ligand_name", "").strip() == "Lithocholic Acid"]

    print(f"\nTier1 binders:")
    print(f"  Plain LCA:  {len(lca_binders)}")
    print(f"  GlycoLCA:   {len(glca_binders)}")
    print(f"  LCA-3-S:    {len(lca3s_binders)}")

    # ── Extract non-binders from tier4 ──
    tier4_nonbinders = [r for r in tier4 if float(r["label"]) == 0.0]
    tier4_binders = [r for r in tier4 if float(r["label"]) == 1.0]
    print(f"\nTier4: {len(tier4_binders)} binders, {len(tier4_nonbinders)} non-binders")

    # Sample non-binders (same seed as original LCA predictions)
    random.seed(args.seed)
    if len(tier4_nonbinders) > args.n_nonbinders:
        nonbinders = random.sample(tier4_nonbinders, args.n_nonbinders)
        print(f"Sampled {args.n_nonbinders} non-binders (seed={args.seed})")
    else:
        nonbinders = tier4_nonbinders
        print(f"Using all {len(nonbinders)} non-binders (fewer than requested)")

    # ── Cross-reference: ensure no binder signature in non-binder set ──
    nonbinder_sigs = {normalize_sig(r["variant_signature"]) for r in nonbinders}

    for label, binders in [("GlycoLCA", glca_binders), ("LCA-3-S", lca3s_binders)]:
        binder_sigs = {normalize_sig(r["variant_signature"]) for r in binders}
        overlap = binder_sigs & nonbinder_sigs
        if overlap:
            print(f"\n  WARNING: {len(overlap)} {label} binder signatures found in non-binder set!")
            for sig in sorted(overlap):
                print(f"    - {sig}")
            # Remove overlapping non-binders
            nonbinders_clean = [r for r in nonbinders
                                if normalize_sig(r["variant_signature"]) not in overlap]
            print(f"    Removed {len(nonbinders) - len(nonbinders_clean)} overlapping non-binders")
        else:
            print(f"\n  OK: No {label} binder signatures found in non-binder set")

    # Also check LCA binders vs non-binders (for completeness)
    lca_binder_sigs = {normalize_sig(r["variant_signature"]) for r in lca_binders}
    lca_overlap = lca_binder_sigs & nonbinder_sigs
    if lca_overlap:
        print(f"\n  WARNING: {len(lca_overlap)} LCA binder signatures in non-binder set!")
        for sig in sorted(lca_overlap):
            print(f"    - {sig}")

    # Collect ALL binder signatures to remove from non-binder pool
    all_binder_sigs = set()
    for binders in [glca_binders, lca3s_binders, lca_binders, tier4_binders]:
        for r in binders:
            all_binder_sigs.add(normalize_sig(r["variant_signature"]))

    nonbinders_final = [r for r in nonbinders
                        if normalize_sig(r["variant_signature"]) not in all_binder_sigs]

    removed = len(nonbinders) - len(nonbinders_final)
    if removed > 0:
        print(f"\n  Removed {removed} non-binders whose signatures match ANY binder set")
    print(f"  Final non-binder count: {len(nonbinders_final)}")

    # ── Write GLCA CSV ──
    glca_rows = []
    # Binders (keep original pair_id)
    for r in glca_binders:
        glca_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "GlycoLithocholic Acid",
            "ligand_smiles": SMILES["GlycoLithocholic Acid"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "1.0",
            "label_tier": "strong",
            "label_source": r.get("label_source", "experimental"),
            "label_confidence": r.get("label_confidence", "1.0"),
            "affinity_uM": r.get("affinity_uM", ""),
        })
    # Non-binders (override ligand SMILES to GLCA)
    for r in nonbinders_final:
        glca_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "GlycoLithocholic Acid",
            "ligand_smiles": SMILES["GlycoLithocholic Acid"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "0.0",
            "label_tier": "negative",
            "label_source": "LCA_screen",
            "label_confidence": r.get("label_confidence", "0.9"),
            "affinity_uM": "",
        })

    glca_path = out_dir / "boltz_glca_binary.csv"
    with open(glca_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(glca_rows)
    n_b = sum(1 for r in glca_rows if r["label"] == "1.0")
    n_nb = sum(1 for r in glca_rows if r["label"] == "0.0")
    print(f"\nWrote {glca_path}: {len(glca_rows)} rows ({n_b} binders + {n_nb} non-binders)")

    # ── Write LCA CSV (same non-binders, plain LCA SMILES) ──
    lca_rows = []
    for r in lca_binders:
        lca_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "Lithocholic Acid",
            "ligand_smiles": SMILES["Lithocholic Acid"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "1.0",
            "label_tier": "strong",
            "label_source": r.get("label_source", "experimental"),
            "label_confidence": r.get("label_confidence", "1.0"),
            "affinity_uM": r.get("affinity_uM", ""),
        })
    for r in nonbinders_final:
        lca_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "Lithocholic Acid",
            "ligand_smiles": SMILES["Lithocholic Acid"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "0.0",
            "label_tier": "negative",
            "label_source": "LCA_screen",
            "label_confidence": r.get("label_confidence", "0.9"),
            "affinity_uM": "",
        })

    lca_path = out_dir / "boltz_lca_binary.csv"
    with open(lca_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(lca_rows)
    n_b = sum(1 for r in lca_rows if r["label"] == "1.0")
    n_nb = sum(1 for r in lca_rows if r["label"] == "0.0")
    print(f"\nWrote {lca_path}: {len(lca_rows)} rows ({n_b} binders + {n_nb} non-binders)")

    # ── Write LCA-3-S CSV ──
    lca3s_rows = []
    for r in lca3s_binders:
        lca3s_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "Lithocholic Acid 3 -S",
            "ligand_smiles": SMILES["Lithocholic Acid 3 -S"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "1.0",
            "label_tier": "strong",
            "label_source": r.get("label_source", "experimental"),
            "label_confidence": r.get("label_confidence", "1.0"),
            "affinity_uM": r.get("affinity_uM", ""),
        })
    for r in nonbinders_final:
        lca3s_rows.append({
            "pair_id": r["pair_id"],
            "ligand_name": "Lithocholic Acid 3 -S",
            "ligand_smiles": SMILES["Lithocholic Acid 3 -S"],
            "variant_name": r.get("variant_name", ""),
            "variant_signature": r["variant_signature"],
            "label": "0.0",
            "label_tier": "negative",
            "label_source": "LCA_screen",
            "label_confidence": r.get("label_confidence", "0.9"),
            "affinity_uM": "",
        })

    lca3s_path = out_dir / "boltz_lca3s_binary.csv"
    with open(lca3s_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(lca3s_rows)
    n_b = sum(1 for r in lca3s_rows if r["label"] == "1.0")
    n_nb = sum(1 for r in lca3s_rows if r["label"] == "0.0")
    print(f"Wrote {lca3s_path}: {len(lca3s_rows)} rows ({n_b} binders + {n_nb} non-binders)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"LCA:       {len(lca_rows)} predictions ({sum(1 for r in lca_rows if r['label']=='1.0')} binders)")
    print(f"GlycoLCA:  {len(glca_rows)} predictions ({sum(1 for r in glca_rows if r['label']=='1.0')} binders)")
    print(f"LCA-3-S:   {len(lca3s_rows)} predictions ({sum(1 for r in lca3s_rows if r['label']=='1.0')} binders)")
    print(f"Shared non-binders: {len(nonbinders_final)}")
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
