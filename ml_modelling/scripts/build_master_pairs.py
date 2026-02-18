#!/usr/bin/env python3
"""
Build master_pairs.csv from 4 experimental data sources + 2 artificial negative generators.

Sources:
  1. ligand_smiles_signature.csv   - existing strong binders
  2. LCA_full_mutational_profile.csv - LCA mutational screen (binder/non-binder)
  3. pnas_data.csv + pnas_smiles.csv - PNAS Cutler library (Y2H tiers)
  4. affinity_data_win.csv          - WIN SSM (Kd values)
  5. Artificial swap negatives       - mismatched (ligand, sequence) pairs
  6. Artificial alanine scan negatives - key positions reverted to alanine

Output: ml_modelling/data/master_pairs.csv
"""

import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── WT residue identities at design positions (3QN1 PYR1) ──────────────────

WT_RESIDUES_16 = {
    59: 'K', 81: 'V', 83: 'V', 92: 'S', 94: 'E', 108: 'F',
    110: 'I', 117: 'L', 120: 'Y', 122: 'S', 141: 'E', 159: 'F',
    160: 'A', 163: 'V', 164: 'V', 167: 'N',
}

# PNAS data has two extra positions
WT_RESIDUES_18 = {**WT_RESIDUES_16, 87: 'L', 89: 'A'}

# WIN parent variant PYR1^WIN mutations (from WT)
WIN_PARENT_MUTATIONS = {59: 'Q', 141: 'I', 159: 'A', 160: 'I'}


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_signature(mutations_dict):
    """Build semicolon-delimited signature from {pos: aa} dict, sorted by position."""
    if not mutations_dict:
        return ""
    return ";".join(f"{pos}{aa}" for pos, aa in sorted(mutations_dict.items()))


def normalize_signature(sig_str):
    """Normalize any signature format to standard '59Q;81I;...' format.

    Handles:
      - Standard: '59Q;81I;...'
      - Dash format: 'V59-L81-A83-...' (AA before position, dash-separated)
    """
    if not sig_str or not sig_str.strip():
        return ""
    sig_str = sig_str.strip().rstrip("-")

    # Detect dash format: starts with a letter followed by digits then dash
    if re.match(r"[A-Z]\d+-", sig_str):
        mutations = {}
        for part in sig_str.split("-"):
            part = part.strip()
            if not part:
                continue
            match = re.match(r"([A-Z])(\d+)", part)
            if match:
                aa = match.group(1)
                pos = int(match.group(2))
                # Only include if different from WT
                wt = WT_RESIDUES_18.get(pos, "")
                if aa != wt:
                    mutations[pos] = aa
        return build_signature(mutations)

    # Standard semicolon format — already correct
    return sig_str


def parse_signature(sig_str):
    """Parse '59Q;81I;...' into {59: 'Q', 81: 'I', ...}."""
    if not sig_str or not sig_str.strip():
        return {}
    result = {}
    for part in sig_str.split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"(\d+)([A-Z])", part)
        if match:
            result[int(match.group(1))] = match.group(2)
    return result


def load_manual_smiles():
    """Load manual_smiles_lookup.csv into {name_lower: smiles}."""
    lookup = {}
    path = DATA_DIR / "manual_smiles_lookup.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["ligand_name"].strip()
            smiles = row["smiles"].strip()
            if name and smiles:
                lookup[name.lower()] = smiles
    return lookup


# ── Step 1: Existing strong binders ────────────────────────────────────────

def step1_existing_binders():
    """Convert ligand_smiles_signature.csv.

    NOTE: The source CSV has Excel auto-increment corruption on SMILES ring
    closure digits for ligands with multiple rows (Imperatorin, Methoxsalen,
    Osthole, WIN 55,212-2). We fix this by using one canonical SMILES per
    ligand_name: manual_smiles_lookup.csv takes priority, then the first
    occurrence in the CSV.
    """
    # Build canonical SMILES lookup: manual overrides > first occurrence
    manual = load_manual_smiles()
    canonical_smiles = {}  # ligand_name_lower -> smiles

    path = DATA_DIR / "ligand_smiles_signature.csv"

    # First pass: collect first SMILES per ligand
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["ligand_name"].strip()
            key = name.lower()
            if key not in canonical_smiles:
                canonical_smiles[key] = row["ligand_smiles_or_ligand_ID"].strip()

    # Override with manual lookup (authoritative, not corrupted)
    for name_lower, smiles in manual.items():
        if name_lower in canonical_smiles:
            canonical_smiles[name_lower] = smiles

    # Second pass: build pairs with canonical SMILES
    pairs = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ligand_name = row["ligand_name"].strip()
            smiles = canonical_smiles[ligand_name.lower()]
            variant_name = row["PYR1_variant_name"].strip()
            signature = normalize_signature(row["PYR1_variant_signature"])

            key = (smiles, signature)
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "ligand_name": ligand_name,
                "ligand_smiles": smiles,
                "variant_name": variant_name,
                "variant_signature": signature,
                "label": 1.0,
                "label_tier": "strong",
                "label_source": "experimental",
                "label_confidence": 1.0,
                "affinity_uM": "",
            })
    return pairs


# ── Step 2: LCA mutational screen ──────────────────────────────────────────

def step2_lca_screen():
    """Convert LCA_full_mutational_profile.csv."""
    manual = load_manual_smiles()
    lca_smiles = manual.get("lithocholic acid", "")
    if not lca_smiles:
        raise ValueError("Lithocholic Acid SMILES not found in manual_smiles_lookup.csv")

    pairs = []
    path = DATA_DIR / "LCA_full_mutational_profile.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant_name = row["des"].strip()
            binder = row.get("Binder", "").strip().lower()

            # Build mutation signature by comparing to WT
            mutations = {}
            for pos, wt_aa in WT_RESIDUES_16.items():
                col = f"res_{pos}"
                obs_aa = row.get(col, "").strip().upper()
                if obs_aa and obs_aa != wt_aa:
                    mutations[pos] = obs_aa

            signature = build_signature(mutations)

            if binder == "yes":
                label, tier = 1.0, "strong"
            else:
                label, tier = 0.0, "negative"

            pairs.append({
                "ligand_name": "Lithocholic_Acid",
                "ligand_smiles": lca_smiles,
                "variant_name": variant_name,
                "variant_signature": signature,
                "label": label,
                "label_tier": tier,
                "label_source": "LCA_screen",
                "label_confidence": 0.9,
                "affinity_uM": "",
            })
    return pairs


# ── Step 3: PNAS Cutler library ────────────────────────────────────────────

def load_pnas_smiles():
    """Build {name_lower: smiles} from pnas_smiles.csv."""
    lookup = {}
    path = DATA_DIR / "pnas_smiles.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Library_name", "").strip()
            smiles = row.get("canonical_smiles", "").strip()
            if name and smiles:
                lookup[name.lower()] = smiles
    # Hardcode the one missing compound
    lookup["deet pestanal"] = "CCN(CC)C(=O)c1cccc(C)c1"
    return lookup


def step3_pnas():
    """Convert pnas_data.csv using pnas_smiles.csv for SMILES lookup."""
    smiles_lookup = load_pnas_smiles()

    # PNAS mutation column headers → position number
    pnas_mut_cols = {
        "K59": 59, "V81": 81, "V83": 83, "L87": 87, "A89": 89,
        "S92": 92, "E94": 94, "F108": 108, "I110": 110, "L117": 117,
        "Y120": 120, "S122": 122, "E141": 141, "F159": 159, "A160": 160,
        "V163": 163, "V164": 164, "N167": 167,
    }

    tier_map = {
        "1": (0.75, "moderate"),
        "10": (0.25, "weak"),
        "100": (0.0, "negative"),
    }

    pairs = []
    path = DATA_DIR / "pnas_data.csv"
    row_idx = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ligand_name = row["library_name"].strip()
            min_conc = row.get("min_conc", "").strip()

            # Skip uninformative rows
            if min_conc not in tier_map:
                continue

            label, tier = tier_map[min_conc]

            # SMILES lookup
            smiles = smiles_lookup.get(ligand_name.lower(), "")
            if not smiles:
                print(f"  WARNING: No SMILES for PNAS ligand '{ligand_name}', skipping")
                continue

            # Build mutation signature
            mutations = {}
            for col_name, pos in pnas_mut_cols.items():
                mut_aa = row.get(col_name, "").strip().upper()
                if mut_aa:
                    mutations[pos] = mut_aa

            signature = build_signature(mutations)
            row_idx += 1
            variant_name = f"PNAS_{ligand_name}_{row_idx}"

            pairs.append({
                "ligand_name": ligand_name,
                "ligand_smiles": smiles,
                "variant_name": variant_name,
                "variant_signature": signature,
                "label": label,
                "label_tier": tier,
                "label_source": "pnas_cutler",
                "label_confidence": 0.7,
                "affinity_uM": "",
            })
    return pairs


# ── Step 4: WIN SSM ────────────────────────────────────────────────────────

def get_win_smiles():
    """Get WIN 55,212-2 SMILES from ligand_smiles_signature.csv."""
    path = DATA_DIR / "ligand_smiles_signature.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "WIN" in row["ligand_name"].upper():
                return row["ligand_smiles_or_ligand_ID"].strip()
    raise ValueError("WIN 55,212-2 SMILES not found")


def step4_win_ssm():
    """Convert affinity_data_win.csv."""
    win_smiles = get_win_smiles()

    pairs = []
    path = DATA_DIR / "affinity_data_win.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant_str = row["Variant"].strip()
            kd_str = row["Kd"].strip()

            if not kd_str:
                continue
            kd = float(kd_str)

            # Build signature: start with parent mutations, override the SSM position
            muts = dict(WIN_PARENT_MUTATIONS)

            if variant_str == "WT":
                # WT relative to PYR1^WIN parent = the parent itself
                pass
            else:
                # Parse e.g. "E141M" → pos=141, new_aa=M
                match = re.match(r"([A-Z])(\d+)([A-Z])", variant_str)
                if not match:
                    print(f"  WARNING: Cannot parse WIN variant '{variant_str}', skipping")
                    continue
                wt_aa, pos_str, new_aa = match.groups()
                pos = int(pos_str)

                if new_aa == WT_RESIDUES_16.get(pos, ""):
                    # Mutation reverts to WT → remove from signature
                    muts.pop(pos, None)
                else:
                    muts[pos] = new_aa

            signature = build_signature(muts)

            # Classify by Kd
            if kd < 500:
                label, tier = 1.0, "strong"
            elif kd < 2000:
                label, tier = 0.75, "moderate"
            elif kd < 10000:
                label, tier = 0.25, "weak"
            else:
                label, tier = 0.0, "negative"

            variant_name = f"PYR1^WIN_ssm_{variant_str}"

            pairs.append({
                "ligand_name": "WIN 55,212-2",
                "ligand_smiles": win_smiles,
                "variant_name": variant_name,
                "variant_signature": signature,
                "label": label,
                "label_tier": tier,
                "label_source": "win_ssm",
                "label_confidence": 0.95,
                "affinity_uM": f"{kd / 1000:.4f}",
            })
    return pairs


# ── Step 5: Artificial swap negatives ───────────────────────────────────────

def step5_swap_negatives(all_pairs):
    """Create mismatched (ligand, sequence) pairs from strong binders."""
    # Group strong binders by ligand SMILES
    binders_by_ligand = defaultdict(list)
    for p in all_pairs:
        if p["label"] == 1.0:
            binders_by_ligand[p["ligand_smiles"]].append(p)

    # Need at least 2 different ligands to do swaps
    ligand_keys = list(binders_by_ligand.keys())
    if len(ligand_keys) < 2:
        print("  WARNING: <2 ligands with strong binders, skipping swap negatives")
        return []

    existing_keys = {(p["ligand_smiles"], p["variant_signature"]) for p in all_pairs}
    swaps = []

    for lig_smiles, binders in binders_by_ligand.items():
        # Collect sequences from OTHER ligands
        other_seqs = []
        for other_lig, other_binders in binders_by_ligand.items():
            if other_lig != lig_smiles:
                other_seqs.extend(other_binders)

        if not other_seqs:
            continue

        # Sample up to 3 swaps per binder
        for binder in binders:
            candidates = [s for s in other_seqs
                          if (lig_smiles, s["variant_signature"]) not in existing_keys]
            sample_n = min(3, len(candidates))
            if sample_n == 0:
                continue

            for picked in random.sample(candidates, sample_n):
                key = (lig_smiles, picked["variant_signature"])
                if key in existing_keys:
                    continue
                existing_keys.add(key)

                swaps.append({
                    "ligand_name": binder["ligand_name"],
                    "ligand_smiles": lig_smiles,
                    "variant_name": picked["variant_name"] + "_swap",
                    "variant_signature": picked["variant_signature"],
                    "label": 0.0,
                    "label_tier": "negative",
                    "label_source": "artificial_swap",
                    "label_confidence": 0.6,
                    "affinity_uM": "",
                })

    return swaps


# ── Step 6: Artificial alanine scan negatives ───────────────────────────────

def step6_ala_scan_negatives(all_pairs):
    """Replace evolved positions with alanine to break binding."""
    existing_keys = {(p["ligand_smiles"], p["variant_signature"]) for p in all_pairs}
    ala_negs = []

    strong_binders = [p for p in all_pairs if p["label"] == 1.0]

    for binder in strong_binders:
        muts = parse_signature(binder["variant_signature"])
        if not muts:
            continue

        # Identify non-alanine mutated positions (candidates for Ala replacement)
        ala_candidates = [pos for pos, aa in muts.items() if aa != "A"]
        if not ala_candidates:
            continue

        # Generate 1-2 alanine-scanned variants
        n_variants = min(2, len(ala_candidates))
        for v_idx in range(n_variants):
            new_muts = dict(muts)

            if len(ala_candidates) >= 3:
                n_replace = random.randint(2, min(4, len(ala_candidates)))
            else:
                n_replace = len(ala_candidates)

            positions_to_ala = random.sample(ala_candidates, min(n_replace, len(ala_candidates)))

            for pos in positions_to_ala:
                wt_aa = WT_RESIDUES_18.get(pos, "")
                if wt_aa == "A":
                    # Reverting to WT which is already A → remove from signature
                    new_muts.pop(pos, None)
                else:
                    new_muts[pos] = "A"

            new_sig = build_signature(new_muts)
            key = (binder["ligand_smiles"], new_sig)
            if key in existing_keys:
                continue
            existing_keys.add(key)

            ala_negs.append({
                "ligand_name": binder["ligand_name"],
                "ligand_smiles": binder["ligand_smiles"],
                "variant_name": binder["variant_name"] + f"_ala{len(positions_to_ala)}",
                "variant_signature": new_sig,
                "label": 0.0,
                "label_tier": "negative",
                "label_source": "artificial_ala_scan",
                "label_confidence": 0.5,
                "affinity_uM": "",
            })

    return ala_negs


# ── Step 7: Merge, dedup, output ────────────────────────────────────────────

CONFIDENCE_RANK = {
    "experimental": 6,
    "win_ssm": 5,
    "LCA_screen": 4,
    "pnas_cutler": 3,
    "artificial_swap": 2,
    "artificial_ala_scan": 1,
}


FIELDNAMES = [
    "pair_id", "ligand_name", "ligand_smiles", "variant_name",
    "variant_signature", "label", "label_tier", "label_source",
    "label_confidence", "affinity_uM",
]

# Tier definitions: (tier_number, tier_name, filter_function)
# Ordered by recommended execution priority
TIER_DEFS = [
    (1, "strong_binders",    lambda p: p["label_source"] in ("experimental", "win_ssm") and p["label"] == 1.0),
    (2, "win_ssm_graded",    lambda p: p["label_source"] == "win_ssm" and p["label"] != 1.0),
    (3, "pnas_cutler",       lambda p: p["label_source"] == "pnas_cutler"),
    (4, "LCA_screen",        lambda p: p["label_source"] == "LCA_screen"),
    (5, "artificial",        lambda p: p["label_source"] in ("artificial_swap", "artificial_ala_scan")),
]


def write_csv(pairs, out_path):
    """Write pairs list to CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(pairs)


def merge_and_output(all_pairs, split_tiers=True):
    """Deduplicate, assign pair_ids, write master + tier CSVs."""
    # Dedup by (smiles, signature) — keep highest confidence
    best = {}
    for p in all_pairs:
        key = (p["ligand_smiles"], p["variant_signature"])
        rank = CONFIDENCE_RANK.get(p["label_source"], 0)
        if key not in best or rank > CONFIDENCE_RANK.get(best[key]["label_source"], 0):
            best[key] = p

    # Sort: by source, then ligand name
    deduped = sorted(best.values(), key=lambda p: (p["label_source"], p["ligand_name"]))

    # Assign pair_ids
    for i, p in enumerate(deduped, 1):
        p["pair_id"] = f"pair_{i:04d}"

    # Write master CSV
    write_csv(deduped, DATA_DIR / "master_pairs.csv")

    # Write per-tier CSVs
    if split_tiers:
        tier_dir = DATA_DIR / "tiers"
        tier_dir.mkdir(exist_ok=True)
        for tier_num, tier_name, tier_filter in TIER_DEFS:
            tier_pairs = [p for p in deduped if tier_filter(p)]
            if tier_pairs:
                tier_path = tier_dir / f"tier{tier_num}_{tier_name}.csv"
                write_csv(tier_pairs, tier_path)
                print(f"  Tier {tier_num} ({tier_name}): {len(tier_pairs)} pairs -> {tier_path.name}")

    return deduped


def print_summary(pairs):
    """Print source × tier count table."""
    counts = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        counts[p["label_source"]][p["label_tier"]] += 1

    tiers = ["strong", "moderate", "weak", "negative"]
    sources = ["experimental", "LCA_screen", "pnas_cutler", "win_ssm",
               "artificial_swap", "artificial_ala_scan"]

    print(f"\n{'Source':<25} {'Strong':>8} {'Moderate':>10} {'Weak':>8} {'Negative':>10} {'Total':>8}")
    print("-" * 75)
    totals = {t: 0 for t in tiers}
    grand = 0
    for src in sources:
        if src not in counts:
            continue
        row_total = sum(counts[src].values())
        grand += row_total
        vals = []
        for t in tiers:
            c = counts[src].get(t, 0)
            totals[t] += c
            vals.append(c)
        print(f"{src:<25} {vals[0]:>8} {vals[1]:>10} {vals[2]:>8} {vals[3]:>10} {row_total:>8}")

    print("-" * 75)
    print(f"{'TOTAL':<25} {totals['strong']:>8} {totals['moderate']:>10} {totals['weak']:>8} {totals['negative']:>10} {grand:>8}")


# ── Batch splitting ─────────────────────────────────────────────────────────

def split_tier_into_batches(tier_csv, batch_size, output_dir=None):
    """Split a tier CSV into batch CSVs of batch_size pairs each.

    Returns list of batch file paths.
    """
    tier_path = Path(tier_csv)
    if output_dir is None:
        output_dir = tier_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(tier_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_batches = math.ceil(len(rows) / batch_size)
    batch_paths = []
    stem = tier_path.stem

    for b in range(n_batches):
        batch_rows = rows[b * batch_size : (b + 1) * batch_size]
        batch_path = output_dir / f"{stem}_batch{b + 1:02d}.csv"
        write_csv(batch_rows, batch_path)
        batch_paths.append(batch_path)

    return batch_paths


# ── Main ────────────────────────────────────────────────────────────────────

def build_all():
    """Run full pipeline: generate all pairs, merge, split into tiers."""
    print("Step 1: Existing strong binders...")
    step1 = step1_existing_binders()
    print(f"  {len(step1)} pairs")

    print("Step 2: LCA mutational screen...")
    step2 = step2_lca_screen()
    print(f"  {len(step2)} pairs")

    print("Step 3: PNAS Cutler library...")
    step3 = step3_pnas()
    print(f"  {len(step3)} pairs")

    print("Step 4: WIN SSM...")
    step4 = step4_win_ssm()
    print(f"  {len(step4)} pairs")

    experimental = step1 + step2 + step3 + step4
    print(f"\nExperimental total: {len(experimental)}")

    print("\nStep 5: Artificial swap negatives...")
    step5 = step5_swap_negatives(experimental)
    print(f"  {len(step5)} pairs")

    print("Step 6: Artificial alanine scan negatives...")
    step6 = step6_ala_scan_negatives(experimental)
    print(f"  {len(step6)} pairs")

    all_pairs = experimental + step5 + step6

    print("\nStep 7: Merge, dedup, output...")
    final = merge_and_output(all_pairs, split_tiers=True)
    print(f"  {len(final)} pairs after dedup")

    print_summary(final)
    print(f"\nOutput: {DATA_DIR / 'master_pairs.csv'}")
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Build master_pairs.csv from experimental data + artificial negatives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full rebuild (default)
  python build_master_pairs.py

  # Split tier 1 into batches of 50 for SLURM
  python build_master_pairs.py --split-batches tiers/tier1_strong_binders.csv --batch-size 50

  # Append a new CSV of pairs to the master (preserves existing pair_ids)
  python build_master_pairs.py --append new_pairs.csv

  # Regenerate tier CSVs from existing master_pairs.csv (no rebuild)
  python build_master_pairs.py --resplit-tiers
        """,
    )
    parser.add_argument(
        "--split-batches", metavar="TIER_CSV",
        help="Split a tier CSV into batch CSVs of --batch-size pairs each",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Pairs per batch when splitting (default: 50)",
    )
    parser.add_argument(
        "--append", metavar="CSV_FILE",
        help="Append new pairs from a CSV to master_pairs.csv (must have same columns)",
    )
    parser.add_argument(
        "--resplit-tiers", action="store_true",
        help="Regenerate tier CSVs from existing master_pairs.csv without rebuilding",
    )
    args = parser.parse_args()

    if args.split_batches:
        # Split a specific tier CSV into batches
        tier_csv = Path(args.split_batches)
        if not tier_csv.is_absolute():
            tier_csv = DATA_DIR / tier_csv
        batch_paths = split_tier_into_batches(tier_csv, args.batch_size)
        print(f"Split {tier_csv.name} into {len(batch_paths)} batches of <={args.batch_size}:")
        for bp in batch_paths:
            with open(bp) as f:
                n = sum(1 for _ in f) - 1  # minus header
            print(f"  {bp.name}: {n} pairs")

    elif args.append:
        # Append new pairs to existing master
        master_path = DATA_DIR / "master_pairs.csv"
        if not master_path.exists():
            print(f"ERROR: {master_path} does not exist. Run without --append first.")
            return

        with open(master_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing_keys = {(p["ligand_smiles"], p["variant_signature"]) for p in existing}
        max_id = max(int(p["pair_id"].split("_")[1]) for p in existing)

        append_path = Path(args.append)
        if not append_path.is_absolute():
            append_path = DATA_DIR / append_path
        with open(append_path, newline="", encoding="utf-8") as f:
            new_rows = list(csv.DictReader(f))

        added = 0
        for row in new_rows:
            key = (row.get("ligand_smiles", ""), row.get("variant_signature", ""))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            max_id += 1
            row["pair_id"] = f"pair_{max_id:04d}"
            existing.append(row)
            added += 1

        write_csv(existing, master_path)
        print(f"Appended {added} new pairs (skipped {len(new_rows) - added} duplicates)")
        print(f"Master now has {len(existing)} pairs")

        # Re-split tiers
        resplit_tiers_from_master(existing)

    elif args.resplit_tiers:
        master_path = DATA_DIR / "master_pairs.csv"
        with open(master_path, newline="", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))
        # Convert label to float for tier filters
        for p in pairs:
            p["label"] = float(p["label"])
        resplit_tiers_from_master(pairs)

    else:
        build_all()


def resplit_tiers_from_master(pairs):
    """Regenerate tier CSVs from a list of pair dicts."""
    tier_dir = DATA_DIR / "tiers"
    tier_dir.mkdir(exist_ok=True)
    for tier_num, tier_name, tier_filter in TIER_DEFS:
        tier_pairs = [p for p in pairs if tier_filter(p)]
        if tier_pairs:
            tier_path = tier_dir / f"tier{tier_num}_{tier_name}.csv"
            write_csv(tier_pairs, tier_path)
            print(f"  Tier {tier_num} ({tier_name}): {len(tier_pairs)} pairs")


if __name__ == "__main__":
    main()
