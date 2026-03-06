#!/usr/bin/env python3
"""Generate double and triple deleterious mutants from nitazene SSM data.

Picks strongly deleterious single mutations (enrichment < -4) at different
positions, combines them into multi-site variants, and outputs a Boltz2-
compatible CSV for binary predictions.

Assumption: combining individually deleterious mutations produces non-binders.

Usage:
    python ml_modelling/scripts/generate_nitazene_multimutants.py
"""

import csv
import random
from itertools import combinations
from pathlib import Path

# Must match parse_nitazene_ssm.py
WT_PYR1 = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

BASE_POCKET = {
    59: "Q", 81: "V", 83: "V", 92: "M", 94: "E", 108: "V",
    110: "I", 117: "L", 120: "A", 122: "G", 141: "E", 159: "A",
    160: "V", 163: "V", 164: "V", 167: "N",
}

SMILES = {
    "nitazene": "CCN(CC)CCN1C2=C(C=C(C=C2)[N+](=O)[O-])N=C1CC3=CC=CC=C3",
    "menitazene": "CCN(CC)CCN1C2=C(C=C(C=C2)[N+](=O)[O-])N=C1CC3=CC=C(C=C3)OC",
}

ENRICHMENT_CUTOFF = -4.0  # strong non-binders only
N_DOUBLES = 20
N_TRIPLES = 20
N_QUADS = 20
SEED = 42


def make_base_sequence():
    seq = list(WT_PYR1)
    for pos, aa in BASE_POCKET.items():
        seq[pos - 1] = aa
    return "".join(seq)


def make_signature(mutations: dict) -> str:
    """Full 16-position pocket signature with specified mutations."""
    sig = dict(BASE_POCKET)
    sig.update(mutations)
    return ";".join(f"{p}{sig[p]}" for p in sorted(sig))


def make_sequence(base_seq: str, mutations: dict) -> str:
    seq = list(base_seq)
    for pos, aa in mutations.items():
        seq[pos - 1] = aa
    return "".join(seq)


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    # Load SSM data — collect strong non-binders per position
    strong_negatives = {}  # {position: [(aa, enrichment), ...]}
    with open(data_dir / "nitazene_ssm_labeled.csv") as f:
        for row in csv.DictReader(f):
            pos = int(row["position"])
            aa = row["amino_acid"]
            is_base = row["is_base"] == "True"
            if is_base:
                continue
            # Use nitazene enrichment (primary ligand)
            enr = row["enrichment_nitazene"]
            if not enr:
                continue
            enr = float(enr)
            if enr < ENRICHMENT_CUTOFF:
                strong_negatives.setdefault(pos, []).append((aa, enr))

    # Sort each position by enrichment (most deleterious first)
    for pos in strong_negatives:
        strong_negatives[pos].sort(key=lambda x: x[1])

    print(f"Strong non-binders (enrichment < {ENRICHMENT_CUTOFF}):")
    for pos in sorted(strong_negatives):
        aas = [f"{aa}({enr:.1f})" for aa, enr in strong_negatives[pos]]
        print(f"  {pos}: {', '.join(aas)}")

    positions = sorted(strong_negatives.keys())
    print(f"\n{len(positions)} positions with strong non-binders")

    base_seq = make_base_sequence()
    random.seed(SEED)

    # Generate double mutants
    doubles = []
    pos_pairs = list(combinations(positions, 2))
    random.shuffle(pos_pairs)
    for p1, p2 in pos_pairs:
        if len(doubles) >= N_DOUBLES:
            break
        aa1 = random.choice(strong_negatives[p1])[0]
        aa2 = random.choice(strong_negatives[p2])[0]
        mutations = {p1: aa1, p2: aa2}
        name_suffix = f"{p1}{aa1}_{p2}{aa2}"
        doubles.append((name_suffix, mutations))

    # Generate triple mutants
    triples = []
    pos_trips = list(combinations(positions, 3))
    random.shuffle(pos_trips)
    for p1, p2, p3 in pos_trips:
        if len(triples) >= N_TRIPLES:
            break
        aa1 = random.choice(strong_negatives[p1])[0]
        aa2 = random.choice(strong_negatives[p2])[0]
        aa3 = random.choice(strong_negatives[p3])[0]
        mutations = {p1: aa1, p2: aa2, p3: aa3}
        name_suffix = f"{p1}{aa1}_{p2}{aa2}_{p3}{aa3}"
        triples.append((name_suffix, mutations))

    # Generate quadruple mutants
    quads = []
    pos_quads = list(combinations(positions, 4))
    random.shuffle(pos_quads)
    for p1, p2, p3, p4 in pos_quads:
        if len(quads) >= N_QUADS:
            break
        aa1 = random.choice(strong_negatives[p1])[0]
        aa2 = random.choice(strong_negatives[p2])[0]
        aa3 = random.choice(strong_negatives[p3])[0]
        aa4 = random.choice(strong_negatives[p4])[0]
        mutations = {p1: aa1, p2: aa2, p3: aa3, p4: aa4}
        name_suffix = f"{p1}{aa1}_{p2}{aa2}_{p3}{aa3}_{p4}{aa4}"
        quads.append((name_suffix, mutations))

    print(f"\nGenerated {len(doubles)} doubles, {len(triples)} triples, {len(quads)} quads")

    # Write Boltz2-compatible CSV (nitazene only for now)
    output_csv = data_dir / "nitazene_multimutant_boltz.csv"
    rows = []

    # Add base as positive control
    rows.append({
        "name": "nitazene_base",
        "variant_signature": make_signature({}),
        "ligand_smiles": SMILES["nitazene"],
        "ligand_name": "nitazene",
        "label": 1,
        "mutant_type": "base",
    })

    # Add single-site binders as positive controls (enrichment > -1)
    with open(data_dir / "nitazene_ssm_labeled.csv") as f:
        for row in csv.DictReader(f):
            if row["label_nitazene"] == "binder":
                pos = int(row["position"])
                aa = row["amino_acid"]
                rows.append({
                    "name": f"nitazene_ssm_{pos}_{aa}",
                    "variant_signature": row["variant_signature"],
                    "ligand_smiles": SMILES["nitazene"],
                    "ligand_name": "nitazene",
                    "label": 1,
                    "mutant_type": "single_binder",
                })

    # Add double mutants as negatives
    for name_suffix, mutations in doubles:
        rows.append({
            "name": f"nitazene_dbl_{name_suffix}",
            "variant_signature": make_signature(mutations),
            "ligand_smiles": SMILES["nitazene"],
            "ligand_name": "nitazene",
            "label": 0,
            "mutant_type": "double_negative",
        })

    # Add triple mutants as negatives
    for name_suffix, mutations in triples:
        rows.append({
            "name": f"nitazene_trp_{name_suffix}",
            "variant_signature": make_signature(mutations),
            "ligand_smiles": SMILES["nitazene"],
            "ligand_name": "nitazene",
            "label": 0,
            "mutant_type": "triple_negative",
        })

    # Add quadruple mutants as negatives
    for name_suffix, mutations in quads:
        rows.append({
            "name": f"nitazene_qud_{name_suffix}",
            "variant_signature": make_signature(mutations),
            "ligand_smiles": SMILES["nitazene"],
            "ligand_name": "nitazene",
            "label": 0,
            "mutant_type": "quad_negative",
        })

    fields = ["name", "variant_signature", "ligand_smiles",
              "ligand_name", "label", "mutant_type"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"\nWrote {len(rows)} rows to {output_csv.name}")
    print(f"  Positives: {n_pos} (base + single binders)")
    print(f"  Negatives: {n_neg} (doubles + triples)")
    print(f"\nSample doubles:")
    for name_suffix, mutations in doubles[:5]:
        muts = ", ".join(f"{BASE_POCKET[p]}{p}{aa}" for p, aa in sorted(mutations.items()))
        print(f"  {muts}")
    print(f"\nSample triples:")
    for name_suffix, mutations in triples[:5]:
        muts = ", ".join(f"{BASE_POCKET[p]}{p}{aa}" for p, aa in sorted(mutations.items()))
        print(f"  {muts}")
    print(f"\nSample quads:")
    for name_suffix, mutations in quads[:5]:
        muts = ", ".join(f"{BASE_POCKET[p]}{p}{aa}" for p, aa in sorted(mutations.items()))
        print(f"  {muts}")


if __name__ == "__main__":
    main()
