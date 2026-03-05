#!/usr/bin/env python3
"""Parse nitazene SSM enrichment data and generate Boltz2-compatible input CSV.

Input:  ml_modelling/data/nitazene_ssm_data.csv
Output: ml_modelling/data/nitazene_ssm_labeled.csv  (analysis-friendly, both ligands per row)
        ml_modelling/data/nitazene_boltz_binary.csv  (Boltz2-compatible, one row per prediction)
        ml_modelling/data/nitazene_ssm_fastas/       (one FASTA per unique sequence)

Label scheme (enrichment = Log2(Variant/Ref) - Log2(WT_variant/WT_Ref)):
  binder:     enrichment > -1
  non-binder: enrichment < -3
  ambiguous:  -3 <= enrichment <= -1
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WT PYR1 sequence (181 residues, 3QN1 stabilized construct)
# Must match scripts/prepare_boltz_yamls.py WT_PYR1_SEQUENCE
WT_PYR1 = (
    "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSV"
    "EQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRI"
    "WTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
)

# Nitazene base pocket AAs (Boltz numbering, 1-indexed on 181-AA sequence)
BASE_POCKET = {
    59: "Q", 81: "V", 83: "V", 92: "M", 94: "E", 108: "V",
    110: "I", 117: "L", 120: "A", 122: "G", 141: "E", 159: "A",
    160: "V", 163: "V", 164: "V", 167: "N",
}

SMILES = {
    "nitazene":    "CCN(CC)CCN1C2=C(C=C(C=C2)[N+](=O)[O-])N=C1CC3=CC=CC=C3",
    "menitazene":  "CCN(CC)CCN1C2=C(C=C(C=C2)[N+](=O)[O-])N=C1CC3=CC=C(C=C3)OC",
}

BINDER_THRESHOLD = -1.0
NONBINDER_THRESHOLD = -3.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_base_sequence() -> str:
    """Apply base pocket mutations to PYR1 WT."""
    seq = list(WT_PYR1)
    for pos, aa in BASE_POCKET.items():
        seq[pos - 1] = aa
    return "".join(seq)


def make_variant_signature(pos: int, aa: str) -> str:
    """Full 16-position pocket signature with one substitution."""
    sig = dict(BASE_POCKET)
    sig[pos] = aa
    return ";".join(f"{p}{sig[p]}" for p in sorted(sig))


def assign_label(enrichment):
    """Classify enrichment into binder / non-binder / ambiguous / missing."""
    if enrichment is None:
        return "missing"
    if enrichment > BINDER_THRESHOLD:
        return "binder"
    if enrichment < NONBINDER_THRESHOLD:
        return "non-binder"
    return "ambiguous"


def parse_enrichment(val: str):
    """Parse enrichment value; return None for missing data."""
    val = val.strip()
    if val in ("#N/A", "", "NA", "N/A"):
        return None
    return float(val)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    input_csv = data_dir / "nitazene_ssm_data.csv"
    output_labeled = data_dir / "nitazene_ssm_labeled.csv"
    output_boltz = data_dir / "nitazene_boltz_binary.csv"
    fasta_dir = data_dir / "nitazene_ssm_fastas"
    fasta_dir.mkdir(exist_ok=True)

    base_seq = make_base_sequence()
    assert len(base_seq) == 181, f"Expected 181 AA, got {len(base_seq)}"

    # Verify base pocket residues
    for pos, aa in BASE_POCKET.items():
        actual = base_seq[pos - 1]
        assert actual == aa, f"Position {pos}: expected {aa}, got {actual}"

    print(f"PYR1 WT length:  {len(WT_PYR1)} AA")
    print(f"Base sequence:   {base_seq[:20]}...{base_seq[-10:]}")
    print(f"Mutations from WT: ", end="")
    wt_muts = []
    for pos in sorted(BASE_POCKET):
        wt_aa = WT_PYR1[pos - 1]
        base_aa = BASE_POCKET[pos]
        if wt_aa != base_aa:
            wt_muts.append(f"{wt_aa}{pos}{base_aa}")
    print(", ".join(wt_muts) if wt_muts else "(none)")

    # ------------------------------------------------------------------
    # Parse input CSV
    # ------------------------------------------------------------------
    rows = []
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # row 1: ,,Nitazene,Menitazene
        next(reader)  # row 2: Position,AminoAcid,enrichment_col,...
        for row in reader:
            if len(row) < 3 or not row[0].strip():
                continue
            pos = int(row[0].strip())
            aa = row[1].strip().upper()
            enr_nit = parse_enrichment(row[2]) if len(row) > 2 else None
            enr_men = parse_enrichment(row[3]) if len(row) > 3 else None
            rows.append((pos, aa, enr_nit, enr_men))

    print(f"\nParsed {len(rows)} rows from {input_csv.name}")

    # Verify base AAs from 0.00 enrichment rows
    detected_base = {}
    for pos, aa, enr_nit, enr_men in rows:
        if enr_nit == 0.0 or enr_men == 0.0:
            detected_base[pos] = aa

    mismatches = 0
    for pos in sorted(BASE_POCKET):
        expected = BASE_POCKET[pos]
        detected = detected_base.get(pos, "?")
        ok = "OK" if detected == expected else "MISMATCH"
        if detected != expected:
            mismatches += 1
        print(f"  {pos:>3d}: detected={detected} expected={expected} {ok}")

    if mismatches:
        print(f"WARNING: {mismatches} base AA mismatches!", file=sys.stderr)

    # ------------------------------------------------------------------
    # Generate labeled CSV (analysis-friendly: one row per variant)
    # ------------------------------------------------------------------
    labeled_rows = []
    sequences_written = set()
    stats = {lig: {"binder": 0, "non-binder": 0, "ambiguous": 0, "missing": 0}
             for lig in ("nitazene", "menitazene")}

    for pos, aa, enr_nit, enr_men in rows:
        is_base = (aa == BASE_POCKET.get(pos))

        # Build variant sequence
        variant_seq = list(base_seq)
        variant_seq[pos - 1] = aa
        variant_seq = "".join(variant_seq)

        # Variant signature (all 16 pocket positions)
        sig = make_variant_signature(pos, aa)

        # Mutation descriptor
        if is_base:
            mutation = f"{pos}{aa}(base)"
        else:
            mutation = f"{BASE_POCKET[pos]}{pos}{aa}"

        # Labels
        label_nit = assign_label(enr_nit)
        label_men = assign_label(enr_men)
        stats["nitazene"][label_nit] += 1
        stats["menitazene"][label_men] += 1

        labeled_rows.append({
            "position": pos,
            "amino_acid": aa,
            "mutation": mutation,
            "is_base": is_base,
            "enrichment_nitazene": enr_nit if enr_nit is not None else "",
            "enrichment_menitazene": enr_men if enr_men is not None else "",
            "label_nitazene": label_nit,
            "label_menitazene": label_men,
            "variant_signature": sig,
            "full_sequence": variant_seq,
        })

        # Write FASTA (one per unique sequence)
        seq_key = f"{pos}_{aa}"
        if seq_key not in sequences_written:
            fasta_path = fasta_dir / f"{seq_key}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">PYR1_{mutation}\n{variant_seq}\n")
            sequences_written.add(seq_key)

    # Write labeled CSV
    labeled_fields = [
        "position", "amino_acid", "mutation", "is_base",
        "enrichment_nitazene", "enrichment_menitazene",
        "label_nitazene", "label_menitazene",
        "variant_signature", "full_sequence",
    ]
    with open(output_labeled, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=labeled_fields)
        writer.writeheader()
        writer.writerows(labeled_rows)

    print(f"\nWrote {len(labeled_rows)} rows to {output_labeled.name}")
    print(f"Wrote {len(sequences_written)} FASTAs to {fasta_dir.name}/")

    # ------------------------------------------------------------------
    # Generate Boltz2-compatible CSV (one row per prediction)
    #   Columns: name, variant_signature, ligand_smiles, ligand_name, label
    #   Two rows per variant (one per ligand), skipping ambiguous/missing
    # ------------------------------------------------------------------
    boltz_rows = []
    for row in labeled_rows:
        sig = row["variant_signature"]
        for lig_key, label_col in [("nitazene", "label_nitazene"),
                                    ("menitazene", "label_menitazene")]:
            label = row[label_col]
            if label in ("ambiguous", "missing"):
                continue
            name = f"{lig_key}_{row['position']}_{row['amino_acid']}"
            boltz_rows.append({
                "name": name,
                "variant_signature": sig,
                "ligand_smiles": SMILES[lig_key],
                "ligand_name": lig_key,
                "label": 1 if label == "binder" else 0,
                "enrichment": (row["enrichment_nitazene"] if lig_key == "nitazene"
                               else row["enrichment_menitazene"]),
            })

    boltz_fields = ["name", "variant_signature", "ligand_smiles",
                    "ligand_name", "label", "enrichment"]
    with open(output_boltz, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=boltz_fields)
        writer.writeheader()
        writer.writerows(boltz_rows)

    print(f"Wrote {len(boltz_rows)} rows to {output_boltz.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    for lig in ("nitazene", "menitazene"):
        s = stats[lig]
        total_labeled = s["binder"] + s["non-binder"]
        print(f"\n{lig.upper()} (base Kd = {'250' if lig == 'nitazene' else '383'} nM):")
        print(f"  Binder (>{BINDER_THRESHOLD}):      {s['binder']:>3d}")
        print(f"  Non-binder (<{NONBINDER_THRESHOLD}): {s['non-binder']:>3d}")
        print(f"  Ambiguous:             {s['ambiguous']:>3d}")
        print(f"  Missing (#N/A):        {s['missing']:>3d}")
        print(f"  Labeled total:         {total_labeled:>3d}")

    # Boltz summary by ligand
    boltz_nit = [r for r in boltz_rows if r["ligand_name"] == "nitazene"]
    boltz_men = [r for r in boltz_rows if r["ligand_name"] == "menitazene"]
    print(f"\nBoltz2 CSV summary:")
    print(f"  Nitazene:    {sum(1 for r in boltz_nit if r['label']==1)} binders, "
          f"{sum(1 for r in boltz_nit if r['label']==0)} non-binders "
          f"({len(boltz_nit)} total predictions)")
    print(f"  Menitazene:  {sum(1 for r in boltz_men if r['label']==1)} binders, "
          f"{sum(1 for r in boltz_men if r['label']==0)} non-binders "
          f"({len(boltz_men)} total predictions)")


if __name__ == "__main__":
    main()
