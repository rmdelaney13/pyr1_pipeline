#!/usr/bin/env python3
"""
Merge MD simulation candidates with NGS sequencing enrichment data.

Reconstructs the 16-position pocket sequence from variant_signature,
matches to All_Sequences_Enrichment.csv, and computes a binding score.

Score logic:
  - binding_ratio   = Count_Bind / Count_Input   (enrichment in binding gate)
  - constitutive_ratio = Count_Const / Count_Input (enrichment in constitutive gate)
  - binder_score    = binding_ratio - constitutive_ratio  (high binding, low constitutive = good)
  - Also report log2 enrichments for each gate
"""

import csv
import math
import os

# ── Configuration ──────────────────────────────────────────────────────────
POSITIONS = ['59', '81', '83', '92', '94', '108', '110', '117',
             '120', '122', '141', '159', '160', '163', '164', '167']
WT_RESIDUES = ['K', 'V', 'V', 'S', 'E', 'F', 'I', 'L',
               'Y', 'S', 'E', 'F', 'A', 'V', 'V', 'N']
POS_TO_IDX = {pos: i for i, pos in enumerate(POSITIONS)}
POS_TO_WT = {pos: wt for pos, wt in zip(POSITIONS, WT_RESIDUES)}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

MD_GUIDE = os.path.join(SCRIPT_DIR, 'md_candidate_guide.csv')
ENRICHMENT = os.path.join(PROJECT_ROOT, 'ml_modelling', 'data',
                          'All_Sequences_Enrichment.csv')
OUTPUT = os.path.join(SCRIPT_DIR, 'md_candidates_with_enrichment.csv')


def signature_to_sequence(signature: str) -> str:
    """Convert variant_signature like '59A;81L;83L;...' to 16-char sequence."""
    seq = list(WT_RESIDUES)  # start with wildtype
    for mut in signature.split(';'):
        mut = mut.strip()
        if not mut:
            continue
        pos = mut[:-1]   # e.g. '59'
        aa = mut[-1]     # e.g. 'A'
        if pos in POS_TO_IDX:
            seq[POS_TO_IDX[pos]] = aa
    return ''.join(seq)


def log2_enrichment(gate_count, input_count, pseudocount=1):
    """Log2 fold-enrichment of gate vs input with pseudocount."""
    return math.log2((gate_count + pseudocount) / (input_count + pseudocount))


def main():
    # ── Load enrichment data keyed by sequence ─────────────────────────────
    enrichment = {}
    with open(ENRICHMENT) as f:
        for row in csv.DictReader(f):
            seq = row['Sequence']
            enrichment[seq] = {
                'confidence': row['Confidence'],
                'count_bind': int(row['Count_Bind']),
                'count_input': int(row['Count_Input']),
                'count_const': int(row['Count_Const']),
            }
    print(f"Loaded {len(enrichment)} sequences from enrichment data")

    # ── Load MD candidates and merge ───────────────────────────────────────
    with open(MD_GUIDE) as f:
        md_rows = list(csv.DictReader(f))

    # Filter to binders only (LCA, not conjugates)
    binders = [r for r in md_rows if r['md_group'] == 'binder']
    print(f"Found {len(binders)} binders in MD candidate guide")

    results = []
    matched = 0
    for row in binders:
        seq = signature_to_sequence(row['variant_signature'])
        enr = enrichment.get(seq)

        entry = {
            'pair_id': row['pair_id'],
            'variant_name': row['variant_name'],
            'variant_signature': row['variant_signature'],
            'label_tier': row['label_tier'],
            'pocket_sequence': seq,
            'binary_plddt_ligand': row['binary_plddt_ligand'],
            'binary_plddt_pocket': row['binary_plddt_pocket'],
            'binary_hbond_distance': row['binary_hbond_distance'],
        }

        if enr:
            matched += 1
            cb = enr['count_bind']
            ci = enr['count_input']
            cc = enr['count_const']

            bind_ratio = cb / ci if ci > 0 else 0
            const_ratio = cc / ci if ci > 0 else 0

            entry.update({
                'ngs_confidence': enr['confidence'],
                'count_bind': cb,
                'count_input': ci,
                'count_const': cc,
                'bind_ratio': round(bind_ratio, 4),
                'const_ratio': round(const_ratio, 4),
                'log2_bind_enrich': round(log2_enrichment(cb, ci), 3),
                'log2_const_enrich': round(log2_enrichment(cc, ci), 3),
                'binder_score': round(bind_ratio - const_ratio, 4),
            })
        else:
            entry.update({
                'ngs_confidence': 'not_found',
                'count_bind': '',
                'count_input': '',
                'count_const': '',
                'bind_ratio': '',
                'const_ratio': '',
                'log2_bind_enrich': '',
                'log2_const_enrich': '',
                'binder_score': '',
            })

        results.append(entry)

    print(f"Matched {matched}/{len(binders)} binders to enrichment data")

    # Sort by binder_score descending (unmatched at bottom)
    results.sort(key=lambda r: (
        r['binder_score'] if isinstance(r['binder_score'], (int, float)) else -999
    ), reverse=True)

    # ── Write output ───────────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(OUTPUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\nWrote {OUTPUT}")
    print(f"\n{'='*80}")
    print(f"{'Variant':<12} {'Sequence':<18} {'Bind':>6} {'Input':>6} {'Const':>6} "
          f"{'B/I':>6} {'C/I':>6} {'Score':>7}")
    print(f"{'='*80}")
    for r in results:
        if r['binder_score'] == '':
            score_str = '  N/A'
        else:
            score_str = f"{r['binder_score']:>7.3f}"
        cb = r['count_bind'] if r['count_bind'] != '' else '-'
        ci = r['count_input'] if r['count_input'] != '' else '-'
        cc = r['count_const'] if r['count_const'] != '' else '-'
        br = f"{r['bind_ratio']:.2f}" if r['bind_ratio'] != '' else '-'
        cr = f"{r['const_ratio']:.2f}" if r['const_ratio'] != '' else '-'
        print(f"{r['variant_name']:<12} {r['pocket_sequence']:<18} "
              f"{str(cb):>6} {str(ci):>6} {str(cc):>6} "
              f"{br:>6} {cr:>6} {score_str}")


if __name__ == '__main__':
    main()
