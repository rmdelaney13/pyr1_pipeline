# MD test structures — 4 binder / ligand pairs

Starting structures for MD, one **ligand-responsive binder per ligand** across the four
bile-acid / conjugate ligands. Each pair ships a matched **binary** (PYR1 + ligand) and
**ternary** (PYR1 + ligand + HAB1) Boltz-2 prediction. Both come from the *same* run
(`boltz_sort_20260818`) and the *same* sequence, so binary and ternary differ only in the
presence of HAB1 — they are independent predictions, not one derived from the other.

## Contents
```
<LIGAND>__<pocket16>/
  binary.pdb    # chains A=PYR1, B=ligand (LIG)          <- output_holo (scratch)
  ternary.pdb   # chains A=PYR1, B=ligand (LIG), C=HAB1   <- output_ternary (in repo)
test_structures_stats.csv   # aggregated Boltz + NGS stats for all 4
```

## The four pairs
| Ligand | Pocket (16-mer) | ER_mean | fold enrich | primary q | notes |
|--------|-----------------|--------:|------------:|-----------|-------|
| LCA    | FDLMVVILYSKILGFM | 7.27 | 154 | 1e-77 | strongest LCA binder |
| LCA3S  | VLLAVGVVLWQILGFM | 7.53 | 185 | 3e-112 | strongest LCA-3-S binder |
| GLCA   | IDLMVAVVLTKILGVI | 2.63 | 6.2 | 9e-6 | best GLCA (weak ligand overall) |
| CDCA   | IDLMLAVVLTKIMGII | 2.68 | 6.4 | 1e-4 | most significant CDCA binder |

## Selection criteria
From `structural_modeling_all_measured_pairs.csv`: `primary_candidate == True`
(direction-matched FDR<1 responder), **excluding** `constitutive_high_confidence` variants
(want ligand-dependent binding, not constitutive HAB1 recruitment), ranked by `ER_mean`.
GLCA/CDCA have far fewer confident binders than LCA/LCA-3-S, so their ER values are lower —
these are the best confident ligand responders available for those two ligands.

All four have high Boltz confidence and ligand-ipTM in both binary and ternary
(`bin_confidence` / `ter_confidence` ≈ 0.93–0.95, ligand-ipTM ≈ 0.95–0.99).
Note `bin_affinity_prob` (Boltz affinity head) is near-random for GLCA/CDCA — expected,
the affinity head is weak for these; NGS enrichment is the real binder label.

## Provenance
- Binary source: `/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo/`
- Ternary source: `ml_modelling/data/20260818_sort_data/output_ternary/`
- Rebuild: `python /tmp/assemble_test_structures.py` (script also saved intent inline; see
  git history / `ml_modelling/analysis/test_structures/` if re-running).

See `ml_modelling/PREDICTION_DATA_MAP.md` for the full catalog of where every Boltz run lives.
