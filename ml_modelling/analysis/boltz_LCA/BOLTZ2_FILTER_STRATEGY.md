# Boltz2 Filter Strategy for Iterative MPNN Design Pipeline

## Recommended Filter (Strategy H)

```
1. Pre-filter:  binary_plddt_ligand  >= 0.80   (remove low-confidence predictions)
2. Structural:  binary_hbond_distance <= 4.0 A  (remove wrong binding mode)
3. Rank by:     binary_plddt_pocket             (higher = better)
4. Select:      top 100 from entire pool
```

All binary-only. No ternary metrics needed. No Z-scores. Cross-batch stable thresholds.

### Performance (binders captured in top 100)

| Dataset | Binders | AUC | Optimal MCC | B@100 | Recall |
|---------|---------|-----|-------------|-------|--------|
| LCA | 52 | 0.887 | 0.594 | 33/52 | 63% |
| GLCA | 18 | 0.829 | 0.354 | 14/18 | 78% |
| LCA-3-S | 59 | 0.576 | 0.150 | 21/59 | 36% |
| **Pooled** | **129** | **0.720** | **0.302** | **35/129** | **27%** |

## Validation Datasets

- **LCA** (lithocholic acid): 52 binders / 500 non-binders. Most relevant proxy for CA/DCA/UDCA/CDCA targets (same steroid scaffold + carboxylate tail, differs only in hydroxylation).
- **GLCA** (glycolithocholic acid): 18 binders / 500 non-binders. Glycine-conjugated variant, validates cross-conjugate generalization.
- **LCA-3-S** (lithocholic acid 3-sulfate): 59 binders / 500 non-binders. Sulfate group changes H-bond geometry — hardest case, less relevant to Round 2 targets.

## 8 Strategies Tested

| ID | Strategy | LCA AUC | GLCA AUC | Pooled B@100 |
|----|----------|---------|----------|--------------|
| **H** | **pLDDT_lig>=0.80 + hbond<=4A, rank pocket pLDDT** | **0.887** | **0.829** | **35/129** |
| A | binary_plddt_pocket (no filter) | 0.746 | 0.828 | 32/129 |
| C | Gate plddt_lig>=0.80, Z(tern_ipLDDT+tern_hbond) | 0.898 | 0.839 | 31/129 |
| B | Z(pocket_pLDDT) + Z(tern_ipLDDT) | 0.784 | 0.732 | 17/129 |
| G | binary_boltz_score | 0.677 | 0.725 | 19/129 |
| D | Gate + Z(tern_ipLDDT + Trp211) | 0.679 | 0.483 | 15/129 |
| F | Gate + Z(tern_ipLDDT + HAB1_pLDDT) | 0.780 | 0.605 | 8/129 |
| E | Gate + tern_ipLDDT (raw) | 0.782 | 0.519 | 8/129 |

## Key Findings

### 1. Binary pocket pLDDT is the most robust ranking metric

`binary_plddt_pocket` is the only metric that achieves >60% binder capture across all 3 ligands. It measures Boltz2's confidence in the pocket region of the binary (PYR1+ligand) prediction.

### 2. H-bond distance gate removes wrong binding modes

`binary_hbond_distance <= 4.0 A` ensures the ligand's H-bond acceptor (OH or COO-) is positioned within hydrogen-bonding range of the conserved water. Designs where the acceptor is >4A from the water have the wrong binding mode and are structural false positives — they may score well on pocket pLDDT but cannot form the water-mediated network required for gate closure.

Adding this gate improves LCA AUC from 0.746 to 0.887 (+0.14) and pooled B@100 from 32 to 35 (+3 binders).

### 3. Ternary metrics are unreliable across ligands

Strategies using ternary Boltz2 metrics (C-F) showed inconsistent performance:
- **ternary_hbond_distance** (Strategy C): Best AUC on LCA (0.898) but biologically suspect — binders show ~5A distances, not real H-bonds. Crashes on LCA-3-S (AUC=0.515). This is a Boltz2 artifact where HAB1 engagement displaces the ligand from the conserved water in silico.
- **ternary_trp211_ligand_distance** (Strategy D): Biologically grounded (Trp lock proximity) but weak discriminator (pooled AUC=0.569). Fails on GLCA (2/18 B@100).
- **ternary_complex_iplddt** (Strategy E): Anti-predictive on GLCA within the gated subset (2/18 B@100). Gating removes informative variance.
- **ternary_plddt_hab1** (Strategy F): Weak across all ligands.

### 4. COO-R116 salt bridge is NOT a false positive signal

We tested filtering out designs with binary_coo_to_r116_dist < 4A (predicted carboxylate-R116 salt bridge). Counter to expectation, binders are MORE likely (15%) to show this contact than non-binders (7%). Filtering on it removes proportionally more binders and decreases AUC. The COO being near R116 reflects the carboxylate tail naturally extending toward the latch region.

### 5. Ligand pLDDT gate removes garbage predictions

`binary_plddt_ligand >= 0.80` removes predictions where Boltz2 couldn't confidently place the ligand. This pre-filter alone improves AUC from 0.714 to 0.733 (pooled) and is the foundation for all gated strategies.

## Figures

All figures in `ml_modelling/analysis/boltz_LCA/figures/`.

### Fig 1: ROC Curves (`fig1_roc_comparison.png`)
ROC curves for all 8 strategies across 4 datasets (LCA, GLCA, LCA-3-S, Pooled). Strategy H (cyan) dominates on LCA and is competitive on GLCA. Ternary strategies (C-F) show high variance across ligands.

### Fig 2: Enrichment at Top-100 (`fig2_enrichment_top100.png`)
Left: enrichment curves on pooled data showing what fraction of binders are captured as you include more top-ranked designs. Right: bar chart of binder recall at top-100 across all datasets and strategies.

### Fig 3: Gate Threshold Optimization (`fig3_gate_optimization.png`)
Sweeps `binary_plddt_ligand` threshold from 0.50 to 0.92 for each ligand. Shows within-gate AUC for ternary metrics and % binders retained. Key insight: gating improves ternary_hbond_distance AUC on LCA but ternary_complex_iplddt AUC drops on GLCA at higher thresholds.

### Fig 4: Ternary H-bond Distance Artifact (`fig4_hbond_distance_concern.png`)
Left: binary vs ternary H-bond distance scatter. Binders have ternary distances >>3.5A (above the H-bond cutoff), confirming this is a Boltz2 artifact, not a real interaction. Right: Trp211 lock distance violin — the biologically grounded alternative has AUC=0.569 (too weak to use).

### Fig 5: Strategy Summary (`fig5_strategy_summary.png`)
Grouped bar chart: % binders captured at top-100 for all 8 strategies across 4 datasets. Strategy H has the tallest pooled bar and is consistent across LCA/GLCA.

### Fig 6: Matthews Correlation Coefficient (`fig6_mcc_comparison.png`)
Left: optimal MCC (best binary classification threshold) per strategy per dataset. Right: MCC at the practical top-100 selection cutoff. Strategy H achieves the best or near-best MCC across LCA (0.594), GLCA (0.354), and pooled (0.302). Strategy C has the highest single-dataset MCC on LCA (0.654) but is inconsistent across ligands.

## Implementation Notes

- **Cross-batch stability**: Both gate thresholds (pLDDT >= 0.80, hbond <= 4.0A) are absolute values that don't change as new sequences are added to the pool. The ranking metric (pocket pLDDT) is also an absolute value — no Z-score recomputation needed.
- **Iterative workflow**: Start with ~100 sequences → MPNN generates ~300 → Boltz2 binary predictions → apply gates → rank by pocket pLDDT → keep running top 100 → repeat.
- **LCA-3-S caveat**: The hbond distance gate works for carboxylic acid bile acids (LCA, GLCA, CA, DCA, UDCA, CDCA) but may need adjustment for sulfated or other conjugated forms where the H-bond acceptor geometry differs.

## Script

Analysis script: `ml_modelling/analysis/boltz_LCA/analyze_filter_recommendation.py`
Summary CSV: `ml_modelling/analysis/boltz_LCA/filter_strategy_comparison.csv`

Generated: 2026-03-03
