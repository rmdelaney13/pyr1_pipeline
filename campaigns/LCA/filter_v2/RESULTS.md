# LCA Boltz Filter v2 — Results

*Running log of results for the plan in `PLAN.md`. The per-feature AUC / filter
results are added once the holo jobs finish (run `/finish-lca-filter`). This first
entry is a dataset-characterization result that holds now.*

---

## R0 — Why the hard negatives matter (sequence diversity of the negative sets)

**Figure:** `analysis/figures/fig_seq_diversity.png`

We measured how close each negative set sits to the confirmed binders, by minimum
Hamming distance (16-mer pocket) to the nearest binder:

| group | n | %DLM motif | median Hamming to nearest binder |
|---|---|---|---|
| binders (to each other) | 50 | 76% | **1** |
| **new confirmed — MOTIF non-binders** | 15 | 100% | **2** |
| new confirmed non-binders (all) | 52 | 29% | 4 |
| **500 screen non-binders** | 499 | 0% | **7** |

**Read:**
- **The 500 screen negatives are *easy*.** At a median **7 mutations** from any binder
  and **0% motif**, they are far from the binder cloud — sequence distance alone
  separates them. A high filter AUC against the 500 does **not** prove the filter
  sees *binding* rather than sequence dissimilarity.
- **The new confirmed motif non-binders are the *hard* negatives — the real test.**
  They sit a median **2 mutations** from a binder (the binders sit 1 from each other)
  and carry the same D81/L83/M92 motif, so neither sequence nor motif can separate
  them. If the **structural** features (confidence, geometry, BUNs, packing) separate
  these from binders, the filter is reading biophysics, not sequence similarity.
- PCA of the pocket sequences (panel B) shows binders and the new motif non-binders
  overlapping, with the 500 screen spread across the rest of the space.

**Consequence for v2:** the headline discrimination metric is **binders vs the new
motif (hard) non-binders**, not the pooled AUC against the 500. The 500 gives power
and an operating point; the hard negatives give the honest, deployment-relevant
number — they are the only negatives close enough to binders to make the test fair.

*(Source: `analysis/fig_seq_diversity.py`. Binders/new non-binders = NGS depth≥10,
corrected metric; 500 = `inputs_lca500/labels.csv`.)*

---

## R1 — Phase A first-pass: do the structural features separate binders from the HARD negatives? (MD set)

*Run 2026-06-12 on the MD set (50 binders / 139 non-binders). Fresh Boltz output for
the 69 new rows; the 120 old rows use the original 552 metrics + committed PDBs.*
**Caveat — read first:** binders are **mostly from the original 552 predictions**; the
hard (new motif) non-binders are **fresh**. A prediction-settings offset could inflate
any binder-vs-hard separation. Phase B (uniform fresh predictions for all) is required
to confirm the confidence numbers. Packing (sequence-derived) is settings-independent.

**Stratified AUC — binders vs three negative tiers** (easy = old assumed, far in
sequence; hard = new motif non-binders, ~2 Hamming from binders):

| feature | vs ALL neg | vs HARD (motif) | vs EASY (assumed) |
|---|---|---|---|
| **pocket pLDDT** ↑ | 0.74 | **0.83** | 0.68 |
| ligand pLDDT ↑ | 0.59 | 0.72 | 0.50 |
| pocket packing (Σ side-chain MW) ↓ | 0.75 | 0.64 | **0.86** |

**Reads:**
1. **Pocket pLDDT separates the *hard* motif-matched negatives strongly (0.83) — *more* than
   the easy ones (0.68).** If it holds under Phase B, this is the key positive result: pocket
   pLDDT is reading *binding*, not sequence distance or the motif (the hard negatives are 2
   mutations from binders and carry the motif, yet it still separates them). **But** the
   settings caveat above could be inflating it — Phase B decides.
2. **Pocket packing is a *sequence-distance* discriminator, not a binding one** — it nails the
   easy negatives (0.86, they're bulkier/farther) but is weak on the hard ones (0.64). Useful
   as a cheap pre-filter for obviously-wrong designs, not for the hard call. (Settings-independent,
   so this split is trustworthy.)
3. **Ligand pLDDT** is weak overall but separates the hard negatives (0.72) — same caveat as pocket.

**BUN (ligand polar unsatisfied), fresh rows only:**
| group | n | mean lig-BUN | % with ≥1 unsat |
|---|---|---|---|
| binders | 18 | 0.28 | 28% |
| hard motif non-binders | 15 | 0.27 | 27% |
| all non-binders | 51 | 0.41 | 41% |

- **Ligand-BUN separates binders from the *broad* negatives (41% vs 28%) but NOT the hard ones
  (27% ≈ 28%).** The hard motif non-binders are as OH/COO-satisfied as binders yet don't bind —
  so BUN is necessary-ish but not the discriminator for the hard call. (Small n; confirm in Phase B.)

**Phase A conclusion (provisional):** pocket pLDDT is the candidate that may carry real binding
signal even on the hard negatives; packing and BUN separate the easy/broad negatives but fade on
the hard ones. **Nothing is adopted until Phase B re-runs this on uniform fresh predictions for all
189 (and adds diffusion RMSD + the powered 500).**
