# LCA / LCA-3-S MD set — 228 structures, geometry-gated, per-ligand balanced

228 Boltz-2 binary structures of PYR1 designs with a bile acid, for MD validation of
co-folding false positives. Each structure is one **(design, ligand) arm**:

| arm | binders | non-binders | total | balanced on |
|---|---|---|---|---|
| **LCA** (`LCAM`, charge −1) | 31 | 80 | **111** | 9 scores |
| **LCA-3-S** (`LCA3S`, charge −2) | 22 | 95 | **117** | 7 scores |

174 distinct designs: 54 appear in both arms, 120 in one arm only.

> **Provenance.** Generated in the analysis repo at
> `/projects/ryde3462/lca_lca3s_modelling` (commit `2ae113c`). The scripts in
> `scripts/` here are copies of that repo's `scripts/`, so the reproduce commands
> below are written relative to it. Predictions live on scratch at
> `/gpfs/alpine1/scratch/ryde3462/lca_lca3s_5seed/seed_{1..5}/boltz_out`.


The question this set is built to answer: **co-folding cannot tell these binders from
these non-binders, so can MD?** Every co-folding score that could give the answer away
is balanced to AUC ≈ 0.5 within each arm.

---

## This replaces the earlier 137-design / 274-PDB set. Here is why

The previous hand-off staged, for each design, the *medoid of the majority binding-mode
cluster* across 5 seeds. That picked poses by **consensus** without ever checking the
ligand **was the right molecule**. It was not:

> **106 of its 274 structures had an inverted stereocentre — 103 of them at C3**, the
> carbon bearing the 3α-OH (LCA) or the sulfate (LCA-3-S). An inverted C3 is the
> **3β-epimer**: a different compound, not the ligand we are modelling.

Nothing in the pipeline caught it. `binary_lig_oh_stereo_ok` was `NaN` for all 14,260
predictions while `pass_oh_stereo` reported `1` throughout — a silent all-clear. Ring
pucker only inspects the steroid rings; pose-consistency compares seeds to each other,
so five seeds agreeing on the *wrong* epimer looks like high confidence.

Do not use the previous set. Its structures are superseded by the ones here.

## The ligand-geometry gate

A seed is usable only if its ligand is the modelled molecule. Three independent
intra-ligand checks, all enforced (`scripts/ligand_geometry_scan.py`, merged into
`scripts/aggregate_seeds.py`):

1. **Stereochemistry** — signed volume at every sp3 centre (C3, C5, C8, C9, C14, C17,
   C20) against the AF3 `userCCD` reference, which is built from the same RDKit mol as
   the Boltz pickle. Any sign mismatch is an inverted centre.
2. **Collapsed atoms** — no two ligand heavy atoms closer than 1.15 Å (observed as low
   as 0.04 Å between carboxylate oxygens).
3. **Ring pucker** — no boat/twist steroid ring (Cremer-Pople, 30° < θ < 150°).
   Orthogonal to the other two: 27 structures pass 1 and 2 and fail this.

Failure rates over the 14,260 labelled-pool predictions:

| ligand | valid | stereo wrong | of which C3 | collapsed | ring distorted |
|---|---|---|---|---|---|
| LCA (`LCAM`) | 74.5% | 25.2% | 22.6% | 0.4% | 0.7% |
| LCA-3-S | 53.0% | 46.1% | 43.8% | 1.1% | 0.9% |

Invalid seeds are dropped from **everything** — the median features, the majority
binding-mode vote, and the choice of representative. So `n_seeds` is the valid-seed
count and `n_seeds_total` is how many were predicted. Of 2,852 arms, **2,407 keep at
least one valid seed and 445 keep none**; the latter are excluded from selection (their
rows survive in the aggregate tables, flagged `n_valid_seeds=0`).

**Every shipped structure was re-verified from the staged file on disk**, not trusted
because the gate had passed it: 228/228 geometry-valid, minimum intra-ligand distance
1.206 Å. See `stage_verification.csv`.

## Why arms, not design pairs

The gate kills single arms, not whole designs — a design can have a usable LCA structure
and no usable LCA-3-S structure. Requiring pairs would have discarded 6 binders whose LCA
arm is perfectly good. MD does not need paired structures, so each arm is selected on its
own and a design may ship with one ligand or both.

Arms are also **labelled per ligand**: a design that binds LCA but not LCA-3-S is a
**positive in the LCA arm and a negative in the LCA-3-S arm**, because that is what the
EC50 data says. This is why the binder counts differ (32 designs bind LCA, 25 bind
LCA-3-S; after the gate, 31 and 22).

## The balance proof

Each arm is its own subset-balancing problem (`scripts/balance_select.py`): choose k
non-binders minimising the worst deviation of any constraint from its target, where each
score contributes an **AUC** constraint (target 0.5) and nine **CDF** constraints (target
= the binders' own quantiles, so the whole distribution matches, not just the mean rank).

Both arms are at the largest size where **every constraint is reachable** — 0 of 90
unsatisfiable for LCA at k=80, 0 of 70 for LCA-3-S at k=95 (`--report-bounds` proves
this against the pool, independently of the search).

| score | LCA: all-NB → balanced | LCA-3-S: all-NB → balanced |
|---|---|---|
| `plddt_ligand` | 0.921 → **0.553** | 0.851 → **0.492** |
| `plddt_pocket` | 0.956 → **0.576** | 0.930 → **0.530** |
| `plddt_protein` | 0.926 → **0.536** | 0.904 → **0.509** |
| `iptm` | 0.852 → **0.530** | 0.680 → **0.492** |
| `affinity_probability_binary` | 0.806 → **0.575** | 0.815 → **0.543** |
| `n_interface_unsatisfied` | 0.300 → **0.439** | 0.331 → **0.433** |
| `n_valid_seeds` | 0.654 → **0.581** | 0.618 → **0.581** |
| `geometry_score` | 0.690 → **0.590** | *declared, see below* |
| `hbond_distance` | 0.466 → **0.490** | *declared, see below* |

Worst deviation from target: 0.153 (LCA), 0.134 (LCA-3-S).

`n_valid_seeds` is balanced deliberately — binders keep more valid seeds than non-binders
(AUC 0.65), so without it "how often Boltz got the stereochemistry right" would remain
available as a proxy for binding. It lands at 0.581 rather than 0.5 only because it is a
discrete 1–5 variable.

### Declared, not balanced: LCA-3-S water-network geometry

`geometry_score` and `hbond_distance` measure the distance from the **3QN1 conserved-water
position** to the ligand's nearest O/N, scored as a Gaussian centred on **2.7 Å** — they
assume the water is retained and the ligand hydrogen-bonds to it.

For **LCA** that is exactly the situation (bare 3α-OH; binders 1.96 Å vs non-binders
2.59 Å, AUC 0.690), so both scores are balanced in the LCA arm.

For **LCA-3-S** it is not. The C3 sulfate oxygens sit **0.37–1.6 Å from the water
position** — the sulfate is *in* the water site:

| design | closest ligand atoms to the conserved-water site |
|---|---|
| BA_PYR1_0085 | **OS2 0.37 Å**, OS1 2.09, O1B 2.39 |
| BA_PYR1_0084 | **OS2 0.89 Å**, OS1 1.60, O1B 2.17 |
| BA_PYR1_0266 | **OS2 0.97 Å**, OS1 1.66, O1B 1.85 |
| BA_PYR1_0592 | **OS1 1.09 Å**, OS2 1.37, O1B 2.26 |

So the 2.7 Å ideal is meaningless for a 3-O-sulfate, and the score inverts: LCA-3-S
binders score 0.096 against non-binders' 0.382 (AUC 0.124). Balancing a meaningless
score would mean discarding real structures to equalise an artefact, so these two ship
as **declared covariates** for the LCA-3-S arm (`geometry_score` AUC 0.206,
`hbond_distance` 0.294 in the selected set) and are in the CSV for auditing.

This also explains the apparent "water-network geometry inverts between ligands"
(0.63 LCA vs 0.31 LCA-3-S) seen earlier — it is the metric, not the biology.

> **Hypothesis for MD to test:** in LCA-3-S the sulfate **displaces** the conserved
> gate water and does its job, rather than hydrogen-bonding to a retained water. Explicit-
> solvent MD can settle this directly: watch whether that water site stays occupied by
> water or by sulfate oxygen. For LCA the water should persist and accept from the 3α-OH.

## Before production MD: restrained minimisation

**59 of 228** structures carry `needs_restrained_min` — a non-polar heavy-atom contact
below 2.8 Å (LCA 21/111, LCA-3-S 38/117; minimum 2.19 Å). Co-folding does not enforce
inter-molecular van der Waals, so these need restrained minimisation first. They are
**declared, not dropped** — dropping them would bias the set, since the flag is itself
more common in the LCA-3-S arm.

Polar contacts are separated from steric ones in the audit (`min_polar`, `n_polar_lt24`),
so a short hydrogen bond is not mistaken for a clash.

## Files

| file | what |
|---|---|
| `md_arm_set.csv` | one row per arm (228): class, per-ligand label, sequences, every balanced and declared score, full seed/geometry QC, contact audit, PDB path |
| `md_arm_set_long.csv` | every **eligible** arm with `in_selection`, for selected-vs-pool figures |
| `stage_verification.csv` | per-staged-file geometry re-check (stereo + collapse) |
| `pdbs/<design>__<LCAM\|LCA3S>.pdb` | the staged structure, chain A protein / chain B ligand |

Each staged PDB is the `rep_seed` structure: the medoid of the majority binding-mode
cluster **among geometry-valid seeds only**. Seeds are evenly used (44/44/61/40/39 for
seeds 1–5), so no single seed dominates.

## Reproducing

```bash
# 1. ligand geometry over every seed of every prediction (SLURM array)
sbatch results/ligand_geometry_scan/submit_scan.sh
python3 scripts/summarize_ligand_geometry.py

# 2. consensus per arm, with the gate applied
python3 scripts/aggregate_seeds.py --jobs 16

# 3. per-arm balanced selection (defaults = the shipped set)
python3 scripts/balance_select.py --report-bounds --write

# 4. write + stage + re-verify
python3 scripts/build_md_arm_set.py
```

## Caveats, stated plainly

- **Marginal balance only.** Every score sits at AUC ≈ 0.5 *one at a time*. A
  multivariate classifier on the same scores can still separate the groups. "Balanced"
  means no single co-folding score gives the answer away — not that no co-folding
  information remains.
- **Population statistics are converged at 5 seeds; per-design calls are not.** Use the
  class-level comparison, and treat an individual design's pose as one sample with QC
  attached (`pose_rmsd_spread`, `seed_agreement`, `n_valid_seeds`).
- **The non-binder pool caps the arm sizes.** LCA cannot exceed 111 and LCA-3-S 117
  while keeping every constraint reachable. Larger sets would require giving up the
  distribution match.
- **Arms are selected independently**, so only 54 designs carry both ligands. Within-design
  LCA-vs-LCA-3-S comparison is possible for those 54, not for the set as a whole.
- **Constitutives are excluded** from this set entirely; they are a third class with their
  own signature (ESM3dG apo ΔG separates them) and would confound a binder-vs-non-binder
  MD comparison.
