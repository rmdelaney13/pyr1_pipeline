# MD hand-off — LCA / LCA-3-S co-folding false positives

**137 designs = 37 experimentally-labeled binders + 100 non-binders**, each with an
LCA (`LCAM`) and an LCA-3-S (`LCA3S`) Boltz holo structure. **274 PDBs** in `pdbs/`.

> **This replaces the previous 200-design hand-off that lived at this path.** The old set
> was built from a **single** Boltz sample per prediction and was never actually balanced —
> it retained pocket-pLDDT AUC **0.739**, so a Boltz-only classifier could still separate
> the classes and MD separation could not have been called new signal. It also had no
> ligand-geometry QC. **Do not simulate the old set.** It remains in git history if needed.

---

## What this set is — and what it is NOT

The 100 non-binders are a deliberate **false-positive subset**: the minority of
non-binders that co-folding scores *like* binders. They were chosen so that every
co-folding metric is statistically unable to rank binder vs non-binder. MD is being asked
to reject exactly the cases co-folding gets wrong.

**This set is not evidence that co-folding is blind.** On the full labeled pool co-folding
discriminates well — LCA pocket pLDDT reaches **AUC 0.954** over all 1,389 non-binders.
Quoting this set's ~0.5 AUCs as "co-folding cannot distinguish the classes" would be
**circular**, because these 100 were selected on those metrics. The honest baseline is
0.954, and that is the number MD has to beat.

## Balance achieved

16 co-folding scores balanced simultaneously — AUC **and** a 9-point CDF quantile grid per
score, 160 linear constraints (`scripts/balance_select.py --mode both`). Balancing AUC
alone was not enough: it left `LCA3S_plddt_ligand` mean-matched but wrong-shaped (KS 0.302
vs critical 0.262). Hence the CDF constraints.

| | result |
|---|---|
| AUC range, all 16 score × ligand combinations | **0.438 – 0.585** |
| 95% CIs containing 0.5 | **16 / 16** |
| combinations separable by a KS test at p<0.05 | **0 / 16** |

Per-combination, binder (n=37) vs non-binder (n=100):

| score | LCAM AUC | LCAM KS p | LCA3S AUC | LCA3S KS p |
|---|---|---|---|---|
| plddt_ligand | 0.547 | 0.211 | 0.468 | 0.220 |
| plddt_pocket | 0.585 | 0.113 | 0.547 | 0.333 |
| plddt_protein | 0.551 | 0.143 | 0.504 | 0.930 |
| geometry_score | 0.531 | 0.568 | 0.438 | 0.695 |
| hbond_distance | 0.479 | 0.085 | 0.475 | 0.801 |
| iptm | 0.522 | 0.438 | 0.484 | 0.604 |
| affinity_probability_binary | 0.511 | 0.975 | 0.545 | 0.705 |
| n_interface_unsatisfied | 0.444 | 0.988 | 0.471 | 1.000 |

Every one of these columns is in `md_handoff_selection.csv`, so the whole table above can
be recomputed from this directory alone.

(`iptm` and `ligand_iptm` are the same number in this data — `protein_iptm` is 0 for all
2,852 rows, since the complex is one protein chain plus one ligand.)

## Why 100 non-binders and not more

The size is a **property of the pool**, not a tuning choice. Binder pocket pLDDT sits at
the very top of the distribution:

| ligand | binder min | binder median | non-binders ≥ binder median | non-binders ≥ binder min |
|---|---|---|---|---|
| LCA (LCAM) | 0.952 | 0.961 | **54** / 1,389 | 188 / 1,389 |
| LCA-3-S | 0.951 | 0.966 | 92 / 1,389 | 622 / 1,389 |

Only 54 non-binders in the entire pool reach the binder median on LCA. Asking for more
non-binders forces the selection down the distribution and the residual grows with it:
at 163 non-binders, pocket pLDDT **cannot** go below 0.614 at any setting.
**k ≤ 111 is the largest non-binder count where AUC 0.5 is reachable**, so 100 fits with
margin. `balance_select.py --report-bounds` prints the exact achievable range per
constraint — it is a proof about the pool, not a search diagnostic.

## Structures

`pdbs/<sequence_id>__<LCAM|LCA3S>.pdb` is the **rep_seed** model_0 — the medoid of the
majority binding-mode cluster across 5 Boltz seeds, not an arbitrary sample.
**Heavy atoms only; add hydrogens at setup.**

Per-design pose QC travels in `md_handoff_selection.csv` (`n_seeds`, `rep_seed`,
`seed_agreement`, `mode_majority`, `flipped_fraction`, `pose_rmsd_spread`,
`pose_rmsd_max`, `head_clash_fraction`, `r79_clash_fraction`, `pass_pose_consistency`)
and per-structure in `structure_qc.csv`. **Nothing is auto-dropped — flags travel with
the row.**

Across the 5 seeds, **any-seed head clash is 49.7% but all-seed clash is 0.8%**, so nearly
every apparent clash is a sampling artefact rather than a property of the design. A
single-seed run cannot tell those apart — which is why the previous hand-off was rebuilt.

---

## ⚠ Before you simulate — contacts are tight

Raw minimum ligand–protein distance is misleading because it mixes hydrogen bonds with
steric clashes, so they are reported separately. A contact counts as POLAR only if both
atoms are N/O/S; everything else is STERIC.

| | LCAM | LCA3S |
|---|---|---|
| median closest steric contact | 3.04 Å | 2.88 Å |
| structures with a steric contact < 3.0 Å | 43.1% | 74.5% |
| **structures with a steric contact < 2.8 Å (hard)** | **17.5%** | **33.6%** |
| closest polar contact < 2.4 Å (too short for an H-bond) | 26 | 73 |

Normal C···C van der Waals contact is ~3.4–3.7 Å, so **the median shipped structure has a
sub-vdW steric contact**, and LCA-3-S is markedly worse than LCA. Offenders cluster at the
charged pocket positions contacting the carboxylate/sulfate head — **Arg141/Lys141,
Arg116, Arg79, Pro88**.

**These are declared covariates, not defects, and nothing was dropped for them.** They are
a property of co-folding output: Boltz is not a force field and does not enforce
inter-molecular van der Waals. **Equilibrate with restrained minimisation before
production MD.** A structure that blows up on the first step is an artefact of this, not
evidence about the design — if that happens, check `<LIG>_min_steric_contact` before
concluding anything about the design.

`<LIG>_needs_restrained_min` marks the < 2.8 Å cases: **71 of 274** structures.

## Collapsed ligand atoms — found and corrected

Boltz occasionally collapses the C24 carboxylate oxygens onto each other. Four staged
structures had two heavy atoms at **0.04–0.25 Å**, which is physically impossible and
would destroy an MD run immediately.

No pre-existing check caught this: ring-pucker QC only inspects the three steroid rings
(the carboxylate is a tail), pose-consistency compares seeds rather than internal
geometry, and ligand pLDDT was **0.94** on two of them — `BA_PYR1_1508/LCA3S` even
*passed* `pass_pose_consistency`.

For all four, other seeds were clean, so the representative was re-picked among
geometry-valid seeds of the same binding mode (medoid by Cα-superposed ligand RMSD):

| design | ligand | was | now |
|---|---|---|---|
| BA_PYR1_0399 | LCAM | seed 3, 0.254 Å | seed 5, 1.240 Å |
| BA_PYR1_0808 | LCA3S | seed 3, 0.053 Å | seed 1, 1.218 Å |
| BA_PYR1_1194 | LCA3S | seed 2, 0.148 Å | seed 1, 1.223 Å |
| BA_PYR1_1508 | LCA3S | seed 3, 0.040 Å | seed 2, 1.226 Å |

Minimum intra-ligand distance across all 274 is now **1.211 Å** (a normal carboxylate C–O
bond). `<LIG>_rep_seed_repicked` marks the four; `<LIG>_lig_min_intra` carries the number.
`aggregate_seeds.py` now bars defective seeds from becoming rep_seed, so this cannot recur.

## Why 5 seeds, and what 5 seeds does *not* buy

**Population statistics are converged.** Binder-vs-non-binder pocket-pLDDT AUC moves only
**0.949 → 0.954** (LCA) from 1 seed to 5, with across-subset sd 0.001 at k=4. More seeds
would not move any population number here.

**Per-design calls are not, and more seeds would largely not fix it.** A design that
clashes in 1 of 5 seeds has a Clopper–Pearson 95% CI of **[0.01, 0.72]**. 12.5% of designs
split 3–2 on binding mode, but under a true p=0.5 a 3–2 split is the *expected* outcome
62% of the time — so most of those are **genuinely bimodal designs rather than
undersampled ones**. More seeds would characterise them as two-state, not elect a winner.

**Treat `mode_majority` as a population descriptor, not a per-design truth claim.**

---

## Files

| file | contents |
|---|---|
| `pdbs/` | 274 structures, `<sequence_id>__<LCAM\|LCA3S>.pdb` |
| `md_handoff_selection.csv` | 137 rows; labels, pocket/protein sequence, all 16 balanced scores, pose QC, structure QC |
| `structure_qc.csv` | 274 rows; per-structure contact and ligand-geometry audit |
| `md_handoff_figure_data_long.csv` | long-form scores over the full 2,852-row labeled pool, for the balance figures |
| `scripts/` | `balance_select.py` (selection), `aggregate_seeds.py` (5-seed consensus), `build_md_handoff.py` (staging), `qc_md_handoff.py` (structure QC) |

## Reproduce

```bash
python3 scripts/balance_select.py --n-total 137 --mode both --restarts 10 \
  --feats plddt_ligand,plddt_pocket,plddt_protein,geometry_score,hbond_distance,iptm,affinity_probability_binary,n_interface_unsatisfied \
  --write --out md_set_nonbinder_ids.csv
python3 scripts/build_md_handoff.py --select-ids md_set_nonbinder_ids.csv --outdir .
python3 scripts/qc_md_handoff.py --apply
```

`qc_md_handoff.py` runs last and never changes the selection, so the balance above is
unaffected by it. Run it without `--apply` for a dry report.

Note: these scripts' default input paths point at the cluster working tree
(`/projects/ryde3462/lca_lca3s_modelling`, with the 5-seed prediction trees on
`/gpfs/alpine1/scratch`). They are included here as the **record of how this set was
built**; re-running them elsewhere needs those paths repointed.

## Known scope limits

- **Constitutive designs are not included.** All 248 × 2 ligands × 5 seeds have been
  predicted but are not yet scored or aggregated, so every constitutive number in this
  project is still single-seed and not comparable seed-for-seed with these rows. The
  three-class benchmark cannot be built yet. Folding them in may also populate the
  high-pocket-pLDDT region the selection is starved of, which would permit a larger
  balanced set than k=100.
- **`esm_apo_dg` is present** (5-seed median, `esm_apo_dg_n_seeds` = 5 throughout) but is
  deliberately **not** in the balanced feature set. ESM3dG is one of the orthogonal
  oracles the paper claims *succeeds*, so balancing it would engineer out the signal being
  tested. Co-folding metrics get balanced; oracles get audited and stay free to
  discriminate.
- These predictions use the **prd004 manifest scaffold**, which differs from the
  `pyr1_pipeline` `MASEL…` background at 8 non-pocket positions, so they are **not**
  directly comparable to earlier pipeline Boltz runs.
