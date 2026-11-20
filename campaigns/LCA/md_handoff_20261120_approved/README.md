# Approved holo PYR1 MD hand-off — 280 starting structures, 2026-11-20

280 Boltz-2 binary PYR1–bile-acid starting structures for MD, reconciled against the
previous GitHub delivery. Each structure is one **(design, ligand) arm**: all eligible
clonal responders plus the approved matched negative controls.

| arm | binders | matched negatives | total |
|---|---:|---:|---:|
| **LCA** (`LCAM`, charge −1) | 51 | 94 | **145** |
| **LCA-3-S** (`LCA3S`, charge −2) | 36 | 99 | **135** |
| **Total** | **87** | **193** | **280** |

217 distinct protein sequences: 53 were in the previous delivery, 164 are new.
All 280 arm keys are unique and all 280 expected starting files are present.

> **Scope.** Holo PYR1 with LCA or LCA-3-S, raw Boltz `model_0` predictions only.
> No preparation or minimisation output is in this package; that belongs elsewhere.

---

## What is here

| file | contents |
|---|---|
| `pdbs/` | the 280 approved starting structures, `<sequence_id>__<ligand>__seed<N>.pdb` |
| `approved_holo_arms.csv` | the frozen roster: sequence, ligand, role, seed, scores, source path, destination name |
| `stage_verification.csv` | per-file source, byte count, SHA-256, sequence check, ligand check, geometry gate |
| `previous_delivery_crosswalk.csv` | outer join to the previous delivery by full protein sequence + ligand, with the previous Git PDB SHA-256. **Previous-only arms live here and nowhere else.** |
| `arm_identities.csv` | which arms are new, shared and previous-only |
| `shared_pose_RMSD.csv` | one row per shared arm: both hashes, fit statistics, ligand and core RMSD |
| `transfer_actions.csv` | per-arm transfer decision and the evidence behind it |
| `md_run_provenance.csv` | what MD actually exists for each arm (see the open question below) |
| `daily_clonal_records.csv` | daily clonal titration records, linked by `sequence_id` + `ligand` |
| `selection_audit.csv`, `matching_AUROCs.csv` | the approved selection, as delivered in the instruction bundle |
| `reconciliation_summary.json` | every count and summary statistic in this README, machine-readable |
| `SHA256SUMS.txt` | one checked hash per delivered file |
| `scripts/` | the code that produced the comparisons |

**The transfer list** is the 260 rows of `transfer_actions.csv` with `in_transfer_list=1`.
The other 20 are byte-identical to the previous delivery and the prior copy can supply
them after hash verification.

---

## 1. Verification of the starts

Each selected PDB was copied byte-for-byte from its original scratch prediction path
(source and staged SHA-256 both recorded and asserted equal). On the **staged** bytes:

| check | result |
|---|---|
| protein sequence == roster `protein_sequence` | **280 / 280** |
| residue numbering contiguous (1–181) | **280 / 280** |
| ligand heavy-atom set == reference (27 LCA / 31 LCA-3-S) | **280 / 280** |
| sulfate group present (LCA-3-S) | **135 / 135** |
| 7 stereocentres agree with reference | **280 / 280** |
| closest heavy-atom pair ≥ 1.15 Å | **280 / 280** (tightest observed 1.209 Å) |
| no six-membered ring with 30° < θ < 150° | **280 / 280** |
| **full geometry gate** | **280 / 280** |

All 280 staged files have distinct SHA-256 (34.3 MB total). Every arm had all five seed
indices present; the selected seed is the highest-confidence gate-passing seed, ties
broken by lowest seed index. 8 arms had a tied maximum confidence.

The geometry gate here is the archived gate, not a reimplementation: it reads the same
AF3 userCCD CIFs (`ligands/af3_userccd/{LCAM,LCA3S}.cif`) that
`ligand_geometry_scan.py` used to gate seed selection. The 7 stereocentres
(C3 C5 C8 C9 C14 C17 C20) and the 3 six-membered rings are *derived* from that bond
graph, not hard-coded. C10 and C13 are quaternary and cannot epimerise.

## 2. Reconciliation with the previous delivery

Previous delivery: the 228-row roster at
[`e7d6a69:campaigns/LCA/md_handoff`](https://github.com/rmdelaney13/pyr1_pipeline/tree/e7d6a6967b43a5e93bb0bb0386e9551362aaf40a/campaigns/LCA/md_handoff),
read from the committed blobs. Not the all-candidate `md_arm_set_long.csv`, and not the
267-arm 30 October package. Arms are matched on **full protein sequence + ligand**;
identical sequence alone is not a match. `LCAM`↔LCA and `LCA3S`↔LCA-3-S.

| Ligand | Previous | Binders | Controls | Shared | New | Previous only | Same seed idx | Diff seed idx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LCA | 111 | 51 | 94 | 42 | 103 | 69 | 7 | 35 |
| LCA-3-S | 117 | 36 | 99 | 38 | 97 | 79 | 13 | 25 |
| **Total** | **228** | **87** | **193** | **80** | **200** | **148** | **20** | **60** |

Recomputed from the staged manifests; matches the specification exactly. Of the 200 new
arms, 6 use a previously delivered sequence with the *other* ligand. The 148
previous-only systems are history and are not in `pdbs/`.

**Old and current class labels are carried separately.** Seven shared arms changed class:

| sequence_id | ligand | previous | current | ligand RMSD (Å) |
|---|---|---|---|---:|
| BA_PYR1_0004 | LCA-3-S | nonbinder | binder | 0.244 |
| BA_PYR1_0086 | LCA-3-S | nonbinder | binder | 0.000 |
| BA_PYR1_0595 | LCA-3-S | nonbinder | binder | 0.220 |
| BA_PYR1_1328 | LCA-3-S | nonbinder | binder | 0.741 |
| BA_PYR1_1391 | LCA-3-S | nonbinder | binder | 0.356 |
| BA_PYR1_1392 | LCA-3-S | nonbinder | binder | 0.405 |
| BA_PYR1_1256 | LCA | binder | matched_negative | 0.237 |

## 3. Pose change in the 80 shared arms

All 80 measured; none unresolved. **80 / 80 previous blob hashes verified** against the
crosswalk. 20 pairs are byte-identical — exactly the 20 same-seed-index arms — and
record a verified zero RMSD. All 60 different-seed-index arms were examined
coordinate-wise.

**Method.** Current PYR1 is superposed onto previous PYR1 by least-squares rigid fit of
the **16 pocket Cα** atoms at full-sequence positions 59, 81, 83, 92, 94, 108, 110, 117,
120, 122, 141, 159, 160, 163, 164, 167. Residue identities were verified against the
full sequence before fitting (16/16 Cα mapped and verified in all 80 arms). The same
transform is applied to the ligand; **the ligands are never superposed on each other.**

Ligand RMSD is over **all** heavy atoms on a complete one-to-one map — 27 for LCA, 31 for
LCA-3-S, never a partial intersection — minimised over the symmetry-equivalent mappings
the reference chemistry permits. Those are the graph automorphisms of the reference
heavy-atom bond graph: |Aut| = 2 for LCA (carboxylate {O4, O4A}) and 12 for LCA-3-S
(carboxylate × the three sulfate terminals {OS1, OS2, OS3}). **`O1B`** — the C3 hydroxyl
in LCA, the sulfate-ester bridge in LCA-3-S — is chemically distinct and is never
equivalenced to either set. Every stereocentre and every core carbon is fixed by every
automorphism, so symmetry minimisation cannot launder a stereochemical difference into a
low RMSD. Both the plain name-mapped and the symmetry-corrected value are reported;
symmetry reduced the value in 25 of 80 arms, by at most 0.624 Å.

**Ligand heavy-atom RMSD after the pocket fit (Å), symmetry-corrected:**

| group | n | median | IQR | max |
|---|---:|---:|---:|---:|
| all measured | 80 | 0.287 | 0.092 – 0.534 | 3.672 |
| LCA | 42 | 0.273 | 0.146 – 0.554 | 3.672 |
| LCA-3-S | 38 | 0.332 | 0.000 – 0.486 | 1.343 |
| same seed index | 20 | 0.000 | 0.000 – 0.000 | 0.000 |
| different seed index | 60 | 0.380 | 0.256 – 0.559 | 3.672 |

**Review bins** — descriptive only. No automatic rerun and no equivalence claim follows
from them:

| ligand | <1 Å | 1–<2 Å | ≥2 Å |
|---|---:|---:|---:|
| LCA | 36 | 2 | 4 |
| LCA-3-S | 36 | 2 | 0 |

Fit quality: pocket-fit Cα RMSD median 0.108 Å (max 0.211). The secondary whole-PYR1 Cα
fit gives median 0.259 Å (max 0.444). It shifts the ligand RMSDs by at most 0.065 Å
(12 arms by more than 0.02 Å) and moves no arm across a review-bin boundary, so the
conclusions do not depend on the choice of fit.

**Steroid-core RMSD** over the 17 fused-ring carbons C1–C17 after the same fit: median
0.194 Å, max 3.495 Å. For all four ≥2 Å arms the core RMSD tracks the all-atom RMSD
(3.14–3.50 Å), so the **whole steroid has moved** in the pocket — this is not a tail or
sulfate wag around a held core:

| sequence_id | ligand | role | seed (new→old) | ligand RMSD | core RMSD |
|---|---|---|---|---:|---:|
| BA_PYR1_1375 | LCA | binder | 2 ← 1 | 3.672 | 3.495 |
| BA_PYR1_0592 | LCA | binder | 4 ← 3 | 3.269 | 3.300 |
| BA_PYR1_1392 | LCA | binder | 1 ← 2 | 3.170 | 3.217 |
| BA_PYR1_1328 | LCA | binder | 4 ← 5 | 3.140 | 3.143 |

The 1–<2 Å arms are BA_PYR1_0598 (LCA-3-S, 1.343; core 0.911) and BA_PYR1_0715
(LCA-3-S, 1.073; core 0.663) — here the core moved less than the whole ligand, i.e. tail
and sulfate reorientation — plus BA_PYR1_0655 (LCA, 1.281) and BA_PYR1_0266 (LCA, 1.072).

The previous manifest's `pose_rmsd_spread` and `pose_rmsd_max` are an across-seed spread
and a mean distance to a medoid. They are **different quantities** and are not used here
as the old-versus-new pairwise RMSD.

## 4. Transfer decisions

| evidence | arms | in package | in transfer list |
|---|---:|---:|---:|
| new sequence–ligand arm | 200 | 200 | 200 |
| shared arm, different starting coordinates | 60 | 60 | 60 |
| shared arm, identical file | 20 | 20 | 0 |
| **total** | **280** | **280** | **260** |

Every shared arm with different starting coordinates is in the transfer list **even when
the ligand RMSD is small**, and any previous trajectory for it started from a different
file. No arm fell into the "identical coordinates, hash differs on formatting alone"
case: every zero-RMSD pair here is byte-identical.

`delivered_previously`, `simulation_started` / `simulation_completed`, and
`approved_current_start` are three separate fields in `transfer_actions.csv` and are
never collapsed into one.

## 5. Daily experimental evidence

`daily_clonal_records.csv` is copied verbatim (hash-checked) and links to the roster by
`sequence_id` + `ligand`; all 212 rows link. 95 arms carry records across 19 observed
dates, with every observed date, hit call, source QC, exclusion, comment, confidence
interval and gain preserved.

`reported_daily_EC50_nM` is the reportable per-date value (184 rows).
`archived_fit_EC50_nM` is an audit field only. Missing and excluded values are blank —
including the excluded **14 September LCA-3-S** series, whose 7 rows carry no reported
EC₅₀.

The roster separates the two kinds of absence: 87 arms `reported daily EC50`, 8
`assayed; no reported daily EC50`, 185 `no linked clonal assay record`. No potency is
inferred for non-enriched candidates. A sequence that responds to only one ligand keeps
its ligand-specific labels on both arms.

---

## Open question for the PI: no MD-run provenance exists

**No existing simulation can be attributed to any arm in this package.** Searched:

| location | result |
|---|---|
| `/gpfs/alpine1/scratch/ryde3462/holo_md` | directory skeleton only — one backbone (`bb01`), **0 files** |
| `/gpfs/alpine1/scratch/ryde3462/apo_md` | directory skeleton only, **0 files**; apo, not holo, and keyed `pair_NNNN` / `const_NN`, a namespace that appears nowhere in the roster or crosswalk |
| `/projects/ryde3462/lca_lca3s_modelling` | no trajectory, topology, preparation record or run log (`*.xtc *.dcd *.prmtop *.parm7 *.mdp *.tpr *.rst7 *.inpcrd topol.top`: 0 each) |

So `md_run_provenance.csv` records all 280 arms as
`unresolved: no MD-run provenance found`, and the 20 byte-identical shared arms —
the only arms where reuse could even be considered — **cannot** have existing dynamics
attributed to them. A matching start file shows the coordinates match; it does not show
that a simulation ran, nor that the simulated system was identical. **Decision needed:**
either locate the MD run records and preparation provenance for those 20 arms, or treat
all 280 as requiring fresh simulation.

For a primary analysis described as starting from the highest-confidence passing model,
use MD from that start, or document and separately analyse the exception.

---

## Reproducing

```bash
cd campaigns/LCA/md_handoff_20261120_approved/scripts
BUNDLE=/projects/ryde3462/lca_lca3s_modelling/62_Approved_holo_MD_cluster_handoff_20261120
PKG=$(cd .. && pwd)

python3 stage_starts.py   --roster $BUNDLE/tables/approved_holo_arms.csv --pkg $PKG
python3 compare_poses.py  --crosswalk $BUNDLE/tables/previous_delivery_crosswalk.csv \
                          --pkg $PKG --prior-repo /projects/ryde3462/software/pyr1_pipeline
python3 assemble_handoff.py --bundle $BUNDLE --pkg $PKG
sha256sum -c SHA256SUMS.txt
```

`lca_chem.py` holds the chemistry reference (bond graph, stereocentres, rings, core,
automorphisms) and the PDB parser; a path and a committed git blob go through the same
parser. `audit_holo.py` in the instruction bundle froze and audited the roster and is not
re-run here: the roster is frozen, hash-verified, and its selection is already reproduced
in `selection_audit.csv`.

Versions: see `versions` in `reconciliation_summary.json`.

## Methods sentence

For each ligand, we kept all eligible clonal responders and selected the largest subset
of candidate negatives for which each of three scores (PYR1 pLDDT, ligand pLDDT and
affinity-classifier probability) individually gave a binder-versus-control AUROC of
0.45–0.55, with ties receiving half credit. Non-enriched candidates and clonal titration
negatives were eligible on equal terms. This balances the marginal ranking of these
scores by design. Quantitative EC₅₀, pose similarity to prior starts, and availability of
an existing trajectory did not determine membership.
