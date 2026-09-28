# MD hand-off set — LCA / LCA-3-S, binders + matched nonbinders

**Built:** 2026-09-28 · **Contact:** Ryan (rmdelaney13@gmail.com)
**For:** apo / holo MD comparison of PYR1 designs against **LCA** and **LCA-3-S**.

## What this is

200 PYR1 designs, each supplied with **two** Boltz-2 holo structures — one bound to
**LCA** (`LCAM`, protonation −1) and one bound to **LCA-3-S** (`LCA3S`, −2):

| class | n | source |
|-------|----|--------|
| **binder** | 35 | experimentally labeled LCA / LCA-3-S / both binders (sort floor5 K125) |
| **nonbinder** | 165 | round-1 depletion negatives, **distribution-matched** to the binders (see below) |
| constitutive | 0 | deliberately set aside for now |

So the deliverable is **400 PDBs** (200 designs × 2 ligands).

## Why the nonbinders were chosen this way (the important part)

The nonbinders are **not** a random draw. They were selected so that a **Boltz-only
classifier cannot tell them apart from the binders** on the two features that normally
discriminate binding: **ligand pLDDT** and **water-network geometry score**, evaluated
for *both* ligands (4 features total).

Matching = greedy nearest-neighbour (1:k, without replacement) in standardized 4-D
feature space. Effect on binder-vs-nonbinder separability (Mann-Whitney AUC, 0.5 = indistinguishable):

| feature | vs. **full** NB pool | vs. **matched** NB set (sent) |
|---------|:---:|:---:|
| LCA ligand pLDDT | 0.834 | **0.558** |
| LCA geometry score | 0.632 | **0.501** |
| LCA-3-S ligand pLDDT | 0.808 | **0.476** |
| LCA-3-S geometry score | 0.305 | **0.459** |

**Interpretation for the experiment:** Boltz already separates binders from the *average*
nonbinder. These 165 negatives are the ones Boltz thinks look just as good as the binders.
**Any binder/non-binder discrimination the MD achieves (apo gate/latch dynamics, pocket
collapse, etc.) is therefore signal that the static Boltz structure did not contain** — which
is the whole point of running the dynamics.

## Ligand distortion QC (all 400 structures pass)

Every ligand was checked for non-physical ring puckering — the main Boltz failure mode
(boat / twist / flattened rings). Each structure's 6-membered rings are detected
topologically (robust to atom naming) and tested with the Cremer-Pople theta angle;
a ring is flagged distorted when 30 < theta < 150 (i.e. not a chair).

**Result: 0 / 400 distorted.** All 400 structures show 3 chair 6-membered rings
(steroid A/B/C). Per-structure values are in `ligand_distortion_check.csv`
(`rings6`, `rings_distorted`, `ring_pucker_max`, and the raw `thetas`), and the same
`*_rings_distorted` / `*_ring_pucker_max` columns are in the main selection CSV per ligand.
Scope note: this checks the three 6-membered rings (the rings Boltz has historically
flattened); the 5-membered D-ring and bond-length distortion are not scored, but ring
connectivity is intact for all 400 (rings6 = 3 everywhere). Re-run with
`scripts/check_ligand_distortion.py <pdb_dir>`.

## Files

- `md_handoff_selection.csv` — one row per design (200). Columns: `sequence_id`, `class`,
  `manifest_label`, `pocket_sequence`, `protein_sequence`, `match_distance` (0 for binders),
  then per-ligand Boltz metrics (`LCAM_*`, `LCA3S_*`: ligand pLDDT, geometry & sub-scores,
  ipTM, ligand-ipTM, H-bond dist/angle, confidence, affinity prob, binding mode, ring pucker,
  ESM apo ΔG) and the PDB path (`LCAM_pdb`, `LCA3S_pdb`, relative to this dir).
- `md_handoff_figure_data_long.csv` — **figure-ready**, one row per (design × ligand) for
  the *entire* labeled set (35 binders + 1,389 nonbinders). Columns: `sequence_id`, `class`,
  `in_selection` (1 = sent to MD), `ligand`, `plddt_ligand`, `geometry_score`. Filter on
  `class`/`in_selection` to reproduce the QC figure or build your own.
- `md_handoff_qc.png` — QC scatter (ligand pLDDT vs geometry) showing matched NBs overlapping
  binders while the un-sent pool separates.
- `ligand_distortion_check.csv` — per-structure ring-pucker QC for all 400 PDBs (0 distorted).
- `pdbs/` — 400 structures named `<sequence_id>__<LCAM|LCA3S>.pdb` (Boltz `model_0`,
  181-residue PYR1 monomer + ligand, numbered 1–181; P88 = gate, R116, L117 = latch).

## Provenance / caveats

- Structures = Boltz-2, no template, 5 diffusion seeds, `model_0` written; MSA = shared
  reference PYR1 unpaired a3m, query-stamped per design (background = prd004; differs from the
  legacy pyr1_pipeline `MASEL` background at 8 non-pocket positions).
- Nonbinder label = "naive-supported round-1 depletion; weak negative, **not** confirmed
  biochemical non-binding." They are depleted, not clonally validated as non-binders.
- LCA-3-S geometry scores are low across the board (binders included) — the water-network
  geometry term was tuned to LCA's 3-OH and does not transfer cleanly to the 3-sulfate; rely on
  ligand pLDDT for LCA-3-S, and use the geometry column mainly for LCA.
- Regenerate everything with `scripts/build_md_handoff.py` then `scripts/plot_md_handoff_qc.py`.
