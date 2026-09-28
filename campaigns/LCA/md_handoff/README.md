# PYR1 structures for LCA / LCA-3-S MD

Built 2026-09-28 by Ryan (rmdelaney13@gmail.com), for the apo/holo MD comparison
of PYR1 designs against LCA and LCA-3-S.

## The set

200 PYR1 designs. Each one has two Boltz-2 holo structures, one with LCA
(`LCAM`, charge -1) and one with LCA-3-S (`LCA3S`, charge -2), so 400 PDBs total.

The 200 are 35 binders and 165 nonbinders:

| class | n | where it comes from |
|-------|----|--------|
| binder | 35 | confirmed LCA / LCA-3-S / both binders (sort floor5 K125) |
| nonbinder | 165 | round-1 depletion negatives, matched to the binders (below) |
| constitutive | 0 | left out for now |

It's the same 165 sequences for both ligands, so you can look at LCA vs LCA-3-S
within a design rather than comparing two different sets.

## How the nonbinders were picked

They aren't random. I chose them so Boltz can't separate them from the binders on
the two things it uses to call binding, ligand pLDDT and the water-network geometry
score, looking at both ligands at once (four numbers per sequence). The matching is
a greedy nearest-neighbour draw (1:k, no reuse) in standardized 4-D feature space.

Here's what that does to how well each feature separates binders from nonbinders
(Mann-Whitney AUC, 0.5 means you can't tell them apart):

| feature | full nonbinder pool | matched set (sent) |
|---------|:---:|:---:|
| LCA ligand pLDDT | 0.834 | 0.558 |
| LCA geometry | 0.632 | 0.501 |
| LCA-3-S ligand pLDDT | 0.808 | 0.476 |
| LCA-3-S geometry | 0.305 | 0.459 |

So Boltz easily separates binders from an average nonbinder, but these 165 are the
ones it thinks look as good as the real binders. If the MD can tell them apart on
gate/latch motion or pocket collapse, that's dynamics telling us something the static
structure didn't, which is the reason to run it.

## Ligand distortion

I checked every ligand for busted ring geometry, which is the usual way Boltz goes
wrong (rings coming out as boat/twist instead of chair). The 6-membered rings are
found by connectivity (so atom naming doesn't matter) and scored with the
Cremer-Pople theta angle; a ring counts as distorted if theta is between 30 and 150.

All 400 came back clean, 3 chair rings each (steroid A/B/C), nothing distorted.
Numbers per structure are in `ligand_distortion_check.csv` (`rings6`,
`rings_distorted`, `ring_pucker_max`, and the raw `thetas`), and the same columns
are in the main CSV per ligand. This only covers the three 6-membered rings, not the
5-membered D-ring or bond-length issues, but the ring connectivity is intact
everywhere (rings6 = 3 for all 400). Re-run with
`scripts/check_ligand_distortion.py <pdb_dir>`.

## Files

- `md_handoff_selection.csv` — one row per design. Sequence, class, manifest label,
  pocket sequence, full protein sequence, match distance (0 for binders), then the
  Boltz metrics for each ligand (`LCAM_*` / `LCA3S_*`: ligand pLDDT, geometry and its
  sub-scores, ipTM, ligand-ipTM, H-bond distance/angle, confidence, affinity
  probability, binding mode, ring pucker, ESM apo dG) and the PDB path.
- `md_handoff_figure_data_long.csv` — long format for plotting, one row per
  design × ligand, covering the whole labeled set (35 binders + 1,389 nonbinders).
  Columns: sequence, class, `in_selection` (1 = sent to MD), ligand, ligand pLDDT,
  geometry. Filter this to redo the QC figure or make your own.
- `md_handoff_qc.png` — ligand pLDDT vs geometry, showing the matched nonbinders
  sitting on the binders while the rest of the pool falls away.
- `ligand_distortion_check.csv` — the per-structure ring check for all 400.
- `pdbs/` — the structures, `<sequence_id>__<LCAM|LCA3S>.pdb` (Boltz model_0,
  181-residue PYR1 monomer plus ligand, numbered 1-181; P88 gate, R116/L117 latch).

## A few things to know

- Structures are Boltz-2, no template, 5 diffusion seeds, model_0. MSA is a shared
  reference PYR1 unpaired a3m, query-stamped per design. Background is prd004, which
  differs from the old pyr1_pipeline MASEL background at 8 non-pocket positions.
- The nonbinder label is round-1 depletion ("weak negative, not confirmed biochemical
  non-binding"), so they're depleted, not clonally verified.
- LCA-3-S geometry is low for everyone, binders included. That score was tuned to
  LCA's 3-OH and doesn't carry over to the sulfate, so use ligand pLDDT for LCA-3-S
  and keep the geometry column mainly for LCA.
- To rebuild: `scripts/build_md_handoff.py` then `scripts/plot_md_handoff_qc.py`.
