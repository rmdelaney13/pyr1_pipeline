# MD pucker-recovery test — can relaxation/MD fix a flattened Boltz ligand?

8 **binary** (PYR1 + ligand, chains A/B) Boltz-2 cofolds — for each of the 4 ligands, one
with a **correctly bent** steroid nucleus and one **flattened** by cofolding. Same metric for
both so you can measure recovery after minimization / MD.

| file | ligand | kind | core_maxdev (Å) |
|------|--------|------|----------------:|
| LCA_correct_MLAAEAVLYWQFVFVT_maxdev1.23.pdb   | LCA   | correct   | 1.23 |
| LCA_flattened_LVFAEMILYEQYVFVN_maxdev0.29.pdb | LCA   | flattened | 0.29 |
| LCA3S_correct_IDLMVAVVLTKILGVI_maxdev1.23.pdb | LCA3S | correct   | 1.23 |
| LCA3S_flattened_DVFAVAVLFWQFVLVI_maxdev0.30.pdb | LCA3S | flattened | 0.30 |
| GLCA_correct_CLAAEGLLYWQHVLVT_maxdev1.23.pdb  | GLCA  | correct   | 1.23 |
| GLCA_flattened_VVAAEALLFWTFVFVT_maxdev0.29.pdb | GLCA | flattened | 0.29 |
| CDCA_correct_MLAGEALLFWGFVFVA_maxdev1.23.pdb  | CDCA  | correct   | 1.23 |
| CDCA_flattened_YLVAEALLFWTFVFVT_maxdev0.31.pdb | CDCA | flattened | 0.31 |

`manifest.csv` has the same info machine-readable. Chains: A = PYR1, B = ligand (resname LIG).

## The metric (`core_maxdev`)
Max out-of-plane deviation (Å) of the 17 fused-ring carbons from their SVD best-fit plane.
Physical 5β bile-acid nucleus (RDKit reference) ≈ **1.2 Å** (bent/kinked). Flattened cofolds
collapse toward **~0.3 Å** (all three 6-rings coplanar). NOTE: the *individual* rings stay
proper chairs (Cremer–Pople Q ≈ 0.58 Å) even when flat — the defect is in the **ring-fusion
geometry / overall bend**, not ring puckering per se. So "unpucker" is really "un-bend."

## Will minimization / MD fix it? (expectation)
**Mostly yes for geometry, with one important caveat.**
- The flat nucleus is a **strained, high-energy** conformation (bad sp3 ring-fusion torsions
  and valence angles). A proper small-molecule force field (GAFF2 / OpenFF) penalizes this, so
  **energy minimization** will relax it partway and **a short MD equilibration (~tens of ps,
  300 K)** should let it cross the small torsional barriers and recover the bent shape.
  Minimization *alone* often only partially recovers (nearest local minimum); add brief MD.
- **Caveat — chirality vs geometry.** MD/min can only repair *conformation*, never
  *configuration*. If Boltz flattened by effectively **inverting a ring-fusion stereocenter**
  (a true chirality error), MD cannot fix it. Which way it goes depends on how you parametrize:
  - Parametrize the ligand **from the intended SMILES** (OpenFF/RDKit template) → the FF's
    improper/chirality terms will actively push the flat coords back to correct → recovers.
  - Parametrize **from the flattened coordinates** (antechamber/GAFF read off this PDB) → the
    FF locks in whatever geometry/stereo is present → it relaxes to a self-consistent minimum
    that may still be wrong. **Avoid this.**

### How to score recovery
After min / MD, **don't only re-measure planarity** — also check ring-fusion stereo:
1. `core_maxdev` back toward ~1.2 Å  → bend recovered.
2. Ring-junction H–C–C–H dihedrals / per-center chirality unchanged from reference → no epimer.
Recompute `core_maxdev` with `../ligand_geometry_check/pucker3.py` logic (chain B + CONECT, or
distance bonds if MD output drops CONECT).

**Recommendation:** parametrize from SMILES, run min + short NVT, re-measure. If a flattened
case does *not* recover, suspect a stereocenter inversion → rebuild the ligand from SMILES
(RDKit ETKDG) and re-dock rather than trusting the cofold.

## Provenance
Binary source: `/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo` (`<pocket>__<LIG>`),
model_0. Selection: `/tmp/select_pucker.py` (correct = closest to 1.23 Å with maxdev ≥ 1.15;
flattened = global minimum maxdev per ligand). See `../LIGAND_GEOMETRY_FINDINGS.md`.
