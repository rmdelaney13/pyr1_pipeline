# Ligand geometry QC — Boltz flattens bile-acid sterol cores (worst: LCA-3-S & all ternary)

**TL;DR:** Boltz-2 cofolding sometimes **flattens the steroid nucleus** — collapsing the bent
5β bile-acid nucleus into a near-planar slab. This is a **general cofolding limitation**, not
LCA-3-S-specific, but it has two clear risk axes: (1) the **3-sulfate (LCA-3-S)** raises the
binary rate to 20% (vs 1–9% for the others), and (2) **adding HAB1 (ternary) makes it much
worse for every ligand** (pooled 9% binary → 28% ternary). Individual rings stay proper chairs
(Q≈0.58 Å); the defect is the **ring-fusion / overall-bend geometry**. The originally-selected
LCA3S MD structure (`VLLAVGVVLWQILGFM`) was one of the flattened ones and was swapped out
(→ `FDLMVVILYSKILGFM`, core_maxdev 1.33 Å).

## Method
For the ligand (chain B) in each Boltz model, using the deposited CONECT bonds:
- **Per-ring Cremer–Pople puckering amplitude Q** — chair ≈ 0.55–0.63 Å, flat/planar ≈ 0.
- **Overall nucleus planarity** — SVD best-fit plane through the 17 fused-ring carbons;
  report max out-of-plane deviation (`core_maxdev`, Å) and inter-ring mean-plane tilt angles.
Scripts: `ligand_geometry_check/` (`pucker*.py`, `sweep.py`, `ref.py`).

## Results
Individual rings are chairs in **every** ligand/model (Q ≈ 0.56–0.60 Å) — nothing wrong at
the ring level. The defect is at the nucleus level. **Full-population sweep** (model_0, every
design; `sweep_full.py`), % with a flattened core (core_maxdev < 0.7 Å):

| Ligand | n | binary median (Å) | binary %flat | binary %severe(<0.5) | ternary %flat | ternary %severe |
|--------|--:|------------------:|-------------:|---------------------:|--------------:|----------------:|
| LCA    | 381 | 1.20 |  6% |  4% | 38% | 13% |
| LCA3S  | 476 | 1.18 | **20%** | 12% | 34% | 27% |
| GLCA   | 249 | 1.19 |  9% |  7% | 30% | 23% |
| CDCA   | 404 | 1.14 |  1% |  1% | 10% |  6% |
| **pooled** | | | **9%** | | **28%** | |

Two effects: **(1) the sulfate** — LCA-3-S has the worst *binary* rate (20% vs 1–9%);
**(2) ternary cofolding** — adding HAB1 flattens the ligand ~3× more for every ligand
(9%→28% pooled), because model capacity shifts to the protein–protein interface. CDCA is the
most robust on both axes. *(An earlier 25-design alphabetical sample reported 52% for LCA3S —
that sample was clustered on flattening-prone pocket families; the population rate is 20%.)*

**Ground truth** (RDKit reference LCA conformers, `campaigns/LCA/conformers/conformers_final`):
`core_maxdev ≈ 1.23 Å`, inter-ring tilt `≈ [2, 64, 64]°`. Bile acids are 5β (cis A/B fusion) →
a *bent* nucleus is correct. Correctly-modeled cofolds reproduce this; flattened ones collapse
all three 6-ring planes near-parallel (`~[2, 4, 5]°`, core_maxdev ≈ 0.3 Å).

## Cause
Not an input error: the LCA3S input SMILES has fully-defined stereochemistry, and its ring
stereocenters are **byte-identical** to LCA — the only difference is `C3-O` → `C3-OSO3⁻`.
Boltz's structure module mishandles the sulfated bile acid, planarizing the fused-ring system
despite correct stereo input. (Related to the known Boltz 3-OH stereo inversion on LCA;
see memory `project_boltz_oh_stereo_contamination`.)

## Implications
- **Filter on `core_maxdev` before trusting any cofold's ligand pose/geometry.** Contamination
  is ~20% for LCA3S binary and ~28–38% for *ternary* of any bile acid — always QC ternary
  ligands, not just LCA3S. Suggested gate: `core_maxdev > ~0.9 Å`.
- **For MD:** a flattened ligand starts in a strained, non-physical conformation and will
  behave badly. Either (a) pick an LCA3S design whose core is not flattened, and/or
  (b) rebuild/energy-minimize the ligand (RDKit ETKDG conformer aligned into the pocket, or
  restrained ligand minimization) before launching.

## Clean LCA3S binders (not flattened), from the ligand-responsive shortlist
| pocket | ER_mean | core_maxdev | note |
|--------|--------:|------------:|------|
| VLLAVGVVLWQILGFM | 7.53 | 0.38 | **FLAT — original pick, avoid** |
| FDLMVVILYSKILGFM | 6.24 | 1.33 | clean; same pocket as the LCA pick (matched OH-vs-sulfate pair) |
| NDLMVSVLFWKILGVI | 5.15 | 1.21 | clean; distinct pocket |
| HDLMLAVVLTKIMGII | 4.39 | 1.18 | clean; distinct pocket |
