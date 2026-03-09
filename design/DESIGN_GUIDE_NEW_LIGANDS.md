# PYR1 Design Pipeline: New Ligand Setup Guide

Step-by-step protocol for setting up a design campaign for any new ligand targeting the PYR1 ABA receptor pocket. Based on lessons from LCA Round 1 (0.6% hit rate) and Boltz2 retrospective.

---

## Overview

```
Conformers (RDKit) → Dock (glycine-shaved) → LigandMPNN → Rosetta relax/score
    → Filter (~1000) → [Expansion rounds] → Boltz2 filter → ~500 Twist pool
```

## Prerequisites

- SMILES string for your ligand
- Template PDBs: `<LIGAND>_H2O.pdb` (with ligand + water) and `<LIGAND>_nolig_H2O.pdb` (without ligand) in `docking/ligand_alignment/files_for_PYR1_docking/`
- PyRosetta + RDKit + LigandMPNN environments on Alpine

---

## Step 1: Conformer Generation

**Script**: `scripts/generate_<ligand>_conformers.py` (or use `ligand_conformers` package directly)

**API**:
```python
from ligand_conformers.config import ConformerConfig
from ligand_conformers.core import generate_conformer_set

cfg = ConformerConfig(num_confs=500, k_final=10, selection_policy="diverse")
result = generate_conformer_set(
    input_spec={"type": "smiles", "value": YOUR_SMILES},
    outdir=Path("conformers/<LIGAND>/"),
    cfg=cfg,
)
```

**Parameters**:
- `num_confs=500`: Initial embedding count (ETKDGv3)
- `k_final=10`: Final diverse conformers
- `cluster_rmsd_cutoff=1.25`: Butina clustering threshold (A)
- `selection_policy="diverse"`: Maximize coverage of conformational space

**Output**: `conformers_final.sdf` + individual PDB/SDF files + `.params` (Rosetta)

**Validation**: Open in PyMOL, check ring puckers and functional group orientations match expected chemistry.

---

## Step 2: Docking (Glycine-Shaved)

**Config**: `docking/templates/config_<LIGAND>.txt`
**Script**: `docking/scripts/grade_conformers_glycine_shaved.py`

**Key config settings**:
```ini
PrePDBFileName  = .../files_for_PYR1_docking/<LIGAND>_H2O.pdb
PostPDBFileName = .../files_for_PYR1_docking/<LIGAND>_nolig_H2O.pdb
GlycineShavePositions = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
EnableHBondGeometryFilter = True
ClusterRMSDCutoff = 1.0    # Keep lenient to retain max poses
```

**Critical**: The glycine-shaved positions are in **docked PDB numbering** (2 residues missing near position 70 in 3QN1 crystal). Boltz uses full sequence numbering (+2 offset for positions >= 72).

**Validation**: Check that docked poses show water-mediated H-bonds (ligand acceptor → water → P88/R116). Inspect cluster count — more is better for diversity.

---

## Step 3: LigandMPNN Design

### Config Files Needed

For each ligand, create in `design/mpnn/`:

1. **Omit JSON** (`<LIGAND>_omit_rosetta.json`): Lists amino acids **forbidden** at specific positions
   ```json
   {"A115": "F", "A157": "R"}
   ```
   - Listed AAs are EXCLUDED (cannot be sampled)
   - Positions not listed = all 20 AA allowed

2. **Bias JSON** (`<LIGAND>_bias_rosetta.json`): Positive/negative log-odds bias
   ```json
   {"A139": {"K": 3.0}, "A79": {"D": 1.0}, "A157": {"H": 1.5}}
   ```
   - Positive = encourage, negative = penalize
   - Typical biases: K at carboxylate-facing position, D for electrostatic network, H for gate H-bond

3. **Boltz versions** (`<LIGAND>_omit_boltz.json`, `<LIGAND>_bias_boltz.json`): Same restrictions with positions shifted +2 for positions >= 72.

### Numbering Offset (PDB vs Boltz)

The 3QN1 crystal structure is missing 2 residues near position 70. All existing codebase numbering matches the docked PDB. Boltz uses the full protein sequence, so positions >= 72 shift by +2.

| PDB (Rosetta) | Boltz | Common roles |
|---|---|---|
| 59 | 59 | gate |
| 79 | 81 | pocket |
| 81 | 83 | pocket |
| 90 | 92 | pocket |
| 92 | 94 | pocket |
| 106 | 108 | pocket |
| 108 | 110 | pocket |
| 115 | 117 | latch |
| 118 | 120 | pocket |
| 120 | 122 | pocket |
| 139 | 141 | carboxylate-facing |
| 157 | 159 | pocket |
| 158 | 160 | pocket |
| 159 | 161 | pocket |
| 161 | 163 | pocket |
| 162 | 164 | pocket |
| 165 | 167 | pocket |

### MPNN Shell Script

Copy `design/instructions/ligand_alignment_mpnni_grouped.sh` and modify:
- `--batch_size 20` (sequences per parent dock)
- `--redesigned_residues` (PDB numbering for initial, Boltz for expansion)
- `--omit_AA_per_residue` → your omit JSON
- `--bias_AA_per_residue` → your bias JSON
- `--temperature 0.3` (conservative; increase for more diversity)

### Design Pipeline Config

Create `design/<LIGAND>_config.txt` from `design/CONFIG_TEMPLATE.txt`:
```ini
[DEFAULT]
PIPE_ROOT = /projects/ryde3462/pyr1_pipeline
CAMPAIGN_ROOT = /projects/ryde3462/pyr1_pipeline/conformers/<LIGAND>
SCRATCH_ROOT = /scratch/alpine/ryde3462/<LIGAND>_design

[design]
DesignIterationRounds = 1
MPNNBatchSize = 20
DesignResidues = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
MPNNOmitDesignFile = %(PIPE_ROOT)s/design/mpnn/<LIGAND>_omit_rosetta.json
MPNNBiasFile = %(PIPE_ROOT)s/design/mpnn/<LIGAND>_bias_rosetta.json
LigandParams = %(CAMPAIGN_ROOT)s/<LIGAND>.params
LigandSMILES = <YOUR_SMILES>
FilterTargetN = 1000
FilterMaxUnsats = 1
FilterMaxPerParent = 20
```

### Run

```bash
python design/scripts/run_design_pipeline.py design/<LIGAND>_config.txt --wait
```

---

## Step 4: Rosetta Relax + Scoring

Handled automatically by the pipeline orchestrator. Key metrics computed:

| Metric | Description | Filter |
|--------|-------------|--------|
| `dG_sep` | Interface binding energy | Lower is better (rank) |
| `buried_unsatisfied_polars` | Broken H-bonds in interface | <= 1 |
| `O1_polar_contact` | Ligand polar atom 1 contacted | yes required |
| `O2_polar_contact` | Ligand polar atom 2 contacted | yes required |
| `charge_satisfied` | Carboxylate/amine groups satisfied | yes required |

**Red flags**: High `buried_unsats` = broken water network. Good binary but poor ternary = latch/lock failure. Ligand RMSD > 3A = wrong binding mode.

---

## Step 5: Filtering

**Script**: `design/instructions/relax_2_filter__allpolar_unsats.py`

Filtering hierarchy:
1. `buried_unsatisfied_polars <= 1` (hard cutoff — water network integrity)
2. **Bucket 1** (priority): All polar contacts satisfied + charge satisfied
3. **Bucket 2** (fill): At least one key polar contact (O1 or O2)
4. Diversity cap: max N designs per parent dock (default 20)
5. Rank by `dG_sep` within each bucket
6. Keep top `FilterTargetN` (default 1000)

---

## Step 6: Expansion Strategy

After initial round produces ~1000 designs:

1. **Select ~100 parents**: Top 50 by dG_sep + 50 from different ligand pose clusters (ensures sampling diverse OH contact modes)
2. **Re-run MPNN** on Boltz-predicted structures (CIF → PDB):
   - Use `<LIGAND>_omit_boltz.json` and `<LIGAND>_bias_boltz.json`
   - Use Boltz numbering for `--redesigned_residues`
   - 20 seqs per parent → 2000 new sequences
3. **Rosetta filter** → keep ~1000
4. **Combine** initial + expansion → ~2000 candidates

---

## Step 7: Boltz2 Filtering

**Mode**: MSA (not template) — MSA >> template for Boltz2 predictions.

**Best composite score** (from LCA retrospective, 0.74 pooled AUC):
```
Composite = Z(P(binder)) + Z(Interface pLDDT)
```

**Known issues**:
- 59R is a trivial negative — pre-filter before scoring
- Ternary Boltz2 has HAB1 bias — skip for filtering, use binary only
- LCA-3-S (sulfate) is hardest to predict correctly

**Pipeline**: Score all ~2000 candidates → rank by composite → select top ~500 for Twist pool.

---

## Step 8: Twist Synthesis Pool

- Target ~500 sequences (Twist 2-4 day turnaround)
- Include diversity: don't just take top 500 — ensure coverage of:
  - Multiple parent docking poses
  - Different position 59 identities (now fully open)
  - Both OH contact modes (for dihydroxy bile acids)
- Readouts: split-reporter FACS + NGS enrichment

---

## Bile Acid-Specific Notes

| Bile acid | OHs | Key consideration |
|-----------|-----|-------------------|
| LCA | 3α-OH only | Simplest; D79 bias 2.2 works |
| CDCA | 3α-OH + 7α-OH | 7α-OH changes pocket near pos 79; reduce D79 bias |
| DCA | 3α-OH + 12α-OH | 12α-OH on different ring face; may need flipped orientation |
| CA | 3α-OH + 7α-OH + 12α-OH | Most polar; hardest to satisfy all contacts |
| UDCA | 3α-OH + 7β-OH | 7β-OH equatorial (vs CDCA 7α axial); D79 bias = 0 |

**Cross-reactivity**: LCA library shows zero cross-reactivity to CA/CDCA. Each hydroxylation class needs separate design.

---

## File Checklist for New Ligand

```
design/mpnn/<LIGAND>_omit_rosetta.json       # Omit for initial round
design/mpnn/<LIGAND>_omit_boltz.json         # Omit for expansion (Boltz numbering)
design/mpnn/<LIGAND>_bias_rosetta.json       # Bias for initial round
design/mpnn/<LIGAND>_bias_boltz.json         # Bias for expansion
design/instructions/ligand_alignment_mpnni_grouped_<LIGAND>_rosetta.sh
design/instructions/ligand_alignment_mpnni_grouped_<LIGAND>_boltz.sh
design/<LIGAND>_config.txt                    # Pipeline config
docking/templates/config_<LIGAND>.txt         # Docking config
scripts/generate_<ligand>_conformers.py       # Conformer wrapper
docking/ligand_alignment/files_for_PYR1_docking/<LIGAND>_H2O.pdb
docking/ligand_alignment/files_for_PYR1_docking/<LIGAND>_nolig_H2O.pdb
conformers/<LIGAND>/conformers_final.sdf      # Generated conformers
conformers/<LIGAND>/<LIGAND>.params           # Rosetta params
```

---

## Lessons from LCA Round 1

1. **0.6% hit rate** (6/1000) — expect low initial hit rate, design for volume
2. **Position 59 matters**: Was restricted to exclude A/G/L/V — opening it up may improve diversity
3. **Buried unsats are the best Rosetta filter**: Directly indicates water network integrity
4. **Boltz2 MSA mode dominates**: Template mode performs worse
5. **Composite Z-score** (P(binder) + Interface pLDDT) = best single metric for filtering
6. **Assembly errors gave ~40 extra "hits"**: Validates that the binding pocket is designable
7. **NGS enrichment data** confirms computational predictions
