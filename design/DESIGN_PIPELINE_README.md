# PYR1 Pipeline: Two Workflows, Shared Components

This document describes the **two distinct workflows** built on the pyr1_pipeline codebase and clarifies how shared components (especially Rosetta relax) serve each one.

---

## At a Glance

| | Ligand Design Pipeline | ML Dataset Pipeline |
|---|---|---|
| **Goal** | Design new PYR1 variants for a specific ligand | Generate training data for ML binding prediction |
| **Input** | Docked PDBs (from docking pipeline) + config.txt | Pairs CSV (ligand SMILES + variant signature) |
| **Orchestrator** | `design/scripts/run_design_pipeline.py` | `ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py` |
| **Scope** | One ligand-variant complex at a time | Hundreds of (ligand, variant) pairs |
| **Core loop** | MPNN → Rosetta relax → filter → AF3 prep | Conformers → thread → relax → dock → cluster → relax → AF3 |
| **Output** | AF3-ready JSON inputs (~1000 designs) | Feature table (40+ metrics per pair) for ML training |
| **Relax script** | `design/rosetta/relax_general_universal.py` | Stage 4: `ml_modelling/scripts/constrained_relax.py` (backbone); Stage 7: `relax_general_universal.py` (interface) |

---

## Workflow 1: Ligand Design Pipeline

**When to use:** You have a good docked pose and want to design optimized PYR1 variants for that ligand, then validate with AlphaFold3.

### Flow

```
Clustered docked PDBs (from docking/)
    |
    v
[1] LigandMPNN sequence design
    |  - Design residues specified in config
    |  - 40 sequences per parent template (configurable)
    |  - Temperature 0.3 (configurable)
    v
[2] Rosetta relax & scoring (relax_general_universal.py)
    |  - Thread MPNN sequence onto parent PDB
    |  - FastRelax interface region
    |  - Score: dG_sep, buried_unsats, polar contacts, charge satisfaction
    v
[3] Filter designs
    |  - Remove buried unsats > threshold
    |  - Require polar contacts and charge satisfaction
    |  - Enforce diversity (max designs per parent dock)
    |  - Keep top N (default 1000)
    v
[4] (Optional) Iterate: feed filtered PDBs back into MPNN
    v
[5] Generate FASTA + extract SMILES from SDF
    v
[6] Create AF3 JSON inputs (binary + ternary)
    v
AF3 execution (manual, GPU cluster)
```

### Key Files

| File | Role |
|---|---|
| `design/scripts/run_design_pipeline.py` | Main orchestrator (1165 lines) |
| `design/rosetta/relax_general_universal.py` | Relax + scoring (universal, auto-detects ligand) |
| `design/rosetta/relax_filter_universal.py` | Filter by Rosetta metrics |
| `design/instructions/ligand_alignment_mpnni_grouped.sh` | MPNN SLURM template |
| `design/instructions/submit_pyrosetta_general_threading_relax.sh` | Rosetta SLURM template |
| `design/instructions/aggregate_scores.py` | Score CSV aggregation |
| `design/instructions/split_and_mutate_to_fasta.py` | FASTA generation |

### Config (add `[design]` section to config.txt)

```ini
[design]
DesignRoot = design
DesignIterationRounds = 1
DesignResidues = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
LigandParams = %(CAMPAIGN_ROOT)s/conformers/0/0.params
LigandSDF = %(CAMPAIGN_ROOT)s/conformers/0/0.sdf
FilterTargetN = 1000
FilterMaxUnsats = 1
FilterMaxPerParent = 20
```

### Running

```bash
# Full pipeline (one command)
python design/scripts/run_design_pipeline.py config.txt

# Skip stages
python design/scripts/run_design_pipeline.py config.txt --skip-mpnn
python design/scripts/run_design_pipeline.py config.txt --af3-prep-only

# Dry run (generates scripts without submitting)
python design/scripts/run_design_pipeline.py config.txt --dry-run
```

### Output

```
$SCRATCH_ROOT/design/
├── iteration_1/
│   ├── mpnn_output/        # LigandMPNN sequences
│   ├── rosetta_output/     # Relaxed PDBs + score files
│   ├── scores/             # Aggregated CSV
│   └── filtered/           # Filtered PDBs + FASTA
├── iteration_2/            # If DesignIterationRounds=2
└── af3_inputs/
    ├── binary/             # AF3 JSON inputs (protein + ligand)
    └── ternary/            # AF3 JSON inputs (protein + ligand + HAB1)
```

### More Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick reference
- [INDIVIDUAL_STEPS_GUIDE.md](INDIVIDUAL_STEPS_GUIDE.md) - Run stages individually
- [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) - First-time setup

---

## Workflow 2: ML Dataset Pipeline

**When to use:** You need a feature-rich dataset of many (ligand, variant) pairs for training ML models to predict PYR1-ligand binding.

### Flow

```
Pairs CSV (ligand_name, smiles, variant_signature, label)
    |
    v
FOR EACH (ligand, variant) PAIR:
    |
    [1] Conformer generation (RDKit ETKDG, 10 per ligand)
    |
    [2] Alignment table (H-bond acceptor identification)
    |
    [3] Thread mutations onto WT PYR1 template → mutant.pdb
    |
    [4] Backbone-constrained relax (constrained_relax.py)
    |   - CoordinateConstraints on N, CA, C, O (SD=0.5 A)
    |   - Relieves threading strain, keeps backbone near crystal
    |
    [5] Dock conformers to mutant pocket (50 repeats)
    |   - Direct-to-mutant (no glycine shaving)
    |   - SVD alignment using template ligand
    |
    [6] Cluster docked poses + extract convergence metrics
    |   - RMSD cutoff 0.75 A
    |   - Convergence ratio, best score, clash flag
    |
    [7] Rosetta relax best docked poses (relax_general_universal.py)
    |   - SKIPPED if clash detected (best_score > 0)
    |   - Top 20 poses otherwise
    |
    [8] AF3 predictions (binary + ternary) [not yet implemented]
    |
    v
AGGREGATE: features_table.csv (40+ columns per pair)
```

### Key Files

| File | Role |
|---|---|
| `ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py` | Main orchestrator (1379 lines) |
| `ml_modelling/scripts/constrained_relax.py` | Backbone-constrained FastRelax (Stage 4) |
| `ml_modelling/scripts/aggregate_ml_features.py` | Feature extraction → CSV |
| `ml_modelling/scripts/prepare_pairs_dataset.py` | Build pairs CSV from ligand/variant data |
| `ml_modelling/scripts/submit_relax_ml.sh` | SLURM wrapper for Stage 7 relax |
| `ml_modelling/scripts/submit_constrained_relax.sh` | SLURM wrapper for Stage 4 relax |
| `ml_modelling/data/ligand_smiles_signature.csv` | 287 validated binder pairs |

### Running

```bash
# Local mode (small tests)
python ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py \
    --pairs-csv ml_modelling/data/ligand_smiles_signature.csv \
    --cache-dir /scratch/ml_dataset_cache \
    --template-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_nolig_H2O.pdb \
    --reference-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb \
    --docking-repeats 50 \
    --max-pairs 3

# SLURM mode (production)
python ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py \
    --pairs-csv ml_modelling/data/ligand_smiles_signature.csv \
    --cache-dir /scratch/ml_dataset_cache \
    --template-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_nolig_H2O.pdb \
    --reference-pdb docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb \
    --docking-repeats 50 --use-slurm --docking-arrays 10

# Aggregate features after all pairs complete
python ml_modelling/scripts/aggregate_ml_features.py \
    --cache /scratch/ml_dataset_cache \
    --output ml_modelling/results/features_table.csv
```

### Dataset Strategy

| Tier | Label | Definition | Example |
|---|---|---|---|
| P1 | Positive | EC50 < 1 uM | WIN + WT PYR1 |
| P2 | Positive | EC50 1-10 uM | nitazene + PYR1^nitav2 |
| N1 | Negative | Expression+/Activation- (FACS) | Hard negatives |
| N2 | Negative | 1-2 mutations from a positive | Near-neighbor |
| N3 | Negative | Random pocket variants | Calibration |

### Output

```
ml_modelling/cache/{pair_id}/
├── conformers/conformers_final.sdf
├── mutant.pdb
├── mutant_relaxed.pdb
├── docking/hbond_geometry_summary.csv
├── relax/relaxed_*.pdb
└── metadata.json            # Stage completion tracking

ml_modelling/results/
├── features_table.csv       # 40+ columns per pair
└── train_val_test_splits/   # Ligand-stratified
```

### More Documentation

- [../ml_modelling/README.md](../ml_modelling/README.md) - ML pipeline overview
- [../ml_modelling/docs/REVISED_PROJECT_PLAN.md](../ml_modelling/docs/REVISED_PROJECT_PLAN.md) - Full project plan
- [../ml_modelling/docs/IMPLEMENTATION_GUIDE.md](../ml_modelling/docs/IMPLEMENTATION_GUIDE.md) - Technical details
- [../ml_modelling/docs/END_TO_END_WORKFLOW.md](../ml_modelling/docs/END_TO_END_WORKFLOW.md) - Step-by-step walkthrough

---

## Shared Components

### relax_general_universal.py

**Location:** `design/rosetta/relax_general_universal.py`

**Used by both pipelines** for interface relaxation and scoring of protein-ligand complexes.

| Feature | What it does |
|---|---|
| Auto-detect polar atoms | Scans ligand for N, O, S atoms; generates per-atom scoring columns |
| Detect charged groups | Identifies carboxylates, amines, sulfonates, phosphates |
| FastRelax | Relaxes interface residues (10 A shell around ligand) |
| Interface scoring | Runs `interface_scoring.xml` for dG_sep, buried_unsats, shape_comp, etc. |
| Polar contact check | Per-atom distance to nearest protein atom (< 3.5 A = contact) |
| Charge satisfaction | Carboxylates need 2 H-bonds or salt bridge; amines need 1 |
| Water handling | Keeps water FIXED; optional water constraint mode |

**Usage in Design Pipeline (Stage 2):**
- Relaxes MPNN-designed sequences threaded onto parent docked PDBs
- Scores to decide which designs pass filtering
- Called via `design/instructions/submit_pyrosetta_general_threading_relax.sh`

**Usage in ML Pipeline (Stage 7):**
- Relaxes best docked poses to extract Rosetta features
- Skipped entirely if clash detected in docking
- Called via `ml_modelling/scripts/submit_relax_ml.sh`

**Arguments:**
```bash
python relax_general_universal.py input.pdb output.pdb ligand.params \
    --xml_path docking/ligand_alignment/scripts/interface_scoring.xml \
    --ligand_chain B \
    --water_chain D \
    --skip_water_constraints  # Use when water positions are uncertain
```

### constrained_relax.py (ML Pipeline only)

**Location:** `ml_modelling/scripts/constrained_relax.py`

This is a **different script** from `relax_general_universal.py`. It relaxes protein-only structures (no ligand) after mutation threading.

| Feature | What it does |
|---|---|
| CoordinateConstraints | Backbone N, CA, C, O atoms pinned to crystal positions (SD=0.5 A) |
| FastRelax | Full relax with constraint weights |
| No ligand | Protein-only (runs before docking) |
| Purpose | Relieve steric strain from threading while preserving backbone |

**Only used by ML pipeline Stage 4.** The design pipeline does not thread mutations itself (it receives pre-docked PDBs and uses MPNN for sequence design).

### Docking Scripts

Both pipelines rely on the docking machinery in `docking/ligand_alignment/scripts/`:
- `grade_conformers_dock_to_sequence_v2.py` - Main docking script
- `cluster_and_extract_best_v2.py` - Cluster docked poses
- `interface_scoring.xml` - Rosetta XML scoring protocol

The **design pipeline** uses docking output as its starting point (clustered PDBs). The **ML pipeline** runs docking internally as Stage 5.

### Template PDBs

| File | Contents | Used by |
|---|---|---|
| `docking/.../3QN1_nolig_H2O.pdb` | WT PYR1, no ligand, with waters | ML pipeline (threading template) |
| `docking/.../3QN1_H2O.pdb` | WT PYR1 with ABA ligand + waters | Both (alignment reference for docking) |

---

## How the Pipelines Relate

```
                    ┌──────────────────────────┐
                    │   DOCKING PIPELINE        │
                    │   (runs first)            │
                    │                           │
                    │  Ligand SDF + Mutant PDB  │
                    │        |                  │
                    │  Dock → Cluster → PDBs    │
                    └──────────┬───────────────┘
                               |
              ┌────────────────┼──────────────────┐
              |                                    |
              v                                    v
┌──────────────────────────┐       ┌───────────────────────────────┐
│  DESIGN PIPELINE          │       │  ML DATASET PIPELINE           │
│  (one complex)            │       │  (many pairs)                  │
│                           │       │                                │
│  Docked PDBs              │       │  Pairs CSV                     │
│    |                      │       │    |                           │
│  MPNN design              │       │  Thread + constrained relax    │
│    |                      │       │    |                           │
│  relax_general_universal  │       │  Dock (internally)             │
│    |                      │       │    |                           │
│  Filter by Rosetta        │       │  Cluster                       │
│    |                      │       │    |                           │
│  FASTA + AF3 JSONs        │       │  relax_general_universal       │
│                           │       │    |                           │
│  → 1000 designs for AF3   │       │  AF3 (future)                  │
│                           │       │    |                           │
│                           │       │  features_table.csv            │
│                           │       │  → ML model training           │
└──────────────────────────┘       └───────────────────────────────┘
```

**Design pipeline** answers: "Given this docked ligand, what PYR1 sequences bind it best?"

**ML pipeline** answers: "Across many ligand-variant pairs, what structural features predict binding?"

They can run independently. The design pipeline requires prior docking output. The ML pipeline runs docking internally for each pair.

---

## Quick Reference: Which Script Does What

### Relax Scripts

| Script | Input | Output | Ligand? | Constraints | Used by |
|---|---|---|---|---|---|
| `relax_general_universal.py` | Protein-ligand PDB + params | Relaxed PDB + scores | Yes | None (interface only) | Both pipelines |
| `constrained_relax.py` | Protein-only PDB | Relaxed PDB | No | Backbone coordinate | ML pipeline only |

### Orchestrators

| Script | Scope | SLURM? | Caching? |
|---|---|---|---|
| `design/scripts/run_design_pipeline.py` | Single complex, iterative design | Yes (auto-generates scripts) | No (stage skipping via flags) |
| `ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py` | Many pairs, feature extraction | Yes (array jobs) | Yes (metadata.json per pair) |

### SLURM Wrappers

| Script | Submits | Pipeline |
|---|---|---|
| `design/instructions/ligand_alignment_mpnni_grouped.sh` | MPNN array job | Design |
| `design/instructions/submit_pyrosetta_general_threading_relax.sh` | Rosetta relax array job | Design |
| `ml_modelling/scripts/submit_constrained_relax.sh` | Backbone relax array job | ML |
| `ml_modelling/scripts/submit_relax_ml.sh` | Interface relax array job | ML |

---

## SLURM Notes (Alpine Cluster)

Both pipelines target the same cluster:
- **Partition:** `amilan`
- **QOS:** `normal`
- **Account:** `ucb472_asc2`
- **Scratch:** `/scratch/alpine/ryde3462/`
- **PyRosetta:** 2025.13

SLURM jobs start in `$HOME`, not the submit directory. All scripts use `cd "$SLURM_SUBMIT_DIR"` or absolute paths.

---

## Common Tasks

### "I want to design variants for a new ligand"

1. Run docking first: `python docking/scripts/run_docking_workflow.py config.txt`
2. Run design: `python design/scripts/run_design_pipeline.py config.txt`
3. Submit AF3 JSONs from `$SCRATCH_ROOT/design/af3_inputs/`

### "I want to add new pairs to the ML dataset"

1. Add rows to `ml_modelling/data/ligand_smiles_signature.csv`
2. Re-run orchestrator (it will skip completed pairs via cache)
3. Re-aggregate features

### "I want to change how relax scoring works for both pipelines"

Edit `design/rosetta/relax_general_universal.py`. Both pipelines call this script for interface relaxation.

### "I want to change backbone constraints for threading"

Edit `ml_modelling/scripts/constrained_relax.py`. Only the ML pipeline uses this (design pipeline doesn't thread).
