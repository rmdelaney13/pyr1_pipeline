# PYR1 ML Dataset Generation

Machine learning dataset generation for PYR1 biosensor design prediction.

## Overview

This component extends the pyr1_pipeline to generate high-quality ML training datasets for predicting PYR1-ligand binding. It integrates conformer generation, Rosetta docking/relax, and AlphaFold3 predictions to create multi-feature representations of (ligand, PYR1 variant) pairs.

## Project Goals

1. **Generate balanced dataset** of PYR1 variant-ligand pairs (positives + negatives)
2. **Extract structural features** from Rosetta and AF3 predictions
3. **Train ML models** for binder/non-binder classification and affinity ranking
4. **Enable prospective design** of novel PYR1 biosensor variants

## Directory Structure

```
ml_modelling/
├── README.md                   # This file
├── data/                       # Input data and ligand information
│   ├── ligand_smiles_signature.csv     # Positive pairs (ligand + variant)
│   ├── manual_smiles_lookup.csv        # Manual SMILES curation
│   └── *.xlsx                          # Excel data files (gitignored)
├── scripts/                    # ML pipeline orchestration
│   ├── orchestrate_ml_dataset_pipeline.py  # Main orchestrator
│   ├── aggregate_ml_features.py            # Feature aggregation
│   ├── process_ligand_smiles.py            # SMILES processing
│   ├── validate_ligand_smiles.py           # SMILES validation
│   ├── fetch_missing_smiles.py             # Auto-fetch from PubChem
│   └── merge_ligand_data.py                # Data merging utilities
├── docs/                       # Project planning and documentation
│   ├── REVISED_PROJECT_PLAN.md             # Current project plan
│   ├── IMPLEMENTATION_GUIDE.md             # Implementation details
│   ├── IMPLEMENTATION_COMPLETE.md          # Completed milestones
│   └── *.md                                # Other planning docs
├── cache/                      # Computational outputs (gitignored)
│   └── {pair_id}/                          # Per-pair results
│       ├── conformers/
│       ├── docking/
│       ├── relax/
│       ├── af3_binary/
│       └── af3_ternary/
└── results/                    # Final datasets (gitignored)
    ├── features_pilot.csv
    ├── features_table.csv
    └── train_val_test_splits/
```

## Key Features

### Affinity-Stratified Dataset Design
- **P1 positives**: EC50 < 1 μM (high-confidence binders)
- **P2 positives**: EC50 1-10 μM (moderate binders)
- **N1 negatives**: Expression+/Activation- (hard negatives from FACS)
- **N2 negatives**: Near-neighbor variants (1-2 mutations from positives)
- **N3 negatives**: Random pocket variants (calibration)

### Feature Extraction Pipeline
1. **Conformer generation** (RDKit ETKDG)
2. **Rosetta docking** with variant threading
3. **Rosetta relax** for energy minimization
4. **AlphaFold3** binary and ternary predictions
5. **Feature aggregation** (40+ metrics per pair)

### Computational Efficiency
- **Phase 1 pilot**: 200 pairs, ~1 day on 500 cores + 4 A100s
- **Phase 2 production**: 1,500 pairs, ~3 days
- **AF3 with templates**: 20-30 seconds per prediction (vs 30+ min without)

## Dependencies

This module depends on:
- `../docking/` - Docking pipeline scripts
- `../design/` - Design pipeline components (optional)
- `../ligand_conformers/` - Conformer generation
- `../scripts/thread_variant_to_pdb.py` - Mutation threading
- `../templates/` - Configuration templates

## Quick Start

### 1. Prepare Input Data
```bash
# Ensure ligand_smiles_signature.csv contains your positive pairs
head ml_modelling/data/ligand_smiles_signature.csv
```

### 2. Run Pipeline
```bash
# From pyr1_pipeline root directory
python ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py \
    --input ml_modelling/data/ligand_smiles_signature.csv \
    --output ml_modelling/cache \
    --config templates/unified_config_template.txt
```

### 3. Aggregate Features
```bash
python ml_modelling/scripts/aggregate_ml_features.py \
    --cache ml_modelling/cache \
    --output ml_modelling/results/features_table.csv
```

## Project Phases

### Phase 0: Foundation (2 weeks)
- [x] Affinity annotation and stratification
- [x] Mutation threading script
- [ ] Negative dataset curation
- [ ] Pilot validation (30 pairs)

### Phase 1: Pilot Dataset (2 weeks)
- [ ] 200 pairs across 3 ligand families
- [ ] Feature table with 40+ metrics
- [ ] Quality control analysis
- [ ] Preliminary ML models

### Phase 2: Production Dataset (4 weeks)
- [ ] 1,500 pairs across all validated ligands
- [ ] Train/val/test splits
- [ ] Baseline ML models (logistic regression, RF, XGBoost, MLP)
- [ ] Feature importance analysis
- [ ] Dataset publication

## Key Scripts

### orchestrate_ml_dataset_pipeline.py
Main orchestration script that:
- Processes ligand SMILES
- Threads variant mutations onto PYR1 template
- Runs conformer generation → docking → relax → AF3
- Tracks progress and caching

### aggregate_ml_features.py
Feature extraction and aggregation:
- Rosetta scores (dG_sep, buried_unsats, sasa, hbonds)
- AF3 metrics (ipTM, pLDDT, interface PAE, ligand RMSD)
- Conformer properties (MMFF energy, rotatable bonds)
- Affinity annotations (EC50, positive_tier)

## Documentation

See `docs/` for detailed planning:
- [REVISED_PROJECT_PLAN.md](docs/REVISED_PROJECT_PLAN.md) - Complete project plan
- [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Implementation details
- [IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md) - Progress tracking

## Expected Outputs

### Feature Table (features_table.csv)
Columns include:
- `pair_id`, `ligand_name`, `variant_signature`
- `positive_tier` (P1/P2/P3 or null for negatives)
- `negative_tier` (N1/N2/N3 or null for positives)
- `EC50_uM`, `assay_type`
- `rosetta_dG_sep`, `rosetta_buried_unsats`, `rosetta_sasa`
- `af3_binary_ipTM`, `af3_binary_pLDDT`, `af3_ternary_ipTM`
- `conformer_mmff_energy`, `num_rotatable_bonds`
- ... (40+ features total)

### Train/Val/Test Splits
- Ligand-stratified splits (60/20/20)
- Affinity-aware stratification
- Variant family hold-outs

## Success Criteria

### Phase 1 → Phase 2 Gate
- [ ] Pilot AUC (P1 vs N2) ≥ 0.65
- [ ] Affinity correlation |r| ≥ 0.3
- [ ] Pipeline wall time < 1 day
- [ ] < 5% missing data

### Phase 2 → Publication Gate
- [ ] Test set AUC ≥ 0.75
- [ ] Cross-tier consistency
- [ ] ≥1,350 complete pairs
- [ ] Top-3 features identified

## Related Documentation

- [../README.md](../README.md) - Main pyr1_pipeline README
- [../QUICK_START.md](../QUICK_START.md) - Pipeline quick start
- [../docking/WORKFLOW_README.md](../docking/WORKFLOW_README.md) - Docking details
- [../design/DESIGN_PIPELINE_README.md](../design/DESIGN_PIPELINE_README.md) - Design details

## Citation

If you use this dataset or pipeline, please cite:
- [Your lab's publications]
- PYR1 Pipeline: https://github.com/rmdelaney13/pyr1_pipeline

---

**Status**: Phase 0 (Foundation) - In Progress

**Last Updated**: 2026-02-16
