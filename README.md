# PYR1 Pipeline: Automated Docking + Design

Complete automated workflow from ligand SDF files to AF3-ready protein designs.

---

## 🎯 What This Does

**Input:** SMILES, PubChem CID, or ligand structure file

**Output:** Filtered protein sequences ready for AlphaFold3 prediction

**Automation:**
- ✅ Conformer generation (SMILES/CID/file → diverse low-energy conformers)
- ✅ Docking (SDF → Clustered poses)
- ✅ LigandMPNN sequence design
- ✅ Rosetta relaxation and filtering
- ✅ AF3 JSON generation with SMILES

**Manual steps:** AF3 execution (GPU cluster), AF3 analysis

---

## 🚀 Quick Start

**5-minute setup, 2 commands to run:**

```bash
# 1. Copy and edit config
cp templates/unified_config_template.txt /scratch/youruser/project/config.txt
vim /scratch/youruser/project/config.txt

# 2. Run docking
python docking/scripts/run_docking_workflow.py config.txt

# 3. Run design
python design/scripts/run_design_pipeline.py config.txt
```

**See:** [QUICK_START.md](QUICK_START.md)

---

## 📚 Documentation

### **Getting Started**

| Document | Use When |
|----------|----------|
| [QUICK_START.md](QUICK_START.md) | You want to run the pipeline NOW |
| [templates/CONFIG_GUIDE.md](templates/CONFIG_GUIDE.md) | You need to understand config parameters |
| [design/SETUP_CHECKLIST.md](design/SETUP_CHECKLIST.md) | First-time setup on cluster |
| [INTEGRATED_PIPELINE_SUMMARY.md](INTEGRATED_PIPELINE_SUMMARY.md) | You want the big picture |

### **Detailed References**

| Document | What It Covers |
|----------|----------------|
| [docking/WORKFLOW_README.md](docking/WORKFLOW_README.md) | Complete docking pipeline |
| [docking/QUICKSTART.md](docking/QUICKSTART.md) | Docking quick reference |
| [design/DESIGN_PIPELINE_README.md](design/DESIGN_PIPELINE_README.md) | Complete design pipeline |
| [design/QUICKSTART.md](design/QUICKSTART.md) | Design quick reference |
| [design/scripts/README.md](design/scripts/README.md) | Script details and debugging |
| [ml_modelling/README.md](ml_modelling/README.md) | ML dataset generation |
| [ml_modelling/docs/REVISED_PROJECT_PLAN.md](ml_modelling/docs/REVISED_PROJECT_PLAN.md) | ML project plan |
| [templates/README.md](templates/README.md) | Config templates overview |

---

## 📁 Repository Structure

```
pyr1_pipeline/
├── ligand_conformers/          # Conformer generation (pre-docking)
│   ├── __init__.py                        Public API
│   ├── __main__.py                        CLI entrypoint
│   ├── config.py                          ConformerConfig dataclass
│   ├── core.py                            Generation funnel engine
│   ├── io_utils.py                        I/O (resolve input, PDB/SDF writers)
│   └── openmm_refine.py                   Optional OpenMM refinement
│
├── docking/                    # Docking pipeline
│   ├── scripts/
│   │   ├── run_docking_workflow.py          Main docking orchestrator
│   │   ├── create_table.py                  SDF → params/alignment
│   │   └── cluster_docked_post_array.py     Post-docking clustering
│   ├── templates/
│   │   └── config.txt                       Docking-only config (legacy)
│   └── WORKFLOW_README.md                   Docking documentation
│
├── design/                     # Design pipeline
│   ├── scripts/
│   │   ├── run_design_pipeline.py          ⭐ Main design orchestrator
│   │   ├── extract_smiles.py                SMILES extraction helper
│   │   └── update_template_smiles.py        Template updater
│   ├── instructions/
│   │   ├── ligand_alignment_mpnni_grouped.sh    MPNN submit template
│   │   ├── submit_pyrosetta_general_threading_relax.sh  Rosetta template
│   │   ├── aggregate_scores.py              Score aggregation
│   │   └── relax_2_filter__allpolar_unsats.py   Filtering script
│   ├── templates/
│   │   ├── pyr1_binary_template.json        AF3 binary template
│   │   └── pyr1_ternary_template.json       AF3 ternary template
│   └── DESIGN_PIPELINE_README.md            Design documentation
│
├── ml_modelling/               # 🆕 ML dataset generation (NEW!)
│   ├── data/                                Ligand and variant data
│   ├── scripts/
│   │   ├── orchestrate_ml_dataset_pipeline.py  ⭐ Main ML orchestrator
│   │   ├── aggregate_ml_features.py            Feature extraction
│   │   └── *.py                                Data processing scripts
│   ├── docs/                                Project planning and guides
│   ├── cache/                               Computational outputs
│   ├── results/                             Final datasets
│   └── README.md                            ML component documentation
│
├── templates/                  # Unified config templates (RECOMMENDED)
│   ├── unified_config_template.txt         ⭐ Complete config template
│   ├── CONFIG_GUIDE.md                      Parameter explanations
│   └── README.md                            Templates overview
│
├── scripts/                    # Shared utility scripts
│   ├── thread_variant_to_pdb.py            Mutation threading
│   └── PYR1_NUMBERING_GUIDE.md             Residue numbering reference
│
├── QUICK_START.md              ⭐ Start here for quickest setup
├── INTEGRATED_PIPELINE_SUMMARY.md  Complete workflow overview
└── README.md                   This file
```

---

## 🎓 For Different Users

### **New Lab Member**

1. Read: [QUICK_START.md](QUICK_START.md)
2. Copy: `templates/unified_config_template.txt`
3. Edit: 3 config sections (paths, ligand input)
4. Run: 2 commands (docking, design)

### **Experienced User**

- **Quick reference:** [docking/QUICKSTART.md](docking/QUICKSTART.md) and [design/QUICKSTART.md](design/QUICKSTART.md)
- **Advanced options:** [design/scripts/README.md](design/scripts/README.md)
- **Custom filtering:** Edit `[design]` section in config

### **Pipeline Developer**

- **Code structure:** [design/scripts/README.md](design/scripts/README.md)
- **Extending pipeline:** See "Advanced" sections in DESIGN_PIPELINE_README
- **Integration points:** [INTEGRATED_PIPELINE_SUMMARY.md](INTEGRATED_PIPELINE_SUMMARY.md)

---

## 🔄 Workflow Overview

```
SMILES / PubChem CID / SDF / PDB
    │
    ├─► CONFORMER GENERATION (ligand_conformers/)
    │   ├─ ETKDGv3 embed (500 confs)
    │   ├─ MMFF94s minimise + energy filter
    │   ├─ Butina RMSD cluster
    │   ├─ (Optional) OpenMM anneal
    │   └─ Select K diverse conformers → SDF + PDB
    │
    ▼
SDF Files
    │
    ├─► DOCKING PIPELINE
    │   ├─ create_table.py (params, alignment)
    │   ├─ grade_conformers (dock + filter)
    │   └─ cluster_docked (cluster poses)
    │
    ▼
Clustered Docked PDBs
    │
    ├─► DESIGN PIPELINE
    │   ├─ LigandMPNN (sequence design)
    │   ├─ Rosetta relax (energy minimize)
    │   ├─ Filter (by metrics)
    │   ├─ (Optional) Iterate
    │   ├─ Generate FASTA
    │   └─ Prepare AF3 JSONs (auto SMILES)
    │
    ▼
AF3-Ready Sequences
    │
    ├─► AF3 EXECUTION (Manual - GPU)
    │
    ▼
AF3 Predictions
```

---

## 🧬 ML Dataset Generation (NEW!)

**Generate training datasets for machine learning models that predict PYR1-ligand binding.**

The `ml_modelling/` component extends the pipeline to create high-quality ML datasets by:
- Processing variant-ligand pairs with affinity annotations
- Threading mutations onto PYR1 templates
- Running full conformer → docking → relax → AF3 pipeline
- Extracting 40+ structural features per pair
- Stratifying by binding affinity (P1/P2/P3 tiers)

### Quick Start (ML Dataset)
```bash
# Run ML dataset generation pipeline
python ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py \
    --input ml_modelling/data/ligand_smiles_signature.csv \
    --output ml_modelling/cache

# Aggregate features
python ml_modelling/scripts/aggregate_ml_features.py \
    --cache ml_modelling/cache \
    --output ml_modelling/results/features_table.csv
```

**See:** [ml_modelling/README.md](ml_modelling/README.md) for detailed documentation

---

## ⚙️ Configuration

**One unified config** controls both docking and design.

### **Essential Sections:**

```ini
[DEFAULT]
CAMPAIGN_ROOT = /projects/youruser/your_ligand
SCRATCH_ROOT = /scratch/youruser/output

[create_table]
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/*.sdf

[grade_conformers]
ArrayTaskCount = 10
MaxScore = -300

[design]
DesignResidues = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
LigandParams = %(CAMPAIGN_ROOT)s/conformers/0/0.params
FilterTargetN = 1000
```

**See:** [templates/unified_config_template.txt](templates/unified_config_template.txt)

**Guide:** [templates/CONFIG_GUIDE.md](templates/CONFIG_GUIDE.md)

---

## 🎨 Key Features

### **Before This Pipeline:**
- ❌ Manual script editing for every ligand
- ❌ ~2 hours of tedious work per ligand
- ❌ Easy to make mistakes (typos, forgotten SMILES)
- ❌ Difficult to teach to new lab members
- ❌ Hard to scale to multiple ligands

### **With This Pipeline:**
- ✅ **Config-driven** (no script editing)
- ✅ **5 minutes of setup** per ligand
- ✅ **Zero manual errors** (automatic path management)
- ✅ **Easy to train** (simple documentation)
- ✅ **Scales to 10+ ligands** in parallel

---

## 📊 Performance

### **Docking Stage:**
- Input: 100 conformers
- Parallel: 10 SLURM array tasks
- Output: ~50-200 clustered poses
- Time: 2-6 hours

### **Design Stage:**
- Input: ~100 docked PDBs
- MPNN: 100 jobs × 40 seqs = 4000 designs
- Rosetta: 4000 relax jobs
- Filter: ~1000 final designs
- Time: 4-8 hours total

### **Total Time:**
- Setup: 5 minutes (config editing)
- Execution: 6-14 hours (fully automated)
- Manual work saved: ~2 hours per ligand

---

## 🔍 Common Operations

### Generate Conformers (New Pre-Docking Stage)
```bash
# From SMILES
python -m ligand_conformers \
  --input "NC(CC(=O)c1ccccc1N)C(O)=O" --input-type smiles \
  --outdir /scratch/user/campaign/conformers --ligand-id kynurenine

# From PubChem name
python -m ligand_conformers \
  --input kynurenine --input-type pubchem \
  --outdir /scratch/user/campaign/conformers

# From existing SDF
python -m ligand_conformers \
  --input ligand.sdf --input-type sdf \
  --outdir /scratch/user/campaign/conformers

# With pipeline config and OpenMM refinement
python -m ligand_conformers \
  --config config.txt --input "C1=CC=CC=C1" --input-type smiles \
  --outdir /scratch/user/campaign/conformers --openmm-refine --nprocs 4
```

Outputs: `conformers_final.sdf`, `conformers_final/conf_*.{sdf,pdb}`, `conformer_report.csv`, `metadata.json`

See: [ligand_conformers/](ligand_conformers/) and [CONFORMER_GENERATION.md](CONFORMER_GENERATION.md)

### Run Complete Pipeline
```bash
python docking/scripts/run_docking_workflow.py config.txt
python design/scripts/run_design_pipeline.py config.txt
```

### Adjust Filtering Stringency
```ini
[design]
FilterTargetN = 2000    # Keep more designs
FilterMaxUnsats = 2     # More permissive
```

### Enable Two Iterations
```ini
[design]
DesignIterationRounds = 2
```

### Dry Run (Test Without Submitting)
```bash
python design/scripts/run_design_pipeline.py config.txt --dry-run
```

### Skip to Specific Stage
```bash
# Only MPNN → Rosetta (docking done)
python design/scripts/run_design_pipeline.py config.txt

# Only AF3 prep (MPNN/Rosetta done)
python design/scripts/run_design_pipeline.py config.txt --af3-prep-only
```

---

## 🆘 Troubleshooting

### Quick Fixes

| Problem | Solution |
|---------|----------|
| "Section not found" | Use `templates/unified_config_template.txt` |
| "No PDB files found" | Check docking completed: `ls $SCRATCH_ROOT/docked/clustered_final/` |
| "SMILES not updated" | Add `LigandSDF` to config, install RDKit |
| Too few designs | Increase `FilterMaxUnsats` in `[design]` |
| SLURM job fails | Check logs in `$SCRATCH_ROOT/design/iteration_1/*/` |

### Detailed Troubleshooting

See:
- [docking/WORKFLOW_README.md](docking/WORKFLOW_README.md#troubleshooting)
- [design/DESIGN_PIPELINE_README.md](design/DESIGN_PIPELINE_README.md#troubleshooting)

---

## 📞 Support & Documentation

### Quick References
- [QUICK_START.md](QUICK_START.md) - Fastest way to get started
- [docking/QUICKSTART.md](docking/QUICKSTART.md) - Docking quick ref
- [design/QUICKSTART.md](design/QUICKSTART.md) - Design quick ref

### Complete Guides
- [INTEGRATED_PIPELINE_SUMMARY.md](INTEGRATED_PIPELINE_SUMMARY.md) - Complete overview
- [docking/WORKFLOW_README.md](docking/WORKFLOW_README.md) - Docking details
- [design/DESIGN_PIPELINE_README.md](design/DESIGN_PIPELINE_README.md) - Design details

### Setup & Config
- [templates/CONFIG_GUIDE.md](templates/CONFIG_GUIDE.md) - Config parameters
- [design/SETUP_CHECKLIST.md](design/SETUP_CHECKLIST.md) - First-time setup

---

## 🎓 Training Materials

**For onboarding new lab members:**

1. **Overview:** Show [INTEGRATED_PIPELINE_SUMMARY.md](INTEGRATED_PIPELINE_SUMMARY.md)
2. **Quick demo:** Walk through [QUICK_START.md](QUICK_START.md)
3. **First run:** Use [design/SETUP_CHECKLIST.md](design/SETUP_CHECKLIST.md)
4. **Reference:** Bookmark [templates/CONFIG_GUIDE.md](templates/CONFIG_GUIDE.md)

**Estimated training time:** 30 minutes to understand, 1 hour to first successful run

---

## 🚧 Future Enhancements

Potential additions (optional):
- SLURM wrapper for full docking → design automation
- AF3 submission integration (JSON batching, submit scripts)
- Convergence detection for iterations
- Advanced filtering with custom metrics
- Automated AF3 analysis integration

---

## 📜 Citation

If you use this pipeline in your research, please cite:

[Your lab's relevant publications]

---

## ✅ Success Metrics

**The pipeline works when:**
1. ✅ You run docking + design with 2 commands (not 20+)
2. ✅ Zero manual script editing required
3. ✅ New lab members can run it after reading docs
4. ✅ SMILES automatically correct in AF3 templates
5. ✅ Reproducible results across different ligands
6. ✅ Easy to scale to 10+ ligands simultaneously

---

**Pipeline Status: Production Ready ✨**

*Last Updated: 2026-02-12*
