# PYR1 Pipeline

Automated workflow from ligand SMILES to AF3-ready protein designs.

```
SMILES / SDF / PubChem CID
    |
    v  ligand_conformers (RDKit ETKDGv3)
Diverse Conformers (SDF)
    |
    v  Docking Pipeline (Rosetta, SLURM arrays)
Clustered Docked Poses (PDB)
    |
    v  Design Pipeline (LigandMPNN + Rosetta + Filter)
Filtered Sequences + AF3 JSON Inputs
    |
    v  AlphaFold3 (GPU)
Validated Designs (pLDDT, ipTM, ligand RMSD)
```

---

## Quick Start

```bash
# 1. Copy config template
cp templates/unified_config_template.txt /projects/youruser/my_ligand/config.txt

# 2. Edit paths, SMILES, and ligand params in config.txt

# 3. Generate conformers
cd /projects/youruser/software/pyr1_pipeline
python -m ligand_conformers --input "SMILES" --input-type smiles \
  --outdir /projects/youruser/my_ligand/conformers --k-final 15

# 4. Run docking (SLURM)
bash docking/scripts/submit_complete_workflow.sh config.txt

# 5. Run design (SLURM, after docking completes)
python design/scripts/run_design_pipeline.py config.txt \
  --skip-af3-submit --skip-af3-analyze --wait
```

Full walkthrough: **[GETTING_STARTED.md](GETTING_STARTED.md)**

---

## Documentation

### Getting Started
| Document | Description |
|----------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Setup, configure, and run end-to-end |
| [templates/CONFIG_GUIDE.md](templates/CONFIG_GUIDE.md) | Config parameter reference |

### Stage References
| Document | Description |
|----------|-------------|
| [CONFORMER_GENERATION.md](CONFORMER_GENERATION.md) | Conformer generation CLI and API |
| [docking/WORKFLOW_README.md](docking/WORKFLOW_README.md) | Docking pipeline details |
| [design/DESIGN_PIPELINE_README.md](design/DESIGN_PIPELINE_README.md) | Design pipeline details (MPNN, Rosetta, AF3) |

### Advanced
| Document | Description |
|----------|-------------|
| [docking/SEQUENCE_DOCKING_GUIDE.md](docking/SEQUENCE_DOCKING_GUIDE.md) | Docking to specific protein sequences from CSV |
| [docking/SAMPLING_GUIDE.md](docking/SAMPLING_GUIDE.md) | RMSD cutoff selection and sampling adequacy |
| [docking/DEBUG_DOCKING_SCORING.md](docking/DEBUG_DOCKING_SCORING.md) | Debugging unrealistic docking scores |
| [design/UNIVERSAL_LIGAND_SUPPORT.md](design/UNIVERSAL_LIGAND_SUPPORT.md) | Universal relax/filter scripts for any ligand |
| [scripts/PYR1_NUMBERING_GUIDE.md](scripts/PYR1_NUMBERING_GUIDE.md) | 3QN1 residue numbering (WT vs PDB) |

### ML Dataset Pipeline (separate workflow)
| Document | Description |
|----------|-------------|
| [ml_modelling/README.md](ml_modelling/README.md) | ML dataset generation overview |

---

## Repository Structure

```
pyr1_pipeline/
├── ligand_conformers/          # Conformer generation (pre-docking)
│   ├── __main__.py             CLI entrypoint
│   ├── core.py                 Generation engine
│   └── config.py               ConformerConfig dataclass
│
├── docking/                    # Docking pipeline
│   └── scripts/
│       ├── run_docking_workflow.py        Main orchestrator
│       ├── submit_complete_workflow.sh    SLURM submit-and-forget
│       ├── create_table.py               SDF -> params/alignment
│       └── cluster_docked_post_array.py  Post-docking clustering
│
├── design/                     # Design pipeline
│   └── scripts/
│       └── run_design_pipeline.py        Main orchestrator (MPNN -> Rosetta -> Filter -> AF3)
│
├── templates/
│   ├── unified_config_template.txt       Config template (start here)
│   └── CONFIG_GUIDE.md                   Parameter reference
│
├── scripts/                    # Shared utilities
│   └── PYR1_NUMBERING_GUIDE.md
│
└── ml_modelling/               # ML dataset generation (separate workflow)
```
