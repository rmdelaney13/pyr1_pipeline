# Unified Config Guide

This directory contains the **unified config template** that controls both docking and design pipelines.

---

## Quick Start

### 1. Copy Template to Your Campaign

```bash
# On cluster
cp /projects/ryde3462/pyr1_pipeline/templates/unified_config_template.txt \
   /scratch/alpine/ryde3462/your_ligand/config.txt
```

### 2. Edit Key Sections

```bash
vim /scratch/alpine/ryde3462/your_ligand/config.txt
```

**Minimum required edits:**

#### **Section: [DEFAULT]**
```ini
CAMPAIGN_ROOT = /projects/ryde3462/your_ligand
SCRATCH_ROOT = /scratch/alpine/ryde3462/your_ligand_output
```

#### **Section: [create_table]**
```ini
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/*.sdf
TargetAtomTriplets = O2-C11-C9; O2-C9-C11  # Update based on your ligand
```

#### **Section: [design]**
```ini
DesignResidues = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
LigandParams = %(CAMPAIGN_ROOT)s/conformers/0/0.params
LigandSDF = %(CAMPAIGN_ROOT)s/conformers/0/0.sdf
```

### 3. Run Pipeline

```bash
# Step 1: Docking
python /projects/ryde3462/pyr1_pipeline/docking/scripts/run_docking_workflow.py \
    /scratch/alpine/ryde3462/your_ligand/config.txt

# Step 2: Design
python /projects/ryde3462/pyr1_pipeline/design/scripts/run_design_pipeline.py \
    /scratch/alpine/ryde3462/your_ligand/config.txt
```

---

## Config Sections Explained

### **[DEFAULT]** - Root Paths

Defines where everything lives. Update these for each new campaign.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `PIPE_ROOT` | Pipeline code location | `/projects/youruser/pyr1_pipeline` |
| `CAMPAIGN_ROOT` | Your project directory | `/projects/youruser/xanthurenic_acid` |
| `SCRATCH_ROOT` | High-speed scratch storage | `/scratch/youruser/xan_output` |

**Tip:** Use `%(PIPE_ROOT)s` and `%(CAMPAIGN_ROOT)s` in later sections for portable paths.

---

### **[create_table]** - Docking Input

Specifies ligand conformers and how to align them.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MoleculeSDFs` | Input SDF files (supports wildcards) | `%(CAMPAIGN_ROOT)s/conformers/*.sdf` |
| `TargetAtomTriplets` | Alignment atoms on template ligand | `O2-C11-C9; O2-C9-C11` |
| `MaxDynamicAlignments` | Max auto-generated alignments | `20` |

**Tip:** Use `DynamicAcceptorAlignment = True` to auto-detect H-bond acceptors.

---

### **[grade_conformers]** - Docking Parameters

Controls the docking simulation.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `OutputDir` | Docking output location | `%(SCRATCH_ROOT)s/docked` |
| `ArrayTaskCount` | Number of parallel SLURM tasks | `10` (large runs), `1` (small) |
| `MaxScore` | Energy cutoff for passing poses | `-300` |
| `ClusterRMSDCutoff` | Clustering threshold (Å) | `0.75` |
| `GlycineShavePositions` | Residues to relax | `59 79 81 90 92 ...` |

**Tip:** Set `ArrayTaskCount = 10` for 100+ conformers, `1` for testing.

---

### **[design]** - Design Pipeline

Controls MPNN → Rosetta → Filtering workflow.

#### **Core Settings**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `DesignRoot` | Output directory name | `design` |
| `DesignIterationRounds` | Number of refinement rounds | `1` or `2` |

#### **MPNN Settings**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MPNNBatchSize` | Sequences per parent structure | `40` |
| `MPNNTemperature` | Sampling temperature | `0.3` (conservative) |
| `DesignResidues` | Residues to redesign | `59 79 81 90 92 ...` |

**Tip:** Match `DesignResidues` to `GlycineShavePositions` for consistency.

#### **Filtering Settings**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `FilterTargetN` | Final number of designs | `1000` |
| `FilterMaxUnsats` | Max buried unsats (lower = stricter) | `1` |
| `FilterMaxPerParent` | Max designs per parent (diversity) | `20` |

**Tip:** If too few designs pass, increase `FilterMaxUnsats` to `2` or `3`.

#### **AF3 Preparation Settings**

| Parameter | Description | Required? |
|-----------|-------------|-----------|
| `LigandParams` | Ligand params file | ✅ Yes |
| `LigandSMILES` | SMILES string for AF3 (highest priority) | Either this or `LigandSDF` |
| `LigandSDF` | SDF for auto SMILES extraction (fallback) | Either this or `LigandSMILES` |
| `AF3BinaryTemplate` | Binary complex template JSON | ✅ Yes |
| `AF3TernaryTemplate` | Ternary complex template JSON | ✅ Yes |

**Tip:** Set `LigandSMILES` directly if you know the SMILES. Otherwise set `LigandSDF` and SMILES will be auto-extracted using RDKit.

#### **AF3 Submission Settings**

Controls how AF3 GPU inference jobs are batched and submitted to SLURM.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `AF3BatchSize` | JSONs per batch directory (each batch = 1 GPU task) | `60` |
| `AF3Partition` | SLURM GPU partition | `aa100` |
| `AF3TimeLimit` | Wall clock time per array task (HH:MM:SS) | `01:00:00` |
| `AF3Account` | SLURM account for job billing | `ucb472_asc2` |
| `AF3QOS` | Quality of service | `normal` |
| `AF3NTasks` | CPU tasks per node | `10` |
| `AF3ModelParamsDir` | Path to AF3 model parameters | `/projects/.../af3` |
| `AF3RunDataPipeline` | Run MSA data pipeline (`true`/`false`) | `false` |
| `AF3JobNamePrefix` | Prefix for SLURM job names | `af3` |

**Tip:** Set `AF3RunDataPipeline = false` when using pre-computed templates (most common). Set `AF3BatchSize` based on how many JSONs each GPU can process within `AF3TimeLimit`.

#### **AF3 Analysis Settings**

Controls quality filtering of AF3 predictions. Requires BioPython and NumPy.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `AF3RefModel` | Reference CIF structure for distance/RMSD calculations | ✅ Required |
| `AF3PLDDTCutoff` | Minimum ligand pLDDT score | `60` |
| `AF3IPTMCutoff` | Minimum ligand ipTM score | `0.65` |
| `AF3LigandChain` | Ligand chain ID in AF3 output | `B` |
| `AF3LigandResID` | Ligand residue ID in AF3 output | `1` |
| `AF3RefChain` | Reference chain for distance measurement | `A` |
| `AF3RefResID` | Reference residue ID | `201` |
| `AF3RefAtom` | Reference atom name (measures min distance to ligand oxygens) | `O1` |

**Tip:** Both binary AND ternary predictions must pass the pLDDT and ipTM cutoffs. If too few designs pass, try lowering `AF3PLDDTCutoff` to `50` or `AF3IPTMCutoff` to `0.5`.

---

## Common Customizations

### Adjust Docking Stringency

**More permissive (more passing poses):**
```ini
[grade_conformers]
MaxScore = -250              # Less negative = more permissive
HBondDistanceIdealBuffer = 1.0  # Larger buffer
```

**More stringent (fewer, better poses):**
```ini
[grade_conformers]
MaxScore = -350              # More negative = stricter
HBondDistanceIdealBuffer = 0.5  # Tighter geometry
```

---

### Adjust Filtering Stringency

**More permissive (keep more designs):**
```ini
[design]
FilterTargetN = 2000
FilterMaxUnsats = 3
FilterMaxPerParent = 50
```

**More stringent (keep only best):**
```ini
[design]
FilterTargetN = 500
FilterMaxUnsats = 0
FilterMaxPerParent = 10
```

---

### Ligand-Specific Filters

If your ligand doesn't have specific polar atoms:

```ini
[design]
FilterIgnoreO1 = True        # Don't require O1 contact
FilterIgnoreO2 = True        # Don't require O2 contact
FilterIgnoreCharge = False   # Still require charge satisfaction
```

---

### Two Design Iterations

For refinement:

```ini
[design]
DesignIterationRounds = 2

# Iteration 1: More permissive
FilterTargetN = 2000
FilterMaxUnsats = 2

# (Script will use same filters for iter 2, or you can manually adjust after iter 1)
```

---

## Path Variable Reference

Use these variables for portable configs:

| Variable | Defined In | Use For |
|----------|-----------|---------|
| `%(PIPE_ROOT)s` | `[DEFAULT]` | Pipeline scripts, templates |
| `%(CAMPAIGN_ROOT)s` | `[DEFAULT]` | Input files, params |
| `%(SCRATCH_ROOT)s` | `[DEFAULT]` | Output directories |

**Example:**
```ini
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/*.sdf
OutputDir = %(SCRATCH_ROOT)s/docked
MPNNScript = %(PIPE_ROOT)s/design/instructions/ligand_alignment_mpnni_grouped.sh
```

---

## Validation Checklist

Before running, verify:

- [ ] `MoleculeSDFs` points to existing SDF files
- [ ] `LigandParams` exists and matches your ligand
- [ ] `DesignResidues` match your protein target
- [ ] `SCRATCH_ROOT` is on high-speed storage
- [ ] `ArrayTaskCount` matches your workload (1 for testing, 10+ for production)
- [ ] Templates exist: `AF3BinaryTemplate`, `AF3TernaryTemplate`

**Quick test:**
```bash
python -c "
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('/scratch/alpine/ryde3462/your_ligand/config.txt')
print('CAMPAIGN_ROOT:', cfg.get('DEFAULT', 'CAMPAIGN_ROOT'))
print('OutputDir:', cfg.get('grade_conformers', 'OutputDir'))
print('DesignResidues:', cfg.get('design', 'DesignResidues'))
"
```

---

## Example Configs

### Small Test Run

```ini
[DEFAULT]
CAMPAIGN_ROOT = /projects/ryde3462/test_ligand
SCRATCH_ROOT = /scratch/alpine/ryde3462/test_output

[create_table]
MoleculeSDFs = %(CAMPAIGN_ROOT)s/single_conformer.sdf

[grade_conformers]
ArrayTaskCount = 1
MaxScore = -250

[design]
DesignIterationRounds = 1
FilterTargetN = 100
```

### Production Run (100+ conformers)

```ini
[DEFAULT]
CAMPAIGN_ROOT = /projects/ryde3462/production_ligand
SCRATCH_ROOT = /scratch/alpine/ryde3462/production_output

[create_table]
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/*.sdf

[grade_conformers]
ArrayTaskCount = 10
MaxScore = -300

[design]
DesignIterationRounds = 2
FilterTargetN = 1000
FilterMaxUnsats = 1
```

---

## Troubleshooting

### "KeyError: 'design'"

**Cause:** Missing `[design]` section
**Fix:** Ensure all sections from template are present

### "Path does not exist"

**Cause:** Incorrect `CAMPAIGN_ROOT` or `SCRATCH_ROOT`
**Fix:** Verify paths exist: `ls $CAMPAIGN_ROOT`

### "No conformers found"

**Cause:** Wrong `MoleculeSDFs` path
**Fix:** Test glob: `ls /path/to/*.sdf`

### "SMILES not updating"

**Cause:** Missing `LigandSDF` or RDKit not installed
**Fix:** Add `LigandSDF` path, install RDKit: `conda install -c conda-forge rdkit`

---

## See Also

- **Docking docs:** [../docking/WORKFLOW_README.md](../docking/WORKFLOW_README.md)
- **Design docs:** [../design/DESIGN_PIPELINE_README.md](../design/DESIGN_PIPELINE_README.md)
- **Quick start:** [../design/QUICKSTART.md](../design/QUICKSTART.md)
- **Setup guide:** [../design/SETUP_CHECKLIST.md](../design/SETUP_CHECKLIST.md)
