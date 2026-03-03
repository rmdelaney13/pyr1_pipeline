# Docking Workflow

SDF conformers -> docked poses -> clustered representatives.

---

## Running

### Option 1: SLURM (Recommended)

```bash
bash scripts/submit_complete_workflow.sh config.txt
```

Reads `ArrayTaskCount` from config, submits array jobs, auto-queues clustering with `--dependency=afterok`. No intervention needed.

### Option 2: Local (Testing)

```bash
python scripts/run_docking_workflow.py config.txt
```

### Option 3: Local Array Simulation

```bash
python scripts/run_docking_workflow.py config.txt --local-arrays 4
```

---

## What Gets Run

Three stages, automatically chained:

1. **create_table.py** -- Read SDF files, generate Rosetta params/PDB, detect H-bond acceptors, create alignment CSV
2. **grade_conformers** -- Dock conformers into PYR1 pocket with perturbation + H-bond filtering (runs as SLURM array)
3. **cluster_docked_post_array.py** -- Cluster all passing poses by ligand heavy-atom RMSD, keep best per cluster

---

## Configuration

Key settings in `config.txt`:

```ini
[DEFAULT]
PIPE_ROOT = /projects/youruser/software/pyr1_pipeline
CAMPAIGN_ROOT = /projects/youruser/my_ligand
SCRATCH_ROOT = /scratch/alpine/youruser/my_ligand

[create_table]
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/*.sdf
DynamicAcceptorAlignment = True
MaxDynamicAlignments = 20
TargetAtomTriplets = O2-C11-C9; O2-C9-C11

[grade_conformers]
ArrayTaskCount = 10
OutputDir = %(SCRATCH_ROOT)s/docked
MaxScore = -300
EnableHBondGeometryFilter = True
ClusterRMSDCutoff = 0.75
GlycineShavePositions = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
```

Full parameter reference: [../templates/CONFIG_GUIDE.md](../templates/CONFIG_GUIDE.md)

---

## SLURM Array Jobs

### Why Arrays?

Each array task processes a different subset of conformers in parallel. With `ArrayTaskCount=10`:
- Task 0: conformers 0, 10, 20, 30, ...
- Task 1: conformers 1, 11, 21, 31, ...
- etc.

### Manual Two-Step (if `submit_complete_workflow.sh` doesn't fit your needs)

```bash
# Step 1: Submit array job
sbatch --array=0-9 scripts/submit_docking_workflow.sh config.txt

# Step 2: After all tasks complete, cluster
sbatch scripts/run_clustering_only.sh config.txt
```

---

## Workflow Stages

### Stage 1: Table Creation (create_table.py)

- Reads SDF conformer files
- Generates Rosetta params and PDB files
- Detects H-bond acceptor atoms using RDKit
- Creates alignment CSV with atom triplets

Skip if already done:
```bash
python scripts/run_docking_workflow.py config.txt --skip-create-table
```

### Stage 2: Docking

- Loads conformers from alignment table
- SVD alignment to template ligand position
- Rigid-body perturbation (rotation + translation)
- H-bond geometry filtering (distance, angle)
- Rosetta scoring with energy cutoff

For SLURM arrays, each task writes to the shared output directory with array-prefixed filenames.

### Stage 3: Clustering

- Collects all passing PDBs from all array tasks
- Clusters by ligand heavy-atom RMSD (default 0.75 A)
- Keeps lowest-energy representative per cluster
- Writes `cluster_summary.csv` with scores

---

## Output

### After Docking (per array task)

```
$SCRATCH_ROOT/docked/
  a0000_rep_0001.pdb
  a0000_rep_0002.pdb
  a0001_rep_0001.pdb
  hbond_geometry_summary_array0000.csv
  hbond_geometry_summary_array0001.csv
```

### After Clustering

```
$SCRATCH_ROOT/docked/clustered_final/
  cluster_0001_a0003_rep_0042.pdb
  cluster_0002_a0007_rep_0123.pdb
  cluster_summary.csv
```

These clustered PDBs are the input to the design pipeline.

---

## Common Operations

### Re-run clustering with different RMSD cutoff

```bash
# Edit config.txt: ClusterRMSDCutoff = 1.0
python scripts/run_docking_workflow.py config.txt --skip-create-table --skip-docking
```

### Re-run a specific failed array task

```bash
python scripts/run_docking_workflow.py config.txt --array-index 5 --skip-create-table --skip-clustering
```

### Only create table (don't dock)

```bash
python scripts/run_docking_workflow.py config.txt --prepare-only
```

---

## Troubleshooting

### "No candidate PDB files found"

No conformers passed docking filters. Try:
- Check `hbond_geometry_summary*.csv` to see why conformers failed
- Relax `MaxScore` (e.g., -250 instead of -300)
- Increase `MaxPerturbTries` (more sampling attempts)
- Widen `HBondDistanceIdealBuffer` (more tolerant H-bond geometry)

### "create_table.py failed"

- Ensure RDKit is installed: `conda install -c conda-forge rdkit`
- Validate SDF files in a molecular viewer
- Check `MoleculeSDFs` glob pattern matches your files

### Array job only runs 1 task

Forgot `--array` flag. Use `submit_complete_workflow.sh` (handles this automatically) or:
```bash
sbatch --array=0-9 scripts/submit_docking_workflow.sh config.txt
```

### Out of memory

- Increase `--mem` in `submit_docking_workflow.sh`
- Remove distant waters from PostPDBFileName
- Reduce `MaxPerturbTries`

---

## Related

- [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) -- RMSD cutoff selection and sampling adequacy
- [DEBUG_DOCKING_SCORING.md](DEBUG_DOCKING_SCORING.md) -- Debugging unrealistic scores
- [SEQUENCE_DOCKING_GUIDE.md](SEQUENCE_DOCKING_GUIDE.md) -- Docking to specific protein sequences from CSV
