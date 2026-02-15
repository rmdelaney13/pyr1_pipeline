# Conformer Generation Stage

Pre-docking stage that produces a small set (K=5–20) of diverse, low-strain
ligand conformers from SMILES, PubChem identifiers, or structure files.

---

## Quick Start

```bash
# From SMILES (most common)
python -m ligand_conformers \
  --input "NC(CC(=O)c1ccccc1N)C(O)=O" \
  --input-type smiles \
  --outdir /scratch/user/campaign/conformers \
  --ligand-id kynurenine

# From PubChem CID or name
python -m ligand_conformers \
  --input 846 --input-type pubchem \
  --outdir /scratch/user/campaign/conformers

# From existing structure file
python -m ligand_conformers \
  --input ligand.sdf --input-type sdf \
  --outdir /scratch/user/campaign/conformers
```

---

## Pipeline

```
SMILES / PubChem CID / SDF / MOL2 / PDB
  │
  ├─ 1. Resolve input → RDKit Mol
  ├─ 2. ETKDGv3 embed (default 500 conformers, pruneRmsThresh 0.5 Å)
  ├─ 3. MMFF94s minimise (UFF fallback)
  ├─ 4. Energy pre-filter (keep top 100)
  ├─ 5. Butina RMSD cluster (heavy atoms, cutoff 1.25 Å)
  ├─ 6. Select lowest-energy rep per cluster
  ├─ 7. (Optional) OpenMM implicit-solvent minimise + MD anneal
  └─ 8. Final diverse K-selection
         │
         ▼
  conformers_final.sdf  +  conf_000.sdf/pdb  +  report.csv  +  metadata.json
```

---

## Outputs

| File | Description |
|------|-------------|
| `conformers_raw.sdf` | All conformers after embedding + minimisation |
| `conformers_clustered.sdf` | Cluster representatives (before final selection) |
| `conformers_final.sdf` | Final K conformers (feed to docking) |
| `conformers_final/conf_000.sdf` | Per-conformer SDF files |
| `conformers_final/conf_000.pdb` | Per-conformer PDB files (deterministic atom names) |
| `conformer_report.csv` | Energies, RMSD, cluster IDs per selected conformer |
| `metadata.json` | Full provenance: parameters, versions, timings |
| `conformer_generation.log` | Detailed log |

---

## How Outputs Feed Docking

The `conformers_final.sdf` and per-conformer `conf_*.sdf` files are
drop-in replacements for the SDF files expected by `create_table.py`:

```ini
[create_table]
MoleculeSDFs = %(CAMPAIGN_ROOT)s/conformers/conformers_final/conf_*.sdf
```

The per-conformer `conf_*.pdb` files use deterministic atom names
(element + index, e.g. `C1`, `C2`, `O1`, `N1`, ...) with residue name `LIG`
and chain `X`, suitable for downstream Rosetta and AF3 workflows.

---

## CLI Reference

```
python -m ligand_conformers --input <value> --input-type <type> --outdir <path> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--input` | SMILES string, PubChem CID/name, or file path |
| `--input-type` | `smiles`, `pubchem`, `sdf`, `mol2`, or `pdb` |
| `--outdir` | Output directory (created if absent) |

### Embedding

| Flag | Default | Description |
|------|---------|-------------|
| `--num-confs` | 500 | Number of ETKDGv3 conformers to embed |
| `--seed` | 42 | Random seed (deterministic output) |
| `--prune-rms` | 0.5 | RMSD prune threshold during embedding (Å) |

### Force Field

| Flag | Default | Description |
|------|---------|-------------|
| `--ff-variant` | MMFF94s | `MMFF94s` or `UFF` |
| `--ff-max-iters` | 500 | Max minimisation iterations |
| `--energy-pre-filter-n` | 100 | Keep top-N by energy before clustering |

### Clustering

| Flag | Default | Description |
|------|---------|-------------|
| `--cluster-rmsd` | 1.25 | Butina heavy-atom RMSD cutoff (Å) |
| `--no-align` | False | Skip alignment before RMSD computation |

### Selection

| Flag | Default | Description |
|------|---------|-------------|
| `-k`, `--k-final` | 10 | Number of final conformers to output |
| `--selection-policy` | diverse | `diverse` (best-per-cluster) or `energy` (global) |

### OpenMM Refinement (Optional)

| Flag | Default | Description |
|------|---------|-------------|
| `--openmm-refine` | off | Enable OpenMM implicit-solvent refinement |
| `--openmm-steps` | 25000 | MD anneal steps (50 ps at 2 fs/step) |

### Performance

| Flag | Default | Description |
|------|---------|-------------|
| `--nprocs` | 1 | Parallel workers for MMFF minimisation |

### Config File

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | none | Pipeline INI config with `[conformer_generation]` section |
| `--ligand-id` | ligand | Identifier for output naming |

CLI flags always override config-file values.

---

## Config File Section

Add this to your unified config (`config.txt`) to set defaults:

```ini
[conformer_generation]
NumConfs = 500
Seed = 42
PruneRmsThresh = 0.5
FFVariant = MMFF94s
EnergyPreFilterN = 100
ClusterRMSDCutoff = 1.25
KFinal = 10
SelectionPolicy = diverse
OpenMMRefine = False
NProcs = 1
```

See `templates/unified_config_template.txt` for the full section with comments.

---

## Python API

For programmatic use from other pipeline stages:

```python
from pathlib import Path
from ligand_conformers import ConformerConfig, generate_conformer_set

cfg = ConformerConfig(
    num_confs=500,
    seed=42,
    k_final=10,
    cluster_rmsd_cutoff=1.25,
    ligand_id="kynurenine",
)

result = generate_conformer_set(
    input_spec={"type": "smiles", "value": "NC(CC(=O)c1ccccc1N)C(O)=O"},
    outdir=Path("/scratch/user/campaign/conformers"),
    cfg=cfg,
)

if result.success:
    print(f"Generated {len(result.selected_ids)} conformers")
    print(f"Energies: {result.energies_mmff}")
```

### PDB Export Helper

```python
from ligand_conformers.io_utils import export_pdb_with_atom_names

# Export a specific conformer to PDB with deterministic atom names
export_pdb_with_atom_names(mol, confId=0, path=Path("conf.pdb"))
```

---

## Dependencies

**Required:**
- Python >= 3.9
- RDKit (with ETKDGv3 support)
- NumPy

**Optional:**
- `pubchempy` — for PubChem CID/name lookups (`pip install pubchempy`)
- `openmm` + `openmmforcefields` + `openff-toolkit` — for OpenMM refinement
  (`conda install -c conda-forge openmm openmmforcefields openff-toolkit`)

The module works without the optional dependencies; missing packages produce
clear error messages with installation instructions.

---

## Testing

```bash
python -m pytest tests/test_ligand_conformers.py -v
```

Tests cover:
- SMILES input for kynurenine → non-empty `conformers_final.sdf`
- Deterministic output with same random seed
- PubChem CID lookup (skipped gracefully if `pubchempy` is absent)
- CLI entrypoint smoke test

---

## Design Decisions

- **ETKDGv3 over ETKDG/ETKDGv2**: Better torsion angle distributions for
  drug-like molecules; produces more experimentally realistic conformers.
- **MMFF94s over MMFF94**: The "s" variant uses a static model for conjugation
  that handles the planar geometries of aromatic/amide systems better.
- **Butina clustering**: O(N^2) memory but well-suited for the 100-conformer
  pre-filtered set; produces centroids that represent distinct binding modes.
- **Diversity-first selection**: Guarantees coverage of conformational space;
  energy ranking is used as a tiebreaker within each cluster.
- **Parallel minimisation strategy**: Energy ranking is done in parallel across
  processes, then the top candidates are re-minimised serially to get correct
  coordinates (avoids pickling large Mol objects).
- **Tautomer/protonation**: Not enumerated automatically (placeholder hook
  provided). Preprocess with Dimorphite-DL or RDKit TautomerEnumerator before
  feeding SMILES if enumeration is needed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No conformers generated" | Increase `--num-confs` or decrease `--prune-rms` |
| "RDKit could not parse SMILES" | Check SMILES validity; try canonical SMILES |
| "pubchempy not installed" | `pip install pubchempy` or use `--input-type smiles` |
| "MMFF params missing" | Script auto-falls back to UFF; check log for warnings |
| "OpenMM import failed" | Install: `conda install -c conda-forge openmm openmmforcefields` |
| Too few clusters | Decrease `--cluster-rmsd` (e.g. 0.75 Å) |
| Too many clusters | Increase `--cluster-rmsd` (e.g. 2.0 Å) |
