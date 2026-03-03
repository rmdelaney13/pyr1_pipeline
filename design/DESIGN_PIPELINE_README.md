# Design Pipeline

Clustered docked PDBs -> MPNN sequence design -> Rosetta relax -> filter -> AF3-ready inputs.

---

## Running

```bash
# Full pipeline (MPNN -> Rosetta -> filter -> AF3 prep)
python design/scripts/run_design_pipeline.py config.txt --wait

# Stop before AF3 GPU submission
python design/scripts/run_design_pipeline.py config.txt \
  --skip-af3-submit --skip-af3-analyze --wait

# Dry run (generate scripts, don't submit)
python design/scripts/run_design_pipeline.py config.txt --dry-run
```

The `--wait` flag polls SLURM between stages so everything chains automatically.

---

## Pipeline Stages

```
Clustered docked PDBs (from docking/)
    |
    v
[1] LigandMPNN sequence design
    |  - 40 sequences per parent template
    |  - Temperature 0.3
    |  - 16 designable pocket positions
    v
[2] Rosetta relax + scoring (relax_general_universal.py)
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
[5] Generate FASTA + AF3 JSON inputs (binary + ternary)
    v
AF3-ready sequences
```

---

## Configuration

Add a `[design]` section to your `config.txt`:

```ini
[design]
DesignRoot = design
DesignIterationRounds = 1
DesignResidues = 59 79 81 90 92 106 108 115 118 120 139 157 158 161 162 165
LigandParams = %(CAMPAIGN_ROOT)s/conformers/0/0.params
LigandSDF = %(CAMPAIGN_ROOT)s/conformers/0/0.sdf
LigandSMILES = YOUR_SMILES_HERE
FilterTargetN = 1000
FilterMaxUnsats = 1
FilterMaxPerParent = 20
```

Full parameter reference: [../templates/CONFIG_GUIDE.md](../templates/CONFIG_GUIDE.md)

---

## Output

```
$SCRATCH_ROOT/design/
  iteration_1/
    mpnn_output/          # LigandMPNN .fa files
    rosetta_output/       # Relaxed PDBs + .sc score files
    scores/               # Aggregated CSV
    filtered/             # Top N PDBs + filtered.csv + filtered.fasta
  af3_inputs/
    binary/*.json         # AF3 protein+ligand inputs
    ternary/*.json        # AF3 protein+ligand+HAB1 inputs
```

---

## Flags

| Flag | Effect |
|------|--------|
| `--wait` | Poll SLURM between stages (recommended) |
| `--dry-run` | Generate scripts without submitting |
| `--iteration N` | Run only iteration N |
| `--skip-mpnn` | Skip LigandMPNN |
| `--skip-rosetta` | Skip Rosetta relax |
| `--skip-aggregate` | Skip score aggregation |
| `--skip-filter` | Skip filtering |
| `--skip-af3-prep` | Skip AF3 JSON generation |
| `--skip-af3-submit` | Stop before AF3 GPU jobs |
| `--skip-af3-analyze` | Skip AF3 analysis |
| `--af3-prep-only` | Only FASTA + AF3 JSON (skip MPNN/Rosetta) |
| `--af3-submit-only` | Only batch and submit AF3 GPU jobs |
| `--af3-analyze-only` | Only analyze AF3 results |
| `--rosetta-to-af3` | Skip MPNN/Rosetta, aggregate -> filter -> AF3 |

---

## Running Individual Steps

If a stage fails and you need to re-run just that step:

### MPNN only
```bash
# Wrapper script
bash design/scripts/run_mpnn_only.sh \
  /scratch/.../clustered_final \
  /scratch/.../design/mpnn_output \
  50

# Or via orchestrator (skip later stages)
python design/scripts/run_design_pipeline.py config.txt \
  --skip-rosetta --skip-aggregate --skip-filter --skip-af3-prep
```

### Rosetta only (MPNN done)
```bash
# Wrapper script
bash design/scripts/run_rosetta_only.sh \
  /scratch/.../clustered_final \
  /scratch/.../mpnn_output \
  /scratch/.../rosetta_output \
  /projects/.../ligand.params \
  500

# Or via orchestrator
python design/scripts/run_design_pipeline.py config.txt --skip-mpnn
```

### Score aggregation only
```bash
python design/instructions/aggregate_scores.py \
  /scratch/.../rosetta_output \
  --output /scratch/.../scores/all_scores.csv
```

### Re-filter with different settings
```bash
python design/instructions/relax_2_filter__allpolar_unsats.py \
  scores.csv rosetta_output filtered_output \
  --target_n 500 --max_unsat 0 --max_per_parent 10
```

### AF3 prep only
```bash
python design/scripts/run_design_pipeline.py config.txt --af3-prep-only
```

---

## Key Files

| File | Role |
|------|------|
| `design/scripts/run_design_pipeline.py` | Main orchestrator |
| `design/rosetta/relax_general_universal.py` | Relax + scoring (auto-detects ligand polar atoms) |
| `design/instructions/ligand_alignment_mpnni_grouped.sh` | MPNN SLURM template |
| `design/instructions/submit_pyrosetta_general_threading_relax.sh` | Rosetta SLURM template |
| `design/instructions/aggregate_scores.py` | Score CSV aggregation |
| `design/instructions/relax_2_filter__allpolar_unsats.py` | Filtering script |
| `design/scripts/extract_smiles.py` | SMILES extraction from SDF |
| `design/scripts/update_template_smiles.py` | Update SMILES in AF3 JSON templates |

---

## How the Orchestrator Works

For each iteration:

1. **MPNN**: Reads template MPNN script, substitutes paths, calculates array count from input PDB count, writes custom submit script, submits to SLURM
2. **Rosetta**: Same pattern -- template script, path substitution, array count from FASTA count, submit
3. **Aggregate**: Scans Rosetta output for `.sc` files, combines into single CSV
4. **Filter**: Applies dG_sep, buried_unsats, polar contacts, charge satisfaction filters; limits per-parent diversity; keeps top N
5. **AF3 Prep**: Generates FASTA from filtered PDBs, extracts SMILES, creates binary + ternary AF3 JSONs

No manual directory creation or script editing needed.

---

## Extending the Pipeline

### Custom filter script

1. Create your filter script (must accept same CLI args as `relax_2_filter__allpolar_unsats.py`)
2. Update config:
   ```ini
   [design]
   FilterScript = %(PIPE_ROOT)s/design/instructions/my_custom_filter.py
   ```

### Manual SLURM submission (if auto-submit fails)

After `--dry-run`, the generated scripts are at:
```bash
$SCRATCH_ROOT/design/iteration_1/mpnn_output/submit_mpnn.sh
$SCRATCH_ROOT/design/iteration_1/rosetta_output/submit_rosetta.sh
```

Submit manually:
```bash
sbatch $SCRATCH_ROOT/design/iteration_1/mpnn_output/submit_mpnn.sh
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No docked PDBs found | Check docking completed: `ls $SCRATCH_ROOT/docked/clustered_final/` |
| MPNN SLURM job fails | Check `mpnn_output/*.err`; verify `ligandmpnn_env` conda env exists |
| Rosetta SLURM job fails | Check `rosetta_output/*.err`; verify `ligand_alignment` conda env |
| Too few designs pass filter | Increase `FilterMaxUnsats` (2 or 3); decrease `FilterTargetN` |
| SMILES not in AF3 JSONs | Set `LigandSMILES` directly in config (avoids RDKit dependency) |

---

## Related

- [UNIVERSAL_LIGAND_SUPPORT.md](UNIVERSAL_LIGAND_SUPPORT.md) -- Universal relax/filter scripts for any ligand
- [../templates/CONFIG_GUIDE.md](../templates/CONFIG_GUIDE.md) -- Config parameter reference
- [../GETTING_STARTED.md](../GETTING_STARTED.md) -- End-to-end setup guide
