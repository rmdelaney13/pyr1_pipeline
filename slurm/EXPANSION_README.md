# Iterative Neural Expansion Pipeline

Iteratively improve bile acid PYR1 designs by cycling through:
**Select top structures → LigandMPNN redesign → Boltz2 predict → score → merge → repeat**

## Overview

Starting from ~2000 initial designs per ligand (CA, CDCA, UDCA, DCA), each round:

1. **Select** top 100 by `binary_total_score`
2. **Redesign** each with LigandMPNN → 3 new sequences per structure (300 total)
3. **Predict** new sequences with Boltz2 (binary, MSA mode, affinity)
4. **Score** predictions (ipTM, pLDDT, P(binder), H-bond geometry, composite)
5. **Merge** into cumulative scores CSV and repeat

After 3–4 rounds the pool grows from ~2000 → ~3200 designs, progressively enriching
high-scoring variants through backbone-aware sequence diversification.

## Quick Start

```bash
cd /projects/ryde3462/software/pyr1_pipeline

# Round 0: Score initial predictions (CA example, must be complete)
bash slurm/run_expansion.sh ca 0

# Round 1 — run the same command 3 times, waiting for SLURM jobs between:
bash slurm/run_expansion.sh ca 1   # Phase A: select top 100 + submit MPNN
squeue -u $USER                    # wait for MPNN jobs to finish
bash slurm/run_expansion.sh ca 1   # Phase B: MPNN→CSV→YAML + submit Boltz
squeue -u $USER                    # wait for Boltz jobs to finish
bash slurm/run_expansion.sh ca 1   # Phase C: score + merge → "Round 1 complete"

# Rounds 2–4: same pattern
bash slurm/run_expansion.sh ca 2
# ...
```

All 4 ligands in parallel:
```bash
for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 0; done
for lig in ca cdca udca dca; do bash slurm/run_expansion.sh $lig 1; done
```

## How It Works

The script `slurm/run_expansion.sh` is **re-entrant** — it detects the current state
by checking which directories/files exist, then runs the next phase:

| Phase | Trigger (what's missing) | What it does |
|-------|--------------------------|--------------|
| **A** | No `selected_pdbs/`      | Select top N PDBs, submit MPNN SLURM array |
| **B** | No `boltz_inputs/`       | Convert MPNN FASTA → CSV → Boltz YAML, submit Boltz array |
| **C** | No `cumulative_scores.csv` | Score new predictions, merge with previous rounds |

If something goes wrong, delete the relevant directory to re-run that phase.

## Directory Layout

```
/scratch/alpine/ryde3462/expansion_{ligand}/
  round_0/
    scores.csv                     ← initial ~2000 scored predictions
  round_1/
    selected_pdbs/                 ← top 100 Boltz PDBs (Phase A)
    selected_manifest.txt
    mpnn_output/                   ← LigandMPNN FASTA outputs (Phase A, SLURM)
    expansion.csv                  ← new sequences as Boltz CSV (Phase B)
    boltz_inputs/                  ← YAML files + manifest (Phase B)
    boltz_output/                  ← Boltz predictions (Phase B, SLURM)
    new_scores.csv                 ← scores for this round only (Phase C)
    cumulative_scores.csv          ← all rounds merged, sorted by score (Phase C)
  round_2/ ...
```

## LigandMPNN Settings

**16 designable pocket positions** (Boltz/natural numbering, 181-residue PYR1):
```
A59 A81 A83 A92 A94 A108 A110 A117 A120 A122 A141 A159 A160 A163 A164 A167
```

> **Numbering shift:** Boltz uses full-length PYR1 (181 aa). The old Rosetta model
> had a 2-residue deletion near position 65, so all positions ≥65 are +2 vs Rosetta
> (e.g., Rosetta 79 → Boltz 81, Rosetta 139 → Boltz 141). Position 59 is unchanged.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sequences per PDB | 3 | `--batch_size 3 --number_of_batches 1` |
| Temperature | 0.3 | Conservative sampling |
| Omit | `expansion_omit.json` | Light: A59 → {A,G,L,V}; A159 → no {R,M,K} |
| Bias | `expansion_bias.json` | A141 → K (weight 2.0) for COO⁻ salt bridge |
| Partition | `amilan` (CPU) | ~20 min per PDB |

## Boltz2 Settings

Same as the initial `run_boltz_bile_acids.sh` predictions:

| Parameter | Value |
|-----------|-------|
| Mode | Binary (PYR1 + ligand) |
| MSA | WT PYR1 a3m with query patching |
| max_msa_seqs | 32 |
| diffusion_samples | 5 |
| affinity | yes |
| Partition | `aa100` (GPU) |

## Scoring

Designs are ranked by `binary_total_score` (range 0–3):

```
binary_total_score = boltz_score + geometry_score

  boltz_score (0–2)    = plddt_ligand + P(binder)
  geometry_score (0–1)  = 0.7 × dist_score + 0.3 × angle_score
    dist_score: Gaussian at 2.7 Å (σ=0.8) — conserved water H-bond distance
    angle_score: Gaussian at 90.5° (σ=25) — Pro88:O–water–ligand angle
```

## Files

| File | Purpose |
|------|---------|
| `slurm/run_expansion.sh` | Master orchestrator (re-entrant) |
| `slurm/submit_mpnn_expansion.sh` | SLURM array for LigandMPNN |
| `scripts/expansion_select.py` | Select top N, copy PDBs |
| `scripts/expansion_mpnn_to_csv.py` | MPNN FASTA → Boltz CSV (with dedup) |
| `scripts/expansion_merge.py` | Merge round scores into cumulative CSV |
| `design/mpnn/expansion_omit.json` | Omit config (Boltz numbering) |
| `design/mpnn/expansion_bias.json` | Bias config (K at A141) |

Existing scripts used as-is:
- `scripts/prepare_boltz_yamls.py` — generate Boltz YAML inputs
- `scripts/analyze_boltz_output.py` — score predictions
- `slurm/submit_boltz.sh` — submit Boltz array jobs

## Prerequisites

- Initial Boltz predictions complete (`/scratch/alpine/ryde3462/boltz_bile_acids/output_{ligand}_binary`)
- `boltz_env` conda environment (for Boltz2 + scoring)
- `ligandmpnn_env` conda environment (for LigandMPNN)
- WT PYR1 MSA at `/scratch/alpine/ryde3462/boltz_lca/wt_prediction/boltz_results_pyr1_wt_lca/msa/pyr1_wt_lca_unpaired_tmp_env/uniref.a3m`
