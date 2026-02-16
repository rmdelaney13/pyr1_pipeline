# PYR1 ML Dataset Pipeline: Implementation Guide
**Companion to:** `PYR1_ML_DATASET_PROJECT_PLAN.md`
**Version:** 1.0
**Date:** 2026-02-16

---

## QUICK START COMMANDS

### Phase 0: Setup

```bash
# 1. Create negative dataset
cd ml_modelling/scripts
python generate_negatives.py \
    --positives ../ligand_smiles_signature.csv \
    --tier1-screens ../../library_screens/2024-11-15_FACS.csv \
    --tier2-count 10 \
    --tier3-count 10 \
    --output ../negatives_curated.csv

# 2. Run pilot validation (1 ligand, 30 pairs)
python orchestrate_ml_pipeline.py \
    --positives ../ligand_smiles_signature.csv \
    --negatives ../negatives_curated.csv \
    --pilot-ligand "WIN-55212-2" \
    --cache-dir $SCRATCH_ROOT/ml_dataset/cache \
    --output ../features_pilot.csv

# 3. Check pilot results
python analyze_pilot.py --features ../features_pilot.csv
```

### Phase 1: Pilot Dataset

```bash
# Run on 10 ligands × 30 pairs = 300 pairs
python orchestrate_ml_pipeline.py \
    --positives ../ligand_smiles_signature.csv \
    --negatives ../negatives_curated.csv \
    --ligand-list pilot_ligands.txt \
    --max-positives 10 \
    --tier1-per-ligand 5 \
    --tier2-per-ligand 10 \
    --tier3-per-ligand 5 \
    --cache-dir $SCRATCH_ROOT/ml_dataset/cache \
    --output ../features_pilot.csv \
    --slurm

# Generate QC report
python generate_qc_report.py \
    --features ../features_pilot.csv \
    --output ../pilot_qc_report.html
```

### Phase 2: Production Dataset

```bash
# Run on all ligands
python orchestrate_ml_pipeline.py \
    --positives ../ligand_smiles_signature.csv \
    --negatives ../negatives_curated.csv \
    --all-ligands \
    --cache-dir $SCRATCH_ROOT/ml_dataset/cache \
    --output ../features_table.csv \
    --slurm

# Split dataset
python split_dataset.py \
    --features ../features_table.csv \
    --train-frac 0.6 \
    --val-frac 0.2 \
    --test-frac 0.2 \
    --stratify ligand_class \
    --output-dir ../splits/

# Train baseline models
python baseline_models.py \
    --train ../splits/train_set.csv \
    --val ../splits/val_set.csv \
    --test ../splits/test_set.csv \
    --output ../baseline_results.csv \
    --figures ../figures/
```

---

## CODE TEMPLATES

### 1. `generate_negatives.py`

```python
#!/usr/bin/env python3
"""
Generate Tier 1/2/3 negative dataset for PYR1 biosensor ML training.

Usage:
    python generate_negatives.py \
        --positives ligand_smiles_signature.csv \
        --tier1-screens library_screens/*.csv \
        --tier2-count 10 \
        --tier3-count 10 \
        --output negatives_curated.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
import hashlib

# Pocket positions (PYR1 binding site)
POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]
AA_ALPHABET = list('ACDEFGHIKLMNPQRSTVWY')


def parse_variant_signature(signature: str) -> dict:
    """
    Parse variant signature string into {position: amino_acid} dict.
    Example: "59K;120A;160G" → {59: 'K', 120: 'A', 160: 'G'}
    """
    if pd.isna(signature) or signature == '':
        return {}
    mutations = {}
    for mut in signature.split(';'):
        if '_' in mut:  # Handle underscore format
            mut = mut.split('_')[-1]
        pos = int(''.join(filter(str.isdigit, mut)))
        aa = mut[-1]
        mutations[pos] = aa
    return mutations


def signature_to_string(mutations: dict) -> str:
    """Convert {position: aa} dict to signature string."""
    return ';'.join([f"{pos}{aa}" for pos, aa in sorted(mutations.items())])


def generate_tier1_from_screens(screen_csvs: List[str], positives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Tier 1 hard negatives from historical FACS screens.

    Criteria:
    - Expression score > threshold (passed QC)
    - Activation score < threshold (failed activation)
    - Not in positives dataset (avoid false negatives)
    """
    tier1_negatives = []

    # Load all screen CSVs
    for csv_path in screen_csvs:
        screen_df = pd.read_csv(csv_path)

        # Filter: high expression, low activation
        candidates = screen_df[
            (screen_df['expression_score'] > screen_df['expression_score'].quantile(0.5)) &
            (screen_df['activation_AUC'] < 0.6)
        ].copy()

        # Exclude known positives
        positive_signatures = set(positives_df['PYR1_variant_signature'].dropna())
        candidates = candidates[~candidates['variant_signature'].isin(positive_signatures)]

        # Add provenance
        candidates['label'] = 0
        candidates['label_tier'] = 'T1'
        candidates['label_source'] = f"FACS_screen_{csv_path.split('/')[-1]}"
        candidates['label_confidence'] = 'HIGH'
        candidates['date_added'] = datetime.now().isoformat()

        tier1_negatives.append(candidates[['ligand_name', 'ligand_smiles', 'variant_name',
                                           'variant_signature', 'label', 'label_tier',
                                           'label_source', 'label_confidence', 'date_added']])

    if tier1_negatives:
        return pd.concat(tier1_negatives, ignore_index=True)
    else:
        print("WARNING: No Tier 1 negatives found in screens!")
        return pd.DataFrame()


def generate_tier2_near_neighbors(positives_df: pd.DataFrame, n_per_ligand: int = 10) -> pd.DataFrame:
    """
    Generate Tier 2 soft negatives: 1–2 mutations from positive variants.
    """
    tier2_negatives = []

    # Group by ligand
    for ligand_name, group in positives_df.groupby('ligand_name'):
        ligand_smiles = group.iloc[0]['ligand_smiles_or_ligand_ID']

        # Sample one positive variant as template
        for idx, row in group.sample(min(3, len(group))).iterrows():  # Use up to 3 templates
            parent_signature = parse_variant_signature(row['PYR1_variant_signature'])

            for i in range(n_per_ligand):
                # Mutate 1–2 random pocket positions
                n_muts = np.random.choice([1, 2], p=[0.6, 0.4])
                positions = np.random.choice(POCKET_POSITIONS, size=n_muts, replace=False)

                new_signature = parent_signature.copy()
                for pos in positions:
                    # Sample random amino acid (different from parent)
                    current_aa = parent_signature.get(pos, 'WT')
                    new_aa = np.random.choice([aa for aa in AA_ALPHABET if aa != current_aa])
                    new_signature[pos] = new_aa

                # Generate unique variant name
                sig_string = signature_to_string(new_signature)
                variant_hash = hashlib.md5(sig_string.encode()).hexdigest()[:6]
                variant_name = f"{ligand_name}_T2_{variant_hash}"

                tier2_negatives.append({
                    'ligand_name': ligand_name,
                    'ligand_smiles': ligand_smiles,
                    'variant_name': variant_name,
                    'variant_signature': sig_string,
                    'label': 0,
                    'label_tier': 'T2',
                    'label_source': f"near_neighbor_from_{row['PYR1_variant_name']}",
                    'label_confidence': 'MEDIUM',
                    'date_added': datetime.now().isoformat()
                })

    return pd.DataFrame(tier2_negatives)


def generate_tier3_random(ligand_list: List[Tuple[str, str]], n_per_ligand: int = 10) -> pd.DataFrame:
    """
    Generate Tier 3 easy negatives: 3–6 random pocket mutations.
    """
    tier3_negatives = []

    for ligand_name, ligand_smiles in ligand_list:
        for i in range(n_per_ligand):
            # Random 3–6 mutations
            n_muts = np.random.randint(3, 7)
            positions = np.random.choice(POCKET_POSITIONS, size=n_muts, replace=False)

            new_signature = {}
            for pos in positions:
                new_signature[pos] = np.random.choice(AA_ALPHABET)

            # Generate unique variant name
            sig_string = signature_to_string(new_signature)
            variant_hash = hashlib.md5(sig_string.encode()).hexdigest()[:6]
            variant_name = f"{ligand_name}_T3_{variant_hash}"

            tier3_negatives.append({
                'ligand_name': ligand_name,
                'ligand_smiles': ligand_smiles,
                'variant_name': variant_name,
                'variant_signature': sig_string,
                'label': 0,
                'label_tier': 'T3',
                'label_source': 'random_pocket_mutations',
                'label_confidence': 'LOW',
                'date_added': datetime.now().isoformat()
            })

    return pd.DataFrame(tier3_negatives)


def main():
    parser = argparse.ArgumentParser(description='Generate negative dataset')
    parser.add_argument('--positives', required=True, help='Path to ligand_smiles_signature.csv')
    parser.add_argument('--tier1-screens', nargs='+', help='Paths to FACS screen CSVs')
    parser.add_argument('--tier2-count', type=int, default=10, help='Tier 2 negatives per ligand')
    parser.add_argument('--tier3-count', type=int, default=10, help='Tier 3 negatives per ligand')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load positives
    positives_df = pd.read_csv(args.positives)
    print(f"Loaded {len(positives_df)} positive pairs")

    # Get unique ligands
    unique_ligands = positives_df[['ligand_name', 'ligand_smiles_or_ligand_ID']].drop_duplicates()
    ligand_list = list(zip(unique_ligands['ligand_name'], unique_ligands['ligand_smiles_or_ligand_ID']))
    print(f"Found {len(ligand_list)} unique ligands")

    # Generate negatives
    negatives = []

    if args.tier1_screens:
        print("Generating Tier 1 negatives from screens...")
        tier1_df = generate_tier1_from_screens(args.tier1_screens, positives_df)
        print(f"  Generated {len(tier1_df)} Tier 1 negatives")
        negatives.append(tier1_df)

    print(f"Generating Tier 2 negatives ({args.tier2_count} per ligand)...")
    tier2_df = generate_tier2_near_neighbors(positives_df, args.tier2_count)
    print(f"  Generated {len(tier2_df)} Tier 2 negatives")
    negatives.append(tier2_df)

    print(f"Generating Tier 3 negatives ({args.tier3_count} per ligand)...")
    tier3_df = generate_tier3_random(ligand_list, args.tier3_count)
    print(f"  Generated {len(tier3_df)} Tier 3 negatives")
    negatives.append(tier3_df)

    # Merge all negatives
    negatives_df = pd.concat(negatives, ignore_index=True)

    # Remove duplicates (same variant signature + ligand)
    negatives_df = negatives_df.drop_duplicates(subset=['ligand_name', 'variant_signature'])

    # Save
    negatives_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(negatives_df)} negatives to {args.output}")

    # Summary
    print("\nSummary:")
    print(negatives_df.groupby('label_tier').size())


if __name__ == '__main__':
    main()
```

---

### 2. `orchestrate_ml_pipeline.py`

```python
#!/usr/bin/env python3
"""
Master orchestrator for PYR1 ML dataset generation pipeline.

Workflow:
1. Load positives + negatives → merge dataset
2. For each (ligand, variant) pair:
   a. Check cache (skip if complete)
   b. Generate conformers (or load from cache)
   c. Submit docking SLURM array job
   d. Submit relax job (dependency: docking done)
   e. Submit AF3 jobs (dependency: relax done)
3. Aggregate all outputs → features_table.csv
"""

import argparse
import pandas as pd
import os
import json
import subprocess
from pathlib import Path
import hashlib
from typing import Dict, List


def get_pair_cache_key(ligand_smiles: str, variant_signature: str) -> str:
    """Generate unique cache key for (ligand, variant) pair."""
    combined = f"{ligand_smiles}_{variant_signature}"
    return hashlib.md5(combined.encode()).hexdigest()


def is_pair_complete(pair_id: str, cache_dir: Path) -> bool:
    """Check if pair has completed all pipeline stages."""
    metadata_path = cache_dir / pair_id / 'metadata.json'
    if not metadata_path.exists():
        return False

    metadata = json.load(open(metadata_path))
    required_stages = ['conformer_generation', 'docking', 'relax', 'af3_binary', 'af3_ternary']

    for stage in required_stages:
        if stage not in metadata.get('stages', {}):
            return False
        if metadata['stages'][stage]['status'] != 'complete':
            return False

    return True


def submit_conformer_generation(ligand_name: str, ligand_smiles: str, cache_dir: Path) -> Path:
    """
    Generate ligand conformers using existing ligand_conformers module.
    Returns path to conformers_final.sdf.
    """
    ligand_hash = hashlib.md5(ligand_smiles.encode()).hexdigest()
    output_dir = cache_dir / f"ligand_{ligand_hash}" / "conformers"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if (output_dir / 'conformers_final.sdf').exists():
        print(f"  Conformers already generated for {ligand_name}")
        return output_dir / 'conformers_final.sdf'

    # Run conformer generation
    cmd = [
        'python', '-m', 'ligand_conformers',
        '--input', ligand_smiles,
        '--input-type', 'smiles',
        '--output', str(output_dir),
        '--num-confs', '10',
        '--cluster-rmsd-cutoff', '1.0',
        '--k-final', '10'
    ]

    print(f"  Generating conformers for {ligand_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Conformer generation failed: {result.stderr}")

    return output_dir / 'conformers_final.sdf'


def submit_docking_job(pair: Dict, conformers_sdf: Path, cache_dir: Path, slurm: bool = False) -> str:
    """
    Submit docking job for (ligand, variant) pair.
    Returns job ID (if SLURM) or 'local'.
    """
    pair_id = pair['pair_id']
    pair_cache = cache_dir / pair_id
    pair_cache.mkdir(parents=True, exist_ok=True)

    # Prepare config
    config = {
        'ligand_sdf': str(conformers_sdf),
        'variant_signature': pair['variant_signature'],
        'output_dir': str(pair_cache / 'docking'),
        'docking_repeats': 50,
        'cluster_rmsd_cutoff': 2.0
    }

    config_path = pair_cache / 'docking_config.json'
    json.dump(config, open(config_path, 'w'), indent=2)

    if slurm:
        # Submit SLURM array job
        cmd = [
            'sbatch',
            '--job-name', f"dock_{pair_id}",
            '--output', str(pair_cache / 'docking_%a.log'),
            '--array', '0-9',  # 10 array tasks × 5 repeats = 50 total
            'submit_docking_workflow.sh',
            str(config_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted docking job {job_id} for {pair_id}")
        return job_id
    else:
        # Run locally (for testing)
        cmd = ['python', 'run_docking_workflow.py', '--config', str(config_path)]
        subprocess.run(cmd)
        return 'local'


def submit_relax_job(pair: Dict, cache_dir: Path, dependency_job_id: str = None, slurm: bool = False) -> str:
    """Submit Rosetta relax job."""
    pair_id = pair['pair_id']
    pair_cache = cache_dir / pair_id

    config = {
        'docked_pose': str(pair_cache / 'docking' / 'clustered_final' / 'best_pose.pdb'),
        'variant_signature': pair['variant_signature'],
        'output_dir': str(pair_cache / 'relax')
    }

    config_path = pair_cache / 'relax_config.json'
    json.dump(config, open(config_path, 'w'), indent=2)

    if slurm:
        cmd = ['sbatch', '--job-name', f"relax_{pair_id}"]
        if dependency_job_id:
            cmd += ['--dependency', f"afterok:{dependency_job_id}"]
        cmd += ['submit_pyrosetta_general_threading_relax.sh', str(config_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted relax job {job_id} for {pair_id}")
        return job_id
    else:
        cmd = ['python', 'general_relax.py', '--config', str(config_path)]
        subprocess.run(cmd)
        return 'local'


def submit_af3_jobs(pair: Dict, cache_dir: Path, dependency_job_id: str = None, slurm: bool = False) -> Dict[str, str]:
    """Submit AF3 binary and ternary jobs."""
    pair_id = pair['pair_id']
    pair_cache = cache_dir / pair_id

    # Generate AF3 JSONs
    binary_json = pair_cache / 'af3_binary_input.json'
    ternary_json = pair_cache / 'af3_ternary_input.json'

    # (Use existing make_af3_jsons.py logic here)
    # ...

    job_ids = {}

    if slurm:
        # Submit binary job
        cmd = ['sbatch', '--job-name', f"af3_bin_{pair_id}"]
        if dependency_job_id:
            cmd += ['--dependency', f"afterok:{dependency_job_id}"]
        cmd += ['af3_gpu_submit.sh', str(binary_json), str(pair_cache / 'af3_binary')]

        result = subprocess.run(cmd, capture_output=True, text=True)
        job_ids['binary'] = result.stdout.strip().split()[-1]

        # Submit ternary job
        cmd = ['sbatch', '--job-name', f"af3_tern_{pair_id}"]
        if dependency_job_id:
            cmd += ['--dependency', f"afterok:{dependency_job_id}"]
        cmd += ['af3_gpu_submit.sh', str(ternary_json), str(pair_cache / 'af3_ternary')]

        result = subprocess.run(cmd, capture_output=True, text=True)
        job_ids['ternary'] = result.stdout.strip().split()[-1]

        print(f"  Submitted AF3 jobs {job_ids} for {pair_id}")
    else:
        # Run locally
        subprocess.run(['python', 'run_af3.py', '--input', str(binary_json)])
        subprocess.run(['python', 'run_af3.py', '--input', str(ternary_json)])
        job_ids = {'binary': 'local', 'ternary': 'local'}

    return job_ids


def main():
    parser = argparse.ArgumentParser(description='Orchestrate ML pipeline')
    parser.add_argument('--positives', required=True, help='Path to positives CSV')
    parser.add_argument('--negatives', required=True, help='Path to negatives CSV')
    parser.add_argument('--pilot-ligand', help='Run pilot on single ligand')
    parser.add_argument('--ligand-list', help='File with ligands to process (one per line)')
    parser.add_argument('--all-ligands', action='store_true', help='Process all ligands')
    parser.add_argument('--cache-dir', required=True, help='Cache directory')
    parser.add_argument('--output', required=True, help='Output features CSV')
    parser.add_argument('--slurm', action='store_true', help='Use SLURM submission')
    parser.add_argument('--max-positives', type=int, help='Max positives per ligand')
    parser.add_argument('--tier1-per-ligand', type=int, default=5)
    parser.add_argument('--tier2-per-ligand', type=int, default=10)
    parser.add_argument('--tier3-per-ligand', type=int, default=5)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    positives_df = pd.read_csv(args.positives)
    negatives_df = pd.read_csv(args.negatives)

    # Add label column to positives
    positives_df['label'] = 1
    positives_df['label_tier'] = 'positive'
    positives_df['label_source'] = 'validated_binder'
    positives_df['label_confidence'] = 'HIGH'

    # Standardize column names
    positives_df = positives_df.rename(columns={
        'ligand_smiles_or_ligand_ID': 'ligand_smiles',
        'PYR1_variant_name': 'variant_name',
        'PYR1_variant_signature': 'variant_signature'
    })

    # Filter ligands
    if args.pilot_ligand:
        positives_df = positives_df[positives_df['ligand_name'] == args.pilot_ligand]
        negatives_df = negatives_df[negatives_df['ligand_name'] == args.pilot_ligand]
    elif args.ligand_list:
        ligand_filter = open(args.ligand_list).read().splitlines()
        positives_df = positives_df[positives_df['ligand_name'].isin(ligand_filter)]
        negatives_df = negatives_df[negatives_df['ligand_name'].isin(ligand_filter)]

    # Sample negatives per ligand
    negatives_sampled = []
    for ligand in positives_df['ligand_name'].unique():
        lig_negs = negatives_df[negatives_df['ligand_name'] == ligand]
        t1 = lig_negs[lig_negs['label_tier'] == 'T1'].sample(min(args.tier1_per_ligand, len(lig_negs[lig_negs['label_tier'] == 'T1'])))
        t2 = lig_negs[lig_negs['label_tier'] == 'T2'].sample(min(args.tier2_per_ligand, len(lig_negs[lig_negs['label_tier'] == 'T2'])))
        t3 = lig_negs[lig_negs['label_tier'] == 'T3'].sample(min(args.tier3_per_ligand, len(lig_negs[lig_negs['label_tier'] == 'T3'])))
        negatives_sampled.append(pd.concat([t1, t2, t3]))

    negatives_df = pd.concat(negatives_sampled, ignore_index=True)

    # Merge positives + negatives
    dataset_df = pd.concat([positives_df, negatives_df], ignore_index=True)
    dataset_df['pair_id'] = dataset_df.apply(
        lambda row: get_pair_cache_key(row['ligand_smiles'], row['variant_signature']), axis=1
    )

    print(f"\nDataset summary:")
    print(f"  Total pairs: {len(dataset_df)}")
    print(f"  Positives: {len(dataset_df[dataset_df['label'] == 1])}")
    print(f"  Negatives: {len(dataset_df[dataset_df['label'] == 0])}")
    print(dataset_df.groupby('label_tier').size())

    # Process each pair
    job_tracker = {}

    for idx, row in dataset_df.iterrows():
        pair_id = row['pair_id']

        # Check cache
        if is_pair_complete(pair_id, cache_dir):
            print(f"[{idx+1}/{len(dataset_df)}] Skipping {pair_id} (already complete)")
            continue

        print(f"\n[{idx+1}/{len(dataset_df)}] Processing {row['ligand_name']} × {row['variant_name']}")

        # Step 1: Generate conformers
        conformers_sdf = submit_conformer_generation(
            row['ligand_name'], row['ligand_smiles'], cache_dir
        )

        # Step 2: Submit docking
        dock_job = submit_docking_job(row.to_dict(), conformers_sdf, cache_dir, args.slurm)

        # Step 3: Submit relax (depends on docking)
        relax_job = submit_relax_job(row.to_dict(), cache_dir, dock_job, args.slurm)

        # Step 4: Submit AF3 (depends on relax)
        af3_jobs = submit_af3_jobs(row.to_dict(), cache_dir, relax_job, args.slurm)

        job_tracker[pair_id] = {
            'docking': dock_job,
            'relax': relax_job,
            'af3': af3_jobs
        }

    # Save job tracker
    json.dump(job_tracker, open(cache_dir / 'job_tracker.json', 'w'), indent=2)
    print(f"\nSubmitted {len(job_tracker)} pipeline jobs")
    print(f"Job tracker saved to {cache_dir / 'job_tracker.json'}")

    if args.slurm:
        print("\nMonitor jobs with: squeue -u $USER")
        print(f"Aggregate results when complete with: python aggregate_ml_features.py --cache-dir {cache_dir} --output {args.output}")
    else:
        print("\nRunning aggregation...")
        # Call aggregation directly
        subprocess.run(['python', 'aggregate_ml_features.py', '--cache-dir', str(cache_dir), '--output', args.output])


if __name__ == '__main__':
    main()
```

---

### 3. `aggregate_ml_features.py` (Skeleton)

```python
#!/usr/bin/env python3
"""
Aggregate all Rosetta + AF3 outputs into a single feature table.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List


def extract_rosetta_features(pair_cache: Path) -> Dict:
    """Extract features from Rosetta score file."""
    sc_file = pair_cache / 'relax' / 'rosetta_scores.sc'
    if not sc_file.exists():
        return {}

    # Parse Rosetta score file (space-delimited, skip header lines)
    df = pd.read_csv(sc_file, delim_whitespace=True, comment='#', skiprows=1)

    return {
        'rosetta_total_score': df['total_score'].min(),
        'rosetta_dG_sep': df['dG_separated'].min(),
        'rosetta_buried_unsats': df['buried_unsatisfied_hbonds'].min(),
        'rosetta_sasa_interface': df['interface_sasa'].max(),
        'rosetta_hbonds_interface': df['interface_hbonds'].max(),
        # ... (add more features)
    }


def extract_af3_features(pair_cache: Path, mode: str = 'binary') -> Dict:
    """Extract features from AF3 output."""
    af3_dir = pair_cache / f'af3_{mode}'
    summary_json = af3_dir / 'summary.json'

    if not summary_json.exists():
        return {}

    summary = json.load(open(summary_json))

    return {
        f'af3_{mode}_ipTM': summary['ipTM'],
        f'af3_{mode}_pLDDT_protein': summary['mean_pLDDT_protein'],
        f'af3_{mode}_pLDDT_ligand': summary['mean_pLDDT_ligand'],
        f'af3_{mode}_interface_PAE': summary['mean_interface_PAE'],
        f'af3_{mode}_ligand_RMSD': summary['ligand_RMSD_to_template'],
        # ... (add more)
    }


def main():
    parser = argparse.ArgumentParser(description='Aggregate ML features')
    parser.add_argument('--cache-dir', required=True, help='Cache directory')
    parser.add_argument('--output', required=True, help='Output CSV')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Find all completed pairs
    pair_dirs = [d for d in cache_dir.glob('*') if d.is_dir() and (d / 'metadata.json').exists()]

    features_list = []

    for pair_dir in pair_dirs:
        metadata = json.load(open(pair_dir / 'metadata.json'))

        # Extract features
        features = {
            'pair_id': metadata['pair_id'],
            'ligand_name': metadata['ligand_name'],
            'ligand_smiles': metadata['ligand_smiles'],
            'variant_name': metadata['variant_name'],
            'variant_signature': metadata['variant_signature'],
            'label': metadata['label'],
            'label_tier': metadata['label_tier'],
            'label_source': metadata['label_source'],
            'label_confidence': metadata['label_confidence'],
        }

        # Add Rosetta features
        features.update(extract_rosetta_features(pair_dir))

        # Add AF3 features
        features.update(extract_af3_features(pair_dir, 'binary'))
        features.update(extract_af3_features(pair_dir, 'ternary'))

        features_list.append(features)

    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(args.output, index=False)

    print(f"Aggregated {len(features_df)} pairs to {args.output}")
    print(f"Completeness: {(~features_df.isna()).mean().mean() * 100:.1f}%")


if __name__ == '__main__':
    main()
```

---

## SLURM TEMPLATES

### Docking Array Job (`submit_docking_workflow.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=dock_array
#SBATCH --output=logs/dock_%A_%a.log
#SBATCH --error=logs/dock_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=normal

# Load config
CONFIG_FILE=$1

# Extract parameters
LIGAND_SDF=$(jq -r '.ligand_sdf' $CONFIG_FILE)
VARIANT_SIG=$(jq -r '.variant_signature' $CONFIG_FILE)
OUTPUT_DIR=$(jq -r '.output_dir' $CONFIG_FILE)
REPEATS_PER_TASK=5

# Run docking
python grade_conformers_glycine_shaved.py \
    --ligand-sdf $LIGAND_SDF \
    --variant-signature "$VARIANT_SIG" \
    --output-dir $OUTPUT_DIR \
    --task-id $SLURM_ARRAY_TASK_ID \
    --repeats $REPEATS_PER_TASK
```

### AF3 GPU Job (`af3_gpu_submit.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=af3_gpu
#SBATCH --output=logs/af3_%j.log
#SBATCH --error=logs/af3_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

INPUT_JSON=$1
OUTPUT_DIR=$2

# Load AF3 environment
module load alphafold3
source activate af3

# Run AF3
run_alphafold \
    --json_path=$INPUT_JSON \
    --output_dir=$OUTPUT_DIR \
    --model_dir=/shared/alphafold3/models \
    --db_dir=/shared/alphafold3/databases
```

---

## MONITORING & DEBUGGING

### Check Pipeline Progress

```bash
# Count completed pairs
find $CACHE_DIR -name "metadata.json" -exec grep -l '"status": "complete"' {} \; | wc -l

# Check for errors
find $CACHE_DIR -name "metadata.json" -exec grep -l '"errors"' {} \;

# Check SLURM queue
squeue -u $USER --format="%.10i %.30j %.8T %.10M %.6D"

# Monitor AF3 GPU jobs
squeue -u $USER --partition=gpu --format="%.10i %.30j %.8T %.10M %.6D"
```

### Debug Failed Pair

```bash
PAIR_ID="abc123def456"
cat $CACHE_DIR/$PAIR_ID/metadata.json | jq '.errors'
tail -n 50 $CACHE_DIR/$PAIR_ID/docking_0.log
tail -n 50 $CACHE_DIR/$PAIR_ID/af3_binary/af3.log
```

### Resubmit Failed Jobs

```python
# In orchestrate_ml_pipeline.py, add --retry-failed flag
parser.add_argument('--retry-failed', action='store_true', help='Retry failed pairs')

# Check metadata for incomplete stages
if args.retry_failed:
    for pair_dir in cache_dir.glob('*'):
        metadata_path = pair_dir / 'metadata.json'
        if metadata_path.exists():
            metadata = json.load(open(metadata_path))
            for stage, info in metadata['stages'].items():
                if info['status'] == 'failed':
                    # Resubmit this stage
                    print(f"Retrying {pair_dir.name} stage {stage}")
```

---

## VALIDATION CHECKLIST

### Phase 0 Acceptance

- [ ] `generate_negatives.py` runs without errors
- [ ] `negatives_curated.csv` has ≥30 Tier 1 + Tier 2/3 generated
- [ ] Pilot (1 ligand × 30 pairs) submitted successfully
- [ ] All SLURM jobs start (no immediate failures)
- [ ] Check one pair's outputs manually:
  - [ ] Conformers SDF exists and has 10 structures
  - [ ] Docking produces clustered poses
  - [ ] Relax outputs PDB + score file
  - [ ] AF3 binary/ternary complete with summary JSON

### Phase 1 Acceptance

- [ ] `features_pilot.csv` has ≥270/300 rows (90% completion)
- [ ] No systematic failures (e.g., all Tier 3 failing)
- [ ] Score distributions show separation (plot histograms)
- [ ] Single-feature AUC (T1 vs T3) ≥ 0.85

### Phase 2 Acceptance

- [ ] `features_table.csv` has ≥1,800/2,000 rows
- [ ] Train/val/test splits are balanced
- [ ] Baseline XGBoost test AUC ≥ 0.75
- [ ] Feature importance analysis complete
- [ ] Final report rendered (HTML/PDF)

---

**End of Implementation Guide**
