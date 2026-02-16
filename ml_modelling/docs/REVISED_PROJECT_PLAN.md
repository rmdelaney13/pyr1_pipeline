# PYR1 Biosensor ML Dataset: REVISED Project Plan
**Version:** 2.0 (Revised)
**Date:** 2026-02-16
**Critical Revisions Based on Lab Feedback**

---

## EXECUTIVE SUMMARY

### Objective
Build a reproducible, HPC-scalable pipeline to generate a high-quality ML training dataset for PYR1 biosensor design. The pipeline integrates conformer generation (RDKit), Rosetta docking/relax, and AlphaFold3 predictions to produce multi-feature representations of (ligand, PYR1 variant) pairs for binder/non-binder classification and ranking.

### Critical Challenges

#### 1. **POSITIVE DATA QUALITY (NEW - CRITICAL)**
**Problem:** Not all "positive" hits are structurally modelable.

**Background:**
- **Your lab (yeast surface display):** EC50s < 1 μM (strong binders, clear structural signal)
- **Cutler lab (yeast 2-hybrid):** LOD 0.1–100 μM (weak binders, noisy structural signal)
- **System architecture:** PYR1-ligand (KD1) drives specificity, but HAB1-PYR1 (KD2 = 2 nM) dominates system EC50
  - If Y2H reports 10–100 μM system EC50 → actual PYR1-ligand KD1 is likely ≫100 μM
  - These weak binders may lack clear Rosetta/AF3 signatures

**Solution:** Stratify positives by affinity and prioritize high-quality data:
- **Tier P1 (Strong positives):** EC50 < 1 μM (surface display validated)
- **Tier P2 (Moderate positives):** EC50 1–10 μM (likely modelable)
- **Tier P3 (Weak positives):** EC50 > 10 μM (low confidence, use sparingly)

**Impact on Dataset Design:**
- Pilot phase: Use only P1+P2 positives (EC50 < 10 μM) for training
- Production phase: Include P3 as separate evaluation set (test model limits)

#### 2. **NEGATIVE DATA ACQUISITION**
Current dataset (287 pairs) contains only positives. Must generate/curate negatives:
- **Tier N1 (Hard):** Expression+ / Activation– from FACS screens
- **Tier N2 (Medium):** Near-neighbor variants (1–2 mutations from P1/P2 binders)
- **Tier N3 (Easy):** Random pocket variants (3–6 mutations, calibration only)

### Proposed Scope (Revised)
- **Phase 0 (Foundation):** Infrastructure setup, affinity filtering, mutation threading script, pilot validation
- **Phase 1 (Pilot):** 3 validated ligand families × balanced dataset = **~200 pairs**
  - WIN-55212-2 (Nature paper, strong P1 binders)
  - Nitazenes (Nature paper, strong P1 binders)
  - Lithocholic acid + 2 conjugates (in-progress, P1/P2 binders)
- **Phase 2 (Production):** All P1+P2 ligands + bile acid expansion = **~1,500 pairs**

### Compute Estimate (Phase 1 Pilot) - CORRECTED
- **Total CPU-hours:** ~1,250 (conformers + docking + relax)
- **Total GPU-hours:** ~2.3 (AF3 inference with templates - MUCH FASTER than estimated!)
- **Wall time (500 cores + 4 A100s):** ~1–2 days

**Key Correction:** AF3 with structural templates is 90–100× faster than estimated:
- Binary (PYR1 + ligand): ~20 seconds/prediction
- Ternary (PYR1 + ligand + HAB1): ~30 seconds/prediction

### Key Deliverables
1. **Affinity-stratified positive dataset** (P1/P2/P3 tiers)
2. **Mutation threading script** (`thread_variant_to_pdb.py`) to apply "59K;120A;160G" signatures
3. **Curated negative dataset** with provenance tracking
4. **Feature table** with 40+ Rosetta + AF3 metrics per pair
5. **Train/validation/test splits** (ligand-stratified + affinity-aware)
6. **Baseline ML benchmarks** (AUC, precision@k, cross-tier performance)

---

## CRITICAL REVISIONS

### Revision 1: Affinity-Based Positive Stratification

#### Current Dataset Analysis Needed
Before proceeding, we must:
1. **Audit `ligand_smiles_signature.csv`** for affinity data
2. **Link to experimental results** (FACS, surface display, Y2H)
3. **Classify each positive** into P1/P2/P3 tiers

**Proposed Classification:**

| Tier | EC50 Range | Assay | Confidence | Training Use | Count Target (Phase 1) |
|------|------------|-------|------------|--------------|------------------------|
| **P1** | < 1 μM | Surface display | HIGH | Primary training | 80 |
| **P2** | 1–10 μM | Surface display or Y2H | MEDIUM | Training + validation | 40 |
| **P3** | > 10 μM | Y2H (weak binders) | LOW | Hold-out test only | 20 |

**Impact on Model Training:**
- **Train on P1+P2 only** (strong structural signal)
- **Evaluate on P3 separately** (test model limits, expect lower accuracy)
- **Report performance by tier** (P1 vs P2 vs P3 AUC)

#### Implementation Steps
1. Create `affinity_annotation.csv` linking variants to EC50 data
2. Merge with `ligand_smiles_signature.csv` → add `positive_tier` column
3. Filter Phase 1 pilot to P1+P2 only (EC50 < 10 μM)

---

### Revision 2: Mutation Threading Script (NEW DELIVERABLE)

#### Problem Statement
**Current gap:** You have glycine-shaved docking working, but no script to thread variant signatures (e.g., "59K;120A;160G") onto the PYR1 template before docking.

**Existing capability (found in codebase):**
- `pyrosetta.toolbox.mutate_residue(pose, position, amino_acid)` - works
- Full-sequence threading in `general_relax.py` - works
- CSV-based position threading in `grade_conformers_sequence_csv_docking_multiple_slurm.py` - works

**Missing:** Parser for "59K;120A;160G" semicolon-separated format.

#### Script Design: `thread_variant_to_pdb.py`

```python
#!/usr/bin/env python3
"""
Thread PYR1 variant signature onto template PDB.

Usage:
    python thread_variant_to_pdb.py \
        --template 3QN1_nolig_H2O.pdb \
        --signature "59K;120A;160G" \
        --output mutant_59K_120A_160G.pdb

Input format: "59K;120A;160G" means:
    - Position 59 → Lysine (K)
    - Position 120 → Alanine (A)
    - Position 160 → Glycine (G)
"""

import argparse
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.toolbox import mutate_residue
import re


def parse_variant_signature(signature: str) -> dict:
    """
    Parse "59K;120A;160G" → {59: 'K', 120: 'A', 160: 'G'}

    Also handles underscore format: "K59Q_Y120A_A160G"
    """
    mutations = {}

    if not signature or signature == '':
        return mutations

    # Handle semicolon format: "59K;120A;160G"
    if ';' in signature:
        for mut in signature.split(';'):
            match = re.match(r'(\d+)([A-Z])', mut.strip())
            if match:
                pos, aa = match.groups()
                mutations[int(pos)] = aa

    # Handle underscore format: "K59Q_Y120A_A160G"
    elif '_' in signature or len(signature.split()) > 1:
        parts = signature.replace('_', ' ').split()
        for mut in parts:
            match = re.match(r'[A-Z](\d+)([A-Z])', mut.strip())
            if match:
                pos, aa = match.groups()
                mutations[int(pos)] = aa

    # Handle space-separated: "59K 120A 160G"
    else:
        for mut in signature.split():
            match = re.match(r'(\d+)([A-Z])', mut.strip())
            if match:
                pos, aa = match.groups()
                mutations[int(pos)] = aa

    return mutations


def apply_mutations(pose, mutations: dict, chain='A'):
    """
    Apply mutations to pose using PyRosetta.

    Args:
        pose: PyRosetta Pose object
        mutations: {position: amino_acid} dict
        chain: Chain ID (default 'A')
    """
    # Get PDB → Rosetta numbering mapping
    pdb_info = pose.pdb_info()

    for pdb_position, target_aa in sorted(mutations.items()):
        # Convert PDB numbering to Rosetta pose numbering
        rosetta_position = pdb_info.pdb2pose(chain, pdb_position)

        if rosetta_position == 0:
            print(f"WARNING: Position {pdb_position} not found in chain {chain}")
            continue

        # Get current amino acid
        current_aa = pose.residue(rosetta_position).name1()

        if current_aa == target_aa:
            print(f"  Position {pdb_position}: already {target_aa} (no change)")
        else:
            print(f"  Position {pdb_position}: {current_aa} → {target_aa}")
            mutate_residue(pose, rosetta_position, target_aa)

    return pose


def main():
    parser = argparse.ArgumentParser(description='Thread variant signature onto PDB')
    parser.add_argument('--template', required=True, help='Template PDB (e.g., 3QN1_nolig_H2O.pdb)')
    parser.add_argument('--signature', required=True, help='Variant signature (e.g., "59K;120A;160G")')
    parser.add_argument('--output', required=True, help='Output mutated PDB')
    parser.add_argument('--chain', default='A', help='Chain ID (default: A)')
    args = parser.parse_args()

    # Initialize PyRosetta
    pyrosetta.init('-mute all')

    # Load template
    print(f"Loading template: {args.template}")
    pose = pose_from_pdb(args.template)

    # Parse mutations
    mutations = parse_variant_signature(args.signature)
    print(f"\nParsed {len(mutations)} mutations:")
    for pos, aa in sorted(mutations.items()):
        print(f"  {pos}{aa}")

    # Apply mutations
    print("\nApplying mutations...")
    pose = apply_mutations(pose, mutations, args.chain)

    # Save output
    pose.dump_pdb(args.output)
    print(f"\nSaved mutated structure: {args.output}")


if __name__ == '__main__':
    main()
```

#### Integration into Workflow
**Before docking:**
```bash
# 1. Thread mutations onto template
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "59K;120A;160G" \
    --output mutant_59K_120A_160G.pdb

# 2. Use mutant structure for docking
python grade_conformers_glycine_shaved.py \
    --receptor mutant_59K_120A_160G.pdb \
    --ligand-sdf WIN_conformers.sdf \
    --output docking_results/
```

**Deliverable:** Add this script to `pyr1_pipeline/scripts/` in Phase 0.

---

### Revision 3: AF3 Timing Correction (MAJOR SPEEDUP)

#### Original Estimates (WRONG)
- Binary: 30 min/prediction → 150 GPU-hours for 300 pairs
- Ternary: 45 min/prediction → 225 GPU-hours for 300 pairs
- **Total Phase 1:** 375 GPU-hours

#### Corrected Estimates (with structural templates)
- Binary: **20 seconds**/prediction → 1.67 GPU-hours for 300 pairs
- Ternary: **30 seconds**/prediction → 2.5 GPU-hours for 300 pairs
- **Total Phase 1:** **4.17 GPU-hours** (90× faster!)

**Why so fast?** You're using structural templates for PYR1 and HAB1, which bypass the expensive MSA generation and reduce inference time dramatically.

#### Revised Compute Budget (Phase 1: 200 pairs)

| Stage | Unit Time | Units | Parallelism | CPU-hours | GPU-hours | Wall Time |
|-------|-----------|-------|-------------|-----------|-----------|-----------|
| Conformers | 2 min | 3 lig | 3 cores | 0.1 | 0 | <1 min |
| Docking | 5 min × 50 | 200 pairs | 200 cores | 833 | 0 | ~4 h |
| Relax | 15 min | 200 pairs | 100 cores | 50 | 0 | ~30 min |
| AF3 (binary) | 20 sec | 200 pairs | 4 A100s | 0 | 1.1 | ~17 min |
| AF3 (ternary) | 30 sec | 200 pairs | 4 A100s | 0 | 1.7 | ~25 min |
| **Total** | — | — | — | **883** | **2.8** | **~5 h** |

**Wall time with 500 cores + 4 A100s:** ~1 day (including queue time)

#### Revised Compute Budget (Phase 2: 1,500 pairs)

| Stage | Unit Time | Units | Parallelism | CPU-hours | GPU-hours | Wall Time |
|-------|-----------|-------|-------------|-----------|-----------|-----------|
| Conformers | 2 min | 15 lig | 15 cores | 0.5 | 0 | <1 min |
| Docking | 5 min × 50 | 1,500 pairs | 500 cores | 6,250 | 0 | ~13 h |
| Relax | 15 min | 1,500 pairs | 300 cores | 375 | 0 | ~1.3 h |
| AF3 (binary) | 20 sec | 1,500 pairs | 4 A100s | 0 | 8.3 | ~2 h |
| AF3 (ternary) | 30 sec | 1,500 pairs | 4 A100s | 0 | 12.5 | ~3 h |
| **Total** | — | — | — | **6,625** | **20.8** | **~19 h** |

**Wall time with 500 cores + 4 A100s:** ~3 days (including queue time)

**Cost Estimate (Phase 2):**
- CPU: 6,625 hrs × $0.02 = **$133**
- GPU (A100): 20.8 hrs × $1.50 = **$31**
- **Total: ~$164** (vs original estimate of $4,000!)

**Conclusion:** This project is **dramatically more feasible** than originally estimated. AF3 is no longer the bottleneck.

---

### Revision 4: Pilot Ligand Selection (Validated Families)

#### Original Pilot Plan
10 diverse ligands across all classes (cannabinoids, organophosphates, coumarins, steroids, nitazenes)

#### Revised Pilot Plan (Validated Families Only)
**Focus on 3 ligand families with published/validated data:**

| Ligand Family | Status | Ligands | P1 Binders | P2 Binders | Data Source |
|---------------|--------|---------|------------|------------|-------------|
| **WIN-55212-2** | Nature paper (published) | 1 | ~20 | ~10 | Surface display (EC50 < 1 μM) |
| **Nitazenes** | Nature paper (published) | 3–5 | ~30 | ~15 | Surface display (EC50 < 5 μM) |
| **Bile acids (LCA)** | In progress | 3 (LCA + 2 conjugates) | ~15 | ~10 | Current project (EC50 < 10 μM) |
| **Total** | — | **~9 ligands** | **~65** | **~35** | — |

**Pilot Dataset Design (Phase 1 Revised):**

| Component | Count | Notes |
|-----------|-------|-------|
| P1 positives | 65 | EC50 < 1 μM (high confidence) |
| P2 positives | 35 | EC50 1–10 μM (moderate confidence) |
| **Total positives** | **100** | Only high-quality binders |
| N1 negatives | 30 | Curated from FACS screens (if available) |
| N2 negatives | 50 | Near-neighbor variants (1–2 mutations from P1) |
| N3 negatives | 20 | Random pocket variants (calibration) |
| **Total negatives** | **100** | Balanced 1:1 ratio |
| **PILOT TOTAL** | **200 pairs** | Feasible for 1-day run |

**Rationale:**
- Prioritize **validated, high-affinity binders** (P1/P2) for clear structural signal
- Use published data (WIN, nitazenes) for reproducibility
- Include in-progress data (LCA) to demonstrate extensibility
- Defer weaker ligands (coumarins, organophosphates) to Phase 2

#### Future Expansion (Phase 2)
Once bile acid project completes:
- Add 4 additional bile acids (total 7 bile acids)
- Expand to other coumarin/organophosphate P1/P2 binders
- Target: ~1,500 total pairs

---

## PHASE PLAN (REVISED)

### **PHASE 0: Foundation & Validation** (2 weeks)

#### Deliverables
1. **Affinity Annotation**
   - Create `affinity_annotation.csv` linking variants to EC50/KD data
   - Classify all positives into P1/P2/P3 tiers
   - Filter `ligand_smiles_signature.csv` to P1+P2 only (EC50 < 10 μM)
   - Document affinity data sources (FACS CSVs, literature, lab notebooks)

2. **Mutation Threading Script**
   - Implement `thread_variant_to_pdb.py` (see Revision 2 above)
   - Test on 5 example variants (WIN, nitazene, LCA)
   - Validate output structures (visual inspection + Rosetta score)
   - Integrate into docking workflow

3. **Negative Dataset Curation**
   - N1: Extract ≥30 hard negatives from WIN/nitazene FACS screens
   - N2: Generate 50 near-neighbor variants (1–2 mutations from P1 binders)
   - N3: Generate 20 random pocket variants (3–6 mutations)
   - Output: `negatives_curated.csv` with provenance tracking

4. **Infrastructure Setup**
   - Extend `run_docking_workflow.py` to:
     - Accept affinity-annotated CSV input
     - Call `thread_variant_to_pdb.py` before docking
     - Track positive_tier and negative_tier in metadata
   - Add caching layer (skip completed pairs on re-runs)

5. **Pilot Validation (WIN-55212-2 × 30 pairs)**
   - 10 P1 positives + 10 P2 positives + 5 N2 + 5 N3 negatives
   - Run full pipeline (conformers → threading → docking → relax → AF3)
   - Validate:
     - Conformer quality (TREMD comparison if reference available)
     - Threading correctness (manual inspection of 3 mutants)
     - Rosetta score separation (P1 vs N3 effect size d > 0.8)
     - AF3 predictions complete in <1 hour (20s binary + 30s ternary × 30)

#### Acceptance Criteria
- [ ] ≥100 positives classified into P1/P2 tiers with documented EC50
- [ ] `thread_variant_to_pdb.py` produces correct mutants (validated on 5 structures)
- [ ] ≥30 N1 negatives curated OR 50 N2 negatives generated (fallback)
- [ ] Pilot run (30 pairs) completes in <4 hours wall time
- [ ] Score distributions show separation (P1 vs N3: d > 0.8, P1 vs N2: d > 0.4)

#### Risk Mitigations
- **Risk:** Affinity data missing for many positives
  **Mitigation:** Focus Phase 1 on WIN/nitazene (known high-affinity); defer ambiguous cases to Phase 2
- **Risk:** Threading script breaks on edge cases
  **Mitigation:** Add extensive input validation + unit tests for multiple signature formats
- **Risk:** Insufficient N1 negatives in historical data
  **Mitigation:** Use N2 as primary negatives (scientifically valid assumption)

---

### **PHASE 1: Pilot Dataset** (2 weeks - SHORTENED)

#### Deliverables
1. **Pilot Dataset Generation**
   - 3 validated ligand families (WIN, nitazenes, LCA)
   - 100 P1+P2 positives + 100 N1+N2+N3 negatives = **200 pairs**
   - Full pipeline: conformers → threading → docking → relax → AF3

2. **Feature Aggregation**
   - Merge Rosetta scores (dG_sep, buried_unsats, sasa, hbonds)
   - Merge AF3 metrics (ipTM, pLDDT, interface PAE, ligand RMSD)
   - Add conformer features (MMFF energy, rotatable bonds)
   - Add provenance (positive_tier, negative_tier, affinity, label_source)
   - Output: `features_pilot.csv` (200 rows × ~50 columns)

3. **Quality Control Report**
   - **By-tier distributions:** P1 vs P2 vs N1 vs N2 vs N3 (violin plots)
   - **Cross-tier AUCs:** P1 vs N1, P1 vs N2, P1 vs N3, P2 vs N2 (single best feature)
   - **Feature correlation heatmap** (identify redundant features)
   - **Missing data analysis** (% completion per stage)
   - **Affinity correlation:** EC50 vs rosetta_dG_sep, EC50 vs af3_ipTM (Spearman r)

4. **Preliminary Model**
   - Simple logistic regression (top 5 features)
   - Report AUC for: P1+P2 vs N1+N2, P1 vs N2 (hardest test)
   - Identify top 3 discriminative features

#### Acceptance Criteria
- [ ] ≥180/200 pairs complete all stages (90% success rate)
- [ ] **P1 vs N3 separation:** AUC ≥ 0.90 (sanity check - easy negatives)
- [ ] **P1 vs N2 separation:** AUC ≥ 0.65 (hard test - near neighbors)
- [ ] **P1 vs P2 separation:** AUC ≥ 0.55 (hardest test - both binders)
- [ ] **Affinity correlation:** EC50 vs dG_sep: |r| ≥ 0.4 (moderate correlation expected)
- [ ] <5% missing data in critical features (dG_sep, ipTM, pLDDT)
- [ ] AF3 wall time <1 hour (4 A100s) - validates timing correction

#### Dataset Design (Phase 1 Revised)

| Ligand Family | Positives (P1+P2) | N1 | N2 | N3 | Total |
|---------------|-------------------|----|----|----|----|
| WIN-55212-2 | 30 | 10 | 15 | 5 | 60 |
| Nitazenes (3) | 45 | 10 | 20 | 5 | 80 |
| LCA + conjugates (3) | 25 | 10 | 15 | 5 | 55 |
| **Total** | **100** | **30** | **50** | **15** | **195** |

*(Round to 200 with buffer)*

---

### **PHASE 2: Production Dataset** (4 weeks - SHORTENED)

#### Deliverables
1. **Full Dataset Generation**
   - All P1+P2 positives from validated families (~250)
   - Add 4 new bile acids when data available (+100 P1+P2)
   - Balanced negatives: 1:1 ratio (50% N1+N2, 50% N3)
   - **Total:** ~1,500 pairs (350 positives + 350 N1+N2 + 350 N3 × 1.3 redundancy)

2. **Train/Val/Test Splits**
   - **Ligand hold-out:** 60% train / 20% val / 20% test (stratified by family)
   - **Affinity hold-out:** Reserve 20% of P1 for final test (high-confidence evaluation)
   - **Variant family hold-out:** Additional 10% for unseen mutation patterns
   - Document split logic in `splits_metadata.json`

3. **Baseline ML Models**
   - Logistic regression (L2, top 10 features)
   - Random forest (500 trees, max_depth=10)
   - XGBoost (max_depth=6, early stopping)
   - Neural network (2-layer MLP, 128 units, dropout=0.2)

4. **Comprehensive Analysis**
   - **Feature importance:** SHAP values, permutation importance
   - **Single-feature AUCs:** Rank all 50 features
   - **Cross-tier performance matrix:** P1 vs {N1,N2,N3}, P2 vs {N1,N2,N3}
   - **Affinity prediction:** Regression model (EC50 vs features)
   - **Calibration curves:** Predicted probability vs true label
   - **Error analysis:** Misclassified examples (false positives/negatives)

5. **Final Report & Dataset Release**
   - Methods section (for publication)
   - Feature table with DOI (Zenodo/Dryad)
   - Code repository (GitHub)
   - Trained model weights

#### Acceptance Criteria
- [ ] ≥1,350/1,500 pairs complete (90% success rate)
- [ ] **Held-out test (P1 vs N1+N2):** AUC ≥ 0.75 (best model)
- [ ] **Cross-tier consistency:** P1 vs N3 AUC ~0.95, P1 vs N2 AUC 0.70–0.85
- [ ] **Top-3 features identified:** Hypothesis = dG_sep, ipTM, buried_unsats
- [ ] **Affinity regression:** EC50 vs predicted binding score: R² ≥ 0.3
- [ ] Feature table published with DOI

#### Dataset Design (Phase 2)

| Component | Count | Notes |
|-----------|-------|-------|
| P1 positives | 200 | EC50 < 1 μM (surface display) |
| P2 positives | 150 | EC50 1–10 μM |
| **Total positives** | **350** | High-confidence binders only |
| N1 negatives | 100 | Curated from FACS (expression+/activation–) |
| N2 negatives | 250 | Near-neighbor variants (1–2 muts from P1) |
| N3 negatives | 350 | Random pocket variants (calibration) |
| **Total negatives** | **700** | 2:1 negative:positive ratio |
| Redundancy buffer | +200 | Ensure 90% completion |
| **TOTAL SUBMITTED** | **~1,250** | Target 1,100 complete |

*(Revised down from 2,000 to focus on quality over quantity)*

---

## UPDATED COMPUTE BUDGET

### Phase 1 Pilot (200 pairs, 3 ligands)

| Resource | Original Estimate | Corrected Estimate | Reduction Factor |
|----------|-------------------|-------------------|------------------|
| CPU-hours | 1,325 | 883 | 1.5× |
| GPU-hours | 375 | 2.8 | **134×** |
| Wall time (500 cores + 4 A100s) | 2.5 days | 1 day | 2.5× |

**Cost:** ~$20 (vs $564 original)

### Phase 2 Production (1,500 pairs, 15 ligands)

| Resource | Original Estimate | Corrected Estimate | Reduction Factor |
|----------|-------------------|-------------------|------------------|
| CPU-hours | 8,834 | 6,625 | 1.3× |
| GPU-hours | 2,500 | 20.8 | **120×** |
| Wall time (500 cores + 4 A100s) | 14 days | 3 days | 4.7× |

**Cost:** ~$164 (vs $3,927 original)

### Scenario Analysis (Revised)

| Scenario | Pairs | Docking Repeats | Total CPU-h | Total GPU-h | Wall Time | Cost |
|----------|-------|-----------------|-------------|-------------|-----------|------|
| **Minimal (Pilot)** | 200 | 30 | 500 | 1.9 | 1 day | $13 |
| **Phase 1 (Baseline)** | 200 | 50 | 883 | 2.8 | 1 day | $20 |
| **Phase 2 (Production)** | 1,500 | 50 | 6,625 | 20.8 | 3 days | $164 |
| **Comprehensive** | 2,500 | 100 | 13,542 | 34.7 | 5 days | $323 |

**Conclusion:** Project is **highly feasible** with corrected AF3 timing. GPU cost is negligible.

---

## AFFINITY-AWARE EVALUATION PLAN

### Split Strategy (Revised)

#### Primary Split: Ligand + Affinity Stratified
```python
# Stratify by both ligand family AND positive tier
train_ligands = stratified_sample(
    all_ligands,
    frac=0.6,
    stratify_by=['ligand_family', 'positive_tier']
)

# Ensure P1 and P2 represented in all splits
val_ligands = stratified_sample(remaining, frac=0.5, ...)
test_ligands = remaining
```

#### Secondary Split: Affinity Hold-Out
```python
# Reserve 20% of P1 positives for final evaluation
# (These are highest-confidence binders - gold standard)
p1_holdout = P1_positives.sample(frac=0.2)
```

#### Tertiary Split: Variant Family Hold-Out
```python
# Hold out entire mutation clusters (same as before)
variant_clusters = cluster_by_signature(hamming_distance <= 1)
test_clusters_unseen = sample(variant_clusters, frac=0.1)
```

### Analysis Plan (Revised)

#### 1. Cross-Tier Performance Matrix

| Model Prediction → | P1 (True+) | P2 (True+) | N1 (True–) | N2 (True–) | N3 (True–) |
|-------------------|-----------|-----------|-----------|-----------|-----------|
| **P1 Test Set** | — | AUC(P1 vs P2) | AUC(P1 vs N1) | AUC(P1 vs N2) | AUC(P1 vs N3) |
| **P2 Test Set** | AUC(P2 vs P1) | — | AUC(P2 vs N1) | AUC(P2 vs N2) | AUC(P2 vs N3) |

**Expected results:**
- AUC(P1 vs N3) ≥ 0.95 (easy negatives)
- AUC(P1 vs N2) = 0.70–0.85 (near neighbors - hardest)
- AUC(P1 vs N1) = 0.75–0.90 (experimental negatives)
- AUC(P1 vs P2) = 0.55–0.65 (both binders - affinity discrimination)

#### 2. Affinity Correlation Analysis

**Hypothesis:** Rosetta dG_sep and AF3 ipTM should correlate with affinity.

```python
# Regression: EC50 ~ rosetta_dG_sep + af3_ipTM + ...
from scipy.stats import spearmanr

for feature in ['rosetta_dG_sep', 'af3_binary_ipTM', 'rosetta_buried_unsats']:
    r, p = spearmanr(positives_df['EC50'], positives_df[feature])
    print(f"{feature}: r={r:.3f}, p={p:.2e}")
```

**Success criterion:** |r| ≥ 0.4 for at least one feature (moderate correlation)

**Key insight:** If correlation is weak (|r| < 0.3), this suggests:
- Structural features alone insufficient for affinity prediction
- Need dynamics, entropy, or kinetic features
- Or: Y2H weak binders (P3) are noisy and should be excluded

#### 3. Feature Importance by Tier

**Question:** Do different features matter for P1 vs P2?

```python
# Train separate models on P1-only and P2-only
model_p1 = XGBoost(X_p1, y_p1)  # P1 positives vs all negatives
model_p2 = XGBoost(X_p2, y_p2)  # P2 positives vs all negatives

# Compare SHAP feature rankings
shap_p1 = shap.TreeExplainer(model_p1).shap_values(X_test)
shap_p2 = shap.TreeExplainer(model_p2).shap_values(X_test)

# Hypothesis: P1 has stronger interface signatures (dG_sep, ipTM)
#             P2 may rely more on buried unsats, water networks
```

#### 4. Baseline Models (Same as Before)
- Logistic regression (L2, C=1.0)
- Random Forest (500 trees, max_depth=10)
- XGBoost (max_depth=6, learning_rate=0.05)
- MLP (2 layers, 128 units, ReLU, dropout=0.2)

---

## RISK REGISTER (UPDATED)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Affinity data incomplete** (< 50% positives annotated) | Medium | High | Focus Phase 1 on WIN/nitazene (known); defer ambiguous to Phase 2; use literature EC50s |
| **P3 weak binders dominate dataset** (noisy signal) | Medium | High | Filter to P1+P2 only (EC50 < 10 μM) for training; use P3 as separate evaluation |
| **Threading script edge cases** (unusual signatures) | Low | Medium | Extensive validation on 20+ test cases; add input sanitization + error handling |
| **Insufficient N1 negatives** (< 30) | Medium | Medium | Use N2 as primary (scientifically valid); document assumption in methods |
| **Rosetta docking low convergence** (< 70%) | Low | Medium | Increase repeats to 100; use multiple starting orientations; relax RMSD cutoff |
| **AF3 template lookup fails** (missing PDB) | Low | High | Pre-validate templates exist for PYR1/HAB1; provide fallback ab initio mode |
| **Feature correlation too high** (r > 0.9) | Medium | Low | Use PCA or L1 regularization; report correlation matrix; remove redundant features |
| **Test set leakage** (ligand seen in train) | Low | High | Strict ligand-based splitting; automated validation checks; document in metadata |
| **HPC queue delays** (> 3 days) | High | Low | Submit off-peak; request priority allocation; short wall times (< 8h per job) |
| **Affinity-structure decoupling** (|r| < 0.3) | Medium | Medium | Report as negative result; investigate dynamics; consider ensemble docking |

---

## IMPLEMENTATION NOTES (UPDATED)

### New Code Module: `thread_variant_to_pdb.py`

**Location:** `pyr1_pipeline/scripts/thread_variant_to_pdb.py`

**Dependencies:** PyRosetta, argparse, re

**Testing plan:**
1. Unit tests: Parse 10 signature formats (semicolon, underscore, space-separated)
2. Integration test: Thread 5 known variants, validate with existing structures
3. Edge cases: Empty signature, invalid positions, non-standard AAs

**Integration points:**
- Called by `orchestrate_ml_pipeline.py` before docking
- Output cached in `{cache_dir}/{pair_id}/mutant.pdb`
- Logged in `metadata.json` under `stages.mutation_threading`

### Updated Folder Structure

```
ml_modelling/
├── ligand_smiles_signature.csv          # (Existing) All positive pairs
├── affinity_annotation.csv              # (NEW) EC50 data + P1/P2/P3 classification
├── positives_filtered.csv               # (NEW) P1+P2 only (EC50 < 10 μM)
├── negatives_curated.csv                # (NEW) N1/N2/N3 with provenance
├── pairs_dataset.csv                    # (Generated) Merged positives + negatives
├── features_pilot.csv                   # (Phase 1) 200 pairs × 50 features
├── features_table.csv                   # (Phase 2) 1,500 pairs × 50 features
├── train_set.csv, val_set.csv, test_set.csv
├── splits_metadata.json
├── baseline_results.csv
├── affinity_analysis_report.html        # (NEW) EC50 correlation plots
├── scripts/
│   ├── thread_variant_to_pdb.py         # (NEW) Mutation threading
│   ├── annotate_affinity.py             # (NEW) EC50 annotation from FACS
│   ├── generate_negatives.py
│   ├── orchestrate_ml_pipeline.py
│   ├── aggregate_ml_features.py
│   ├── split_dataset.py
│   ├── baseline_models.py
│   └── analyze_affinity_correlation.py  # (NEW) Affinity vs features
└── cache/
    └── {pair_id}/
        ├── mutant.pdb                   # (NEW) Threaded structure (before docking)
        ├── conformers/
        ├── docking/
        ├── relax/
        ├── af3_binary/
        ├── af3_ternary/
        └── metadata.json                # (Updated) Includes positive_tier, EC50
```

---

## DELIVERABLES CHECKLIST (UPDATED)

### Phase 0 (Foundation) - 2 weeks
- [ ] `affinity_annotation.csv` (≥100 positives with EC50 + P1/P2/P3 tier)
- [ ] `thread_variant_to_pdb.py` (validated on 10 test cases)
- [ ] `negatives_curated.csv` (≥30 N1 OR ≥50 N2 + 20 N3)
- [ ] Pilot validation (WIN × 30 pairs) complete in <4 hours
- [ ] Threading correctness validated (manual inspection of 3 structures)

### Phase 1 (Pilot) - 2 weeks
- [ ] `features_pilot.csv` (200 pairs, ≥180 complete)
- [ ] QC report with by-tier distributions
- [ ] Cross-tier AUC matrix (P1 vs {N1,N2,N3})
- [ ] Affinity correlation analysis (EC50 vs top 5 features)
- [ ] Preliminary logistic regression model (AUC ≥ 0.70)

### Phase 2 (Production) - 4 weeks
- [ ] `features_table.csv` (1,500 pairs, ≥1,350 complete)
- [ ] Train/val/test splits with affinity stratification
- [ ] Baseline models (4 algorithms, test AUC ≥ 0.75)
- [ ] Feature importance analysis (SHAP + permutation)
- [ ] Affinity regression model (EC50 prediction, R² ≥ 0.3)
- [ ] Final report (methods + figures + tables)
- [ ] Dataset published with DOI

---

## TIMELINE SUMMARY (REVISED)

| Phase | Duration | Key Milestone | Compute Time | Team Effort |
|-------|----------|---------------|--------------|-------------|
| Phase 0 | 2 weeks | Pilot validation (30 pairs) | 4 hours | 40 hours |
| Phase 1 | 2 weeks | Pilot dataset (200 pairs) | 1 day | 50 hours |
| Phase 2 | 4 weeks | Production dataset (1,500 pairs) | 3 days | 80 hours |
| **Total** | **8 weeks** | **Public dataset release** | **~4 days** | **~170 hours** |

**Reduction from original plan:** 11 weeks → 8 weeks (AF3 speedup enables faster iteration)

---

## SUCCESS CRITERIA (GO/NO-GO) - UPDATED

### Phase 0 → Phase 1 Gate
- [ ] ≥50 P1 positives annotated with EC50 < 1 μM
- [ ] `thread_variant_to_pdb.py` produces correct mutants (visual + Rosetta validation)
- [ ] Pilot (30 pairs) completes successfully
- [ ] P1 vs N3 score separation: d > 0.8

### Phase 1 → Phase 2 Gate
- [ ] Pilot AUC (P1 vs N2) ≥ 0.65 (hard test)
- [ ] Affinity correlation (EC50 vs dG_sep): |r| ≥ 0.3
- [ ] Pipeline wall time <1 day (validates compute estimates)
- [ ] <5% missing data in critical features

### Phase 2 → Publication Gate
- [ ] Test set AUC ≥ 0.75 (best model, P1+P2 vs N1+N2)
- [ ] Cross-tier consistency (P1 vs N3 AUC ~0.95)
- [ ] Feature table ≥1,350 complete pairs
- [ ] Top-3 features identified and validated (SHAP + permutation)

---

## QUESTIONS FOR YOU (ACTION ITEMS)

### Immediate (This Week)
1. **Affinity data access:** Can you provide EC50 values for WIN/nitazene/LCA variants?
   - Preferred format: CSV with columns `variant_name`, `ligand_name`, `EC50_uM`, `assay_type`, `date`
   - If not centralized: Point me to FACS CSVs or lab notebooks to extract

2. **FACS negative data:** Do you have Expression+/Activation– data from WIN or nitazene screens?
   - Need: Variant sequences that passed expression gate but failed activation
   - Format: Any (I can parse FACS CSVs)

3. **Bile acid data status:** For LCA + 2 conjugates:
   - Which variants are validated binders (EC50 < 10 μM)?
   - When do you expect data for the 4 additional bile acids? (Timeline for Phase 2 expansion)

### Technical Validation (Next Week)
4. **Threading script testing:** Can you provide 3 example variant signatures to test `thread_variant_to_pdb.py`?
   - E.g., one WIN variant, one nitazene variant, one LCA variant
   - Ideally with existing crystal structure or model for validation

5. **AF3 template confirmation:** What PDB/CIF templates are you using for:
   - PYR1 structure? (e.g., 3QN1?)
   - HAB1 structure? (e.g., from ternary complex?)
   - Ligand-bound reference? (for RMSD calculation)

### Strategic (Phase Planning)
6. **Affinity cutoff consensus:** Is EC50 < 10 μM a reasonable P1+P2 threshold?
   - Or should we be more conservative (e.g., EC50 < 5 μM)?

7. **Publication goals:** Is the target:
   - Methods paper (pipeline description + dataset release)?
   - Application paper (ML model for prospective design)?
   - Both (staged publications)?

---

**END OF REVISED PLAN**
