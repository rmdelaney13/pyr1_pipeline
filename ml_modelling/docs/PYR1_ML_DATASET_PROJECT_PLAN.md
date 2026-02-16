# PYR1 Biosensor ML Dataset Generation: Project Plan
**Version:** 1.0
**Date:** 2026-02-16
**Author:** Claude Code Analysis

---

## EXECUTIVE SUMMARY

### Objective
Build a reproducible, HPC-scalable pipeline to generate a high-quality ML training dataset for PYR1 biosensor design. The pipeline integrates conformer generation (RDKit), Rosetta docking/relax, and AlphaFold3 predictions to produce multi-feature representations of (ligand, PYR1 variant) pairs for binder/non-binder classification and ranking.

### Critical Challenge
**Negative data acquisition.** Current dataset (287 pairs from `ligand_smiles_signature.csv`) contains only validated binders. We must generate or curate negatives across three tiers:
- **Tier 1 (Hard):** Variants that passed expression but failed activation in past screens
- **Tier 2 (Medium):** Near-neighbor variants (1–2 mutations from binders)
- **Tier 3 (Easy):** Synthetic random pocket variants (3–6 mutations)

### Proposed Scope
- **Phase 0 (Foundation):** Infrastructure setup, negative dataset curation, pilot validation
- **Phase 1 (Pilot):** 10 ligands × (10 positives + 5 T1 + 10 T2 + 5 T3 negatives) = **300 pairs**
- **Phase 2 (Production):** All unique ligands × expanded dataset = **~2,000 pairs**

### Compute Estimate (Phase 1 Pilot)
- **Total CPU-hours:** ~18,000–36,000 (conformers + docking + relax)
- **Total GPU-hours:** ~150–300 (AF3 inference)
- **Wall time (1,000 cores + 8 A100s):** ~2–5 days

### Key Deliverables
1. Curated negative dataset with provenance tracking
2. Feature table with 40+ Rosetta + AF3 metrics per pair
3. Train/validation/test splits (ligand-stratified + variant-family hold-out)
4. Baseline ML model benchmarks (AUC, precision@k)
5. Production-ready orchestrator with caching and resumability

---

## PHASE PLAN

### **PHASE 0: Foundation & Validation** (2 weeks)

#### Deliverables
1. **Negative Dataset Curation**
   - Tier 1: Extract ≥50 hard negatives from historical screens (prioritize recent campaigns)
   - Tier 2: Generate near-neighbor variants using pocket position sampling
   - Tier 3: Generate random pocket variants (3–6 mutations) as easy negatives
   - Output: `negatives_curated.csv` with columns: `ligand_name`, `variant_name`, `variant_signature`, `label_tier`, `label_source`, `label_confidence`

2. **Infrastructure Setup**
   - Extend `run_docking_workflow.py` to accept negative-labeled CSV input
   - Create `generate_negatives.py` script with tier-specific logic
   - Add provenance tracking to all outputs (`metadata.json` per pair)
   - Set up caching layer (skip completed pairs on re-runs)

3. **Pilot Validation (1 ligand × 30 pairs)**
   - Run full pipeline on WIN-55212-2 (reference ligand with known binders)
   - Validate conformer quality vs TREMD benchmark
   - Check Rosetta score separation (binder vs non-binder)
   - Verify AF3 predictions complete successfully

#### Acceptance Criteria
- [ ] ≥50 Tier 1 negatives curated with documented provenance
- [ ] Tier 2/3 generation scripts produce valid variant signatures
- [ ] Pilot run (30 pairs) completes in <8 hours wall time
- [ ] Score distributions show clear separation (effect size d > 0.8)
- [ ] No hard failures in any pipeline stage

#### Risk Mitigations
- **Risk:** Insufficient Tier 1 negatives in historical data
  **Mitigation:** Lower threshold to ≥30 for Phase 1, prioritize prospective data capture
- **Risk:** AF3 batch job failures
  **Mitigation:** Implement per-JSON retries with exponential backoff

---

### **PHASE 1: Pilot Dataset** (3 weeks)

#### Deliverables
1. **Pilot Dataset Generation**
   - 10 structurally diverse ligands (cannabinoids, organophosphates, coumarins, steroids, nitazenes)
   - Per ligand: 10 positives + 5 T1 + 10 T2 + 5 T3 negatives = 30 pairs/ligand
   - **Total:** 300 pairs

2. **Feature Aggregation**
   - Merge Rosetta scores (dG_sep, buried_unsats, sasa, hbonds)
   - Merge AF3 metrics (ipTM, pLDDT, interface PAE, ligand RMSD)
   - Add conformer features (MMFF energy, rotatable bonds, Tanimoto similarity to centroid)
   - Output: `features_pilot.csv` (300 rows × ~45 columns)

3. **Quality Control Report**
   - Per-tier score distributions (violin plots)
   - Feature correlation heatmap
   - Missing data analysis (% completion per stage)
   - Outlier detection (>3σ from tier median)

#### Acceptance Criteria
- [ ] ≥270/300 pairs complete all pipeline stages (90% success rate)
- [ ] Tier 1 vs Tier 3 separation: AUC ≥ 0.85 (single best feature)
- [ ] Tier 1 vs Tier 2 separation: AUC ≥ 0.65 (harder test)
- [ ] <5% missing data in critical features (dG_sep, ipTM, pLDDT)
- [ ] AF3 inference completes in <48h wall time (8 A100 GPUs)

#### Dataset Design (Phase 1)

| Ligand Class | Ligands | Positives/Lig | T1/Lig | T2/Lig | T3/Lig | Total Pairs |
|--------------|---------|---------------|--------|--------|--------|-------------|
| Cannabinoids | 3       | 10            | 5      | 10     | 5      | 90          |
| Organophosphates | 2   | 10            | 5      | 10     | 5      | 60          |
| Coumarins    | 2       | 10            | 5      | 10     | 5      | 60          |
| Steroids     | 2       | 10            | 5      | 10     | 5      | 60          |
| Nitazenes    | 1       | 10            | 5      | 10     | 5      | 30          |
| **Total**    | **10**  | **100**       | **50** | **100**| **50** | **300**     |

---

### **PHASE 2: Production Dataset** (6 weeks)

#### Deliverables
1. **Full Dataset Generation**
   - All unique ligands from `ligand_smiles_signature.csv` (~40 ligands after de-duplication)
   - Expanded positives: all validated binders from CSV (~287 total)
   - Balanced negatives: 1:1 ratio (positive:negative) with 50% T1+T2, 50% T3
   - **Total:** ~2,000 pairs (287 positives + ~290 T1+T2 + ~290 T3 negatives × ~1.4 redundancy)

2. **Train/Val/Test Splits**
   - **Ligand hold-out:** 60% train / 20% val / 20% test (stratified by ligand class)
   - **Variant family hold-out:** Additional test set with unseen mutation patterns
   - Document split logic in `splits_metadata.json`

3. **Baseline ML Models**
   - Logistic regression (L2 penalty) on top-10 features
   - Random forest (500 trees, max_depth=10)
   - Gradient boosting (XGBoost with early stopping)
   - Neural network (2-layer MLP, 128 hidden units)

4. **Analysis Report**
   - Feature importance rankings (SHAP values)
   - Single-feature AUC comparison (which metrics matter most?)
   - Cross-tier performance (T1 vs T2 vs T3 classification accuracy)
   - Calibration curves (predicted probability vs actual label)

#### Acceptance Criteria
- [ ] ≥1,800/2,000 pairs complete (90% success rate)
- [ ] Held-out test AUC ≥ 0.75 (best model)
- [ ] Top-3 features: Rosetta interface dG, AF3 ipTM, buried unsats (hypothesis)
- [ ] T3 negatives: perfect separation (AUC ~1.0, sanity check)
- [ ] T1+T2 negatives: challenging but learnable (AUC 0.70–0.85)
- [ ] Feature table published with DOI (Zenodo/Dryad)

#### Dataset Design (Phase 2)

| Component | Count | Notes |
|-----------|-------|-------|
| Unique ligands | ~40 | After SMILES de-duplication |
| Positive pairs | 287 | All validated binders from CSV |
| Tier 1 negatives | 150 | Curated from historical screens + prospective data |
| Tier 2 negatives | 290 | Near-neighbor variants (1–2 mutations) |
| Tier 3 negatives | 290 | Random pocket variants (3–6 mutations) |
| Redundancy buffer | +200 | To ensure 90% completion → 2,000 target |
| **Total submitted** | **~2,200** | ~2,000 expected to complete |

---

## COMPUTE BUDGET ESTIMATES

### Assumptions
- **Conformer generation:** 10 conformers/ligand (ETKDG + MMFF), ~2 min/ligand on 1 CPU
- **Docking:** 50 repeats/pair, ~5 min/repeat on 1 CPU (Rosetta glycine-shave protocol)
- **Relax:** 1 pose/pair (best docked), ~15 min/pose on 1 CPU (FastRelax + constraints)
- **AF3 (binary):** ~30 min/prediction on 1 A100 (includes MSA + inference)
- **AF3 (ternary):** ~45 min/prediction on 1 A100 (larger complex)

### Phase 1 Pilot (300 pairs, 10 ligands)

| Stage | Unit Time | Units | Parallelism | CPU-hours | GPU-hours | Wall Time |
|-------|-----------|-------|-------------|-----------|-----------|-----------|
| Conformers | 2 min | 10 lig | 10 cores | 0.3 | 0 | <1 min |
| Docking | 5 min × 50 | 300 pairs | 300 cores | 1,250 | 0 | ~4 h |
| Relax | 15 min | 300 pairs | 150 cores | 75 | 0 | ~30 min |
| AF3 (binary) | 30 min | 300 pairs | 8 A100s | 0 | 150 | ~19 h |
| AF3 (ternary) | 45 min | 300 pairs | 8 A100s | 0 | 225 | ~28 h |
| **Total** | — | — | — | **1,325** | **375** | **~52 h** |

**Wall time with 1,000 cores + 8 A100s:** ~2.5 days (accounting for queue delays)

### Phase 2 Production (2,000 pairs, 40 ligands)

| Stage | Unit Time | Units | Parallelism | CPU-hours | GPU-hours | Wall Time |
|-------|-----------|-------|-------------|-----------|-----------|-----------|
| Conformers | 2 min | 40 lig | 40 cores | 1.3 | 0 | <1 min |
| Docking | 5 min × 50 | 2,000 pairs | 500 cores | 8,333 | 0 | ~17 h |
| Relax | 15 min | 2,000 pairs | 500 cores | 500 | 0 | ~1 h |
| AF3 (binary) | 30 min | 2,000 pairs | 8 A100s | 0 | 1,000 | ~125 h |
| AF3 (ternary) | 45 min | 2,000 pairs | 8 A100s | 0 | 1,500 | ~188 h |
| **Total** | — | — | — | **8,834** | **2,500** | **~330 h** |

**Wall time with 1,000 cores + 8 A100s:** ~14 days (accounting for queue delays + retries)

### Scenario Analysis

| Scenario | Pairs | Docking Repeats | Total CPU-h | Total GPU-h | Wall Time (1K cores + 8 A100s) |
|----------|-------|-----------------|-------------|-------------|-------------------------------|
| **Low (Conservative)** | 1,500 | 30 | 6,000 | 1,875 | ~10 days |
| **Medium (Baseline)** | 2,000 | 50 | 8,834 | 2,500 | ~14 days |
| **High (Comprehensive)** | 3,000 | 100 | 26,500 | 3,750 | ~21 days |

### Cost Estimates (HPC Allocation)
Assuming typical university HPC rates:
- **CPU-hours:** $0.02/core-hour → Phase 2 = $177 (CPU only)
- **GPU-hours (A100):** $1.50/GPU-hour → Phase 2 = $3,750 (GPU only)
- **Total Phase 2:** ~$4,000 (dominated by AF3 inference)

*Note: Many academic HPCs provide free allocations; commercial cloud (AWS, GCP) would be 3–5× higher.*

---

## NEGATIVE DATASET CURATION STRATEGY

### Tier 1: Historical Screen Hard Negatives (Target: ≥150)

#### Minimal Curation Protocol
1. **Identify recent library screens** (last 2 years) with expression + activation data
2. **Filtering criteria:**
   - Expression: GFP intensity > threshold (passed QC gate)
   - Activation: AUC < 0.6 or fold-change < 1.5 (failed activation)
   - Coverage: ≥3 variants per ligand class (ensure diversity)
3. **Data extraction:**
   - Source: FACS CSV files, library design spreadsheets
   - Fields: `variant_sequence`, `ligand_tested`, `expression_score`, `activation_score`
   - Confidence: `HIGH` (experimentally validated)
4. **Effort estimate:** ~8 hours manual curation (script-assisted filtering)

#### Prospective Data Capture (Ongoing)
- **Standardize library design templates** to include:
  - Variant signature (pocket mutations)
  - Ligand SMILES (not just name)
  - `label_tier=T1` flag for failed variants
- **Automate data export** from FACS analysis pipeline to `negatives_prospective.csv`
- **Goal:** Accumulate 50+ new negatives per campaign

### Tier 2: Near-Neighbor Soft Negatives (Target: ~290)

#### Generation Algorithm
```python
def generate_tier2_negatives(positive_variant, n_negatives=10):
    """
    Sample near-neighbor variants (1–2 mutations from positive).
    Assumption: Most 1-mutation variants are non-binders (soft assumption).
    """
    pocket_positions = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]
    aa_alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    negatives = []
    for _ in range(n_negatives):
        n_muts = random.choice([1, 2])  # 1–2 mutations
        positions = random.sample(pocket_positions, n_muts)
        new_variant = mutate_signature(positive_variant, positions, aa_alphabet)
        negatives.append(new_variant)

    return negatives
```

#### Confidence Labeling
- **Confidence:** `MEDIUM` (inferred, not validated)
- **Provenance:** `generated_near_neighbor_from_{parent_variant}`
- **Exclusion list:** Check against known positives to avoid false negatives

### Tier 3: Random Pocket Negatives (Target: ~290)

#### Generation Algorithm
```python
def generate_tier3_negatives(ligand, n_negatives=10):
    """
    Generate random 3–6 pocket mutations (almost surely disrupt binding).
    """
    pocket_positions = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]
    aa_alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    negatives = []
    for _ in range(n_negatives):
        n_muts = random.randint(3, 6)  # 3–6 random mutations
        positions = random.sample(pocket_positions, n_muts)
        new_variant = random_mutate(positions, aa_alphabet)
        negatives.append(new_variant)

    return negatives
```

#### Confidence Labeling
- **Confidence:** `LOW` (synthetic, untested)
- **Provenance:** `random_pocket_mutations`
- **Usage:** Calibration only; exclude from final model evaluation

### Provenance Tracking Schema

```csv
ligand_name,ligand_smiles,variant_name,variant_signature,label,label_tier,label_source,label_confidence,date_added
WIN-55212-2,SMILES_STRING,WIN_neg_001,59R;81M;...,0,T1,FACS_screen_2024-11-15,HIGH,2026-02-16
WIN-55212-2,SMILES_STRING,WIN_neg_002,59K;81V;...,0,T2,near_neighbor_from_WIN_pos_001,MEDIUM,2026-02-16
WIN-55212-2,SMILES_STRING,WIN_neg_003,59D;81P;...,0,T3,random_pocket,LOW,2026-02-16
```

---

## EVALUATION SPLITS & ANALYSIS PLAN

### Split Strategy: Ligand + Variant Family Hold-Out

#### Primary Split (Ligand-Stratified)
```python
# Ensure each ligand class represented in train/val/test
train_ligands = stratified_sample(all_ligands, frac=0.6, stratify_by='ligand_class')
val_ligands   = stratified_sample(remaining, frac=0.5, stratify_by='ligand_class')  # 20% of total
test_ligands  = remaining  # 20% of total

# All (ligand, variant) pairs with ligand ∈ train_ligands → train set
```

**Rationale:** Prevents data leakage where model sees test ligand during training (even with different variant).

#### Secondary Split (Variant Family Hold-Out)
```python
# Hold out entire variant families (e.g., all "59Q + 120A + 160M" mutations)
# Identify variant clusters by signature similarity (Hamming distance ≤ 1)
variant_clusters = cluster_by_signature(all_variants, distance_threshold=1)

# Reserve 10% of clusters for "unseen mutation pattern" test set
test_clusters_unseen = random.sample(variant_clusters, k=int(0.1 * len(variant_clusters)))
```

**Rationale:** Tests generalization to novel mutation combinations (harder test).

#### Stratification Constraints
- Equal positive:negative ratio in all splits
- Equal Tier 1:Tier 2:Tier 3 ratio in all splits
- Minimum 5 examples per ligand class in test set

### Minimal Analysis Plan

#### 1. Score Distribution Separability
**Metric:** Effect size (Cohen's d) between positive and negative distributions per feature.

```python
features_to_check = [
    'rosetta_dG_sep',           # Rosetta interface binding energy
    'rosetta_buried_unsats',    # Buried unsatisfied H-bonds
    'af3_ipTM',                 # AF3 interface predicted TM-score
    'af3_pLDDT_ligand',         # AF3 ligand confidence
    'af3_interface_PAE',        # AF3 interface positional error
    'ligand_RMSD_to_template',  # AF3 ligand geometry accuracy
]

for feature in features_to_check:
    d = cohen_d(positives[feature], negatives[feature])
    print(f"{feature}: d={d:.2f}")  # Expect d > 0.8 for useful features
```

**Success criterion:** ≥3 features with d > 0.8 (large effect size).

#### 2. Single-Feature AUCs
**Goal:** Identify which features are most discriminative.

```python
for feature in features_to_check:
    auc = roc_auc_score(y_true, X[feature])
    print(f"{feature}: AUC={auc:.3f}")

# Rank features by AUC (descending)
```

**Hypothesis:** Top-3 features will be:
1. `rosetta_dG_sep` (direct binding energy)
2. `af3_ipTM` (interface quality)
3. `rosetta_buried_unsats` (H-bond satisfaction)

#### 3. Baseline ML Models
**Models:**
- Logistic Regression (L2, C=1.0)
- Random Forest (n_estimators=500, max_depth=10)
- XGBoost (max_depth=6, learning_rate=0.05, n_estimators=200)
- MLP (2 layers, 128 units, ReLU, dropout=0.2)

**Evaluation Metrics:**
- **AUC-ROC:** Overall discrimination
- **Precision@k (k=50):** Precision in top-50 ranked predictions (mirrors experimental validation budget)
- **F1-score:** Balanced accuracy
- **Calibration:** Brier score, reliability diagram

**Cross-Validation:** 5-fold stratified CV on training set.

#### 4. Cross-Tier Performance Analysis
**Goal:** Understand which negatives are "hard" vs "easy".

```python
# Train on positives + all negatives
# Evaluate separately on T1, T2, T3 test sets

auc_t1 = roc_auc_score(y_test_t1, predictions_t1)  # Expect 0.70–0.80
auc_t2 = roc_auc_score(y_test_t2, predictions_t2)  # Expect 0.75–0.85
auc_t3 = roc_auc_score(y_test_t3, predictions_t3)  # Expect 0.95–1.00
```

**Success criterion:**
- T3 (easy negatives): AUC ≥ 0.95 (sanity check)
- T1 (hard negatives): AUC ≥ 0.70 (challenging but learnable)
- T2 performance between T1 and T3

#### 5. Feature Importance (SHAP)
**Goal:** Understand which features drive predictions (interpretability).

```python
import shap

explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Deliverable:** Feature importance ranking + waterfall plots for example predictions.

---

## RISK REGISTER & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient Tier 1 negatives** (< 50) | Medium | High | Lower threshold to 30 for Phase 1; prioritize prospective data capture; consider Tier 2 as primary negatives |
| **AF3 batch job failures** (>10% fail rate) | Medium | Medium | Implement per-JSON retry logic (max 3 retries); isolate failures to separate logs; reduce batch size from 60 to 30 |
| **Rosetta docking low success rate** (<70% converge) | Low | Medium | Increase docking repeats from 50 to 100; relax clustering RMSD cutoff; use multiple starting orientations |
| **Conformer quality poor** (low TREMD recovery) | Low | Medium | Increase conformer count from 10 to 20; add OpenMM refinement step; validate against experimental structures |
| **Feature correlation too high** (r > 0.9) | Medium | Low | Perform PCA for dimensionality reduction; use L1 regularization to select features; report correlation matrix |
| **Class imbalance** (positives >> negatives) | Low | Medium | Oversample negatives to 1:1 ratio; use class weights in loss function; stratified sampling |
| **Test set leakage** (ligand seen in train) | Low | High | Strict ligand-based splitting; automated validation checks; document split logic in metadata |
| **HPC queue delays** (>7 days wait time) | High | Low | Submit jobs during off-peak hours; request priority allocation; use checkpoint/resume for long jobs |
| **Storage quota exceeded** (>10 TB) | Medium | Medium | Compress intermediate files (gzip PDBs); delete clustered poses after filtering; archive old campaigns |
| **Label noise in Tier 1** (false negatives) | Medium | High | Cross-validate with literature; require ≥2 experimental replicates; flag low-confidence labels |

---

## IMPLEMENTATION NOTES

### Code Modules to Create

#### 1. `generate_negatives.py`
**Location:** `ml_modelling/scripts/`

**Functions:**
- `generate_tier1_from_screens(screen_csvs)` → curate from historical data
- `generate_tier2_near_neighbors(positive_csv, n_per_ligand=10)` → 1–2 mutation variants
- `generate_tier3_random(ligand_list, n_per_ligand=10)` → 3–6 mutation variants
- `merge_negatives_with_provenance(t1, t2, t3)` → single CSV with metadata

**Output:** `negatives_curated.csv` (matches `ligand_smiles_signature.csv` schema + extra columns)

#### 2. `orchestrate_ml_pipeline.py`
**Location:** `ml_modelling/scripts/`

**Workflow:**
```python
1. Load positives from ligand_smiles_signature.csv
2. Load negatives from negatives_curated.csv
3. Merge into pairs_dataset.csv (with label column: 1=binder, 0=non-binder)
4. For each pair:
   a. Check cache: skip if {pair_id}/metadata.json exists and valid
   b. Generate conformers (or load from cache)
   c. Submit docking array job (dependency: conformers done)
   d. Submit relax job (dependency: docking clustered)
   e. Submit AF3 binary + ternary (dependency: relax done)
5. Aggregate all outputs into features_table.csv
6. Generate QC report (missing data, outliers, distributions)
```

**Caching Strategy:**
- Cache key: `{ligand_smiles_hash}_{variant_signature_hash}`
- Cache location: `$SCRATCH_ROOT/ml_dataset/cache/{cache_key}/`
- Cache validation: Check `metadata.json` for stage completion flags

#### 3. `aggregate_ml_features.py`
**Location:** `ml_modelling/scripts/`

**Inputs:**
- Rosetta score files: `{pair_id}/rosetta_scores.sc`
- AF3 output: `{pair_id}/af3_binary/summary.json`, `{pair_id}/af3_ternary/summary.json`
- Conformer reports: `{ligand_id}/conformer_report.csv`

**Output Schema (features_table.csv):**

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `pair_id` | str | — | `{ligand_name}_{variant_name}` |
| `ligand_name` | str | Input | Ligand identifier |
| `ligand_smiles` | str | Input | Canonical SMILES |
| `variant_name` | str | Input | PYR1 variant name |
| `variant_signature` | str | Input | Pocket mutations (e.g., "59K;120A;160G") |
| `label` | int | Input | 1=binder, 0=non-binder |
| `label_tier` | str | Input | T1/T2/T3 (or "positive") |
| `label_source` | str | Input | Provenance (e.g., "FACS_screen_2024-11-15") |
| `label_confidence` | str | Input | HIGH/MEDIUM/LOW |
| `conformer_count` | int | Conformer | Number of final conformers |
| `conformer_min_energy` | float | Conformer | Lowest MMFF energy (kcal/mol) |
| `conformer_rmsd_range` | float | Conformer | Max pairwise RMSD (diversity metric) |
| `docking_best_score` | float | Rosetta | Best interface score from docking |
| `docking_cluster_size` | int | Rosetta | Size of best cluster |
| `rosetta_dG_sep` | float | Rosetta | Interface binding energy (ddG) |
| `rosetta_total_score` | float | Rosetta | Total Rosetta energy |
| `rosetta_buried_unsats` | int | Rosetta | Buried unsatisfied H-bonds |
| `rosetta_sasa_interface` | float | Rosetta | Interface SASA (Å²) |
| `rosetta_hbonds_interface` | int | Rosetta | H-bonds across interface |
| `rosetta_ligand_neighbors` | int | Rosetta | Residues within 4Å of ligand |
| `af3_binary_ipTM` | float | AF3 | Binary interface pTM |
| `af3_binary_pLDDT_protein` | float | AF3 | Mean protein pLDDT |
| `af3_binary_pLDDT_ligand` | float | AF3 | Mean ligand pLDDT |
| `af3_binary_interface_PAE` | float | AF3 | Mean interface PAE (Å) |
| `af3_binary_ligand_RMSD` | float | AF3 | Ligand RMSD to template (Å) |
| `af3_ternary_ipTM` | float | AF3 | Ternary interface pTM |
| `af3_ternary_pLDDT_protein` | float | AF3 | Mean protein pLDDT |
| `af3_ternary_pLDDT_ligand` | float | AF3 | Mean ligand pLDDT |
| `af3_ternary_interface_PAE` | float | AF3 | Mean interface PAE (Å) |
| `af3_ternary_ligand_RMSD` | float | AF3 | Ligand RMSD to template (Å) |
| `af3_ternary_water_hbonds` | int | AF3 | Water-mediated H-bonds |
| ... | ... | ... | (40+ features total) |

#### 4. `split_dataset.py`
**Location:** `ml_modelling/scripts/`

**Functions:**
- `ligand_stratified_split(features_df, train=0.6, val=0.2, test=0.2)`
- `variant_family_holdout(features_df, holdout_frac=0.1)`
- `validate_split_balance(train, val, test)` → check class ratios, ligand coverage

**Output:**
- `train_set.csv`, `val_set.csv`, `test_set.csv`
- `splits_metadata.json` (documents split logic + random seed)

#### 5. `baseline_models.py`
**Location:** `ml_modelling/scripts/`

**Models:**
- `run_logistic_regression(X_train, y_train, X_test, y_test)`
- `run_random_forest(X_train, y_train, X_test, y_test)`
- `run_xgboost(X_train, y_train, X_test, y_test)`
- `run_mlp(X_train, y_train, X_test, y_test)`

**Evaluation:**
- ROC curves (save to `figures/roc_curves.png`)
- Precision-recall curves
- Feature importance plots (SHAP)
- Calibration plots

**Output:** `baseline_results.csv` (model, AUC, precision@50, F1, brier_score)

---

### Folder Structure (Production)

```
ml_modelling/
├── ligand_smiles_signature.csv          # (Existing) Positive pairs
├── negatives_curated.csv                # (New) Curated negatives with provenance
├── pairs_dataset.csv                    # (Generated) Merged positives + negatives
├── features_table.csv                   # (Generated) Full feature matrix
├── train_set.csv, val_set.csv, test_set.csv  # (Generated) Splits
├── splits_metadata.json                 # Split logic + random seed
├── baseline_results.csv                 # ML model benchmarks
├── scripts/
│   ├── generate_negatives.py            # Negative dataset curation
│   ├── orchestrate_ml_pipeline.py       # Master orchestrator
│   ├── aggregate_ml_features.py         # Feature aggregation
│   ├── split_dataset.py                 # Train/val/test splitting
│   └── baseline_models.py               # ML baselines
├── figures/                             # Plots (ROC, distributions, SHAP)
├── logs/                                # Pipeline logs
└── cache/                               # Per-pair cached outputs
    └── {ligand_hash}_{variant_hash}/
        ├── conformers/                  # Conformer SDF/PDB files
        ├── docking/                     # Docked poses
        ├── relax/                       # Relaxed structures
        ├── af3_binary/                  # AF3 binary predictions
        ├── af3_ternary/                 # AF3 ternary predictions
        └── metadata.json                # Provenance + stage completion
```

---

### Logging & Resumability

#### Metadata Schema (`metadata.json` per pair)
```json
{
  "pair_id": "WIN-55212-2_PYR1^4F",
  "ligand_name": "WIN-55212-2",
  "ligand_smiles": "...",
  "variant_name": "PYR1^4F",
  "variant_signature": "83K;115Q;120G;159V;160G",
  "label": 1,
  "label_tier": "positive",
  "label_source": "validated_binder",
  "label_confidence": "HIGH",
  "stages": {
    "conformer_generation": {
      "status": "complete",
      "timestamp": "2026-02-16T10:30:00Z",
      "output_dir": "conformers/",
      "num_conformers": 10
    },
    "docking": {
      "status": "complete",
      "timestamp": "2026-02-16T14:45:00Z",
      "output_dir": "docking/",
      "num_poses": 47
    },
    "relax": {
      "status": "complete",
      "timestamp": "2026-02-16T15:30:00Z",
      "output_file": "relax/relaxed_pose.pdb"
    },
    "af3_binary": {
      "status": "complete",
      "timestamp": "2026-02-16T18:00:00Z",
      "output_dir": "af3_binary/",
      "ipTM": 0.82,
      "pLDDT_ligand": 78.5
    },
    "af3_ternary": {
      "status": "complete",
      "timestamp": "2026-02-16T19:30:00Z",
      "output_dir": "af3_ternary/",
      "ipTM": 0.85,
      "pLDDT_ligand": 81.2
    }
  },
  "errors": []
}
```

#### Resume Logic
```python
def is_pair_complete(pair_id, cache_dir):
    metadata_path = f"{cache_dir}/{pair_id}/metadata.json"
    if not os.path.exists(metadata_path):
        return False

    metadata = json.load(open(metadata_path))
    required_stages = ['conformer_generation', 'docking', 'relax', 'af3_binary', 'af3_ternary']

    for stage in required_stages:
        if metadata['stages'][stage]['status'] != 'complete':
            return False

    return True

# In orchestrator:
for pair in pairs_dataset:
    if is_pair_complete(pair['pair_id'], cache_dir):
        print(f"Skipping {pair['pair_id']} (already complete)")
        continue
    else:
        submit_pipeline_jobs(pair)
```

---

## DELIVERABLES CHECKLIST

### Phase 0 (Foundation)
- [ ] `negatives_curated.csv` with ≥50 Tier 1 + Tier 2/3 generation scripts
- [ ] `generate_negatives.py` validated on 10 ligands
- [ ] `orchestrate_ml_pipeline.py` with caching + resumability
- [ ] Pilot validation (1 ligand × 30 pairs) complete in <8h
- [ ] Score separation analysis (effect size d > 0.8)

### Phase 1 (Pilot)
- [ ] `features_pilot.csv` (300 pairs × 45 features, ≥90% complete)
- [ ] QC report: distributions, correlations, missing data
- [ ] Tier 1 vs Tier 3 AUC ≥ 0.85 (single best feature)
- [ ] Wall time <3 days (1K cores + 8 A100s)

### Phase 2 (Production)
- [ ] `features_table.csv` (2,000 pairs × 45 features, ≥90% complete)
- [ ] `train_set.csv`, `val_set.csv`, `test_set.csv` with metadata
- [ ] `baseline_results.csv` (4 models, test AUC ≥ 0.75)
- [ ] Feature importance analysis (SHAP values + plots)
- [ ] Final report (PDF, 10–15 pages) with figures
- [ ] Dataset published with DOI (Zenodo/Dryad)

---

## TIMELINE SUMMARY

| Phase | Duration | Key Milestone | Team Effort |
|-------|----------|---------------|-------------|
| Phase 0 | 2 weeks | Pilot validation complete | 40 hours (curation + coding) |
| Phase 1 | 3 weeks | Pilot dataset + QC report | 60 hours (pipeline tuning + analysis) |
| Phase 2 | 6 weeks | Production dataset + baselines | 100 hours (full run + modeling) |
| **Total** | **11 weeks** | **Public dataset release** | **~200 hours** |

**Critical Path:** Tier 1 negative curation (blocks Phase 1 start) → AF3 GPU allocation (limits Phase 2 throughput).

---

## SUCCESS CRITERIA (GO/NO-GO)

### Phase 0 → Phase 1 Gate
- [ ] ≥30 Tier 1 negatives curated (minimum viable)
- [ ] Pilot (30 pairs) completes with <10% failures
- [ ] Score distributions show separation (d > 0.5)

### Phase 1 → Phase 2 Gate
- [ ] Pilot AUC ≥ 0.75 (any single feature)
- [ ] Pipeline wall time <3 days (scalability validated)
- [ ] <5% missing data in critical features

### Phase 2 → Publication Gate
- [ ] Test set AUC ≥ 0.75 (best model)
- [ ] Feature table has ≥1,800 complete pairs
- [ ] Cross-tier performance meets expectations (T3 AUC ~1.0, T1 AUC ≥ 0.70)

---

## APPENDIX: KEY METRICS GLOSSARY

| Metric | Source | Description | Expected Range (Binders) |
|--------|--------|-------------|--------------------------|
| `rosetta_dG_sep` | Rosetta | Interface binding energy (kcal/mol) | < -10 (more negative = better) |
| `rosetta_buried_unsats` | Rosetta | Buried unsatisfied H-bond donors/acceptors | < 3 (fewer = better) |
| `rosetta_sasa_interface` | Rosetta | Solvent-accessible surface area at interface (Å²) | > 300 (larger = better) |
| `af3_ipTM` | AF3 | Interface predicted TM-score (0–1) | > 0.7 (higher = better) |
| `af3_pLDDT_ligand` | AF3 | Ligand confidence score (0–100) | > 70 (higher = better) |
| `af3_interface_PAE` | AF3 | Interface positional error (Å) | < 5 (lower = better) |
| `af3_ligand_RMSD` | AF3 | Ligand RMSD to template (Å) | < 2 (lower = better) |

---

**End of Project Plan**
