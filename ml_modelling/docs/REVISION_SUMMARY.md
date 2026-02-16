# Project Plan Revision Summary
**Date:** 2026-02-16
**Based on:** Lab feedback on affinity, threading, and AF3 timing

---

## 🔴 CRITICAL CHANGES

### 1. **Positive Data Quality Stratification** (NEW)

**Problem:** Not all positives are created equal. Y2H weak binders (10–100 μM LOD) likely have poor structural signal.

**Solution:** Stratify positives by affinity:

| Tier | EC50 Range | Confidence | Use Case |
|------|------------|------------|----------|
| **P1** | < 1 μM | HIGH | Primary training |
| **P2** | 1–10 μM | MEDIUM | Training + validation |
| **P3** | > 10 μM | LOW | Hold-out test only (expect lower accuracy) |

**Impact:**
- Phase 1 pilot: Only P1+P2 positives (EC50 < 10 μM)
- Separate evaluation on P3 to test model limits
- Report performance by tier (P1 vs P2 vs P3 AUC)

**Action Required:**
- Link `ligand_smiles_signature.csv` to EC50 data
- Create `affinity_annotation.csv` with P1/P2/P3 classification

---

### 2. **Mutation Threading Script** (NEW DELIVERABLE)

**Problem:** You have glycine-shaved docking, but no script to thread "59K;120A;160G" signatures onto PYR1 template.

**Solution:** Created `thread_variant_to_pdb.py`

**Location:** `pyr1_pipeline/scripts/thread_variant_to_pdb.py`

**Features:**
- Parses multiple signature formats: "59K;120A;160G", "K59Q_Y120A", "59K 120A 160G"
- Uses PyRosetta's `mutate_residue()` (existing capability, new parser)
- Single variant OR batch CSV processing
- Validation mode + mutation logs

**Example usage:**
```bash
# Single variant
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "59K;120A;160G" \
    --output mutant_59K_120A_160G.pdb

# Batch mode
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --csv variants.csv \
    --output-dir mutants/
```

**Integration:** Called before docking in `orchestrate_ml_pipeline.py`

**Testing needed:**
- Provide 3 example variants (WIN, nitazene, LCA) for validation
- Visual inspection of mutant structures

---

### 3. **AF3 Timing Correction** (MASSIVE SPEEDUP)

**Original estimate (WRONG):**
- Binary: 30 min/prediction
- Ternary: 45 min/prediction
- Phase 1 (300 pairs): 375 GPU-hours
- Phase 2 (2,000 pairs): 2,500 GPU-hours

**Corrected estimate (with templates):**
- Binary: **20 seconds**/prediction
- Ternary: **30 seconds**/prediction
- Phase 1 (200 pairs): **2.8 GPU-hours** (134× faster!)
- Phase 2 (1,500 pairs): **20.8 GPU-hours** (120× faster!)

**Why?** Structural templates for PYR1/HAB1 bypass expensive MSA generation.

**Impact:**
- AF3 no longer the bottleneck (was 94% of compute cost, now 0.3%)
- Wall time reduced: Phase 1 = 1 day (was 2.5 days), Phase 2 = 3 days (was 14 days)
- Cost reduced: Phase 2 = $164 (was $4,000!)
- Project is **highly feasible** with existing HPC allocation

---

### 4. **Pilot Ligand Selection** (Validated Families)

**Original:** 10 diverse ligands across all classes

**Revised:** 3 validated families with published/in-progress data

| Family | Status | Ligands | P1+P2 Binders | Data Quality |
|--------|--------|---------|---------------|--------------|
| WIN-55212-2 | Nature paper | 1 | ~30 | Surface display (< 1 μM) |
| Nitazenes | Nature paper | 3–5 | ~45 | Surface display (< 5 μM) |
| Bile acids (LCA) | In progress | 3 | ~25 | Current project (< 10 μM) |
| **Total** | — | **~9** | **~100** | High confidence |

**Rationale:**
- Focus on **high-quality, published data** for reproducibility
- Defer weak/ambiguous ligands to Phase 2
- Include in-progress LCA data to demonstrate extensibility

**Future:** Add 4 bile acids in Phase 2 when data available

---

## 📊 UPDATED DATASET DESIGN

### Phase 1 Pilot (Revised from 300 → 200 pairs)

| Component | Count | Notes |
|-----------|-------|-------|
| P1 positives | 65 | EC50 < 1 μM (primary training) |
| P2 positives | 35 | EC50 1–10 μM (validation) |
| N1 negatives | 30 | FACS Expression+/Activation– |
| N2 negatives | 50 | Near-neighbor (1–2 muts from P1) |
| N3 negatives | 20 | Random pocket (calibration) |
| **Total** | **200** | Balanced, high-quality |

### Phase 2 Production (Revised from 2,000 → 1,500 pairs)

| Component | Count | Notes |
|-----------|-------|-------|
| P1 positives | 200 | Expand with bile acids |
| P2 positives | 150 | Include moderate binders |
| N1 negatives | 100 | Curated from screens |
| N2 negatives | 250 | Near-neighbor variants |
| N3 negatives | 350 | Random pocket |
| Redundancy | +200 | Ensure 90% completion |
| **Total** | **~1,250** | (→ 1,100 complete) |

**Rationale:** Smaller, higher-quality dataset more valuable than large, noisy dataset.

---

## 💰 UPDATED COMPUTE BUDGET

### Phase 1 Pilot (200 pairs)

| Resource | Original | Corrected | Speedup |
|----------|----------|-----------|---------|
| CPU-hours | 1,325 | 883 | 1.5× |
| GPU-hours | 375 | **2.8** | **134×** |
| Wall time | 2.5 days | 1 day | 2.5× |
| Cost | $564 | **$20** | 28× |

### Phase 2 Production (1,500 pairs)

| Resource | Original | Corrected | Speedup |
|----------|----------|-----------|---------|
| CPU-hours | 8,834 | 6,625 | 1.3× |
| GPU-hours | 2,500 | **20.8** | **120×** |
| Wall time | 14 days | 3 days | 4.7× |
| Cost | $3,927 | **$164** | 24× |

**Key insight:** Project is now **trivially feasible** with existing HPC resources.

---

## 📅 UPDATED TIMELINE

| Phase | Duration | Compute Time | Team Effort |
|-------|----------|--------------|-------------|
| Phase 0 | 2 weeks | 4 hours | 40 hours |
| Phase 1 | 2 weeks | 1 day | 50 hours |
| Phase 2 | 4 weeks | 3 days | 80 hours |
| **Total** | **8 weeks** | **~4 days** | **~170 hours** |

**Reduction:** 11 weeks → 8 weeks (AF3 speedup enables faster iteration)

---

## ✅ ACTION ITEMS FOR YOU

### Immediate (This Week)

1. **Affinity Data Collection**
   - [ ] Provide EC50 values for WIN/nitazene/LCA variants
   - [ ] Format: CSV with `variant_name`, `ligand_name`, `EC50_uM`, `assay_type`
   - [ ] Classify each variant into P1/P2/P3 tier

2. **FACS Negative Data**
   - [ ] Share Expression+/Activation– variants from WIN or nitazene screens
   - [ ] Need: Variant sequences that passed expression gate but failed activation
   - [ ] Any format (I can parse FACS CSVs)

3. **Bile Acid Status**
   - [ ] Which LCA variants are validated binders (EC50 < 10 μM)?
   - [ ] Timeline for 4 additional bile acids? (for Phase 2 expansion)

### Next Week (Threading Script Validation)

4. **Test Threading Script**
   - [ ] Provide 3 example variant signatures:
     - One WIN variant (e.g., "83F;115Q;120G")
     - One nitazene variant
     - One LCA variant
   - [ ] Ideally with existing structure or model for validation

5. **AF3 Templates**
   - [ ] Confirm PDB/CIF templates for:
     - PYR1 structure (3QN1?)
     - HAB1 structure
     - Ligand-bound reference (for RMSD)

### Strategic Discussion

6. **Affinity Cutoff Consensus**
   - [ ] Is EC50 < 10 μM a good P1+P2 threshold?
   - [ ] Or should we be more conservative (< 5 μM)?

7. **Publication Goals**
   - [ ] Target: Methods paper + dataset release?
   - [ ] Or application paper (ML model for design)?

---

## 📁 NEW FILES CREATED

1. **`REVISED_PROJECT_PLAN.md`** - Complete revised plan (addresses all 4 feedback points)
2. **`thread_variant_to_pdb.py`** - Mutation threading script (ready to test)
3. **`REVISION_SUMMARY.md`** - This document (quick reference)

**Original plan preserved:** `PYR1_ML_DATASET_PROJECT_PLAN.md` (for reference)

---

## 🎯 KEY DECISIONS NEEDED

### Go/No-Go Criteria

**Phase 0 → Phase 1:**
- ≥50 P1 positives annotated (EC50 < 1 μM)
- Threading script validated on 3 test cases
- Pilot (30 pairs) completes successfully

**Phase 1 → Phase 2:**
- Pilot AUC (P1 vs N2) ≥ 0.65
- Affinity correlation (EC50 vs dG_sep): |r| ≥ 0.3
- Pipeline runs in <1 day (validates compute estimates)

**Phase 2 → Publication:**
- Test AUC ≥ 0.75 (P1+P2 vs N1+N2)
- Top-3 features identified (SHAP + permutation)
- ≥1,350 complete pairs in feature table

---

## 🔬 SCIENTIFIC HYPOTHESES TO TEST

1. **Affinity-structure correlation:**
   - Hypothesis: EC50 correlates with rosetta_dG_sep (r ≥ 0.4)
   - If false: Suggests dynamics/entropy/kinetics matter more than static structure

2. **Tier-specific features:**
   - Hypothesis: P1 and P2 have different discriminative features
   - P1 = strong interface (dG_sep, ipTM)
   - P2 = marginal interface (unsats, water networks)

3. **Near-neighbor hardness:**
   - Hypothesis: N2 (1–2 muts from binders) are genuinely hard to discriminate
   - Target: AUC(P1 vs N2) = 0.65–0.75
   - If AUC > 0.85: N2 too easy (increase mutation distance)
   - If AUC < 0.60: N2 too hard (structural signal insufficient)

4. **Weak binder modelability:**
   - Hypothesis: P3 (EC50 > 10 μM) have poor structural signal
   - Test: Train on P1+P2, evaluate on P3 separately
   - Expect: Lower AUC on P3 vs P1 test set

---

**END OF SUMMARY**

**Next steps:** Please review and provide the requested data (affinity annotations, test variants, bile acid status) so we can proceed to Phase 0 implementation.
