# PYR1 Numbering Fix: CRITICAL UPDATE ✅

**Date:** 2026-02-16
**Issue:** 3QN1 structure has residues 67-68 deleted → numbering mismatch
**Status:** FIXED

---

## 🔴 THE PROBLEM

Your 3QN1 PYR1 docking template has **residues 67-68 REMOVED**.

This creates a numbering offset:
```
Variant signature uses WT numbering:  "81I;120A;167D"
But in the deletion PDB:
  - WT position 81 is actually at PDB position 79 (81 - 2)
  - WT position 120 is actually at PDB position 118 (120 - 2)
  - WT position 167 is actually at PDB position 165 (167 - 2)
```

**Without conversion:** Mutations would be applied to WRONG residues! ❌

---

## ✅ THE FIX

Updated `thread_variant_to_pdb.py` to **automatically convert** WT numbering to PDB numbering:

### **Conversion Rules**

| WT Position Range | PDB Position | Conversion |
|-------------------|--------------|------------|
| 1–66 | Same (1–66) | No change |
| **67–68** | **DELETED** | **Error!** |
| 69+ | WT - 2 | Subtract 2 |

### **New Function: `convert_wt_to_deletion_numbering()`**

```python
def convert_wt_to_deletion_numbering(wt_position: int) -> int:
    """
    Convert WT PYR1 numbering to 3QN1 deletion PDB numbering.

    Examples:
        81 → 79 (subtract 2)
        120 → 118 (subtract 2)
        167 → 165 (subtract 2)
        59 → 59 (no change, before deletion)
        67 → ERROR (deleted!)
    """
    if wt_position < 67:
        return wt_position  # Before deletion
    elif 67 <= wt_position <= 68:
        raise ValueError("Position in deleted region!")
    else:
        return wt_position - 2  # After deletion
```

---

## 🧪 TESTING THE FIX

### **Test 1: Automatic Conversion (Default)**

```bash
python pyr1_pipeline/scripts/thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "81I;120A;167D" \
    --output test_mutant.pdb

# Expected output:
# Position 81 (WT) → 79 (PDB with Δ67-68): V → I
# Position 120 (WT) → 118 (PDB with Δ67-68): Y → A
# Position 167 (WT) → 165 (PDB with Δ67-68): N → D
# ✓ Successfully created mutant: test_mutant.pdb
```

### **Test 2: Error Detection (Position 67 or 68)**

```bash
python pyr1_pipeline/scripts/thread_variant_to_pdb.py \
    --signature "59K;67A;120G" \
    --test

# Expected output:
# ERROR: Position 67 is in the deleted region (residues 67-68 removed from PDB).
# Cannot thread mutations at deleted positions!
```

### **Test 3: Visual Verification in PyMOL**

After threading, verify in PyMOL:

```bash
# Thread a simple variant
python pyr1_pipeline/scripts/thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "81I;120A;160G" \
    --output mutant_test.pdb

# Load both structures
pymol 3QN1_nolig_H2O.pdb mutant_test.pdb

# In PyMOL console:
# Select position 79 (WT 81) - should be Ile
select pos79, resi 79 and chain A and mutant_test
show sticks, pos79
# Verify: Should be ILE (was VAL in WT)

# Select position 118 (WT 120) - should be Ala
select pos118, resi 118 and chain A and mutant_test
show sticks, pos118
# Verify: Should be ALA (was TYR in WT)

# Select position 158 (WT 160) - should be Gly
select pos158, resi 158 and chain A and mutant_test
show sticks, pos158
# Verify: Should be GLY (was ALA in WT)
```

---

## 📊 EXAMPLE CONVERSIONS

### **Common Pocket Positions**

| WT Pos | PDB Pos | Shift | Example |
|--------|---------|-------|---------|
| 59 | 59 | 0 | 59K → position 59 |
| 81 | 79 | -2 | 81I → position 79 |
| 83 | 81 | -2 | 83L → position 81 |
| 92 | 90 | -2 | 92M → position 90 |
| 108 | 106 | -2 | 108V → position 106 |
| 115 | 113 | -2 | 115Q → position 113 |
| 120 | 118 | -2 | 120A → position 118 |
| 122 | 120 | -2 | 122G → position 120 |
| 141 | 139 | -2 | 141D → position 139 |
| 159 | 157 | -2 | 159H → position 157 |
| 160 | 158 | -2 | 160G → position 158 |
| 163 | 161 | -2 | 163W → position 161 |
| 164 | 162 | -2 | 164F → position 162 |
| 167 | 165 | -2 | 167D → position 165 |

---

## 🔧 USAGE

### **Default (Automatic Conversion)**

For 3QN1 structure (default):

```bash
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "YOUR_SIGNATURE" \
    --output mutant.pdb

# Conversion happens automatically!
```

### **Disable Conversion (Full-Length WT)**

If using a **full-length WT PDB** without deletions:

```bash
python thread_variant_to_pdb.py \
    --template WT_full_length.pdb \
    --signature "YOUR_SIGNATURE" \
    --output mutant.pdb \
    --no-deletion  # ← Disables conversion

# Uses WT numbering directly (no shift)
```

### **Custom Deletion Region**

For different deletion (e.g., Δ50-52):

```bash
python thread_variant_to_pdb.py \
    --template custom.pdb \
    --signature "YOUR_SIGNATURE" \
    --output mutant.pdb \
    --deletion-start 50 \
    --deletion-length 3
```

---

## 📋 INTEGRATION WITH PIPELINE

The orchestrator automatically passes deletion parameters:

```python
# In orchestrate_ml_dataset_pipeline.py
run_mutation_threading(
    template_pdb='3QN1_nolig_H2O.pdb',
    variant_signature='81I;120A;167D',
    output_pdb='mutant.pdb',
    chain='A'
    # Deletion conversion happens automatically!
)
```

**No changes needed to your existing workflow!**

---

## ⚠️ IMPORTANT NOTES

1. **Always use WT numbering in variant signatures**
   - Your CSV should use biological numbering (81, 120, 167, etc.)
   - Script handles PDB conversion automatically

2. **Never include positions 67-68 in signatures**
   - These don't exist in 3QN1
   - Script will raise an error

3. **Validate with PyMOL**
   - After threading, always visually check 2-3 positions
   - Compare side-by-side with WT structure

4. **Log files document conversion**
   ```bash
   python thread_variant_to_pdb.py \
       --template 3QN1_nolig_H2O.pdb \
       --signature "81I;120A;167D" \
       --output mutant.pdb \
       --log conversion_log.txt

   # conversion_log.txt will show:
   # Position 81 (WT) → 79 (PDB with Δ67-68)
   # Position 120 (WT) → 118 (PDB with Δ67-68)
   # Position 167 (WT) → 165 (PDB with Δ67-68)
   ```

---

## ✅ VALIDATION CHECKLIST

Before proceeding to Phase 1:

- [ ] Test threading on simple variant (e.g., "81I;120A;160G")
- [ ] Verify positions in PyMOL:
  - [ ] Position 79 (WT 81) is correct amino acid
  - [ ] Position 118 (WT 120) is correct amino acid
  - [ ] Position 158 (WT 160) is correct amino acid
- [ ] Test error detection (try "67A" in signature → should error)
- [ ] Run full pipeline on 1 pair (conformers → threading → docking)
- [ ] Check docking output uses mutant structure (not WT)

---

## 📚 DOCUMENTATION

| File | Purpose |
|------|---------|
| [PYR1_NUMBERING_GUIDE.md](c:\Users\rmdel\OneDrive - UCB-O365\Whitehead Lab\pyr1_pipeline\scripts\PYR1_NUMBERING_GUIDE.md) | Complete numbering reference |
| [thread_variant_to_pdb.py](c:\Users\rmdel\OneDrive - UCB-O365\Whitehead Lab\pyr1_pipeline\scripts\thread_variant_to_pdb.py) | Updated threading script |
| [IMPLEMENTATION_COMPLETE.md](c:\Users\rmdel\OneDrive - UCB-O365\Whitehead Lab\ml_modelling\IMPLEMENTATION_COMPLETE.md) | Full pipeline guide |

---

## 🎯 NEXT STEPS

1. **Test the fix:**
   ```bash
   # Provide 3 example variant signatures
   # I'll validate the threading works correctly
   ```

2. **Visual validation:**
   ```bash
   # Thread 1 variant and inspect in PyMOL
   # Confirm positions are mutated correctly
   ```

3. **Run pilot:**
   ```bash
   # Once validated, proceed with 30-pair pilot
   ```

---

**Status:** ✅ FIXED and ready for testing

**Critical:** Test threading on at least 3 variants before running Phase 1 pilot!
