# PYR1 Numbering Conversion Guide

**Critical Information:** The 3QN1 PYR1 structure used for docking has **residues 67-68 DELETED**.

This creates a numbering mismatch between:
- **WT sequence numbering** (variant signatures, biological numbering)
- **PDB numbering** (3QN1 deletion structure)

---

## 🔢 NUMBERING CONVERSION RULES

### **Rule 1: Positions 1–66**
✅ **No conversion needed** (before deletion)

```
WT position 59 → PDB position 59 (no change)
WT position 66 → PDB position 66 (no change)
```

### **Rule 2: Positions 67–68**
❌ **DO NOT EXIST in deletion PDB**

```
WT position 67 → ERROR (deleted!)
WT position 68 → ERROR (deleted!)
```

**Important:** If a variant signature includes positions 67 or 68, the threading script will raise an error.

### **Rule 3: Positions 69+**
✅ **Subtract 2** (shift left due to deletion)

```
WT position 69 → PDB position 67 (69 - 2 = 67)
WT position 81 → PDB position 79 (81 - 2 = 79)
WT position 120 → PDB position 118 (120 - 2 = 118)
WT position 167 → PDB position 165 (167 - 2 = 165)
```

---

## 📊 CONVERSION TABLE (Common Pocket Positions)

| WT Position | PDB Position (Δ67-68) | Shift | Pocket Residue |
|-------------|----------------------|-------|----------------|
| 59 | 59 | 0 | K59 |
| 66 | 66 | 0 | — |
| **67** | **DELETED** | — | ❌ |
| **68** | **DELETED** | — | ❌ |
| 69 | 67 | -2 | — |
| 81 | 79 | -2 | V81 |
| 83 | 81 | -2 | V83 |
| 92 | 90 | -2 | S92 |
| 94 | 92 | -2 | E94 |
| 108 | 106 | -2 | F108 |
| 110 | 108 | -2 | V110 |
| 115 | 113 | -2 | R115 |
| 117 | 115 | -2 | V117 |
| 120 | 118 | -2 | Y120 |
| 122 | 120 | -2 | S122 |
| 141 | 139 | -2 | K141 |
| 159 | 157 | -2 | F159 |
| 160 | 158 | -2 | A160 |
| 163 | 161 | -2 | V163 |
| 164 | 162 | -2 | V164 |
| 167 | 165 | -2 | N167 |

---

## 🧪 EXAMPLE CONVERSIONS

### **Example 1: Simple Variant**

**Variant signature:** `"59K;120A;160G"`

**Conversion:**
```
Position 59 (K):  59 ≤ 66 → PDB position 59 (no change)
Position 120 (A): 120 ≥ 69 → PDB position 118 (subtract 2)
Position 160 (G): 160 ≥ 69 → PDB position 158 (subtract 2)
```

**Threading output:**
```
Position 59 (WT) → 59 (PDB): K
Position 120 (WT) → 118 (PDB): Y → A
Position 160 (WT) → 158 (PDB): A → G
```

---

### **Example 2: WIN-55212-2 Variant**

**Variant signature:** `"83K;115Q;120G;159V;160G"`

**Conversion:**
```
Position 83 (K):  83 ≥ 69 → PDB position 81 (subtract 2)
Position 115 (Q): 115 ≥ 69 → PDB position 113 (subtract 2)
Position 120 (G): 120 ≥ 69 → PDB position 118 (subtract 2)
Position 159 (V): 159 ≥ 69 → PDB position 157 (subtract 2)
Position 160 (G): 160 ≥ 69 → PDB position 158 (subtract 2)
```

**Threading output:**
```
Position 83 (WT) → 81 (PDB): V → K
Position 115 (WT) → 113 (PDB): R → Q
Position 120 (WT) → 118 (PDB): Y → G
Position 159 (WT) → 157 (PDB): F → V
Position 160 (WT) → 158 (PDB): A → G
```

---

### **Example 3: Nitazene Variant**

**Variant signature:** `"59Q;81I;83L;92M;108V;120A;122G;141D;159H;160V"`

**Conversion:**
```
Position 59 (Q):  59 ≤ 66 → PDB position 59 (no change)
Position 81 (I):  81 ≥ 69 → PDB position 79 (subtract 2)
Position 83 (L):  83 ≥ 69 → PDB position 81 (subtract 2)
Position 92 (M):  92 ≥ 69 → PDB position 90 (subtract 2)
Position 108 (V): 108 ≥ 69 → PDB position 106 (subtract 2)
Position 120 (A): 120 ≥ 69 → PDB position 118 (subtract 2)
Position 122 (G): 122 ≥ 69 → PDB position 120 (subtract 2)
Position 141 (D): 141 ≥ 69 → PDB position 139 (subtract 2)
Position 159 (H): 159 ≥ 69 → PDB position 157 (subtract 2)
Position 160 (V): 160 ≥ 69 → PDB position 158 (subtract 2)
```

---

### **Example 4: ERROR Case (Position 67 or 68)**

**Variant signature:** `"59K;67A;120G"` ❌

**Conversion:**
```
Position 59 (K):  59 ≤ 66 → PDB position 59 (no change)
Position 67 (A):  67 is DELETED → ERROR!
Position 120 (G): (not reached due to error)
```

**Error message:**
```
ERROR: Position 67 is in the deleted region (residues 67-68 removed from PDB).
Cannot thread mutations at deleted positions!
```

---

## 🛠️ USING THE THREADING SCRIPT

### **Default Behavior (3QN1 with Δ67-68)**

The script **automatically handles the conversion** by default:

```bash
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "81I;120A;167D" \
    --output mutant.pdb

# Output:
# Position 81 (WT) → 79 (PDB with Δ67-68): V → I
# Position 120 (WT) → 118 (PDB with Δ67-68): Y → A
# Position 167 (WT) → 165 (PDB with Δ67-68): N → D
```

### **Disable Conversion (Full-Length WT PDB)**

If using a **full-length WT PDB** (without deletion), use `--no-deletion`:

```bash
python thread_variant_to_pdb.py \
    --template WT_PYR1_full_length.pdb \
    --signature "81I;120A;167D" \
    --output mutant.pdb \
    --no-deletion

# Output (no conversion):
# Position 81 (Rosetta 81): V → I
# Position 120 (Rosetta 120): Y → A
# Position 167 (Rosetta 167): N → D
```

### **Custom Deletion**

For a different deletion region (e.g., Δ50-52):

```bash
python thread_variant_to_pdb.py \
    --template custom_deletion.pdb \
    --signature "59K;81I;120A" \
    --output mutant.pdb \
    --deletion-start 50 \
    --deletion-length 3

# Conversion:
# Position 59 (WT) → 56 (PDB): subtract 3 (after Δ50-52)
# Position 81 (WT) → 78 (PDB): subtract 3
# Position 120 (WT) → 117 (PDB): subtract 3
```

---

## ✅ VALIDATION TESTS

### **Test 1: Simple Conversion**

```bash
python thread_variant_to_pdb.py \
    --signature "81I;120A;167D" \
    --test

# Expected output:
# Parsed 3 mutations: {81: 'I', 120: 'A', 167: 'D'}
# Conversion (with Δ67-68):
#   WT 81 → PDB 79
#   WT 120 → PDB 118
#   WT 167 → PDB 165
```

### **Test 2: Error Detection**

```bash
python thread_variant_to_pdb.py \
    --signature "59K;67A;120G" \
    --test

# Expected output:
# Parsed 3 mutations: {59: 'K', 67: 'A', 120: 'G'}
# ERROR: Position 67 is in the deleted region (residues 67-68 removed from PDB)
```

### **Test 3: Visual Verification in PyMOL**

After threading:

```bash
# Thread mutations
python thread_variant_to_pdb.py \
    --template 3QN1_nolig_H2O.pdb \
    --signature "81I;120A;160G" \
    --output mutant_81I_120A_160G.pdb

# Load in PyMOL
pymol 3QN1_nolig_H2O.pdb mutant_81I_120A_160G.pdb

# In PyMOL:
select pos79, resi 79 and chain A  # Should be I (was V at WT position 81)
select pos118, resi 118 and chain A  # Should be A (was Y at WT position 120)
select pos158, resi 158 and chain A  # Should be G (was A at WT position 160)

show sticks, pos79 or pos118 or pos158
label pos79, "WT 81 → I"
label pos118, "WT 120 → A"
label pos158, "WT 160 → G"
```

---

## 📋 QUICK REFERENCE FORMULA

```python
def convert_wt_to_pdb(wt_pos):
    """Convert WT position to 3QN1 deletion PDB position."""
    if wt_pos <= 66:
        return wt_pos  # No change
    elif 67 <= wt_pos <= 68:
        raise ValueError(f"Position {wt_pos} is deleted!")
    else:
        return wt_pos - 2  # Subtract 2
```

---

## ⚠️ IMPORTANT NOTES

1. **Always use WT numbering in variant signatures**
   - Signatures should use biological/WT numbering (e.g., "81I", "167D")
   - The script handles PDB conversion automatically

2. **Never mutate positions 67-68**
   - These positions don't exist in the 3QN1 structure
   - Script will raise an error if attempted

3. **Verify mutations visually**
   - After threading, always check a few positions in PyMOL
   - Compare WT and mutant structures side-by-side

4. **Log files document conversion**
   - Use `--log mutation_log.txt` to save conversion details
   - Useful for debugging and record-keeping

---

## 🔗 RELATED DOCUMENTATION

- **Threading script:** `pyr1_pipeline/scripts/thread_variant_to_pdb.py`
- **Implementation guide:** `ml_modelling/IMPLEMENTATION_COMPLETE.md`
- **Full project plan:** `ml_modelling/REVISED_PROJECT_PLAN.md`

---

**Last updated:** 2026-02-16
