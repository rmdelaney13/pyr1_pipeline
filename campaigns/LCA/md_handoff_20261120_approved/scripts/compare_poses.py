#!/usr/bin/env python3
"""Step 3: measure pose change in the shared arms, old delivery vs approved start.

For each arm present in both deliveries:

  1. Read the previous PDB from the COMMITTED git blob at the pinned commit and
     the approved current PDB from the staged package. Verify the previous blob's
     SHA-256 against the crosswalk. Record both hashes even when the two seed
     indices agree. Byte-identical files record a verified zero RMSD; differing
     hashes are examined coordinate-wise regardless of seed index, because
     formatting alone can change a hash.
  2. Superpose CURRENT PYR1 onto PREVIOUS PYR1 by a least-squares rigid fit of
     the 16 pocket CA atoms at full-sequence positions POCKET. Residue identities
     are verified before fitting. The same transform is applied to the ligand;
     the ligands are never superposed on each other.
  3. Primary result: all-heavy-atom ligand RMSD after the pocket fit, over a
     complete one-to-one map, minimised over the reference chemistry's
     symmetry-equivalent mappings (carboxylate and sulfate oxygens only -- every
     stereocentre is fixed, so stereochemistry is retained). Both the plain
     name-mapped and the symmetry-corrected value are reported.
  4. Steroid-core RMSD over the 17 fused-ring carbons C1-C17 after the same fit,
     to separate a shifted core from a moving tail or sulfate.

An incomplete atom map is reported as missing, never as zero. The previous
manifest's across-seed `pose_rmsd_spread` / `pose_rmsd_max` are a different
quantity and are not used here.

    python3 compare_poses.py --crosswalk C --pkg D --prior-repo R
"""
import argparse
import hashlib
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from lca_chem import refs, read_pdb, PRETTY

COMMIT = "e7d6a6967b43a5e93bb0bb0386e9551362aaf40a"
OLD = "campaigns/LCA/md_handoff"
# 16 pocket CA positions, 1-based in the 181-residue full protein sequence
POCKET = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]


def kabsch(mobile, target):
    """Rigid transform taking `mobile` onto `target`; returns (R, t, rmsd)."""
    mc, tc = mobile.mean(0), target.mean(0)
    P, Q = mobile - mc, target - tc
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    fitted = P @ R + tc
    rmsd = float(np.sqrt(((fitted - target) ** 2).sum(1).mean()))
    return R, mc, tc, rmsd


def apply_fit(X, R, mc, tc):
    return (X - mc) @ R + tc


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


class PriorRepo:
    def __init__(self, repo):
        self.g = ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "show"]

    def blob(self, relpath):
        return subprocess.check_output(self.g + [f"{COMMIT}:{OLD}/{relpath}"])


def symmetry_rmsd(ref, cur, prev):
    """Min RMSD over reference graph automorphisms; also the identity mapping.

    Returns (name_mapped, symmetry_corrected, best_permutation_description).
    """
    names = ref.heavy
    C = np.array([cur[a] for a in names])
    best, best_desc = None, ""
    plain = rmsd(C, np.array([prev[a] for a in names]))
    for m in ref.automorphisms:
        P = np.array([prev[m[a]] for a in names])
        v = rmsd(C, P)
        if best is None or v < best - 1e-12:
            best = v
            moved = [f"{a}->{m[a]}" for a in names if m[a] != a]
            best_desc = ";".join(moved) if moved else "identity"
    return plain, best, best_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosswalk", required=True, type=Path)
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--prior-repo", required=True, type=Path)
    a = ap.parse_args()

    cw = pd.read_csv(a.crosswalk)
    shared = cw[cw.arm_status.eq("in_both")].copy()
    assert len(shared) == 80, f"{len(shared)} shared arms, expected 80"
    repo = PriorRepo(a.prior_repo.resolve())
    R = refs()

    rows = []
    for r in shared.itertuples():
        rec = dict(sequence_id=r.sequence_id, ligand=r.ligand,
                   ligand_name=PRETTY[r.ligand], role=r.role,
                   previous_id=r.previous_id, previous_role=r.previous_role,
                   class_changed=bool(r.class_changed),
                   current_seed=int(r.selected_seed), previous_seed=int(r.previous_seed),
                   seed_index_comparison=r.seed_index_comparison,
                   current_pdb=r.delivery_pdb, previous_pdb=r.previous_pdb)

        cur_path = a.pkg / r.delivery_pdb
        if not cur_path.exists():
            rows.append(rec | dict(comparison_status="unresolved: current file missing"))
            continue
        cur_bytes = cur_path.read_bytes()
        try:
            prev_bytes = repo.blob(r.previous_pdb)
        except subprocess.CalledProcessError:
            rows.append(rec | dict(comparison_status="unresolved: previous blob unreadable"))
            continue

        cur_sha = hashlib.sha256(cur_bytes).hexdigest()
        prev_sha = hashlib.sha256(prev_bytes).hexdigest()
        rec.update(current_pdb_sha256=cur_sha, previous_pdb_sha256=prev_sha,
                   previous_hash_verified=int(prev_sha == r.previous_pdb_sha256),
                   bytes_identical=int(cur_bytes == prev_bytes))
        if prev_sha != r.previous_pdb_sha256:
            rows.append(rec | dict(comparison_status="unresolved: previous hash mismatch"))
            continue

        cu, pv = read_pdb(cur_bytes), read_pdb(prev_bytes)
        ref = R[r.ligand]

        # --- pocket CA fit, with residue-identity verification --------------
        have = [p for p in POCKET if p in cu["ca"] and p in pv["ca"]]
        ident_ok = all(cu["ca"][p][0] == pv["ca"][p][0] == r.protein_sequence[p - 1]
                       for p in have)
        rec.update(n_ca_pocket=len(have), pocket_residues_verified=int(ident_ok))
        if len(have) != len(POCKET) or not ident_ok:
            rows.append(rec | dict(comparison_status="unresolved: pocket CA map incomplete"))
            continue

        Pc = np.array([cu["ca"][p][1] for p in have])
        Pp = np.array([pv["ca"][p][1] for p in have])
        Rm, mc, tc, ca_rmsd = kabsch(Pc, Pp)

        # --- secondary whole-PYR1 CA fit, for sensitivity -------------------
        allpos = [p for p in cu["resnums"] if p in pv["ca"]]
        Ac = np.array([cu["ca"][p][1] for p in allpos])
        Ap = np.array([pv["ca"][p][1] for p in allpos])
        Rm2, mc2, tc2, ca_rmsd_all = kabsch(Ac, Ap)
        rec.update(ca_rmsd_pocket_fit=round(ca_rmsd, 4), n_ca_whole=len(allpos),
                   ca_rmsd_whole_fit=round(ca_rmsd_all, 4))

        # --- ligand map completeness ---------------------------------------
        need = set(ref.heavy)
        if not (need <= set(cu["ligand"]) and need <= set(pv["ligand"])):
            rec.update(n_ligand_heavy=len(need & set(cu["ligand"]) & set(pv["ligand"])),
                       ligand_atoms_expected=len(need),
                       ligand_missing_current=";".join(sorted(need - set(cu["ligand"]))),
                       ligand_missing_previous=";".join(sorted(need - set(pv["ligand"]))))
            rows.append(rec | dict(comparison_status="unresolved: incomplete ligand atom map"))
            continue

        cur_lig = {k: v for k, v in cu["ligand"].items() if k in need}
        prev_lig = {k: v for k, v in pv["ligand"].items() if k in need}
        fit_lig = {k: apply_fit(v[None, :], Rm, mc, tc)[0] for k, v in cur_lig.items()}
        plain, sym, desc = symmetry_rmsd(ref, fit_lig, prev_lig)

        # core: fixed by every automorphism, so name mapping is the only mapping
        Cc = np.array([fit_lig[c] for c in ref.core])
        Cp = np.array([prev_lig[c] for c in ref.core])

        # sensitivity: same ligand RMSD under the whole-protein fit
        fit_lig2 = {k: apply_fit(v[None, :], Rm2, mc2, tc2)[0] for k, v in cur_lig.items()}
        _, sym2, _ = symmetry_rmsd(ref, fit_lig2, prev_lig)

        rec.update(
            comparison_status="measured",
            ligand_atoms_expected=len(need), n_ligand_heavy=len(need),
            mapping_method="reference atom names + graph-automorphism symmetry",
            n_symmetry_mappings=len(ref.automorphisms),
            ligand_rmsd_name_mapped=round(plain, 4),
            ligand_rmsd_symmetry_corrected=round(sym, 4),
            symmetry_mapping_used=desc,
            core_rmsd_C1_C17=round(rmsd(Cc, Cp), 4), n_core_atoms=len(ref.core),
            ligand_rmsd_whole_ca_fit=round(sym2, 4),
        )
        if rec["bytes_identical"]:
            rec["comparison_status"] = "measured: byte-identical (verified zero RMSD)"
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(a.pkg / "shared_pose_RMSD.csv", index=False)
    report(out, a.pkg)


def report(out, pkg):
    m = out[out.comparison_status.str.startswith("measured")]
    print(f"shared arms: {len(out)}; measured: {len(m)}; "
          f"unresolved: {len(out)-len(m)}")
    print(f"previous hashes verified: {int(out.previous_hash_verified.fillna(0).sum())}/{len(out)}")
    print(f"byte-identical pairs: {int(out.bytes_identical.fillna(0).sum())}")
    for s in sorted(set(out.comparison_status) - set(m.comparison_status)):
        print(f"  UNRESOLVED [{s}]: {int((out.comparison_status==s).sum())}")

    col = "ligand_rmsd_symmetry_corrected"
    def summarise(df, label):
        if not len(df):
            print(f"  {label:28s}   n=0"); return
        v = df[col]
        print(f"  {label:28s} n={len(df):3d}  median={v.median():6.3f}  "
              f"IQR={v.quantile(.25):6.3f}-{v.quantile(.75):6.3f}  max={v.max():7.3f}")
    print("\nligand heavy-atom RMSD after pocket CA fit (A), symmetry-corrected:")
    summarise(m, "all measured")
    for lig in ("LCAM", "LCA3S"):
        summarise(m[m.ligand.eq(lig)], f"{PRETTY[lig]}")
    for s in ("same_index", "different_index"):
        summarise(m[m.seed_index_comparison.eq(s)], s)
    print("\nreview bins (symmetry-corrected ligand RMSD, descriptive only):")
    b = pd.cut(m[col], [-0.001, 1, 2, np.inf], labels=["<1 A", "1-<2 A", ">=2 A"])
    tab = pd.crosstab(m.ligand.map(PRETTY), b)
    print(tab.to_string())
    print(f"\n  core C1-C17 RMSD: median={m.core_rmsd_C1_C17.median():.3f}  "
          f"max={m.core_rmsd_C1_C17.max():.3f}")
    print(f"  pocket-fit CA RMSD: median={m.ca_rmsd_pocket_fit.median():.3f}  "
          f"max={m.ca_rmsd_pocket_fit.max():.3f}")
    print(f"  whole-CA-fit CA RMSD (sensitivity): median={m.ca_rmsd_whole_fit.median():.3f}  "
          f"max={m.ca_rmsd_whole_fit.max():.3f}")
    nz = m[m[col].ne(m.ligand_rmsd_name_mapped)]
    print(f"  symmetry correction changed the value in {len(nz)}/{len(m)} arms")
    print(f"\n-> {pkg}/shared_pose_RMSD.csv")


if __name__ == "__main__":
    main()
