#!/usr/bin/env python3
"""Independent audit of the finished hand-off: re-checks the delivered package
without trusting the build that produced it. Run from the package root.

    cd campaigns/LCA/md_handoff_20261120_approved && python3 scripts/verify_package.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

OK = True


def chk(label, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def main():
    r = pd.read_csv("approved_holo_arms.csv")
    cw = pd.read_csv("previous_delivery_crosswalk.csv")
    st = pd.read_csv("stage_verification.csv")
    rm = pd.read_csv("shared_pose_RMSD.csv")
    ta = pd.read_csv("transfer_actions.csv")
    d = pd.read_csv("daily_clonal_records.csv")
    summ = json.load(open("reconciliation_summary.json"))

    print("package integrity")
    chk("280 unique current arm keys", r.drop_duplicates(["sequence_id", "ligand"]).shape[0] == 280)
    chk("280 expected starting files present", len(list(Path("pdbs").glob("*.pdb"))) == 280)
    chk("every roster delivery_pdb exists", all(Path(p).exists() for p in r.delivery_pdb))
    s = st[st.staged.eq(1)].sample(12, random_state=0)
    bad = [x.delivery_pdb for x in s.itertuples()
           if hashlib.sha256(open(x.delivery_pdb, "rb").read()).hexdigest() != x.sha256]
    chk("sampled staged hashes match stage_verification", not bad, f"{len(s)} sampled")
    chk("all staged files pass the geometry gate", st[st.staged.eq(1)].geom_ok.eq(1).all())

    print("\nreconciliation")
    chk("roster roles 87 binders / 193 controls",
        r.role.eq("binder").sum() == 87 and r.role.eq("matched_negative").sum() == 193)
    chk("crosswalk 80 shared / 200 new / 148 previous-only",
        cw.arm_status.value_counts().to_dict() == {"new_arm": 200, "previous_only": 148,
                                                   "in_both": 80})
    chk("217 distinct sequences, 53 previously delivered",
        r.protein_sequence.nunique() == 217 and summ["sequences_in_previous_delivery"] == 53)
    chk("7 class changes", cw.class_changed.fillna(False).astype(bool).sum() == 7)
    chk("previous-only arms absent from pdbs/",
        not any((Path("pdbs") / x).exists()
                for x in cw[cw.arm_status.eq("previous_only")].previous_pdb.fillna("")))

    print("\npose comparison")
    chk("80 shared arms, all measured",
        len(rm) == 80 and rm.comparison_status.str.startswith("measured").all())
    chk("80/80 previous hashes verified", rm.previous_hash_verified.eq(1).all())
    chk("byte-identical == zero RMSD",
        (rm[rm.bytes_identical.eq(1)].ligand_rmsd_symmetry_corrected == 0).all())
    chk("zero RMSD only where byte-identical",
        (rm[rm.ligand_rmsd_symmetry_corrected.eq(0)].bytes_identical == 1).all())
    chk("ligand atom counts 27 / 31",
        rm[rm.ligand.eq("LCAM")].n_ligand_heavy.eq(27).all()
        and rm[rm.ligand.eq("LCA3S")].n_ligand_heavy.eq(31).all())
    chk("16/16 pocket Ca mapped and identity-verified",
        rm.n_ca_pocket.eq(16).all() and rm.pocket_residues_verified.eq(1).all())
    chk("no RMSD reported as 0 for an unresolved map",
        rm.ligand_rmsd_symmetry_corrected.notna().all())
    chk("symmetry-corrected <= name-mapped everywhere",
        (rm.ligand_rmsd_symmetry_corrected <= rm.ligand_rmsd_name_mapped + 1e-9).all())

    print("\ntransfer + evidence separation")
    chk("280 transfer rows, 260 in transfer list",
        len(ta) == 280 and ta.in_transfer_list.sum() == 260)
    chk("all 280 in full package", ta.in_full_package.eq(1).all())
    chk("every different-coordinate shared arm is in the transfer list",
        ta[ta.evidence.eq("shared arm, different starting coordinates")].in_transfer_list.eq(1).all())
    chk("delivered_previously / simulation / current start are distinct columns",
        {"delivered_previously", "simulation_started", "simulation_completed",
         "approved_current_start"} <= set(ta.columns))
    chk("no arm claims reusable MD",
        ta.existing_md_can_represent.str.startswith("unresolved").sum()
        + ta.existing_md_can_represent.str.startswith(("no", "n/a")).sum() == 280)

    print("\ndaily records")
    chk("all record rows link to the roster",
        len(d.merge(r[["sequence_id", "ligand"]], on=["sequence_id", "ligand"])) == len(d))
    chk("excluded 2026-09-14 LCA-3-S series has no reported EC50",
        d[d.date.eq("2026-09-14") & d.ligand.eq("LCA3S")].reported_daily_EC50_nM.isna().all())
    chk("reported EC50 present only where archived fit was included",
        d[d.reported_daily_EC50_nM.notna()]["EC50 included"].eq("Y").all())
    chk("roster distinguishes no-assay from assay-without-EC50",
        set(r.daily_assay_status) == {"reported daily EC50",
                                      "assayed; no reported daily EC50",
                                      "no linked clonal assay record"})

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
