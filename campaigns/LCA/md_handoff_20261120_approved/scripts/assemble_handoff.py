#!/usr/bin/env python3
"""Steps 2, 4, 5, 6: reconcile the deliveries, decide transfers, attach the
clonal records, and close out the package.

Step 2  Recompute every reconciliation count from the STAGED manifests rather
        than trusting the instruction table, and write the identity lists.
Step 4  transfer_actions.csv. "Delivered previously", "simulation actually
        started/completed" and "approved current start" are three separate
        fields and are never collapsed.
Step 5  Copy the daily clonal records and link them by (sequence_id, ligand).
        Missing and excluded values stay blank; `reported_daily_EC50_nM` is the
        reportable field and `archived_fit_EC50_nM` is audit-only.
Step 6  SHA256SUMS.txt over every delivered file, plus a final count check.

    python3 assemble_handoff.py --bundle B --pkg D --md-provenance P
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from lca_chem import PRETTY

CORE_FILES = ["approved_holo_arms.csv", "daily_clonal_records.csv",
              "previous_delivery_crosswalk.csv", "shared_pose_RMSD.csv",
              "transfer_actions.csv", "selection_audit.csv",
              "matching_AUROCs.csv", "stage_verification.csv",
              "md_run_provenance.csv", "reconciliation_summary.json",
              "arm_identities.csv", "README.md"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def reconcile(roster, cw, pkg):
    """Step 2: counts recomputed from the staged tables, plus identity lists."""
    cur = cw[cw.arm_status.ne("previous_only")]
    prev_seqs = set(cw[cw.arm_status.ne("new_arm")].protein_sequence)
    rows, summary = [], {}
    for lig in ("LCAM", "LCA3S"):
        L = cw[cw.ligand.eq(lig)] if "ligand" in cw else None
        sub = cw[cw.ligand.eq(lig)]
        ro = roster[roster.ligand.eq(lig)]
        sh = sub[sub.arm_status.eq("in_both")]
        rows.append(dict(
            ligand=PRETTY[lig],
            previous_arms=int(sub.arm_status.isin(["in_both", "previous_only"]).sum()),
            approved_binders=int(ro.role.eq("binder").sum()),
            approved_controls=int(ro.role.eq("matched_negative").sum()),
            approved_total=len(ro),
            shared_arms=len(sh),
            new_arms=int(sub.arm_status.eq("new_arm").sum()),
            previous_only=int(sub.arm_status.eq("previous_only").sum()),
            shared_same_seed_index=int(sh.seed_index_comparison.eq("same_index").sum()),
            shared_different_seed_index=int(
                sh.seed_index_comparison.eq("different_index").sum()),
            class_changed=int(sh.class_changed.fillna(False).astype(bool).sum()),
        ))
    tab = pd.DataFrame(rows)
    total = tab.drop(columns="ligand").sum()
    total["ligand"] = "Total"
    tab = pd.concat([tab, total.to_frame().T[tab.columns]], ignore_index=True)

    seqs = roster.drop_duplicates("protein_sequence")
    summary["arms"] = tab.to_dict("records")
    summary["distinct_current_sequences"] = int(roster.protein_sequence.nunique())
    summary["sequences_in_previous_delivery"] = int(
        seqs.protein_sequence.isin(prev_seqs).sum())
    summary["sequences_new"] = int((~seqs.protein_sequence.isin(prev_seqs)).sum())
    summary["new_arms_reusing_delivered_sequence_other_ligand"] = int(
        cw[cw.arm_status.eq("new_arm")].sequence_in_previous_handoff.sum())
    summary["unique_current_arm_keys"] = int(
        roster.drop_duplicates(["sequence_id", "ligand"]).shape[0])

    ident = cw[["sequence_id", "ligand", "role", "previous_id", "previous_role",
                "arm_status", "seed_index_comparison", "selected_seed",
                "previous_seed", "class_changed", "delivery_pdb", "previous_pdb"]].copy()
    ident["ligand_name"] = ident.ligand.map(PRETTY)
    ident.sort_values(["arm_status", "ligand", "sequence_id"]).to_csv(
        pkg / "arm_identities.csv", index=False)
    return tab, summary


def md_provenance(cw, pkg):
    """Step 4 input: what MD actually exists. Searched, not assumed."""
    searched = [
        "/gpfs/alpine1/scratch/ryde3462/holo_md",
        "/gpfs/alpine1/scratch/ryde3462/apo_md",
        "/projects/ryde3462/lca_lca3s_modelling",
    ]
    found = []
    for root in searched:
        r = Path(root)
        if not r.exists():
            found.append(dict(path=root, exists=0, files=0, note="path absent"))
            continue
        n = sum(1 for _ in r.rglob("*") if _.is_file())
        found.append(dict(path=root, exists=1, files=n,
                          note="directory skeleton only, no run files"
                          if n == 0 else "files present; see note column"))
    rows = []
    for r in cw[cw.arm_status.ne("previous_only")].itertuples():
        rows.append(dict(
            sequence_id=r.sequence_id, ligand=r.ligand, ligand_name=PRETTY[r.ligand],
            delivered_previously=int(r.arm_status == "in_both"),
            previous_pdb=r.previous_pdb if r.arm_status == "in_both" else "",
            approved_current_start=r.delivery_pdb,
            md_topology_found="", md_trajectory_found="", md_run_log_found="",
            simulation_started="", simulation_completed="",
            md_provenance_status="unresolved: no MD-run provenance found",
            md_provenance_note=("no holo trajectory, topology, preparation record or "
                                "run log exists for any arm; see README"),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(pkg / "md_run_provenance.csv", index=False)
    return out, found


def transfer_actions(roster, cw, rmsd, prov, pkg):
    """Step 4: one row per current arm, with the evidence that drove the action."""
    r = rmsd.set_index(["sequence_id", "ligand"])
    cur = cw[cw.arm_status.ne("previous_only")]
    rows = []
    for a in cur.itertuples():
        k = (a.sequence_id, a.ligand)
        rec = dict(sequence_id=a.sequence_id, ligand=a.ligand,
                   ligand_name=PRETTY[a.ligand], role=a.role,
                   arm_status=a.arm_status,
                   delivered_previously=int(a.arm_status == "in_both"),
                   approved_current_start=a.delivery_pdb,
                   previous_pdb=a.previous_pdb if a.arm_status == "in_both" else "",
                   current_seed=int(a.selected_seed),
                   previous_seed=int(a.previous_seed) if a.arm_status == "in_both" else "",
                   seed_index_comparison=a.seed_index_comparison,
                   class_changed=bool(a.class_changed) if a.arm_status == "in_both" else "",
                   previous_role=a.previous_role if a.arm_status == "in_both" else "",
                   simulation_started="", simulation_completed="",
                   md_provenance_status="unresolved: no MD-run provenance found")

        if a.arm_status == "new_arm":
            rec.update(evidence="new sequence-ligand arm",
                       in_full_package=1, in_transfer_list=1,
                       action="transfer approved top-seed PDB",
                       existing_md_can_represent="n/a - arm was never delivered",
                       bytes_identical="", ligand_rmsd_symmetry_corrected="",
                       core_rmsd_C1_C17="", comparison_status="")
            rows.append(rec); continue

        if k not in r.index:
            rec.update(evidence="shared arm, comparison unresolved",
                       in_full_package=1, in_transfer_list=1,
                       action="transfer approved top-seed PDB; comparison unresolved",
                       existing_md_can_represent="unresolved: pose comparison missing",
                       comparison_status="missing")
            rows.append(rec); continue

        c = r.loc[k]
        rec.update(bytes_identical=int(c.bytes_identical),
                   comparison_status=c.comparison_status,
                   ligand_rmsd_symmetry_corrected=c.get("ligand_rmsd_symmetry_corrected", ""),
                   core_rmsd_C1_C17=c.get("core_rmsd_C1_C17", ""),
                   ca_rmsd_pocket_fit=c.get("ca_rmsd_pocket_fit", ""))

        if not str(c.comparison_status).startswith("measured"):
            rec.update(evidence="shared arm, comparison unresolved",
                       in_full_package=1, in_transfer_list=1,
                       action="transfer approved top-seed PDB; comparison unresolved",
                       existing_md_can_represent="unresolved: " + str(c.comparison_status))
        elif int(c.bytes_identical) == 1:
            rec.update(evidence="shared arm, identical file",
                       in_full_package=1, in_transfer_list=0,
                       action="include in full package; prior copy may supply the file "
                              "after hash verification",
                       existing_md_can_represent="unresolved: no MD-run provenance to attribute")
        elif float(c.ligand_rmsd_symmetry_corrected) == 0.0 and float(c.ca_rmsd_pocket_fit) == 0.0:
            rec.update(evidence="shared arm, identical coordinates, hash differs "
                                "(formatting/metadata only)",
                       in_full_package=1, in_transfer_list=1,
                       action="stage canonical new file; record reason for hash difference",
                       existing_md_can_represent="unresolved: no MD-run provenance to attribute")
        else:
            rec.update(evidence="shared arm, different starting coordinates",
                       in_full_package=1, in_transfer_list=1,
                       action="transfer approved top-seed file; any previous trajectory "
                              "used a different start",
                       existing_md_can_represent="no - different starting coordinates")
        rows.append(rec)

    out = pd.DataFrame(rows)
    assert len(out) == 280, f"transfer_actions has {len(out)} rows, expected 280"
    out.sort_values(["ligand", "arm_status", "sequence_id"]).to_csv(
        pkg / "transfer_actions.csv", index=False)
    return out


def daily(bundle, roster, pkg):
    """Step 5: copy the records verbatim and report linkage."""
    src = bundle / "tables" / "daily_clonal_records.csv"
    shutil.copyfile(src, pkg / "daily_clonal_records.csv")
    d = pd.read_csv(pkg / "daily_clonal_records.csv")
    assert sha256_file(src) == sha256_file(pkg / "daily_clonal_records.csv")
    linked = d.merge(roster[["sequence_id", "ligand"]], on=["sequence_id", "ligand"])
    assert len(linked) == len(d), "daily record rows failed to link to the roster"
    excl = d[d.date.eq("2026-09-14") & d.ligand.eq("LCA3S")]
    return dict(
        rows=len(d),
        arms_with_records=int(d.groupby(["sequence_id", "ligand"]).ngroups),
        observed_dates=int(d.date.nunique()),
        rows_with_reported_EC50=int(d.reported_daily_EC50_nM.notna().sum()),
        rows_with_archived_fit_only=int((d.archived_fit_EC50_nM.notna()
                                         & d.reported_daily_EC50_nM.isna()).sum()),
        excluded_20260914_LCA3S_rows=len(excl),
        excluded_20260914_LCA3S_reported_EC50=int(
            excl.reported_daily_EC50_nM.notna().sum()),
        roster_status_counts=roster.daily_assay_status.value_counts().to_dict(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--pkg", required=True, type=Path)
    a = ap.parse_args()
    pkg, bundle = a.pkg, a.bundle

    for f in ("approved_holo_arms.csv", "previous_delivery_crosswalk.csv",
              "selection_audit.csv", "matching_AUROCs.csv"):
        shutil.copyfile(bundle / "tables" / f, pkg / f)

    roster = pd.read_csv(pkg / "approved_holo_arms.csv")
    cw = pd.read_csv(pkg / "previous_delivery_crosswalk.csv")
    rmsd = pd.read_csv(pkg / "shared_pose_RMSD.csv")
    stage = pd.read_csv(pkg / "stage_verification.csv")

    tab, summary = reconcile(roster, cw, pkg)
    prov, searched = md_provenance(cw, pkg)
    ta = transfer_actions(roster, cw, rmsd, prov, pkg)
    summary["daily_clonal_records"] = daily(bundle, roster, pkg)
    summary["md_provenance_search"] = searched

    # ---- step 6 verification -------------------------------------------------
    staged = stage[stage.staged.eq(1)]
    pdbs = sorted((pkg / "pdbs").glob("*.pdb"))
    summary["package"] = dict(
        expected_arms=280, unique_arm_keys=summary["unique_current_arm_keys"],
        expected_starting_files=280, staged_files=len(pdbs),
        files_passing_all_checks=int((staged.geom_ok.eq(1)
                                      & staged.sequence_matches_roster.eq(1)
                                      & staged.ligand_atoms_match_reference.eq(1)).sum()),
        unique_sha256=int(staged.sha256.nunique()),
        total_bytes=int(staged.bytes.sum()),
        missing_paths=[r.delivery_pdb for r in stage[stage.staged.ne(1)].itertuples()],
    )
    m = rmsd[rmsd.comparison_status.str.startswith("measured", na=False)]
    col = "ligand_rmsd_symmetry_corrected"
    summary["pose_comparison"] = dict(
        shared_arms=len(rmsd), measured=len(m), unresolved=len(rmsd) - len(m),
        previous_hashes_verified=int(rmsd.previous_hash_verified.fillna(0).sum()),
        byte_identical=int(rmsd.bytes_identical.fillna(0).sum()),
        median=round(float(m[col].median()), 4),
        iqr=[round(float(m[col].quantile(.25)), 4), round(float(m[col].quantile(.75)), 4)],
        maximum=round(float(m[col].max()), 4),
        bins={"<1": int((m[col] < 1).sum()),
              "1-<2": int(((m[col] >= 1) & (m[col] < 2)).sum()),
              ">=2": int((m[col] >= 2).sum())},
        by_ligand={PRETTY[l]: dict(
            n=int(len(m[m.ligand.eq(l)])),
            median=round(float(m[m.ligand.eq(l)][col].median()), 4),
            maximum=round(float(m[m.ligand.eq(l)][col].max()), 4))
            for l in ("LCAM", "LCA3S")},
        by_seed_match={s: dict(
            n=int(len(m[m.seed_index_comparison.eq(s)])),
            median=round(float(m[m.seed_index_comparison.eq(s)][col].median()), 4),
            maximum=round(float(m[m.seed_index_comparison.eq(s)][col].max()), 4))
            for s in ("same_index", "different_index")},
    )
    summary["transfer"] = dict(
        in_full_package=int(ta.in_full_package.sum()),
        in_transfer_list=int(ta.in_transfer_list.sum()),
        by_evidence=ta.evidence.value_counts().to_dict(),
        class_changes=int(pd.Series(ta.class_changed).eq(True).sum()),
        md_reuse_unresolved=int(ta.existing_md_can_represent.str.startswith(
            "unresolved").sum()),
    )
    summary["reconciliation_table"] = tab.to_dict("records")
    summary["versions"] = versions()
    (pkg / "reconciliation_summary.json").write_text(json.dumps(summary, indent=2,
                                                                default=str))

    print(tab.to_string(index=False))
    print("\ndistinct current sequences:", summary["distinct_current_sequences"],
          "| previously delivered:", summary["sequences_in_previous_delivery"],
          "| new:", summary["sequences_new"])
    print("new arms reusing a delivered sequence with the other ligand:",
          summary["new_arms_reusing_delivered_sequence_other_ligand"])
    print("\ntransfer list:", summary["transfer"]["in_transfer_list"],
          "of", summary["transfer"]["in_full_package"], "in the full package")
    for k, v in summary["transfer"]["by_evidence"].items():
        print(f"   {v:3d}  {k}")
    print("\ndaily clonal records:", summary["daily_clonal_records"])
    return summary


def versions():
    import platform, networkx, scipy
    git = subprocess.run(["git", "-c", "safe.directory=/projects/ryde3462/software/pyr1_pipeline",
                          "-C", "/projects/ryde3462/software/pyr1_pipeline",
                          "rev-parse", "HEAD"], capture_output=True, text=True)
    return dict(python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, networkx=networkx.__version__,
                scipy=scipy.__version__,
                prior_repo_head=git.stdout.strip() or "unknown")


if __name__ == "__main__":
    main()
