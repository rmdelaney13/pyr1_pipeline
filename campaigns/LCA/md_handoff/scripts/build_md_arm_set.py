#!/usr/bin/env python3
"""Write and stage the PER-ARM MD set from a balance_select.py selection.

WHY THIS IS SEPARATE FROM build_md_handoff.py
build_md_handoff.py is design-paired to its core: it requires both ligands present for
every design, concatenates both into one feature vector, and stages two PDBs per design.
The ligand-geometry gate kills single ARMS, not whole designs, so the deliverable is now
a list of (design, ligand) arms -- a design may ship with LCA, with LCA-3-S, or with both.
Its built-in distance matcher is also superseded by balance_select.py's subset balancing,
so what remains for it to do here is nothing. This script is the single writer for the
per-arm set; build_md_handoff.py stays as-is for the paired set already shipped.

Input  results/md_handoff_5seed/balanced_selection_ids.csv  (sequence_id, ligand, class)
Output <outdir>/md_arm_set.csv            one row per arm, every balanced + declared score
       <outdir>/md_arm_set_long.csv       all eligible arms, in_selection flag, for figures
       <outdir>/pdbs/<sid>__<lig>.pdb     the rep_seed structure
       <outdir>/stage_verification.csv    per-staged-file geometry re-check

The staged files are RE-VERIFIED here, from the PDB on disk, rather than trusted because
the gate said so: every shipped structure must have correct stereochemistry at every
centre and no collapsed atoms. 106 of the 274 structures in the first shipped set were
inverted, so "the pipeline says it is fine" is not the standard any more.
"""
import argparse, csv, glob, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ligand_geometry_scan as LGS           # one implementation of the geometry check

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
BASE = f"{ROOT}/results/md_handoff_5seed"
SCRATCH = "/gpfs/alpine1/scratch/ryde3462/lca_lca3s_5seed"
LIGS = ["LCAM", "LCA3S"]

BINDS = {"both binder":  {"LCAM": 1, "LCA3S": 1},
         "LCA binder":   {"LCAM": 1, "LCA3S": 0},
         "LCA3S binder": {"LCAM": 0, "LCA3S": 1},
         "nonbinder":    {"LCAM": 0, "LCA3S": 0}}

SLIM_COLS = ["plddt_ligand", "plddt_pocket", "geometry_score", "geometry_dist_score",
             "geometry_ang_score", "iptm", "ligand_iptm", "hbond_distance", "hbond_angle",
             "rings6", "rings_distorted", "ring_pucker_max"]
FULL_COLS = ["binary_confidence_score", "binary_plddt_protein", "binary_iptm",
             "binary_ligand_iptm", "binary_affinity_probability_binary",
             "binary_n_interface_unsatisfied", "binary_binding_mode", "esm_apo_dg"]
QC_COLS = ["n_seeds", "n_seeds_total", "n_valid_seeds", "geom_valid_fraction",
           "stereo_ok_fraction", "n_seeds_collapsed", "n_seeds_ring_distorted",
           "invalid_seeds", "inverted_centres", "rep_seed", "seed_agreement",
           "mode_majority", "flipped_fraction", "pose_rmsd_spread", "pose_rmsd_max",
           "head_clash_fraction", "r79_clash_fraction", "pass_pose_consistency",
           "ligand_min_intra_dist"]


def load(a):
    mani = {r["sequence_id"]: r for r in csv.DictReader(open(a.manifest))}
    slim = {}
    for r in csv.DictReader(open(a.slim)):
        slim.setdefault(r["name"].split("__")[0], {})[r["ligand"]] = r
    full = {}
    for r in csv.DictReader(open(a.full)):
        full[(r["name"].split("__")[0], r["ligand"])] = r
    sel = [r for r in csv.DictReader(open(a.select))]
    if "ligand" not in (sel[0] if sel else {}):
        sys.exit(f"{a.select} has no 'ligand' column -- that is a design-paired "
                 "selection, not a per-arm one. Re-run balance_select.py.")
    return mani, slim, full, sel


def stage(a, arms, full, refs):
    """Copy each arm's rep_seed model_0 and re-verify its geometry from disk."""
    pdbdir = f"{a.outdir}/pdbs"
    if os.path.isdir(pdbdir):
        stale = [f for f in os.listdir(pdbdir) if f.endswith(".pdb")]
        for f in stale:
            os.remove(f"{pdbdir}/{f}")
        if stale:
            print(f"cleared {len(stale)} pre-existing PDBs from {pdbdir}")
    else:
        os.makedirs(pdbdir, exist_ok=True)

    ver, missing, bad = [], [], []
    for sid, lig in arms:
        name = f"{sid}__{lig}_binary"
        fr = full.get((sid, lig), {}) or {}
        rep = fr.get("rep_seed", "")
        invalid = {s for s in (fr.get("invalid_seeds") or "").split(";") if s}
        if not rep:
            missing.append(f"{sid}__{lig} (no rep_seed)")
            continue
        if rep in invalid:
            # must not happen: aggregate_seeds.py bars invalid seeds from being rep_seed
            bad.append(f"{sid}__{lig} rep_seed={rep} is in invalid_seeds")
            continue
        hits = sorted(glob.glob(f"{SCRATCH}/seed_{rep}/boltz_out/boltz_results_*/"
                                f"predictions/{name}/{name}_model_0.pdb"))
        if not hits:
            missing.append(f"{sid}__{lig} (rep_seed {rep} file absent)")
            continue
        dst = f"{pdbdir}/{sid}__{lig}.pdb"
        shutil.copyfile(hits[0], dst)

        # re-verify from the COPY that will actually ship
        co = LGS.ligand_coords(dst)
        inverted, evaluable = [], True
        for c, (nbrs, ref_sign) in refs[lig].items():
            v = LGS.signed_volume(co, c, nbrs)
            if v is None or abs(v) < 1e-6:
                evaluable = False
                continue
            if np.sign(v) != ref_sign:
                inverted.append(c)
        X = np.array(list(co.values()))
        D = np.linalg.norm(X[:, None] - X[None, :], axis=2) + np.eye(len(X)) * 1e9
        dmin = float(D.min())
        ok = int(evaluable and not inverted and dmin >= LGS.LIG_MIN_A)
        ver.append(dict(sequence_id=sid, ligand=lig, rep_seed=rep,
                        pdb=f"pdbs/{sid}__{lig}.pdb", evaluable=int(evaluable),
                        n_inverted=len(inverted), inverted_centres=";".join(inverted),
                        min_intra_dist=f"{dmin:.3f}", geometry_ok=ok))
        if not ok:
            bad.append(f"{sid}__{lig} inverted={inverted} dmin={dmin:.2f}")

    with open(f"{a.outdir}/stage_verification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ver[0].keys()) if ver else ["sequence_id"])
        w.writeheader(); w.writerows(ver)

    print(f"\nstaged {len(ver)} PDBs -> {pdbdir}  (requested {len(arms)})")
    nok = sum(r["geometry_ok"] for r in ver)
    print(f"re-verified from disk: {nok}/{len(ver)} geometry-valid"
          f"   min intra-ligand distance {min((float(r['min_intra_dist']) for r in ver), default=float('nan')):.3f} A")
    if missing:
        print(f"  WARNING no structure for {len(missing)}: {missing[:5]}")
    if bad:
        print(f"  *** {len(bad)} STAGED STRUCTURES FAILED VERIFICATION: {bad[:5]}")
        return ver, False
    return ver, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv")
    ap.add_argument("--slim", default=f"{BASE}/boltz_plddt_iptm_geometry_pucker_5seed.csv")
    ap.add_argument("--full", default=f"{BASE}/boltz_full_metrics_5seed.csv")
    ap.add_argument("--select", default=f"{BASE}/balanced_selection_ids.csv")
    ap.add_argument("--outdir", default=f"{ROOT}/results/md_arm_set")
    ap.add_argument("--no-stage", action="store_true", help="write CSVs only")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    mani, slim, full, sel = load(a)
    arms = [(r["sequence_id"], r["ligand"]) for r in sel]
    cls = {(r["sequence_id"], r["ligand"]): r["class"] for r in sel}
    print(f"{len(arms)} arms from {a.select}")
    for lig in LIGS:
        n = [k for k in arms if k[1] == lig]
        nb = sum(1 for k in n if cls[k] == "binder")
        print(f"  {lig:6s} {len(n):4d} arms  ({nb} binder, {len(n) - nb} nonbinder)")

    refs = {lg: LGS.reference(p) for lg, p in LGS.CIF.items()}

    rows = []
    for sid, lig in arms:
        m = mani.get(sid, {})
        s = (slim.get(sid, {}) or {}).get(lig, {}) or {}
        fr = full.get((sid, lig), {}) or {}
        lab = m.get("label", "")
        row = {"sequence_id": sid, "ligand": lig, "class": cls[(sid, lig)],
               "manifest_label": lab,
               "binds_this_ligand": BINDS.get(lab, {}).get(lig, ""),
               "pocket_sequence": m.get("pocket_sequence", ""),
               "protein_sequence": m.get("protein_sequence", ""),
               "pdb": f"pdbs/{sid}__{lig}.pdb"}
        for c in SLIM_COLS:
            row[c] = s.get(c, fr.get(f"binary_{c}", fr.get(c, "")))
        for c in FULL_COLS:
            row[c] = fr.get(c, "")
        for c in QC_COLS:
            row[c] = fr.get(c, "")
        rows.append(row)

    out = f"{a.outdir}/md_arm_set.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} arms, {len(rows[0])} cols)")

    # long form over every ELIGIBLE arm, so figures can show selected vs pool
    longp = f"{a.outdir}/md_arm_set_long.csv"
    insel = set(arms)
    with open(longp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence_id", "ligand", "manifest_label", "binds_this_ligand",
                    "eligible", "in_selection", "plddt_ligand", "plddt_pocket",
                    "geometry_score", "hbond_distance", "n_valid_seeds"])
        for (sid, lig), fr in sorted(full.items()):
            lab = mani.get(sid, {}).get("label", "")
            if lab not in BINDS:
                continue
            s = (slim.get(sid, {}) or {}).get(lig, {}) or {}
            nv = fr.get("n_valid_seeds", "")
            w.writerow([sid, lig, lab, BINDS[lab][lig],
                        int(str(nv).isdigit() and int(nv) >= 1),
                        int((sid, lig) in insel),
                        s.get("plddt_ligand", ""), s.get("plddt_pocket", ""),
                        s.get("geometry_score", ""), s.get("hbond_distance", ""), nv])
    print(f"wrote {longp}")

    if not a.no_stage:
        _, ok = stage(a, arms, full, refs)
        if not ok:
            sys.exit("staging verification FAILED -- do not ship this set")


if __name__ == "__main__":
    main()
