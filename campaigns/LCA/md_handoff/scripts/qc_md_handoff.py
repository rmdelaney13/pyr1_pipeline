#!/usr/bin/env python3
"""Structure-level QC for the MD hand-off set, applied AFTER build_md_handoff.py.

Three jobs, all on the already-selected set (this script never changes the
selection itself, so the balance proven by balance_select.py is untouched):

  1. INTERNAL LIGAND GEOMETRY. Boltz occasionally collapses the C24 carboxylate
     oxygens onto each other (observed 0.04-0.25 A). Ring-pucker QC cannot see
     it (the carboxylate is a tail, not a ring), pose-consistency cannot see it
     (it compares seeds, not internal geometry), and pLDDT can still be ~0.94.
     Such a structure is unusable for MD. Where the staged rep_seed is defective
     we RE-PICK the representative among geometry-clean seeds of the same
     binding mode (medoid by CA-superposed ligand RMSD) and restage it.

  2. CONTACT AUDIT of whatever is finally staged. Raw minimum ligand-protein
     distance conflates hydrogen bonds with steric clashes, so the two are
     reported separately: a contact is POLAR only if both atoms are N/O/S.
     Nothing is dropped on this basis - these are declared covariates that the
     MD collaborator needs in order to set up restrained minimisation.

  3. AUDITABILITY. balance_select.py balances 16 score x ligand combinations but
     the shipped CSV carried only 12, so a reader could not check the other 4.
     Those columns are copied in from the FULL metrics table.

Writes: md_handoff_selection.csv (updated in place, backup kept) and
        structure_qc.csv (per structure, long form).
"""
import argparse, csv, glob, os, shutil
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
SCRATCH = "/gpfs/alpine1/scratch/ryde3462/lca_lca3s_5seed"
LIGS = ["LCAM", "LCA3S"]

LIG_MIN_A = 1.15     # below this, two ligand heavy atoms are collapsed
STERIC_HARD = 2.8    # C-containing contact below this is a hard clash
STERIC_TIGHT = 3.0   # sub-vdW, strained but usually relaxes
POLAR_SHORT = 2.4    # too short even for a strong H-bond
POLAR = set("NOS")

# the 4 balanced features that build_md_handoff.py did not carry through
MISSING_FEATS = {"plddt_protein": "binary_plddt_protein",
                 "n_interface_unsatisfied": "binary_n_interface_unsatisfied"}


def parse_pdb(path):
    """-> (ca{resi:xyz}, prot[(xyz,elem)], lig[(name,xyz,elem)]); heavy atoms only."""
    ca, prot, lig = {}, [], []
    with open(path) as fh:
        for L in fh:
            if not L.startswith(("ATOM", "HETATM")):
                continue
            el = L[76:78].strip().upper()
            if el == "H":
                continue
            xyz = (float(L[30:38]), float(L[38:46]), float(L[46:54]))
            if L.startswith("ATOM"):
                prot.append((xyz, el))
                if L[12:16].strip() == "CA":
                    ca[int(L[22:26])] = xyz
            else:
                lig.append((L[12:16].strip(), xyz, el))
    return ca, prot, lig


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def ligand_metrics(path):
    """Internal + interface geometry of one structure."""
    ca, prot, lig = parse_pdb(path)
    L = np.array([a[1] for a in lig], float)
    out = {"n_lig_atoms": len(lig)}

    if len(L) > 1:
        d = np.linalg.norm(L[:, None, :] - L[None, :, :], axis=2) + np.eye(len(L)) * 1e9
        i, j = np.unravel_index(d.argmin(), d.shape)
        out["lig_min_intra"] = float(d.min())
        out["lig_min_intra_pair"] = f"{lig[i][0]}-{lig[j][0]}"
    else:
        out["lig_min_intra"] = float("inf")
        out["lig_min_intra_pair"] = ""
    out["ligand_geom_bad"] = int(out["lig_min_intra"] < LIG_MIN_A)

    if prot and len(L):
        P = np.array([a[0] for a in prot], float)
        D = np.linalg.norm(L[:, None, :] - P[None, :, :], axis=2)
        pol = np.array([[(lig[i][2] in POLAR) and (prot[j][1] in POLAR)
                         for j in range(len(prot))] for i in range(len(lig))])
        st = np.where(pol, np.inf, D)
        pl = np.where(pol, D, np.inf)
        out["min_steric"] = float(st.min())
        out["min_polar"] = float(pl.min())
        out["n_steric_lt28"] = int((st < STERIC_HARD).sum())
        out["n_steric_lt30"] = int((st < STERIC_TIGHT).sum())
        out["n_polar_lt24"] = int((pl < POLAR_SHORT).sum())
        out["min_any"] = float(D.min())
    return out, ca, lig


def seed_paths(name):
    out = {}
    for s in (1, 2, 3, 4, 5):
        h = sorted(glob.glob(f"{SCRATCH}/seed_{s}/boltz_out/boltz_results_*/"
                             f"predictions/{name}/{name}_model_0.pdb"))
        if h:
            out[s] = h[0]
    return out


def medoid(cands, paths):
    """Seed whose ligand pose is closest to the other candidates (CA-superposed)."""
    if len(cands) == 1:
        return cands[0]
    frames, ref = {}, None
    for s in cands:
        ca, _, lig = parse_pdb(paths[s])
        if ref is None:
            ref = ca
        common = sorted(set(ca) & set(ref))
        if len(common) < 20:
            continue
        R, t = kabsch(np.array([ca[r] for r in common]),
                      np.array([ref[r] for r in common]))
        frames[s] = {a: R @ np.asarray(x, float) + t for a, x, _ in lig}
    tot = {s: 0.0 for s in cands}
    cnt = {s: 0 for s in cands}
    for a in cands:
        for b in cands:
            if a >= b or a not in frames or b not in frames:
                continue
            sh = sorted(set(frames[a]) & set(frames[b]))
            if not sh:
                continue
            A = np.array([frames[a][k] for k in sh])
            B = np.array([frames[b][k] for k in sh])
            r = float(np.sqrt(((A - B) ** 2).sum(-1).mean()))
            tot[a] += r; cnt[a] += 1
            tot[b] += r; cnt[b] += 1
    scored = [(tot[s] / cnt[s], s) for s in cands if cnt[s]]
    return min(scored)[1] if scored else cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", default=f"{ROOT}/results/md_experiment_set")
    ap.add_argument("--full", default=f"{ROOT}/results/md_handoff_5seed/"
                                      "boltz_full_metrics_5seed.csv")
    ap.add_argument("--long", default=f"{ROOT}/results/md_handoff_5seed/"
                                      "boltz_5seed_long.csv")
    ap.add_argument("--apply", action="store_true",
                    help="restage fixed PDBs and rewrite the selection CSV "
                         "(default is a dry run that only reports)")
    a = ap.parse_args()

    sel_path = f"{a.set_dir}/md_handoff_selection.csv"
    pdbdir = f"{a.set_dir}/pdbs"
    rows = list(csv.DictReader(open(sel_path)))
    print(f"selection: {len(rows)} designs from {sel_path}")

    # per-seed binding mode, so a re-pick stays inside the majority mode
    mode = {}
    if os.path.exists(a.long):
        for r in csv.DictReader(open(a.long)):
            mode[(r["name"], int(r["seed"]))] = r.get("binary_binding_mode", "")

    full = {}
    for r in csv.DictReader(open(a.full)):
        full[(r["name"], r["ligand"])] = r

    qc_rows, repicks, flagged = [], [], []
    for r in rows:
        sid = r["sequence_id"]
        for lig in LIGS:
            name = f"{sid}__{lig}_binary"
            staged = f"{pdbdir}/{sid}__{lig}.pdb"
            if not os.path.exists(staged):
                print(f"  MISSING staged PDB {staged}")
                continue
            paths = seed_paths(name)
            per_seed = {s: ligand_metrics(p)[0] for s, p in paths.items()}
            clean = [s for s, m in per_seed.items() if not m["ligand_geom_bad"]]
            cur = int(float(r.get(f"{lig}_rep_seed") or 0) or 0)

            new_rep = cur
            if cur in per_seed and per_seed[cur]["ligand_geom_bad"]:
                maj = r.get(f"{lig}_mode_majority", "")
                cands = [s for s in clean if mode.get((name, s), maj) == maj] or clean
                if cands:
                    new_rep = medoid(sorted(cands), paths)
                    repicks.append((sid, lig, cur, new_rep,
                                    per_seed[cur]["lig_min_intra"],
                                    per_seed[new_rep]["lig_min_intra"]))

            if a.apply and new_rep != cur:
                shutil.copyfile(paths[new_rep], staged)
                r[f"{lig}_rep_seed"] = str(new_rep)
                r[f"{lig}_rep_seed_repicked"] = "1"
            elif f"{lig}_rep_seed_repicked" not in r:
                r[f"{lig}_rep_seed_repicked"] = "0"

            m, _, _ = ligand_metrics(staged if not (a.apply and new_rep != cur)
                                     else paths[new_rep])
            r[f"{lig}_min_steric_contact"] = f"{m['min_steric']:.2f}"
            r[f"{lig}_min_polar_contact"] = f"{m['min_polar']:.2f}"
            r[f"{lig}_n_steric_lt28"] = m["n_steric_lt28"]
            r[f"{lig}_n_steric_lt30"] = m["n_steric_lt30"]
            r[f"{lig}_lig_min_intra"] = f"{m['lig_min_intra']:.3f}"
            r[f"{lig}_ligand_geom_bad"] = m["ligand_geom_bad"]
            r[f"{lig}_needs_restrained_min"] = int(m["min_steric"] < STERIC_HARD)
            if m["min_steric"] < STERIC_HARD or m["ligand_geom_bad"]:
                flagged.append((sid, lig, round(m["min_steric"], 2),
                                m["ligand_geom_bad"]))

            # carry the 4 balanced-but-uncarried columns through
            fr = full.get((name, lig)) or full.get((f"{sid}__{lig}", lig)) or {}
            for short, src in MISSING_FEATS.items():
                if src in fr:
                    r[f"{lig}_{short}"] = fr[src]

            qc_rows.append(dict(sequence_id=sid, ligand=lig, cls=r["class"],
                                rep_seed=new_rep, repicked=int(new_rep != cur),
                                **{k: v for k, v in m.items()}))

    print(f"\nrep_seed re-picks needed: {len(repicks)}")
    for sid, lig, old, new, dold, dnew in repicks:
        print(f"  {sid:16}{lig:6} seed {old} ({dold:.3f} A collapsed) -> "
              f"seed {new} ({dnew:.3f} A)")
    print(f"\nstructures flagged (hard steric < {STERIC_HARD} A or bad ligand geom): "
          f"{len(flagged)} / {len(qc_rows)}")

    if not a.apply:
        print("\nDRY RUN - nothing written. Re-run with --apply.")
        return

    shutil.copyfile(sel_path, sel_path + ".bak_pre_qc")
    cols = list(rows[0].keys())
    with open(sel_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    qc_path = f"{a.set_dir}/structure_qc.csv"
    with open(qc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qc_rows[0].keys()))
        w.writeheader()
        w.writerows(qc_rows)
    print(f"\nwrote {sel_path} (backup .bak_pre_qc)\nwrote {qc_path}")


if __name__ == "__main__":
    main()
