#!/usr/bin/env python3
"""Aggregate the 5-seed Boltz ensemble to one consensus row per (design, ligand).

Reads  results/md_handoff_5seed/boltz_5seed_long.csv  (from merge_5seed.py)
Writes results/md_handoff_5seed/boltz_full_metrics_5seed.csv          (FULL schema + QC)
       results/md_handoff_5seed/boltz_plddt_iptm_geometry_pucker_5seed.csv  (SLIM schema + plddt_pocket)
       results/md_handoff_5seed/seed_qc.csv                           (QC only, for plots)

Aggregation
  numeric column  -> MEDIAN across seeds (robust to one bad sample, which is the
                     whole point of the re-prediction; `_mean`-style averaging
                     would let a single 1.9 A clash pose drag the ensemble)
  categorical     -> MAJORITY vote, ties broken toward the rep_seed's value
  representative  -> rep_seed = medoid of the MAJORITY-MODE cluster, i.e. the seed
                     whose ligand pose is closest to the other majority-mode seeds.
                     Its model_0 is the PDB that gets staged for MD.

Ligand-geometry gate (2026-10-01)
  A seed is usable only if its ligand IS the modelled molecule. Three independent
  intra-ligand checks, all enforced:
    1. stereochemistry  signed volume at every sp3 centre vs the userCCD reference
                        (ligand_geometry_scan.py). C3 inverted = the 3-beta epimer.
                        Fails in 23% of LCA and 42% of LCA-3-S structures.
    2. collapsed atoms  no two ligand heavy atoms < LIG_MIN_A (0.8% of structures)
    3. ring pucker      no boat/twist steroid ring, from rings_distorted in the long
                        table (0.8%; orthogonal -- 27 structures pass 1 and 2 but
                        fail this)
  Invalid seeds are dropped from the median, the majority vote AND the rep_seed, so
  n_seeds is the VALID-seed count and n_seeds_total is how many were predicted. A
  group with no valid seed keeps its row, flagged n_valid_seeds=0, for selection to
  drop. NOTE the pipeline's own binary_lig_oh_stereo_ok / pass_oh_stereo are NOT used
  here: the former is NaN for all 14,260 rows while the latter reports 1 throughout,
  a silent all-clear that is how inverted ligands reached the first shipped set.

QC columns (these are the new information a 5-seed run buys)
  n_seeds                  valid seeds the consensus was computed from
  flipped_fraction         fraction of seeds with binary_binding_mode == 'flipped'
  unknown_fraction         fraction with mode 'unknown'
  mode_majority            'normal' / 'flipped' / 'unknown'
  seed_agreement           fraction of seeds in the majority mode (1.0 = unanimous)
  pose_rmsd_spread         mean pairwise ligand heavy-atom RMSD (A) after chain-A CA
                           superposition -- how much the pose moves between seeds
  pose_rmsd_max            worst pairwise ligand RMSD (A)
  head_clash_fraction      fraction of seeds with ANY ligand heavy atom < CLASH_A of
                           a protein heavy atom
  r79_clash_fraction       same, restricted to the R79 guanidinium (NE/NH1/NH2/CZ) --
                           the specific 1.88 A O1B-R79 failure found in the 200-set QC
  pass_pose_consistency    1 when seed_agreement >= AGREE_MIN and
                           pose_rmsd_spread <= SPREAD_MAX and head_clash_fraction == 0
                           and mode_majority != 'unknown'. The last clause matters:
                           a design whose mode is UNANIMOUSLY 'unknown' scores
                           seed_agreement 1.0 while having no assignable binding mode
                           at all, so without it a consistently BAD pose would read as
                           a trustworthy one.

Nothing is auto-dropped: flags travel with the row and build_md_handoff.py decides.

    python3 aggregate_seeds.py [--jobs N] [--long PATH] [--outdir DIR]
"""
import argparse, csv, os, sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
SCRATCH = "/gpfs/alpine1/scratch/ryde3462/lca_lca3s_5seed"
SEEDS = [1, 2, 3, 4, 5]

CLASH_A = 2.2          # ligand heavy atom vs protein heavy atom
# Minimum credible distance between two bonded ligand heavy atoms. Boltz
# occasionally collapses the C24 carboxylate oxygens onto each other (observed
# at 0.04-0.25 A, i.e. physically impossible). Ring-pucker QC cannot see this
# because the carboxylate is a tail group, and pLDDT can still be high, so this
# is checked explicitly and such seeds are barred from becoming rep_seed.
LIG_MIN_A = 1.15
R79_ATOMS = {"NE", "NH1", "NH2", "CZ"}
R79_RESI = 79          # PDB numbering of the scaffold (matches analyze_boltz_output)
AGREE_MIN = 0.6        # >= 3/5 seeds agree on the binding mode
SPREAD_MAX = 2.0       # A mean pairwise ligand RMSD

# SLIM schema = results/full_binary/boltz_plddt_iptm_geometry_pucker.csv
# + plddt_pocket, which build_md_handoff.py now needs for the 6-D match but which
# the old single-seed SLIM never carried.
SLIM_COLS = ["name", "ligand", "iptm", "ligand_iptm", "plddt_ligand", "plddt_pocket",
             "hbond_distance", "hbond_angle", "geometry_dist_score",
             "geometry_ang_score", "geometry_score",
             "rings6", "rings_distorted", "ring_pucker_max"]
# SLIM col -> FULL col (analyze_boltz_output prefixes its binary metrics)
SLIM_SRC = {c: ("binary_" + c if c not in ("name", "ligand", "rings6",
                                           "rings_distorted", "ring_pucker_max") else c)
            for c in SLIM_COLS}

QC_COLS = ["n_seeds", "rep_seed", "flipped_fraction", "unknown_fraction",
           "mode_majority", "seed_agreement", "pose_rmsd_spread", "pose_rmsd_max",
           "head_clash_fraction", "r79_clash_fraction", "pass_pose_consistency",
           "ligand_geom_bad_fraction", "ligand_geom_bad_seeds", "ligand_min_intra_dist",
           "seeds_present",
           # stereo gate (2026-10-01). n_seeds is now the VALID-seed count the consensus
           # was computed from; n_seeds_total is how many were predicted.
           "n_seeds_total", "n_seeds_scanned", "n_valid_seeds", "geom_valid_fraction",
           "stereo_ok_fraction", "n_seeds_collapsed", "n_seeds_ring_distorted",
           "invalid_seeds", "inverted_centres", "stereo_gated"]


# ---------------------------------------------------------------- PDB geometry
def parse_pdb(path):
    """-> (ca dict resi->xyz, protein heavy atoms Nx3, R79 guanidinium Mx3,
           ligand {atomname: xyz})   chain A = protein, chain B = ligand."""
    ca, prot, r79, lig = {}, [], [], {}
    with open(path) as f:
        for ln in f:
            rec = ln[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            el = ln[76:78].strip() or ln[12:16].strip()[:1]
            if el == "H":
                continue
            name = ln[12:16].strip()
            ch = ln[21]
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            if ch == "B":
                lig[name] = xyz
            elif ch == "A":
                prot.append(xyz)
                try:
                    resi = int(ln[22:26])
                except ValueError:
                    resi = -1
                if name == "CA":
                    ca[resi] = xyz
                if resi == R79_RESI and name in R79_ATOMS:
                    r79.append(xyz)
    return (ca,
            np.asarray(prot, float).reshape(-1, 3),
            np.asarray(r79, float).reshape(-1, 3),
            lig)


def kabsch(P, Q):
    """Rotation+translation mapping P onto Q (same convention as aggregate_models.py)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def min_dist(A, B):
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def pdb_path(seed, name):
    p = f"{SCRATCH}/seed_{seed}/boltz_out/boltz_results_*/predictions/{name}/{name}_model_0.pdb"
    import glob
    h = glob.glob(p)
    return h[0] if h else None


def geom_group(args):
    """Per-(name) cross-seed geometry: pose spread + clash fractions."""
    name, seeds = args
    parsed = {}
    for s in seeds:
        p = pdb_path(s, name)
        if not p:
            continue
        try:
            parsed[s] = parse_pdb(p)
        except Exception:
            continue
    out = dict(name=name, pose_rmsd_spread="", pose_rmsd_max="",
               head_clash_fraction="", r79_clash_fraction="", clash_seeds="",
               ligand_geom_bad_fraction="", ligand_geom_bad_seeds="",
               ligand_min_intra_dist="")
    if not parsed:
        return out

    # clash fractions (per-seed, no superposition needed)
    clash, r79c, bad = [], [], []
    geom_bad, geom_bad_seeds, intra_mins = [], [], []
    for s, (ca, prot, r79, lig) in parsed.items():
        L = np.asarray(list(lig.values()), float).reshape(-1, 3)
        dmin = min_dist(L, prot)
        c = int(np.isfinite(dmin) and dmin < CLASH_A)
        clash.append(c)
        if c:
            bad.append(str(s))
        dr = min_dist(L, r79)
        r79c.append(int(np.isfinite(dr) and dr < CLASH_A))

        # internal ligand sanity: no two heavy atoms closer than LIG_MIN_A
        if len(L) > 1:
            d = np.linalg.norm(L[:, None, :] - L[None, :, :], axis=2)
            d += np.eye(len(L)) * 1e9
            imin = float(d.min())
        else:
            imin = float("inf")
        intra_mins.append(imin)
        gb = int(imin < LIG_MIN_A)
        geom_bad.append(gb)
        if gb:
            geom_bad_seeds.append(str(s))

    out["head_clash_fraction"] = f"{np.mean(clash):.3f}"
    out["r79_clash_fraction"] = f"{np.mean(r79c):.3f}"
    out["clash_seeds"] = ";".join(bad)
    out["ligand_geom_bad_fraction"] = f"{np.mean(geom_bad):.3f}"
    out["ligand_geom_bad_seeds"] = ";".join(geom_bad_seeds)
    finite = [m for m in intra_mins if np.isfinite(m)]
    out["ligand_min_intra_dist"] = f"{min(finite):.3f}" if finite else ""
    out["_geom_ok_seeds"] = [s for s, gb in zip(parsed, geom_bad) if not gb]

    # pairwise ligand RMSD after chain-A CA superposition onto the lowest seed
    ss = sorted(parsed)
    ref_ca = parsed[ss[0]][0]
    frames = {}
    for s in ss:
        ca, prot, r79, lig = parsed[s]
        common = sorted(set(ca) & set(ref_ca))
        if len(common) < 20:
            continue
        P = np.array([ca[r] for r in common])
        Q = np.array([ref_ca[r] for r in common])
        R, t = kabsch(P, Q)
        frames[s] = {a: R @ np.asarray(x, float) + t for a, x in lig.items()}
    pair = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            a, b = frames.get(ss[i]), frames.get(ss[j])
            if not a or not b:
                continue
            shared = sorted(set(a) & set(b))
            if not shared:
                continue
            A = np.array([a[k] for k in shared]); B = np.array([b[k] for k in shared])
            pair.append((float(np.sqrt(((A - B) ** 2).sum(-1).mean())), ss[i], ss[j]))
    if pair:
        d = [p[0] for p in pair]
        out["pose_rmsd_spread"] = f"{np.mean(d):.3f}"
        out["pose_rmsd_max"] = f"{max(d):.3f}"
    out["_pairs"] = pair
    return out


# ---------------------------------------------------------------- aggregation
def is_num(v):
    if v is None or v == "":
        return False
    try:
        float(v); return True
    except ValueError:
        return False


def main():
    global SCRATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", default=f"{ROOT}/results/md_handoff_5seed/boltz_5seed_long.csv")
    ap.add_argument("--outdir", default=f"{ROOT}/results/md_handoff_5seed")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-geometry", action="store_true",
                    help="skip the PDB pose/clash pass (metrics-only, much faster)")
    ap.add_argument("--scratch", default=SCRATCH,
                    help="prediction tree holding seed_<S>/boltz_out (the constitutive "
                         "run lives in a SEPARATE tree, lca_lca3s_5seed_const)")
    ap.add_argument("--geom-scan",
                    default=f"{ROOT}/results/ligand_geometry_scan/ligand_geometry_long.csv",
                    help="ligand_geometry_scan.py output. A seed whose ligand is the wrong "
                         "stereoisomer (C3 inverted = the 3-beta epimer, NOT LCA/LCA-3-S) is "
                         "not the molecule being modelled, so it is excluded from the whole "
                         "consensus -- medians, majority vote and rep_seed alike.")
    ap.add_argument("--no-stereo-gate", action="store_true",
                    help="keep stereo-invalid seeds (pre-2026-10-01 behaviour: the shipped "
                         "137-design set was built this way and 106/274 of its staged "
                         "structures turned out to be inverted)")
    a = ap.parse_args()
    SCRATCH = a.scratch

    # (name, seed) -> scan row. Validity is settled here, without parsing a single PDB,
    # so the same valid-seed list can be handed to the geometry pass and the group loop.
    scan = {}
    if not a.no_stereo_gate:
        if not os.path.exists(a.geom_scan):
            sys.exit(f"--geom-scan not found: {a.geom_scan}  (--no-stereo-gate to skip)")
        for r in csv.DictReader(open(a.geom_scan)):
            if r.get("name") and r.get("seed"):
                scan[(r["name"], int(r["seed"]))] = r
        print(f"stereo gate ON: {len(scan)} scanned structures from {a.geom_scan}")
    else:
        print("stereo gate OFF (--no-stereo-gate)")

    rows = list(csv.DictReader(open(a.long)))
    if not rows:
        sys.exit(f"empty long table: {a.long}")
    cols = list(rows[0].keys())
    groups = {}
    for r in rows:
        groups.setdefault(r["name"], []).append(r)
    print(f"{len(rows)} long rows -> {len(groups)} (design x ligand) groups")

    # which columns are numeric (decided on the data, not hardcoded)
    numeric = set()
    for c in cols:
        if c in ("name", "ligand", "seed"):
            continue
        vals = [r[c] for r in rows[:4000] if r[c] != ""]
        if vals and all(is_num(v) for v in vals):
            numeric.add(c)
    categorical = [c for c in cols if c not in numeric and c not in ("name", "ligand", "seed")]
    print(f"  numeric cols: {len(numeric)}   categorical cols: {len(categorical)}")

    # ---- stereo gate: decide each group's usable seeds BEFORE anything reads a PDB.
    # gate[name] = (all_seeds, valid_seeds, qc dict). A group with zero valid seeds keeps
    # all its seeds so the row still exists and is auditable, but n_valid_seeds = 0 marks
    # the arm as unusable and downstream selection drops it.
    gate = {}
    for n, g in groups.items():
        all_s = sorted(int(r["seed"]) for r in g)
        if not scan:
            gate[n] = (all_s, all_s, {})
            continue
        # Ring pucker is the THIRD intra-ligand check and it is orthogonal to the other
        # two: 27 structures have correct stereo and no collapsed atoms but a boat/twist
        # steroid ring. It lives in the long table (pucker_shard.py), not the scan.
        ring_bad = {}
        for r in g:
            try:
                ring_bad[int(r["seed"])] = int(float(r.get("rings_distorted") or 0)) > 0
            except ValueError:
                ring_bad[int(r["seed"])] = False
        got = [(s, scan.get((n, s))) for s in all_s]
        scanned = [(s, r) for s, r in got if r is not None]
        stereo = [s for s, r in scanned if r.get("stereo_ok") == "1"]
        collapsed = [s for s, r in scanned if r.get("collapsed") == "1"]
        rings = [s for s in all_s if ring_bad.get(s)]
        valid = [s for s, r in scanned
                 if r.get("geom_ok") == "1" and not ring_bad.get(s)]
        inv = sorted({c for _, r in scanned for c in (r.get("inverted_centres") or "").split(";") if c})
        qc = {
            "n_seeds_total": len(all_s),
            "n_seeds_scanned": len(scanned),
            "n_valid_seeds": len(valid),
            "geom_valid_fraction": f"{len(valid) / len(scanned):.3f}" if scanned else "",
            "stereo_ok_fraction": f"{len(stereo) / len(scanned):.3f}" if scanned else "",
            "n_seeds_collapsed": len(collapsed),
            "n_seeds_ring_distorted": len(rings),
            "invalid_seeds": ";".join(str(s) for s in all_s if s not in valid),
            "inverted_centres": ";".join(inv),
            "stereo_gated": 1,
        }
        gate[n] = (all_s, valid or all_s, qc)
    if scan:
        nv = [len(v) for _, v, q in gate.values() if q.get("n_valid_seeds") == 0]
        print(f"  stereo gate: {len(gate) - len(nv)}/{len(gate)} groups have >=1 valid seed; "
              f"{len(nv)} have NONE (kept, flagged n_valid_seeds=0)")

    # cross-seed geometry -- over the VALID seeds only, so pose spread and clash
    # fractions describe the ensemble that actually gets used.
    geo = {}
    if not a.no_geometry:
        work = [(n, gate[n][1]) for n in groups]
        print(f"cross-seed pose/clash pass over {len(work)} groups on {a.jobs} procs ...")
        with Pool(a.jobs) as pool:
            for k, res in enumerate(pool.imap_unordered(geom_group, work, chunksize=16), 1):
                geo[res["name"]] = res
                if k % 500 == 0:
                    print(f"  {k}/{len(work)}")
        print("  done")

    full_rows, slim_rows, qc_rows = [], [], []
    for name, g_all in sorted(groups.items()):
        g_all = sorted(g_all, key=lambda r: int(r["seed"]))
        _all_seeds, valid_seeds, gate_qc = gate[name]
        # Everything below -- medians, majority mode, rep_seed -- sees valid seeds only.
        g = [r for r in g_all if int(r["seed"]) in set(valid_seeds)] or g_all
        seeds = [int(r["seed"]) for r in g]
        by_seed = {int(r["seed"]): r for r in g}

        modes = [r.get("binary_binding_mode", "") for r in g]
        mc = Counter(modes)
        maj, majn = mc.most_common(1)[0]
        maj_seeds = [s for s, r in by_seed.items() if r.get("binary_binding_mode", "") == maj]

        # rep_seed = medoid of the majority-mode cluster by ligand RMSD
        gg = geo.get(name, {})
        pairs = gg.get("_pairs") or []
        # A seed with collapsed ligand atoms must never be the staged structure:
        # it is unusable for MD regardless of how well it agrees with the others.
        # Prefer geometry-clean members of the majority cluster; only if EVERY
        # candidate is defective do we fall back (and the flag columns say so).
        geom_ok = gg.get("_geom_ok_seeds")
        cand = [s for s in maj_seeds if s in geom_ok] if geom_ok else list(maj_seeds)
        if not cand:
            cand = [s for s in seeds if geom_ok and s in geom_ok] or list(maj_seeds)

        rep = cand[0] if cand else seeds[0]
        if len(cand) > 1 and pairs:
            tot = {s: 0.0 for s in cand}
            cnt = {s: 0 for s in cand}
            for d, i, j in pairs:
                if i in tot and j in tot:
                    tot[i] += d; cnt[i] += 1
                    tot[j] += d; cnt[j] += 1
            scored = [(tot[s] / cnt[s], s) for s in cand if cnt[s]]
            if scored:
                rep = min(scored)[1]

        out = {"name": name, "ligand": g[0]["ligand"]}
        for c in cols:
            if c in ("name", "ligand", "seed"):
                continue
            if c in numeric:
                v = [float(r[c]) for r in g if is_num(r[c])]
                out[c] = f"{float(np.median(v)):.6g}" if v else ""
            else:
                vv = [r[c] for r in g if r[c] != ""]
                if not vv:
                    out[c] = ""
                else:
                    cnt2 = Counter(vv)
                    top = cnt2.most_common()
                    best = top[0][1]
                    tied = [v for v, n in top if n == best]
                    rv = by_seed[rep].get(c, "")
                    out[c] = rv if rv in tied else tied[0]

        qc = {
            "n_seeds": len(seeds),
            "rep_seed": rep,
            "seeds_present": ";".join(map(str, seeds)),
            "flipped_fraction": f"{mc.get('flipped', 0) / len(seeds):.3f}",
            "unknown_fraction": f"{mc.get('unknown', 0) / len(seeds):.3f}",
            "mode_majority": maj,
            "seed_agreement": f"{majn / len(seeds):.3f}",
            "pose_rmsd_spread": gg.get("pose_rmsd_spread", ""),
            "pose_rmsd_max": gg.get("pose_rmsd_max", ""),
            "head_clash_fraction": gg.get("head_clash_fraction", ""),
            "r79_clash_fraction": gg.get("r79_clash_fraction", ""),
            "ligand_geom_bad_fraction": gg.get("ligand_geom_bad_fraction", ""),
            "ligand_geom_bad_seeds": gg.get("ligand_geom_bad_seeds", ""),
            "ligand_min_intra_dist": gg.get("ligand_min_intra_dist", ""),
            **gate_qc,
        }
        agree = majn / len(seeds)
        spread = gg.get("pose_rmsd_spread", "")
        hcf = gg.get("head_clash_fraction", "")
        qc["pass_pose_consistency"] = int(
            agree >= AGREE_MIN
            and maj != "unknown"          # unanimous 'unknown' is consistent, not good
            and (spread == "" or float(spread) <= SPREAD_MAX)
            and (hcf == "" or float(hcf) == 0.0)
        )
        out.update(qc)
        full_rows.append(out)

        slim_rows.append({c: out.get(SLIM_SRC[c], "") if c not in ("name", "ligand")
                          else out[c] for c in SLIM_COLS})
        qc_rows.append(dict(name=name, ligand=out["ligand"], **qc))

    os.makedirs(a.outdir, exist_ok=True)
    full_cols = ["name", "ligand"] + [c for c in cols if c not in ("name", "ligand", "seed")] + QC_COLS
    paths = {
        "full": f"{a.outdir}/boltz_full_metrics_5seed.csv",
        "slim": f"{a.outdir}/boltz_plddt_iptm_geometry_pucker_5seed.csv",
        "qc":   f"{a.outdir}/seed_qc.csv",
    }
    for key, (p, rws, cs) in {
        "full": (paths["full"], full_rows, full_cols),
        "slim": (paths["slim"], slim_rows, SLIM_COLS),
        "qc":   (paths["qc"], qc_rows, ["name", "ligand"] + QC_COLS),
    }.items():
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cs, extrasaction="ignore")
            w.writeheader(); w.writerows(rws)
        print(f"wrote {p}  ({len(rws)} rows, {len(cs)} cols)")

    # ---- summary the decisions actually hinge on ----
    def frac(pred):
        n = sum(1 for r in full_rows if pred(r))
        return f"{n} / {len(full_rows)} ({100*n/len(full_rows):.1f}%)"
    print("\n--- seed-ensemble QC ---")
    print(f"unanimous binding mode        : {frac(lambda r: float(r['seed_agreement']) == 1.0)}")
    print(f"mode split (agreement <= 0.6) : {frac(lambda r: float(r['seed_agreement']) <= 0.6)}")
    print(f"any-seed head clash           : {frac(lambda r: r['head_clash_fraction'] not in ('',) and float(r['head_clash_fraction']) > 0)}")
    print(f"all-seed head clash           : {frac(lambda r: r['head_clash_fraction'] not in ('',) and float(r['head_clash_fraction']) == 1.0)}")
    print(f"any-seed R79 clash            : {frac(lambda r: r['r79_clash_fraction'] not in ('',) and float(r['r79_clash_fraction']) > 0)}")
    print(f"pass_pose_consistency         : {frac(lambda r: r['pass_pose_consistency'] == 1)}")
    sp = [float(r["pose_rmsd_spread"]) for r in full_rows if r["pose_rmsd_spread"] != ""]
    if sp:
        print(f"ligand pose RMSD spread (A)   : median {np.median(sp):.2f}  p90 {np.percentile(sp, 90):.2f}  max {max(sp):.2f}")
    # With the stereo gate on, n_seeds < 5 is the EXPECTED result of dropping inverted
    # seeds, not evidence of an unfinished run -- so judge completeness on what was
    # predicted (n_seeds_total) and report the gate's cost separately.
    nmiss = sum(1 for r in full_rows
                if int(r.get("n_seeds_total") or r["n_seeds"]) < len(SEEDS))
    if nmiss:
        print(f"\nWARNING: {nmiss} groups were PREDICTED with fewer than {len(SEEDS)} "
              f"seeds (run still finishing?)")
    if any(r.get("stereo_gated") for r in full_rows):
        gated = sum(1 for r in full_rows if int(r["n_seeds"]) < int(r["n_seeds_total"] or 0))
        none_ok = sum(1 for r in full_rows if int(r.get("n_valid_seeds") or 0) == 0)
        print(f"\nstereo gate: {gated}/{len(full_rows)} groups lost >=1 seed to inverted "
              f"stereochemistry; {none_ok} have NO valid seed (n_valid_seeds=0 -> "
              f"selection must drop these arms)")


if __name__ == "__main__":
    main()
