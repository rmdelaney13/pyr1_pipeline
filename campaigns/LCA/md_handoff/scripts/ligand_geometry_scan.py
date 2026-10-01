#!/usr/bin/env python3
"""Ligand-geometry scan over Boltz model_0 PDBs: stereochemistry + collapsed atoms.

One row per structure. Shards a FIXED manifest by line index (i % N == task), so
the manifest must not change between submissions of the same array.

Checks, all against the source-of-truth ligand definition (the AF3 userCCD, which
is built from the same RDKit mol as the Boltz pickle):

  stereo   For every sp3 carbon with 3 heavy neighbours in the reference (C3, C5,
           C8, C9, C14, C17, C20), the sign of the signed volume of its heavy
           neighbours (sorted by atom name) must match the reference. C10/C13 are
           quaternary and cannot epimerise. A mismatch is an inverted centre --
           e.g. C3 inverted = the 3-beta epimer, not LCA / LCA-3-S.
  collapse Minimum distance between any two ligand heavy atoms. Below LIG_MIN_A
           the ligand is physically impossible (observed: carboxylate O4/O4A at
           0.04 A).

Unlike the pipeline's oh_stereo check (NaN for all 2,852 rows while pass_oh_stereo
reported 1), this FAILS LOUDLY: a structure whose centres cannot be evaluated gets
geom_evaluable=0 and geom_ok=0, never a silent pass.

    python3 ligand_geometry_scan.py --manifest M --out shard.csv --shard 3/20
"""
import argparse, csv, os, re
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
CIF = {"LCAM": f"{ROOT}/ligands/af3_userccd/LCAM.cif",
       "LCA3S": f"{ROOT}/ligands/af3_userccd/LCA3S.cif"}
LIG_MIN_A = 1.15
NAME_RE = re.compile(r"/seed_(\d+)/boltz_out/boltz_results_[^/]+/predictions/"
                     r"(?P<name>(?P<sid>[^/]+?)__(?P<lig>LCAM|LCA3S)_binary)/")


def read_loop(lines, prefix):
    hdr, start = [], None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(prefix + "."):
            hdr.append(s.split(".")[-1]); start = i
        elif hdr and start is not None and not s.startswith("_"):
            start = i; break
    rows = []
    for l in lines[start:]:
        f = l.split()
        if len(f) < len(hdr):
            break
        rows.append({k: f[i].strip('"') for i, k in enumerate(hdr)})
    return rows


def reference(cif):
    lines = open(cif).read().splitlines()
    co = {r["atom_id"]: np.array([float(r["pdbx_model_Cartn_x_ideal"]),
                                  float(r["pdbx_model_Cartn_y_ideal"]),
                                  float(r["pdbx_model_Cartn_z_ideal"])])
          for r in read_loop(lines, "_chem_comp_atom")}
    nb = {}
    for r in read_loop(lines, "_chem_comp_bond"):
        a, b = r["atom_id_1"], r["atom_id_2"]
        nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    heavy_nb = {a: sorted(x for x in v if not x.startswith("H")) for a, v in nb.items()}
    centres = {}
    for a, n in heavy_nb.items():
        if a.startswith("C") and len(n) == 3:
            v = signed_volume(co, a, n)
            if v is not None and abs(v) > 0.5:
                centres[a] = (n, np.sign(v))
    return centres


def signed_volume(co, c, n):
    if c not in co or any(k not in co for k in n):
        return None
    a, b, d = (co[k] - co[c] for k in n)
    return float(np.dot(np.cross(a, b), d))


def ligand_coords(path):
    co = {}
    with open(path) as fh:
        for L in fh:
            if L.startswith("HETATM") and L[76:78].strip().upper() != "H":
                co[L[12:16].strip()] = np.array(
                    [float(L[30:38]), float(L[38:46]), float(L[46:54])])
    return co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/N")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))

    refs = {lg: reference(p) for lg, p in CIF.items()}
    centre_names = sorted(set().union(*[set(r) for r in refs.values()]))

    paths = [l.strip() for l in open(a.manifest) if l.strip()]
    mine = [p for k, p in enumerate(paths) if k % n == i]
    cols = (["path", "name", "sequence_id", "ligand", "seed", "geom_evaluable",
             "stereo_ok", "n_inverted", "inverted_centres", "lig_min_intra",
             "collapsed", "geom_ok"] + [f"inv_{c}" for c in centre_names])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in mine:
            m = NAME_RE.search(p)
            row = {"path": p}
            if not m:
                row.update(geom_evaluable=0, geom_ok=0)
                w.writerow(row); continue
            lg = m["lig"]
            row.update(name=m["name"], sequence_id=m["sid"], ligand=lg, seed=m.group(1))
            try:
                co = ligand_coords(p)
            except OSError:
                row.update(geom_evaluable=0, geom_ok=0)
                w.writerow(row); continue

            evaluable, inverted = True, []
            for c, (nbrs, ref_sign) in refs[lg].items():
                v = signed_volume(co, c, nbrs)
                if v is None or abs(v) < 1e-6:
                    evaluable = False
                    row[f"inv_{c}"] = ""
                    continue
                inv = int(np.sign(v) != ref_sign)
                row[f"inv_{c}"] = inv
                if inv:
                    inverted.append(c)

            if len(co) > 1:
                X = np.array(list(co.values()))
                D = np.linalg.norm(X[:, None] - X[None, :], axis=2) + np.eye(len(X)) * 1e9
                dmin = float(D.min())
            else:
                dmin, evaluable = float("nan"), False

            collapsed = int(np.isfinite(dmin) and dmin < LIG_MIN_A)
            stereo_ok = int(evaluable and not inverted)
            row.update(geom_evaluable=int(evaluable), stereo_ok=stereo_ok,
                       n_inverted=len(inverted), inverted_centres=";".join(inverted),
                       lig_min_intra=f"{dmin:.3f}", collapsed=collapsed,
                       geom_ok=int(stereo_ok and not collapsed))
            w.writerow(row)
    os.replace(tmp, a.out)
    print(f"shard {i}/{n}: {len(mine)} structures -> {a.out}")


if __name__ == "__main__":
    main()
