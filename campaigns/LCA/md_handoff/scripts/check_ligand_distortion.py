#!/usr/bin/env python3
"""Independent ligand-distortion check for the MD hand-off PDBs.

For every staged PDB, detect the ligand's 6-membered rings (topologically, so it
works regardless of Boltz atom naming) and test each for non-chair puckering via
Cremer-Pople theta. A steroid should show 3 chair rings (A/B/C); a ring is
DISTORTED when 30 < theta < 150 (boat/twist/flattened). Writes a per-structure
CSV and prints any offenders.
"""
import csv, glob, os, sys
import numpy as np
import networkx as nx

PIPE = "/projects/ryde3462/software/pyr1_pipeline/scripts"
sys.path.insert(0, PIPE)
from ligand_geometry import parse_pdb_coords, _build_bond_graph

PDBDIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/projects/ryde3462/lca_lca3s_modelling/results/md_handoff/pdbs"
OUT = os.path.join(os.path.dirname(PDBDIR), "ligand_distortion_check.csv")


def cp(c):
    N = len(c); c = c - c.mean(0); j = np.arange(N)
    R1 = (c * np.sin(2 * np.pi * j / N)[:, None]).sum(0)
    R2 = (c * np.cos(2 * np.pi * j / N)[:, None]).sum(0)
    n = np.cross(R1, R2); n /= np.linalg.norm(n); z = c @ n
    q2 = np.hypot(np.sqrt(2 / N) * (z * np.cos(4 * np.pi * j / N)).sum(),
                  -np.sqrt(2 / N) * (z * np.sin(4 * np.pi * j / N)).sum())
    q3 = np.sqrt(1 / N) * (z * (-1) ** j).sum()
    amp = np.hypot(q2, q3)
    return amp, np.degrees(np.arctan2(q2, q3))


def check(path):
    _, lig = parse_pdb_coords(path, ligand_chain='B')
    if not lig:
        return None
    adj = _build_bond_graph(lig)
    G = nx.Graph(); G.add_nodes_from(range(len(lig)))
    for i, nb in enumerate(adj):
        for j in nb:
            G.add_edge(i, j)
    n6 = ndist = 0; worst = 0.0; thetas = []
    for cyc in nx.cycle_basis(G):
        if len(cyc) == 6:
            n6 += 1
            _, th = cp(np.array([lig[k][0] for k in cyc]))
            thetas.append(round(th, 1))
            if 30 < th < 150:
                ndist += 1
            worst = max(worst, min(abs(th), abs(th - 180.0)))
    return n6, ndist, round(worst, 1), thetas


def main():
    rows = []
    for pdb in sorted(glob.glob(f"{PDBDIR}/*.pdb")):
        base = os.path.basename(pdb)[:-4]           # <sid>__<LIG>
        sid, lig = base.split("__")
        r = check(pdb)
        if r is None:
            rows.append(dict(sequence_id=sid, ligand=lig, rings6="", rings_distorted="ERR",
                             ring_pucker_max="", thetas="no-ligand"))
            continue
        n6, ndist, worst, thetas = r
        rows.append(dict(sequence_id=sid, ligand=lig, rings6=n6, rings_distorted=ndist,
                         ring_pucker_max=worst, thetas=";".join(map(str, thetas))))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence_id", "ligand", "rings6",
                                          "rings_distorted", "ring_pucker_max", "thetas"])
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    by_lig = {}
    offenders = []
    fewer_rings = []
    for r in rows:
        by_lig.setdefault(r["ligand"], [0, 0])
        by_lig[r["ligand"]][0] += 1
        if r["rings_distorted"] not in ("", "ERR") and int(r["rings_distorted"]) > 0:
            by_lig[r["ligand"]][1] += 1
            offenders.append(r)
        if r["rings6"] != "" and r["rings6"] != 3:
            fewer_rings.append(r)
    print(f"checked {n} structures -> {OUT}\n")
    print(f"{'ligand':8s} {'n':>5s} {'distorted':>10s}")
    for lig, (tot, dist) in sorted(by_lig.items()):
        print(f"{lig:8s} {tot:5d} {dist:10d}")
    print(f"\nrings6 != 3 (ring not detected / fused-ring parse issue): {len(fewer_rings)}")
    for r in fewer_rings:
        print(f"  {r['sequence_id']} {r['ligand']}: rings6={r['rings6']} thetas={r['thetas']}")
    print(f"\nDISTORTED ligands (rings_distorted>0): {len(offenders)}")
    for r in offenders:
        print(f"  {r['sequence_id']} {r['ligand']}: ndist={r['rings_distorted']} "
              f"pucker_max={r['ring_pucker_max']} thetas={r['thetas']}")


if __name__ == "__main__":
    main()
