#!/usr/bin/env python3
"""Step 1: stage the 280 approved starting structures and verify them.

Byte-preserving copy of each selected Boltz model_0 PDB to its `delivery_pdb`
path, then, on the STAGED bytes:

  * SHA-256, byte count and source path recorded;
  * protein sequence checked against the roster's `protein_sequence`;
  * ligand identity checked -- heavy-atom names must be exactly the reference
    set (27 for LCA, 31 for LCA-3-S), including the sulfate group for LCA-3-S;
  * the archived ligand-geometry gate re-applied: 7 stereocentres, closest
    heavy-atom pair >= 1.15 A, and no six-membered ring with 30 < theta < 150.

Raw prediction files only. Nothing here is minimised or prepared for MD; any
such output belongs in a separate directory.

    python3 stage_starts.py --roster tables/approved_holo_arms.csv --pkg <dir>
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

import lca_chem
from lca_chem import refs, read_pdb


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True, type=Path)
    ap.add_argument("--pkg", required=True, type=Path)
    a = ap.parse_args()

    roster = pd.read_csv(a.roster)
    assert len(roster) == 280, f"roster has {len(roster)} rows, expected 280"
    assert not roster.duplicated(["sequence_id", "ligand"]).any()
    assert roster.delivery_pdb.nunique() == 280
    R = refs()

    rows, missing = [], []
    for r in roster.itertuples():
        src = Path(r.structure_path)
        dst = a.pkg / r.delivery_pdb
        rec = dict(sequence_id=r.sequence_id, ligand=r.ligand, role=r.role,
                   selected_seed=r.selected_seed, delivery_pdb=r.delivery_pdb,
                   source_path=str(src))
        if not src.exists():
            missing.append(rec | dict(reason="source file not found"))
            rows.append(rec | dict(staged=0))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)                      # bytes, no rewrite
        src_sha, dst_sha = sha256(src), sha256(dst)
        assert src_sha == dst_sha, f"copy not byte-preserving: {r.delivery_pdb}"

        st = read_pdb(dst)
        ref = R[r.ligand]
        names = set(st["ligand"])
        expected = set(ref.heavy)
        sulfate = {"S1", "OS1", "OS2", "OS3"}

        ok_eval, inverted, _ = ref.stereo(st["ligand"])
        dmin, dpair = ref.min_intra(st["ligand"])
        thetas, n_distorted, pucker_max = ref.pucker(st["ligand"])

        stereo_ok = int(ok_eval and not inverted)
        collapsed = int(dmin == dmin and dmin < lca_chem.LIG_MIN_A)
        pucker_ok = int(n_distorted == 0) if n_distorted is not None else 0
        geom_ok = int(stereo_ok and not collapsed and pucker_ok)

        rec.update(
            staged=1, bytes=dst.stat().st_size, sha256=dst_sha, source_sha256=src_sha,
            sequence_matches_roster=int(st["sequence"] == r.protein_sequence),
            n_residues=len(st["resnums"]),
            resnums_contiguous=int(st["resnums"] ==
                                   list(range(st["resnums"][0],
                                              st["resnums"][0] + len(st["resnums"])))),
            ligand_resname=st["ligand_resname"],
            n_ligand_heavy=len(st["ligand"]),
            ligand_atoms_match_reference=int(names == expected),
            ligand_missing_atoms=";".join(sorted(expected - names)),
            ligand_extra_atoms=";".join(sorted(names - expected)),
            sulfate_present=int(sulfate <= names) if r.ligand == "LCA3S" else "",
            geom_evaluable=int(ok_eval), stereo_ok=stereo_ok,
            n_inverted=len(inverted), inverted_centres=";".join(inverted),
            lig_min_intra=round(dmin, 3), lig_min_intra_pair=dpair,
            collapsed=collapsed, rings6=len(ref.rings6),
            ring_thetas=";".join(map(str, thetas)) if thetas else "",
            rings_distorted=n_distorted if n_distorted is not None else "",
            ring_pucker_max=pucker_max if pucker_max is not None else "",
            pucker_ok=pucker_ok, geom_ok=geom_ok,
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(a.pkg / "stage_verification.csv", index=False)

    n = len(out)
    staged = out[out.staged.eq(1)]
    print(f"staged {len(staged)}/{n} structures -> {a.pkg}/pdbs")
    if missing:
        print(f"MISSING SOURCES: {len(missing)}")
        for m in missing:
            print(f"  {m['sequence_id']} {m['ligand']}: {m['source_path']}")
    for col, label in [("sequence_matches_roster", "protein sequence == roster"),
                       ("ligand_atoms_match_reference", "ligand atom set == reference"),
                       ("resnums_contiguous", "residue numbering contiguous"),
                       ("stereo_ok", "stereochemistry"), ("pucker_ok", "ring pucker"),
                       ("geom_ok", "FULL GEOMETRY GATE")]:
        bad = staged[staged[col].ne(1)]
        print(f"  {label:34s} {len(staged)-len(bad):3d} pass, {len(bad):3d} fail")
        for b in bad.itertuples():
            print(f"      FAIL {b.sequence_id} {b.ligand} seed{b.selected_seed}")
    s3 = staged[staged.ligand.eq("LCA3S")]
    print(f"  {'sulfate present (LCA-3-S)':34s} "
          f"{int(s3.sulfate_present.eq(1).sum()):3d} pass, "
          f"{int(s3.sulfate_present.ne(1).sum()):3d} fail")
    print(f"  bytes staged: {staged.bytes.sum()/1e6:.1f} MB; "
          f"unique sha256: {staged.sha256.nunique()}/{len(staged)}")
    print(f"  min heavy-atom pair over all staged: {staged.lig_min_intra.min():.3f} A "
          f"(threshold {lca_chem.LIG_MIN_A})")


if __name__ == "__main__":
    main()
