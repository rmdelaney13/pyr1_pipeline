#!/usr/bin/env python3
"""Chemistry reference for LCA / LCA-3-S, read from the AF3 userCCD CIFs.

Single source of truth for every chemical fact this hand-off depends on:
the heavy-atom bond graph, the seven sp3 stereocentres, the three six-membered
steroid rings, the C1-C17 fused-ring core, and the symmetry-equivalent atom
mappings used for ligand RMSD.

These are the same CIFs the archived geometry gate used
(`scripts/ligand_geometry_scan.py` in the analysis repo), so the gate re-applied
to staged coordinates here is the gate that selected the seeds, not a lookalike.

Derived, never hard-coded:
  stereocentres  sp3 carbons with exactly 3 heavy neighbours and a non-degenerate
                 reference signed volume -> C3 C5 C8 C9 C14 C17 C20 (both ligands).
                 C10/C13 are quaternary and cannot epimerise.
  rings          minimum cycle basis -> 3 six-membered (A: C1-C2-C3-C4-C5-C10,
                 B: C5-C6-C7-C8-C9-C10, C: C8-C9-C11-C12-C13-C14) + 5-membered D.
  automorphisms  element-labelled graph automorphisms, bond order ignored.
                 LCA  |Aut|=2  : carboxylate {O4,O4A}
                 LCA3S |Aut|=12: carboxylate {O4,O4A} x sulfate {OS1,OS2,OS3}
                 O1B (C3 hydroxyl / sulfate ester bridge) is NOT equivalent to
                 either set, and every stereocentre and core carbon is fixed by
                 every automorphism, so symmetry minimisation cannot launder a
                 stereochemical difference into a low RMSD.
"""
import re
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

CIF_DIR = "/projects/ryde3462/lca_lca3s_modelling/ligands/af3_userccd"
LIGANDS = ("LCAM", "LCA3S")
# display names; the roster and the previous delivery both use the CIF ids
PRETTY = {"LCAM": "LCA", "LCA3S": "LCA-3-S"}
EXPECTED_HEAVY = {"LCAM": 27, "LCA3S": 31}
CORE = [f"C{i}" for i in range(1, 18)]          # C1-C17 fused-ring carbons
LIG_MIN_A = 1.15                                 # collapse threshold, archived gate
PUCKER_LO, PUCKER_HI = 30.0, 150.0               # distorted when LO < theta < HI


def _read_loop(lines, prefix):
    """Minimal mmCIF loop reader (same shape as the archived scan's)."""
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


def signed_volume(coords, centre, neighbours):
    """Signed volume of a centre's 3 heavy neighbours, sorted by atom name."""
    if centre not in coords or any(k not in coords for k in neighbours):
        return None
    a, b, d = (coords[k] - coords[centre] for k in neighbours)
    return float(np.dot(np.cross(a, b), d))


class Reference:
    """Everything the hand-off needs to know about one ligand."""

    def __init__(self, ligand):
        self.ligand = ligand
        lines = open(f"{CIF_DIR}/{ligand}.cif").read().splitlines()
        atoms = _read_loop(lines, "_chem_comp_atom")
        self.element = {r["atom_id"]: r["type_symbol"] for r in atoms}
        self.ideal = {r["atom_id"]: np.array([float(r["pdbx_model_Cartn_x_ideal"]),
                                              float(r["pdbx_model_Cartn_y_ideal"]),
                                              float(r["pdbx_model_Cartn_z_ideal"])])
                      for r in atoms}
        self.heavy = sorted(a for a, e in self.element.items() if e != "H")
        if len(self.heavy) != EXPECTED_HEAVY[ligand]:
            raise ValueError(f"{ligand}: {len(self.heavy)} heavy atoms, "
                             f"expected {EXPECTED_HEAVY[ligand]}")

        self.graph = nx.Graph()
        for a in self.heavy:
            self.graph.add_node(a, el=self.element[a])
        for r in _read_loop(lines, "_chem_comp_bond"):
            a, b = r["atom_id_1"], r["atom_id_2"]
            if a in self.graph and b in self.graph:
                self.graph.add_edge(a, b)

        self.neighbours = {a: sorted(self.graph[a]) for a in self.heavy}
        self.stereocentres = self._stereocentres()
        self.rings6 = self._rings6()
        self.core = [c for c in CORE if c in self.graph]
        self.automorphisms = self._automorphisms()
        self._validate()

    def _stereocentres(self):
        out = {}
        for a, nb in self.neighbours.items():
            if a.startswith("C") and len(nb) == 3:
                v = signed_volume(self.ideal, a, nb)
                if v is not None and abs(v) > 0.5:
                    out[a] = (nb, float(np.sign(v)))
        return out

    def _rings6(self):
        rings = nx.minimum_cycle_basis(self.graph)
        return [sorted(c) for c in rings if len(c) == 6]

    def _automorphisms(self):
        gm = GraphMatcher(self.graph, self.graph,
                          node_match=categorical_node_match("el", None))
        return [dict(m) for m in gm.isomorphisms_iter()]

    def _validate(self):
        assert len(self.stereocentres) == 7, \
            f"{self.ligand}: {len(self.stereocentres)} stereocentres, expected 7"
        assert len(self.rings6) == 3, \
            f"{self.ligand}: {len(self.rings6)} six-membered rings, expected 3"
        assert len(self.core) == 17, f"{self.ligand}: core has {len(self.core)}/17 carbons"
        # symmetry must not touch stereocentres, core carbons, or ring membership
        fixed = set(self.stereocentres) | set(self.core)
        for m in self.automorphisms:
            for a in fixed:
                assert m[a] == a, f"{self.ligand}: automorphism moves {a}"

    # ---- staged-coordinate checks -------------------------------------------

    def stereo(self, coords):
        """Per-centre inversion verdict against the reference signed volumes."""
        evaluable, inverted, per = True, [], {}
        for c, (nb, ref_sign) in sorted(self.stereocentres.items()):
            v = signed_volume(coords, c, nb)
            if v is None or abs(v) < 1e-6:
                evaluable = False
                per[c] = ""
                continue
            inv = int(np.sign(v) != ref_sign)
            per[c] = inv
            if inv:
                inverted.append(c)
        return evaluable, inverted, per

    def min_intra(self, coords):
        """Closest heavy-atom pair, and which pair it is."""
        names = [a for a in self.heavy if a in coords]
        if len(names) < 2:
            return float("nan"), ""
        X = np.array([coords[a] for a in names])
        D = np.linalg.norm(X[:, None] - X[None, :], axis=2) + np.eye(len(X)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        return float(D[i, j]), f"{names[i]}-{names[j]}"

    def pucker(self, coords):
        """Cremer-Pople theta for each reference six-membered ring.

        Rings come from the reference bond graph by atom name, so a distorted
        pose cannot hide a ring by breaking its distance-inferred bonds.
        """
        thetas, distorted = [], 0
        for ring in self.rings6:
            if any(a not in coords for a in ring):
                return None, None, None
            thetas.append(round(_cp_theta(self._ordered(ring, coords)), 1))
        for th in thetas:
            if PUCKER_LO < th < PUCKER_HI:
                distorted += 1
        worst = max(min(abs(t), abs(t - 180.0)) for t in thetas)
        return thetas, distorted, round(worst, 1)

    def _ordered(self, ring, coords):
        """Ring atoms in connectivity order (Cremer-Pople needs a cyclic walk)."""
        sub = self.graph.subgraph(ring)
        walk = [ring[0]]
        while len(walk) < len(ring):
            nxt = [n for n in sub[walk[-1]] if n not in walk]
            if not nxt:
                raise ValueError(f"{self.ligand}: ring {ring} is not a cycle")
            walk.append(nxt[0])
        return np.array([coords[a] for a in walk])


def _cp_theta(c):
    """Cremer-Pople theta (degrees) for a 6-ring, as in check_ligand_distortion.py."""
    N = len(c); c = c - c.mean(0); j = np.arange(N)
    R1 = (c * np.sin(2 * np.pi * j / N)[:, None]).sum(0)
    R2 = (c * np.cos(2 * np.pi * j / N)[:, None]).sum(0)
    n = np.cross(R1, R2); n /= np.linalg.norm(n); z = c @ n
    q2 = np.hypot(np.sqrt(2 / N) * (z * np.cos(4 * np.pi * j / N)).sum(),
                  -np.sqrt(2 / N) * (z * np.sin(4 * np.pi * j / N)).sum())
    q3 = np.sqrt(1 / N) * (z * (-1) ** j).sum()
    return float(np.degrees(np.arctan2(q2, q3)))


REFS = None


def refs():
    global REFS
    if REFS is None:
        REFS = {lg: Reference(lg) for lg in LIGANDS}
    return REFS


# ---- PDB reading ----------------------------------------------------------

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


def read_pdb(path_or_bytes):
    """Protein CA trace (resseq -> (aa, xyz)) and ligand heavy atoms (name -> xyz).

    Accepts a path or raw bytes, so a committed git blob and a staged file go
    through exactly the same parser.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        text = path_or_bytes.decode("utf-8", "replace")
    else:
        text = open(path_or_bytes, "r").read()
    ca, lig, resname = {}, {}, set()
    for L in text.splitlines():
        if L.startswith("ATOM") and L[12:16].strip() == "CA":
            aa = AA3.get(L[17:20].strip())
            if aa:
                ca[int(L[22:26])] = (aa, np.array([float(L[30:38]), float(L[38:46]),
                                                   float(L[46:54])]))
        elif L.startswith("HETATM") and L[76:78].strip().upper() != "H":
            lig[L[12:16].strip()] = np.array([float(L[30:38]), float(L[38:46]),
                                              float(L[46:54])])
            resname.add(L[17:20].strip())
    nums = sorted(ca)
    seq = "".join(ca[n][0] for n in nums)
    return dict(ca=ca, resnums=nums, sequence=seq, ligand=lig,
                ligand_resname=",".join(sorted(resname)))
