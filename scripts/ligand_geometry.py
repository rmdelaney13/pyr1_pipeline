#!/usr/bin/env python3
"""Ligand geometry validation: steroid ring distortion and stereochemistry check.

Compares predicted ligand geometry against a reference structure using:
  1. Protein CA superposition to put structures in the same frame
  2. Graph isomorphism for chemically correct atom matching
  3. Kabsch alignment on ring core atoms only (excludes flexible tail)
  4. Per-atom deviations and ring planarity metrics

Ring core = ring carbons + angular methyls + ring-bonded OHs.
Tail atoms (side chain + carboxyl) are excluded because they are flexible
and their conformation depends on pocket contacts, not steroid integrity.

Distortion flags:
  - max_dev > 1.0 A → ring atom significantly displaced (likely wrong chirality)
  - planarity_ratio < 0.5 → ring system too flat (collapsed steroid)

Usage as library:
    from ligand_geometry import LigandGeometryChecker
    checker = LigandGeometryChecker("reference.pdb")
    result = checker.check("prediction.pdb")
    # result = {'core_rmsd': 0.12, 'max_dev': 0.3, 'planarity_ratio': 0.95, ...}

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════
# PDB PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_pdb_coords(pdb_path, protein_chain='A', ligand_chain='B'):
    """Extract protein CA coords and ligand heavy-atom coords from PDB.

    Returns:
        ca_coords: dict {resnum: np.array([x,y,z])}
        ligand_atoms: list of (np.array([x,y,z]), element_symbol)
    """
    ca_coords = {}
    ligand_atoms = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resnum_str = line[22:26].strip()
            try:
                resnum = int(resnum_str)
            except ValueError:
                continue

            if ch == protein_chain:
                if atom_name == 'CA' and resnum not in ca_coords:
                    ca_coords[resnum] = np.array([x, y, z])
            elif ch == ligand_chain:
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
                if elem == 'H':
                    continue
                ligand_atoms.append((np.array([x, y, z]), elem))

    return ca_coords, ligand_atoms


# ═══════════════════════════════════════════════════════════════════
# ALIGNMENT
# ═══════════════════════════════════════════════════════════════════

def kabsch_rotation(P, Q):
    """Kabsch algorithm: optimal rotation from P onto Q (Nx3 arrays)."""
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def kabsch_align(coords_mobile, coords_target):
    """Full Kabsch alignment. Returns (R, center_mobile, center_target)."""
    cm = coords_mobile.mean(axis=0)
    ct = coords_target.mean(axis=0)
    m = coords_mobile - cm
    t = coords_target - ct
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    return R, cm, ct


def align_and_transform_ligand(ca_query, ligand_query, ca_ref):
    """Align query to reference by protein CAs, transform ligand coords.

    Returns list of (xyz_aligned, element) or None if alignment fails.
    """
    common = sorted(set(ca_query) & set(ca_ref))
    if len(common) < 10:
        return None

    mobile = np.array([ca_query[r] for r in common])
    fixed = np.array([ca_ref[r] for r in common])

    mobile_center = mobile.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    P = mobile - mobile_center
    Q = fixed - fixed_center
    R = kabsch_rotation(P, Q)

    aligned = []
    for coord, elem in ligand_query:
        new_coord = R @ (coord - mobile_center) + fixed_center
        aligned.append((new_coord, elem))

    return aligned


# ═══════════════════════════════════════════════════════════════════
# BOND GRAPH & RING DETECTION
# ═══════════════════════════════════════════════════════════════════

def _build_bond_graph(atoms, bond_cc=1.65, bond_co=1.55):
    """Build adjacency list from ligand atom coordinates using bond cutoffs."""
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(atoms[i][0] - atoms[j][0])
            ei, ej = atoms[i][1], atoms[j][1]
            if ei == 'C' and ej == 'C' and d < bond_cc:
                adj[i].add(j)
                adj[j].add(i)
            elif {ei, ej} == {'C', 'O'} and d < bond_co:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _find_ring_atoms(atoms, adj):
    """Find atoms in ring systems (size 4-7) using edge-deletion BFS."""
    n = len(atoms)
    ring_atoms = set()
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            visited = {u: None}
            q = deque([u])
            found = False
            while q and not found:
                curr = q.popleft()
                for nb in adj[curr]:
                    if curr == u and nb == v:
                        continue
                    if nb == v:
                        path = [v]
                        node = curr
                        while node is not None:
                            path.append(node)
                            node = visited[node]
                        if 4 <= len(path) <= 7:
                            ring_atoms.update(path)
                        found = True
                        break
                    if nb not in visited:
                        visited[nb] = curr
                        q.append(nb)
    return ring_atoms


def _classify_core_atoms(atoms, adj):
    """Classify ligand atoms into ring core vs tail.

    Core = ring carbons + angular methyls (degree-1 C on ring C) + ring-bonded OHs.
    Returns set of indices for ring core atoms.
    """
    ring_set = _find_ring_atoms(atoms, adj)
    core = set(ring_set)

    # Angular methyls (degree-1 C bonded to ring C)
    for i in range(len(atoms)):
        if i in ring_set:
            continue
        if atoms[i][1] == 'C' and len(adj[i]) == 1:
            nb = list(adj[i])[0]
            if nb in ring_set:
                core.add(i)

    # Ring OHs (O bonded to ring C)
    for i in range(len(atoms)):
        if atoms[i][1] == 'O':
            for nb in adj[i]:
                if nb in ring_set:
                    core.add(i)
                    break

    return core


# ═══════════════════════════════════════════════════════════════════
# GRAPH ISOMORPHISM ATOM MATCHING
# ═══════════════════════════════════════════════════════════════════

def _match_by_topology(ref_atoms, query_atoms, max_solutions=20):
    """Match ligand atoms by bond topology (graph isomorphism).

    Returns list of (ref_idx, query_idx) pairs, or None if no isomorphism.
    """
    n = len(ref_atoms)
    if n != len(query_atoms):
        return None

    adj_r = _build_bond_graph(ref_atoms)
    adj_q = _build_bond_graph(query_atoms)

    elem_r = [a[1] for a in ref_atoms]
    elem_q = [a[1] for a in query_atoms]
    deg_r = [len(a) for a in adj_r]
    deg_q = [len(a) for a in adj_q]

    candidates = []
    for i in range(n):
        candidates.append(
            [j for j in range(n)
             if elem_r[i] == elem_q[j] and deg_r[i] == deg_q[j]])

    if any(len(c) == 0 for c in candidates):
        return None

    order = sorted(range(n), key=lambda i: len(candidates[i]))

    best = [None, float('inf'), 0]
    mapping = {}
    reverse = {}

    def backtrack(depth):
        if best[2] >= max_solutions:
            return
        if depth == n:
            sd = sum(
                np.sum((ref_atoms[i][0] - query_atoms[mapping[i]][0]) ** 2)
                for i in range(n))
            rmsd = np.sqrt(sd / n)
            best[2] += 1
            if rmsd < best[1]:
                best[1] = rmsd
                best[0] = dict(mapping)
            return

        ri = order[depth]
        for qi in candidates[ri]:
            if qi in reverse:
                continue
            ok = True
            for nb_r in adj_r[ri]:
                if nb_r in mapping and mapping[nb_r] not in adj_q[qi]:
                    ok = False
                    break
            if ok:
                for nb_q in adj_q[qi]:
                    if nb_q in reverse and reverse[nb_q] not in adj_r[ri]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[ri] = qi
            reverse[qi] = ri
            backtrack(depth + 1)
            del mapping[ri]
            del reverse[qi]

    backtrack(0)

    if best[0] is None:
        return None
    return [(ri, best[0][ri]) for ri in range(n)]


# ═══════════════════════════════════════════════════════════════════
# PLANARITY
# ═══════════════════════════════════════════════════════════════════

def compute_ring_planarity(coords):
    """Max and mean deviation from best-fit plane (Angstroms)."""
    if len(coords) < 4:
        return 0.0, 0.0
    center = coords.mean(axis=0)
    centered = coords - center
    _, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    deviations = np.abs(centered @ normal)
    return float(deviations.max()), float(deviations.mean())


# ═══════════════════════════════════════════════════════════════════
# MAIN GEOMETRY CHECK
# ═══════════════════════════════════════════════════════════════════

def check_ligand_geometry(pdb_path, ref_ca_coords, ref_ligand_atoms,
                          ref_core_indices, ligand_chain='B'):
    """Check ligand geometry via graph isomorphism + ring-core-only Kabsch.

    Returns dict with metrics, or None if failed.
    """
    from scipy.optimize import linear_sum_assignment

    query_ca, query_lig = parse_pdb_coords(pdb_path, ligand_chain=ligand_chain)
    if not query_lig:
        return None

    aligned_lig = align_and_transform_ligand(query_ca, query_lig, ref_ca_coords)
    if aligned_lig is None:
        return None

    # Try graph isomorphism
    pairs = _match_by_topology(ref_ligand_atoms, aligned_lig)

    if pairs is not None:
        ref_indices = [ri for ri, _ in pairs]
        query_indices = [qi for _, qi in pairs]
        method = 'graph'
    else:
        # Fallback: per-element Hungarian
        from collections import defaultdict
        ref_by_elem = defaultdict(list)
        for idx, (coord, elem) in enumerate(ref_ligand_atoms):
            ref_by_elem[elem].append((idx, coord))
        query_by_elem = defaultdict(list)
        for idx, (coord, elem) in enumerate(aligned_lig):
            query_by_elem[elem].append((idx, coord))

        ref_indices = []
        query_indices = []
        for elem in sorted(ref_by_elem):
            if elem not in query_by_elem:
                continue
            rc = np.array([c for _, c in ref_by_elem[elem]])
            qc = np.array([c for _, c in query_by_elem[elem]])
            cost = np.zeros((len(rc), len(qc)))
            for i in range(len(rc)):
                for j in range(len(qc)):
                    cost[i, j] = np.linalg.norm(rc[i] - qc[j])
            row_ind, col_ind = linear_sum_assignment(cost)
            for ri_h, ci_h in zip(row_ind, col_ind):
                ref_indices.append(ref_by_elem[elem][ri_h][0])
                query_indices.append(query_by_elem[elem][ci_h][0])

        if not ref_indices:
            return None
        method = 'hungarian'

    # Separate core vs tail
    core_pairs = [(ri, qi) for ri, qi in zip(ref_indices, query_indices)
                  if ri in ref_core_indices]
    if len(core_pairs) < 4:
        return None

    ref_core = np.array([ref_ligand_atoms[ri][0] for ri, _ in core_pairs])
    query_core = np.array([aligned_lig[qi][0] for _, qi in core_pairs])
    core_elems = [ref_ligand_atoms[ri][1] for ri, _ in core_pairs]

    # Kabsch on ring core only
    R, cm, cr = kabsch_align(query_core, ref_core)
    aligned_core = (query_core - cm) @ R.T + cr

    per_atom_dev = np.sqrt(np.sum((aligned_core - ref_core) ** 2, axis=1))

    core_rmsd = float(np.sqrt(np.mean(per_atom_dev ** 2)))
    max_dev = float(per_atom_dev.max())
    max_dev_elem = core_elems[int(np.argmax(per_atom_dev))]

    c_mask = np.array([e == 'C' for e in core_elems])
    o_mask = np.array([e == 'O' for e in core_elems])

    c_rmsd = float(np.sqrt(np.mean(per_atom_dev[c_mask] ** 2))) \
        if c_mask.sum() > 0 else 0.0
    o_rmsd = float(np.sqrt(np.mean(per_atom_dev[o_mask] ** 2))) \
        if o_mask.sum() > 0 else 0.0
    max_o_dev = float(per_atom_dev[o_mask].max()) if o_mask.sum() > 0 else 0.0

    # Ring planarity
    ring_c_mask = np.array([e == 'C' and ri in ref_core_indices
                            for (ri, _), e in zip(core_pairs, core_elems)])
    if ring_c_mask.sum() >= 4:
        query_ring_c = aligned_core[ring_c_mask]
        ref_ring_c = ref_core[ring_c_mask]
        max_plan, _ = compute_ring_planarity(query_ring_c)
        ref_max_plan, _ = compute_ring_planarity(ref_ring_c)
        planarity_ratio = round(max_plan / ref_max_plan, 3) \
            if ref_max_plan > 0.01 else None
    else:
        planarity_ratio = None

    distorted = (max_dev > 1.5) or \
                (planarity_ratio is not None and planarity_ratio < 0.5)

    return {
        'core_rmsd': round(core_rmsd, 3),
        'c_rmsd': round(c_rmsd, 3),
        'o_rmsd': round(o_rmsd, 3),
        'max_dev': round(max_dev, 3),
        'max_dev_atom': max_dev_elem,
        'max_o_dev': round(max_o_dev, 3),
        'planarity_ratio': planarity_ratio,
        'n_matched': len(core_pairs),
        'match_method': method,
        'distorted': distorted,
    }


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE CLASS
# ═══════════════════════════════════════════════════════════════════

class LigandGeometryChecker:
    """Reusable checker: parses reference once, checks many predictions."""

    def __init__(self, ref_pdb_path, protein_chain='A', ligand_chain='B'):
        self.ref_ca, self.ref_lig = parse_pdb_coords(
            ref_pdb_path, protein_chain=protein_chain,
            ligand_chain=ligand_chain)
        if not self.ref_lig:
            raise ValueError(f"No ligand atoms found in {ref_pdb_path}")
        if len(self.ref_ca) < 10:
            raise ValueError(f"Only {len(self.ref_ca)} protein CAs in {ref_pdb_path}")

        ref_adj = _build_bond_graph(self.ref_lig)
        self.ref_core_indices = _classify_core_atoms(self.ref_lig, ref_adj)

        n_core = len(self.ref_core_indices)
        n_total = len(self.ref_lig)
        core_C = sum(1 for i in self.ref_core_indices if self.ref_lig[i][1] == 'C')
        core_O = sum(1 for i in self.ref_core_indices if self.ref_lig[i][1] == 'O')
        print(f"  Ligand geometry ref: {n_total} atoms, "
              f"ring core: {n_core} ({core_C}C, {core_O}O), "
              f"tail excluded: {n_total - n_core}")

    def check(self, pdb_path, ligand_chain='B'):
        """Check one prediction. Returns dict or None."""
        return check_ligand_geometry(
            pdb_path, self.ref_ca, self.ref_lig,
            self.ref_core_indices, ligand_chain=ligand_chain)


# ═══════════════════════════════════════════════════════════════════
# HAB1 CLASH CHECK
# ═══════════════════════════════════════════════════════════════════

class HAB1ClashChecker:
    """Check ligand-HAB1 clashes by aligning binary PDB to a ternary reference.

    Binary predictions don't include HAB1, so the ligand can float into the
    space needed for the Trp211 lock. This checker aligns each binary PDB
    to a ternary reference (which has HAB1 on chain C) via protein CAs,
    then measures min distance between ligand and HAB1 heavy atoms.
    """

    def __init__(self, ternary_ref_path, protein_chain='A', hab1_chain='C'):
        self.ref_ca = {}
        self.hab1_atoms = []  # list of (coord, resnum, atom_name)

        with open(ternary_ref_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]

                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue

                if ch == protein_chain and atom_name == 'CA':
                    if resnum not in self.ref_ca:
                        self.ref_ca[resnum] = np.array([x, y, z])
                elif ch == hab1_chain and elem != 'H':
                    self.hab1_atoms.append(
                        (np.array([x, y, z]), resnum, atom_name))

        if not self.hab1_atoms:
            raise ValueError(f"No HAB1 atoms (chain {hab1_chain}) in {ternary_ref_path}")
        if len(self.ref_ca) < 10:
            raise ValueError(f"Only {len(self.ref_ca)} protein CAs in {ternary_ref_path}")

        self.hab1_coords = np.array([a[0] for a in self.hab1_atoms])
        print(f"  HAB1 clash ref: {len(self.ref_ca)} PYR1 CAs, "
              f"{len(self.hab1_atoms)} HAB1 heavy atoms")

    def check(self, pdb_path, protein_chain='A', ligand_chain='B'):
        """Check one binary prediction for HAB1 clashes.

        Returns dict with min_dist, closest residues, or None if failed.
        """
        query_ca = {}
        lig_atoms = []

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue
                ch = line[21]
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]

                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue

                if ch == protein_chain and atom_name == 'CA':
                    if resnum not in query_ca:
                        query_ca[resnum] = np.array([x, y, z])
                elif ch == ligand_chain and elem != 'H':
                    lig_atoms.append((np.array([x, y, z]), atom_name))

        if not lig_atoms or not query_ca:
            return None

        # Align query to reference via protein CAs
        common = sorted(set(query_ca) & set(self.ref_ca))
        if len(common) < 10:
            return None

        mobile = np.array([query_ca[r] for r in common])
        fixed = np.array([self.ref_ca[r] for r in common])
        mc = mobile.mean(axis=0)
        fc = fixed.mean(axis=0)
        R = kabsch_rotation(mobile - mc, fixed - fc)

        # Transform ligand into reference frame
        lig_coords = np.array([R @ (a[0] - mc) + fc for a in lig_atoms])

        # Min distance from any ligand atom to any HAB1 atom
        diffs = lig_coords[:, None, :] - self.hab1_coords[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = float(dists[min_idx])

        return {
            'min_dist': round(min_dist, 3),
            'closest_hab1_res': self.hab1_atoms[min_idx[1]][1],
            'closest_hab1_atom': self.hab1_atoms[min_idx[1]][2],
            'closest_lig_atom': lig_atoms[min_idx[0]][1],
        }


# ═══════════════════════════════════════════════════════════════════
# LATCH LOOP RMSD
# ═══════════════════════════════════════════════════════════════════

def compute_latch_rmsd(pdb_path, ref_pdb_path, protein_chain='A',
                       latch_residues=None):
    """Compute backbone RMSD of latch loop after CA-alignment to reference.

    The latch loop (residues 114-118, centered on R116) must maintain its
    conformation for HAB1 Trp385 insertion. If Boltz2 deforms this loop
    to let sidechains interact with ligand, the design is incompatible
    with ternary complex formation.

    Returns dict with:
      latch_rmsd: backbone RMSD (CA, C, N, O) of latch residues after
                  full-chain CA alignment (Angstroms)
      latch_ca_rmsd: CA-only RMSD of latch residues (simpler metric)
    """
    if latch_residues is None:
        latch_residues = [114, 115, 116, 117, 118]

    result = {'latch_rmsd': None, 'latch_ca_rmsd': None}
    bb_atoms = {'CA', 'C', 'N', 'O'}

    def _parse_chain(pdb_file):
        ca_coords = {}
        latch_bb = {}
        with open(pdb_file) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                ch = line[21]
                if ch != protein_chain:
                    continue
                atom_name = line[12:16].strip()
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z])

                if atom_name == 'CA' and resnum not in ca_coords:
                    ca_coords[resnum] = coord
                if resnum in latch_residues and atom_name in bb_atoms:
                    key = (resnum, atom_name)
                    if key not in latch_bb:
                        latch_bb[key] = coord

        return ca_coords, latch_bb

    ref_ca, ref_latch = _parse_chain(ref_pdb_path)
    pred_ca, pred_latch = _parse_chain(pdb_path)

    if not ref_ca or not pred_ca:
        return result

    common_res = sorted(set(ref_ca) & set(pred_ca))
    if len(common_res) < 10:
        return result

    ref_ca_arr = np.array([ref_ca[r] for r in common_res])
    pred_ca_arr = np.array([pred_ca[r] for r in common_res])

    ref_center = ref_ca_arr.mean(axis=0)
    pred_center = pred_ca_arr.mean(axis=0)

    R = kabsch_rotation(pred_ca_arr - pred_center, ref_ca_arr - ref_center)

    common_latch = sorted(set(ref_latch) & set(pred_latch))
    if not common_latch:
        return result

    ref_latch_arr = np.array([ref_latch[k] for k in common_latch])
    pred_latch_arr = np.array([pred_latch[k] for k in common_latch])

    pred_latch_aligned = ((pred_latch_arr - pred_center) @ R.T) + ref_center

    diff = ref_latch_arr - pred_latch_aligned
    result['latch_rmsd'] = round(
        float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))), 3)

    ca_keys = [k for k in common_latch if k[1] == 'CA']
    if ca_keys:
        ref_ca_l = np.array([ref_latch[k] for k in ca_keys])
        pred_ca_l = np.array([pred_latch[k] for k in ca_keys])
        pred_ca_aligned = ((pred_ca_l - pred_center) @ R.T) + ref_center
        diff_ca = ref_ca_l - pred_ca_aligned
        result['latch_ca_rmsd'] = round(
            float(np.sqrt(np.mean(np.sum(diff_ca ** 2, axis=1)))), 3)

    return result
