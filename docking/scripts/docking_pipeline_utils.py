import math
import os
import time

import numpy as np


def _strip_inline_comment(value):
    """Strip inline # comments from config values."""
    if value is None:
        return value
    # Find # preceded by 2+ spaces or a tab (inline comment convention)
    idx = value.find('  #')
    if idx == -1:
        idx = value.find('\t#')
    if idx != -1:
        value = value[:idx]
    return value.strip()


def cfg_get(config, section, key, fallback=None):
    if section in config and key in config[section]:
        return _strip_inline_comment(config[section][key])
    if key in config["DEFAULT"]:
        return _strip_inline_comment(config["DEFAULT"][key])
    return fallback


def cfg_getbool(config, section, key, fallback=False):
    raw = cfg_get(config, section, key, None)
    if raw is None:
        return fallback
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _xyz_to_np(xyz):
    return np.array([float(xyz.x), float(xyz.y), float(xyz.z)], dtype=float)


def _angle_deg(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def atom_index_by_name(residue, atom_name):
    if not atom_name:
        return None
    target = atom_name.strip()
    for i in range(1, residue.natoms() + 1):
        if residue.atom_name(i).strip() == target:
            return i
    return None


def rename_water_resnames_to_tp3(pdb_path):
    if not os.path.exists(pdb_path):
        return
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    updated = []
    for line in lines:
        if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) >= 20:
            resname = line[17:20]
            if resname in {"WAT", "HOH"}:
                line = f"{line[:17]}TP3{line[20:]}"
        updated.append(line)
    with open(pdb_path, "w", encoding="utf-8") as handle:
        handle.writelines(updated)


def dump_pose_pdb(pose, output_path, rename_water=False):
    pose.dump_pdb(output_path)
    if rename_water:
        rename_water_resnames_to_tp3(output_path)


def ensure_table_ready(config_path, csv_file_name, wait_seconds=1800, poll_seconds=2):
    if os.path.exists(csv_file_name):
        return

    lock_file = f"{csv_file_name}.create_table.lock"
    have_lock = False
    fd = None
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        have_lock = True
    except FileExistsError:
        have_lock = False

    if have_lock:
        try:
            import create_table

            print(f"CSV {csv_file_name} not found. Running create_table.py from config.")
            create_table.main([config_path])
            if not os.path.exists(csv_file_name):
                raise RuntimeError(f"create_table completed but CSV still missing: {csv_file_name}")
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        return

    start = time.time()
    while (time.time() - start) < wait_seconds:
        if os.path.exists(csv_file_name):
            return
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Timed out waiting for CSV generation: {csv_file_name}. "
        f"Lock file observed: {lock_file}"
    )


def evaluate_hbond_geometry(
    pose,
    ligand_res_index,
    acceptor_atom_name=None,
    neighbor_atom_names=None,
    distance_min=1.5,
    distance_max=3.5,
    distance_ideal=2.8,
    donor_angle_min=100.0,
    acceptor_angle_min=90.0,
    quality_min=0.25,
):
    if neighbor_atom_names is None:
        neighbor_atom_names = []

    lig = pose.residue(ligand_res_index)

    acceptor_indices = []
    if acceptor_atom_name:
        idx = atom_index_by_name(lig, acceptor_atom_name)
        if idx is not None:
            acceptor_indices.append(idx)
    if not acceptor_indices:
        for i in range(1, lig.natoms() + 1):
            if lig.atom_type(i).element() in {"O", "N"}:
                acceptor_indices.append(i)

    if not acceptor_indices:
        return {
            "passed": False,
            "quality": 0.0,
            "distance": None,
            "donor_angle": None,
            "acceptor_angle": None,
            "acceptor_atom": None,
            "water_residue": None,
        }

    best = {
        "passed": False,
        "quality": 0.0,
        "distance": None,
        "donor_angle": None,
        "acceptor_angle": None,
        "acceptor_atom": None,
        "water_residue": None,
    }

    half_range = max(distance_ideal - distance_min, distance_max - distance_ideal, 1e-6)

    for acc_idx in acceptor_indices:
        acc_xyz = _xyz_to_np(lig.xyz(acc_idx))

        pref_vec = None
        neighbor_vecs = []
        for n_name in neighbor_atom_names:
            n_idx = atom_index_by_name(lig, n_name)
            if n_idx is None:
                continue
            n_xyz = _xyz_to_np(lig.xyz(n_idx))
            v = acc_xyz - n_xyz
            n = np.linalg.norm(v)
            if n > 1e-8:
                neighbor_vecs.append(v / n)
        if neighbor_vecs:
            pref_vec = np.sum(neighbor_vecs, axis=0)
            pref_n = np.linalg.norm(pref_vec)
            if pref_n <= 1e-8:
                pref_vec = None
            else:
                pref_vec = pref_vec / pref_n

        for resid in range(1, pose.total_residue() + 1):
            water = pose.residue(resid)
            if not water.is_water():
                continue

            o_idxs = [i for i in range(1, water.natoms() + 1) if water.atom_type(i).element() == "O"]
            h_idxs = [i for i in range(1, water.natoms() + 1) if water.atom_type(i).element() == "H"]

            for o_idx in o_idxs:
                o_xyz = _xyz_to_np(water.xyz(o_idx))
                dist = float(np.linalg.norm(o_xyz - acc_xyz))
                if dist < distance_min or dist > distance_max:
                    continue

                donor_angle = 180.0
                if h_idxs:
                    donor_angle = 0.0
                    for h_idx in h_idxs:
                        h_xyz = _xyz_to_np(water.xyz(h_idx))
                        angle = _angle_deg(o_xyz - h_xyz, acc_xyz - h_xyz)
                        if angle is not None and angle > donor_angle:
                            donor_angle = angle
                if donor_angle < donor_angle_min:
                    continue

                acceptor_angle = 180.0
                if pref_vec is not None:
                    angle = _angle_deg(pref_vec, o_xyz - acc_xyz)
                    if angle is None:
                        continue
                    acceptor_angle = angle
                    if acceptor_angle < acceptor_angle_min:
                        continue

                dist_score = max(0.0, 1.0 - abs(dist - distance_ideal) / half_range)
                donor_score = max(0.0, (donor_angle - donor_angle_min) / max(180.0 - donor_angle_min, 1e-6))
                if pref_vec is None:
                    acceptor_score = 1.0
                else:
                    acceptor_score = max(
                        0.0,
                        (acceptor_angle - acceptor_angle_min) / max(180.0 - acceptor_angle_min, 1e-6),
                    )
                quality = dist_score * donor_score * acceptor_score

                if quality > best["quality"]:
                    best = {
                        "passed": quality >= quality_min,
                        "quality": quality,
                        "distance": dist,
                        "donor_angle": donor_angle,
                        "acceptor_angle": acceptor_angle,
                        "acceptor_atom": lig.atom_name(acc_idx).strip(),
                        "water_residue": resid,
                    }

    best["passed"] = best["quality"] >= quality_min
    return best


def ligand_heavy_atom_coords(pose, ligand_res_index):
    lig = pose.residue(ligand_res_index)
    coords = []
    for i in range(1, lig.natoms() + 1):
        if lig.atom_type(i).element() != "H":
            coords.append(_xyz_to_np(lig.xyz(i)))
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(coords)


def ligand_rmsd(coords_a, coords_b):
    if coords_a.shape != coords_b.shape or coords_a.size == 0:
        return float("inf")
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
