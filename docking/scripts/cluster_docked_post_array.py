#!/usr/bin/env python
import argparse
import csv
import os
import random
import shutil
from configparser import ConfigParser

import numpy as np


def _cfg_clean(raw):
    if raw is None:
        return ""
    return str(raw).split("#", 1)[0].strip()


def _cfg_float(section, key, default):
    raw = section.get(key, None)
    cleaned = _cfg_clean(raw)
    if cleaned == "":
        cleaned = str(default)
    return float(cleaned)


def _is_heavy_atom(line):
    elem = line[76:78].strip()
    if not elem:
        name = line[12:16].strip()
        elem = "".join([c for c in name if c.isalpha()])[:1]
    return elem.upper() != "H"


def _parse_residue_key(line):
    chain = line[21:22]
    resseq = line[22:26]
    icode = line[26:27]
    resname = line[17:20]
    return (chain, resseq, icode, resname)


def _extract_ligand_coords_from_pdb(pdb_path):
    residue_heavy_coords = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip().upper()
            if resname in {"HOH", "WAT", "TP3"}:
                continue
            if not _is_heavy_atom(line):
                continue
            key = _parse_residue_key(line)
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            residue_heavy_coords.setdefault(key, []).append([x, y, z])

    if not residue_heavy_coords:
        return np.zeros((0, 3), dtype=float)

    best_key = max(residue_heavy_coords.keys(), key=lambda k: len(residue_heavy_coords[k]))
    return np.asarray(residue_heavy_coords[best_key], dtype=float)


def _ligand_rmsd(coords_a, coords_b):
    if coords_a.shape != coords_b.shape or coords_a.size == 0:
        return float("inf")
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _collect_scores(input_dir, pattern_prefix):
    score_map = {}
    for name in os.listdir(input_dir):
        if not name.startswith(pattern_prefix) or not name.endswith(".csv"):
            continue
        path = os.path.join(input_dir, name)
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    pdb_path = (row.get("output_pdb") or "").strip()
                    score_raw = (row.get("score") or "").strip()
                    if not pdb_path or not score_raw:
                        continue
                    try:
                        score = float(score_raw)
                    except ValueError:
                        continue
                    base = os.path.basename(pdb_path)
                    old = score_map.get(base)
                    if old is None or score < old:
                        score_map[base] = score
        except Exception:
            continue
    return score_map


def _cluster_candidates(candidates, rmsd_cutoff):
    clusters = []
    for item in candidates:
        assigned = None
        for idx, cluster in enumerate(clusters):
            if _ligand_rmsd(item["coords"], cluster["coords"]) <= rmsd_cutoff:
                assigned = idx
                break
        if assigned is None:
            clusters.append(dict(item))
            continue

        old = clusters[assigned]
        old_score = old.get("score")
        new_score = item.get("score")
        if old_score is None and new_score is None:
            continue
        if old_score is None and new_score is not None:
            clusters[assigned] = dict(item)
            continue
        if old_score is not None and new_score is None:
            continue
        if new_score < old_score:
            clusters[assigned] = dict(item)
    return clusters


def _convergence_analysis(candidates, rmsd_cutoff, final_count, n_shuffles=10, seed=42):
    """Compute cluster discovery curves to assess sampling convergence.

    Processes candidates in random order through greedy clustering and tracks
    the number of clusters found vs. poses processed. Repeated with multiple
    random shuffles to give a robust estimate.

    Returns dict with keys: curves, pct_at_quarters, verdict, mean_half_pct.
    """
    n = len(candidates)
    if n < 2 or final_count == 0:
        return {"curves": [], "pct_at_quarters": {}, "verdict": "SKIP", "mean_half_pct": 0.0}

    rng = random.Random(seed)
    all_curves = []

    for _ in range(n_shuffles):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        clusters = []
        curve = []
        for item in shuffled:
            assigned = False
            for cluster in clusters:
                if _ligand_rmsd(item["coords"], cluster["coords"]) <= rmsd_cutoff:
                    assigned = True
                    break
            if not assigned:
                clusters.append({"coords": item["coords"]})
            curve.append(len(clusters))
        all_curves.append(curve)

    # Average curve across shuffles
    avg_curve = [float(np.mean([c[i] for c in all_curves])) for i in range(n)]

    # Percentage of final clusters discovered at 25%, 50%, 75% of poses
    pct_at = {}
    for frac_label, frac in [("25%", 0.25), ("50%", 0.50), ("75%", 0.75)]:
        idx = max(0, int(n * frac) - 1)
        pct_at[frac_label] = avg_curve[idx] / final_count * 100

    mean_half_pct = pct_at["50%"]
    if mean_half_pct >= 90:
        verdict = "CONVERGED"
    elif mean_half_pct >= 75:
        verdict = "LIKELY_CONVERGED"
    else:
        verdict = "NOT_CONVERGED"

    return {
        "curves": all_curves,
        "avg_curve": avg_curve,
        "pct_at_quarters": pct_at,
        "verdict": verdict,
        "mean_half_pct": mean_half_pct,
    }


def _save_convergence_report(conv_result, final_count, n_candidates, output_dir):
    """Save convergence analysis CSV and print summary."""
    # Save discovery curve CSV
    curve_csv = os.path.join(output_dir, "convergence_curve.csv")
    avg_curve = conv_result.get("avg_curve", [])
    with open(curve_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["poses_processed", "avg_clusters_found", "pct_of_final"])
        for i, avg_val in enumerate(avg_curve, start=1):
            writer.writerow([i, f"{avg_val:.1f}", f"{avg_val / final_count * 100:.1f}"])

    # Print summary
    pct = conv_result["pct_at_quarters"]
    print(f"\n--- Convergence Analysis ---")
    print(f"Total poses: {n_candidates}, Final clusters: {final_count}")
    print(f"Clusters found at  25% of poses: {pct.get('25%', 0):.0f}%")
    print(f"Clusters found at  50% of poses: {pct.get('50%', 0):.0f}%")
    print(f"Clusters found at  75% of poses: {pct.get('75%', 0):.0f}%")
    print(f"Verdict: {conv_result['verdict']}")
    if conv_result["verdict"] == "NOT_CONVERGED":
        print(f"  -> Increase NumReps in config and re-dock for better coverage")
    elif conv_result["verdict"] == "LIKELY_CONVERGED":
        print(f"  -> Sampling is probably adequate, but more reps would improve confidence")
    else:
        print(f"  -> Sampling is sufficient; additional reps unlikely to find new clusters")
    print(f"Convergence curve: {curve_csv}")

    # Try to save a plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in conv_result.get("curves", []):
            ax.plot(range(1, len(curve) + 1), curve, color="steelblue", alpha=0.15, linewidth=0.8)
        ax.plot(range(1, len(avg_curve) + 1), avg_curve, color="navy", linewidth=2, label="Mean")
        ax.axhline(y=final_count, color="red", linestyle="--", linewidth=1, label=f"Final ({final_count})")
        ax.set_xlabel("Poses processed")
        ax.set_ylabel("Clusters found")
        ax.set_title(f"Cluster Discovery Curve — {conv_result['verdict']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(output_dir, "convergence_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Convergence plot: {plot_path}")
    except ImportError:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", nargs="?", default="config.txt")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rmsd-cutoff", type=float, default=None)
    parser.add_argument("--pdb-glob-prefix", default="a")
    parser.add_argument("--geometry-csv-prefix", default="hbond_geometry_summary")
    args = parser.parse_args(argv)

    config = ConfigParser()
    config.read(args.config_file)
    def_cfg = config["DEFAULT"]
    spec_cfg = config["grade_conformers"]

    input_dir = (args.input_dir or _cfg_clean(spec_cfg.get("OutputDir", ""))).strip()
    if not input_dir:
        scratch_root = _cfg_clean(def_cfg.get("SCRATCH_ROOT", ""))
        input_dir = os.path.join(scratch_root, "docked") if scratch_root else "output"
    input_dir = os.path.abspath(input_dir)

    output_dir = (args.output_dir or "").strip()
    if not output_dir:
        output_dir = os.path.join(input_dir, "clustered_final")
    output_dir = os.path.abspath(output_dir)

    rmsd_cutoff = args.rmsd_cutoff
    if rmsd_cutoff is None:
        rmsd_cutoff = _cfg_float(spec_cfg, "ClusterRMSDCutoff", 0.75)

    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted(
        [
            name
            for name in os.listdir(input_dir)
            if name.lower().endswith(".pdb") and name.startswith(args.pdb_glob_prefix)
        ]
    )
    if not pdb_files:
        print(f"No candidate PDB files found in {input_dir}")
        return

    score_map = _collect_scores(input_dir, args.geometry_csv_prefix)
    candidates = []
    skipped = 0
    for name in pdb_files:
        path = os.path.join(input_dir, name)
        coords = _extract_ligand_coords_from_pdb(path)
        if coords.size == 0:
            skipped += 1
            continue
        candidates.append(
            {
                "name": name,
                "path": path,
                "coords": coords,
                "score": score_map.get(name),
            }
        )

    clusters = _cluster_candidates(candidates, rmsd_cutoff)

    summary_rows = []
    for i, cluster in enumerate(clusters, start=1):
        out_name = f"cluster_{i:04d}_{cluster['name']}"
        out_path = os.path.join(output_dir, out_name)
        shutil.copyfile(cluster["path"], out_path)
        summary_rows.append(
            {
                "cluster_id": i,
                "source_pdb": cluster["name"],
                "source_path": cluster["path"],
                "score": "" if cluster.get("score") is None else cluster["score"],
                "output_pdb": out_path,
            }
        )

    summary_csv = os.path.join(output_dir, "cluster_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster_id", "source_pdb", "source_path", "score", "output_pdb"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Input directory: {input_dir}")
    print(f"Candidates scanned: {len(pdb_files)}")
    print(f"Candidates with ligand coords: {len(candidates)}")
    print(f"Skipped (no ligand coords): {skipped}")
    print(f"RMSD cutoff: {rmsd_cutoff}")
    print(f"Final clusters: {len(clusters)}")
    print(f"Clustered representatives: {output_dir}")
    print(f"Cluster summary CSV: {summary_csv}")

    # Convergence analysis
    if len(candidates) >= 5 and len(clusters) >= 2:
        conv_result = _convergence_analysis(candidates, rmsd_cutoff, len(clusters))
        _save_convergence_report(conv_result, len(clusters), len(candidates), output_dir)


if __name__ == "__main__":
    main()
