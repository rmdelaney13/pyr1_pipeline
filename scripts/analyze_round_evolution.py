#!/usr/bin/env python3
"""
Analyze how Boltz2 scores evolve across expansion rounds.

Loads boltz_scored.csv for each ligand, parses which round each design
belongs to from the name encoding, and shows per-round statistics.

Usage (on Alpine):
    python scripts/analyze_round_evolution.py
    python scripts/analyze_round_evolution.py --expansion-root /path/to/expansion --ligands ca cdca
    python scripts/analyze_round_evolution.py --plot   # save figures
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_ROOT = "/scratch/alpine/ryde3462/expansion/ligandmpnn"
DEFAULT_LIGANDS = ["ca", "cdca", "dca"]

# Strategy H gates
GATES = {
    "plddt_ligand":   ("binary_plddt_ligand",   ">=", 0.65),
    "hbond_dist":     ("binary_hbond_distance",  "<=", 4.5),
    "latch_rmsd":     ("latch_rmsd",             "<=", 1.25),
}

KEY_METRICS = [
    "binary_plddt_ligand",
    "binary_plddt_pocket",
    "binary_hbond_distance",
    "binary_iptm",
    "binary_total_score",
]


def parse_round(name: str) -> int:
    """Extract expansion round from design name.

    Initial designs (e.g. 'ca_0001') → round 0.
    Expansion designs (e.g. 'ca_exp_r3_...') → round N (leftmost/latest).
    """
    m = re.search(r"_exp_r(\d+)_", name)
    return int(m.group(1)) if m else 0


def apply_gate(series, op, threshold):
    if op == ">=":
        return series >= threshold
    elif op == "<=":
        return series <= threshold
    return pd.Series(True, index=series.index)


def analyze_ligand(lig: str, root: Path, do_plot: bool = False, fig_dir: Path = None):
    scored = root / lig / "boltz_scored.csv"
    if not scored.exists():
        print(f"  {lig.upper()}: boltz_scored.csv not found, skipping")
        return None

    df = pd.read_csv(scored)
    if "name" not in df.columns:
        print(f"  {lig.upper()}: no 'name' column, skipping")
        return None

    df["round"] = df["name"].apply(parse_round)
    rounds = sorted(df["round"].unique())

    print(f"\n{'='*70}")
    print(f"  {lig.upper()}: {len(df)} total designs across rounds {rounds}")
    print(f"{'='*70}")

    # ── Per-round summary table ─────────────────────────────────────────
    rows = []
    for r in rounds:
        sub = df[df["round"] == r]
        row = {"round": r, "n_designs": len(sub)}

        for col in KEY_METRICS:
            if col in sub.columns:
                row[f"{col}_median"] = sub[col].median()
                row[f"{col}_mean"] = sub[col].mean()

        # Gate pass rates
        all_pass = pd.Series(True, index=sub.index)
        for gate_name, (col, op, thresh) in GATES.items():
            if col in sub.columns:
                mask = apply_gate(sub[col], op, thresh)
                row[f"gate_{gate_name}"] = f"{mask.sum()}/{len(sub)} ({100*mask.mean():.0f}%)"
                all_pass &= mask
            else:
                row[f"gate_{gate_name}"] = "N/A"
        row["all_gates_pass"] = f"{all_pass.sum()}/{len(sub)} ({100*all_pass.mean():.0f}%)"

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Print key metrics table
    print(f"\n  Per-round medians:")
    print(f"  {'Round':>5} {'N':>6}  {'pLDDT_lig':>10} {'pLDDT_pkt':>10} {'Hbond_d':>8} {'ipTM':>6} {'TotalScr':>9} {'Gates':>12}")
    print(f"  {'-'*5:>5} {'-'*6:>6}  {'-'*10:>10} {'-'*10:>10} {'-'*8:>8} {'-'*6:>6} {'-'*9:>9} {'-'*12:>12}")
    for _, row in summary.iterrows():
        r = int(row["round"])
        n = int(row["n_designs"])
        plddt_l = row.get("binary_plddt_ligand_median", float("nan"))
        plddt_p = row.get("binary_plddt_pocket_median", float("nan"))
        hbond = row.get("binary_hbond_distance_median", float("nan"))
        iptm = row.get("binary_iptm_median", float("nan"))
        total = row.get("binary_total_score_median", float("nan"))
        gates = row.get("all_gates_pass", "N/A")
        print(f"  {r:>5} {n:>6}  {plddt_l:>10.3f} {plddt_p:>10.3f} {hbond:>8.2f} {iptm:>6.3f} {total:>9.3f} {gates:>12}")

    # Print gate breakdown
    print(f"\n  Gate pass rates per round:")
    print(f"  {'Round':>5}  ", end="")
    for gn in GATES:
        print(f"  {gn:>18}", end="")
    print(f"  {'ALL':>18}")
    for _, row in summary.iterrows():
        r = int(row["round"])
        print(f"  {r:>5}  ", end="")
        for gn in GATES:
            print(f"  {row.get(f'gate_{gn}', 'N/A'):>18}", end="")
        print(f"  {row.get('all_gates_pass', 'N/A'):>18}")

    # ── Trend: are later rounds improving? ──────────────────────────────
    if len(rounds) > 1:
        print(f"\n  Trend (round 0 → round {rounds[-1]}):")
        r0 = df[df["round"] == 0]
        rlast = df[df["round"] == rounds[-1]]
        for col in KEY_METRICS:
            if col in df.columns:
                d = rlast[col].median() - r0[col].median()
                direction = "higher" if "hbond" not in col else "lower"
                good = (d > 0 and "hbond" not in col) or (d < 0 and "hbond" in col)
                symbol = "+" if good else "-"
                print(f"    {col:>30}: {d:+.3f} median shift [{symbol}]")

    # ── Optional plots ──────────────────────────────────────────────────
    if do_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f"{lig.upper()} — Score Evolution Across Rounds", fontsize=14, fontweight="bold")

            plot_metrics = KEY_METRICS + ["all_gates"]
            for idx, (ax, metric) in enumerate(zip(axes.flat, plot_metrics)):
                if metric == "all_gates":
                    # Bar chart of gate pass rates
                    rates = []
                    for r in rounds:
                        sub = df[df["round"] == r]
                        mask = pd.Series(True, index=sub.index)
                        for _, (col, op, thresh) in GATES.items():
                            if col in sub.columns:
                                mask &= apply_gate(sub[col], op, thresh)
                        rates.append(100 * mask.mean() if len(sub) > 0 else 0)
                    ax.bar(rounds, rates, color="steelblue", alpha=0.8)
                    ax.set_ylabel("Pass rate (%)")
                    ax.set_title("All Gates Pass Rate")
                    ax.set_xlabel("Round")
                else:
                    if metric not in df.columns:
                        ax.set_visible(False)
                        continue
                    # Box plot per round
                    data = [df.loc[df["round"] == r, metric].dropna().values for r in rounds]
                    bp = ax.boxplot(data, positions=rounds, widths=0.6, patch_artist=True,
                                   boxprops=dict(facecolor="lightsteelblue", alpha=0.7))
                    medians = [np.median(d) if len(d) > 0 else np.nan for d in data]
                    ax.plot(rounds, medians, "o-", color="darkblue", markersize=5, linewidth=1.5, label="median")

                    # Gate line if applicable
                    for gn, (gcol, gop, gthresh) in GATES.items():
                        if gcol == metric:
                            ax.axhline(gthresh, color="red", linestyle="--", alpha=0.7, label=f"gate={gthresh}")

                    short = metric.replace("binary_", "").replace("_", " ")
                    ax.set_title(short)
                    ax.set_xlabel("Round")
                    ax.legend(fontsize=8)

            # Hide unused axes
            for ax in axes.flat[len(plot_metrics):]:
                ax.set_visible(False)

            plt.tight_layout()
            if fig_dir:
                fig_dir.mkdir(parents=True, exist_ok=True)
                out = fig_dir / f"round_evolution_{lig}.png"
            else:
                out = root / lig / f"round_evolution_{lig}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Figure saved: {out}")
        except ImportError:
            print("  (matplotlib not available, skipping plots)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze score evolution across expansion rounds")
    parser.add_argument("--expansion-root", default=DEFAULT_ROOT)
    parser.add_argument("--ligands", nargs="+", default=DEFAULT_LIGANDS)
    parser.add_argument("--plot", action="store_true", help="Save box-plot figures")
    parser.add_argument("--fig-dir", default=None, help="Directory for figures (default: per-ligand dir)")
    args = parser.parse_args()

    root = Path(args.expansion_root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else None

    print("Round Evolution Analysis")
    print(f"Root: {root}")
    print(f"Ligands: {', '.join(args.ligands)}")

    for lig in args.ligands:
        analyze_ligand(lig, root, do_plot=args.plot, fig_dir=fig_dir)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
