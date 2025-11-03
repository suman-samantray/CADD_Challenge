#!/usr/bin/env python3
"""
plot_rank_summary.py
--------------------------------------------
Independent plotting utility for CADD_flow results.

Usage:
    python plot_rank_summary.py --pdb_id 1PPB

Creates:
    vina_vs_boltz.png
    vina_hist.png
    admet_heatmap.png
    top10_bar.png
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def safe_read_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path, comment="#")
        except Exception:
            try:
                return pd.read_csv(path, sep="\t", comment="#")
            except Exception:
                pass
    return pd.DataFrame()


def plot_vina_vs_boltz(df, out_png):
    vina_col = next((c for c in df.columns if "vina" in c.lower() and "bind" in c.lower()), None)
    boltz_col = next((c for c in df.columns if "boltz" in c.lower() and "conf" in c.lower()), None)

    if vina_col and boltz_col:
        x = pd.to_numeric(df[vina_col], errors="coerce")
        y = pd.to_numeric(df[boltz_col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            print(f"[WARN] Not enough valid points for scatter ({mask.sum()})")
            return
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=x[mask], y=y[mask], s=50, alpha=0.8, edgecolor="k")
        plt.xlabel("Vina Score (kcal/mol, lower=better)")
        plt.ylabel("Boltz Confidence (0–1, higher=better)")
        plt.title("Vina vs Boltz Correlation")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] Scatter plot saved → {out_png}")
    else:
        print("[INFO] Boltz or Vina columns missing; skipping scatter plot.")


def plot_vina_hist(df, out_png):
    vina_col = next((c for c in df.columns if "vina" in c.lower() and "bind" in c.lower()), None)
    if vina_col is None:
        print("[WARN] No Vina score found for histogram.")
        return
    x = pd.to_numeric(df[vina_col], errors="coerce").dropna()
    if x.empty:
        print("[WARN] No valid Vina scores.")
        return
    plt.figure(figsize=(6,5))
    plt.hist(x, bins=15, color="steelblue", alpha=0.75, edgecolor="black")
    plt.xlabel("Vina Score (kcal/mol)")
    plt.ylabel("Count")
    plt.title("Distribution of Vina Binding Energies")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Histogram saved → {out_png}")


def plot_top10_bar(df, out_png):
    if "Ligand" not in df.columns:
        print("[WARN] No Ligand column for top-10 plot.")
        return
    top10 = df.head(10).copy()
    vina_col = next((c for c in top10.columns if "vina" in c.lower() and "bind" in c.lower()), None)
    boltz_col = next((c for c in top10.columns if "boltz" in c.lower() and "conf" in c.lower()), None)

    plt.figure(figsize=(8,5))
    bar_width = 0.35
    indices = np.arange(len(top10))

    if vina_col:
        plt.bar(indices, top10[vina_col], width=bar_width, label="Vina Score", color="salmon", alpha=0.8)
    if boltz_col:
        plt.bar(indices + bar_width, top10[boltz_col], width=bar_width, label="Boltz Confidence", color="skyblue", alpha=0.8)

    plt.xticks(indices + bar_width/2, top10["Ligand"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Top-10 Ligands: Vina vs Boltz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Top-10 bar plot saved → {out_png}")


def plot_admet_heatmap(df, out_png):
    props = [c for c in df.columns if c in
             ["MolWt", "LogP", "TPSA", "RotatableBonds", "HBD", "HBA", "QED", "SA_Score"]]
    if not props:
        print("[WARN] No ADMET-like numeric columns found for heatmap.")
        return
    df_sel = df[props].copy().apply(pd.to_numeric, errors="coerce")
    df_sel = df_sel.dropna(axis=0, how="any")
    if df_sel.empty:
        print("[WARN] Not enough numeric data for heatmap.")
        return
    plt.figure(figsize=(8,6))
    corr = df_sel.corr(method="spearman")
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar=True)
    plt.title("ADMET / PhysChem Correlation Map")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Heatmap saved → {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    root = Path(__file__).resolve().parent.parent
    base = root / "results" / pdb_id

    summary_csv = base / "summary.csv"
    sel10_csv = base / "selected_10.csv"

    df = safe_read_csv(summary_csv)
    if df.empty and sel10_csv.exists():
        df = safe_read_csv(sel10_csv)

    if df.empty:
        print(f"[ERR] No valid CSV found for {pdb_id}")
        return

    # Output files
    vina_vs_boltz_png = base / "vina_vs_boltz.png"
    vina_hist_png = base / "vina_hist.png"
    admet_heatmap_png = base / "admet_heatmap.png"
    top10_bar_png = base / "top10_bar.png"

    print(f"[INFO] Plotting results for {pdb_id}...")

    plot_vina_vs_boltz(df, vina_vs_boltz_png)
    plot_vina_hist(df, vina_hist_png)
    plot_admet_heatmap(df, admet_heatmap_png)
    plot_top10_bar(df, top10_bar_png)

    print("[OK] All plots generated successfully.")


if __name__ == "__main__":
    main()

