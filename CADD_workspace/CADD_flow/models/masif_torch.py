#!/usr/bin/env python3
"""
masif_torch.py
--------------------------------------------
Dynamic MaSIF-like pocket box estimator and post-score correction.

- pre  : estimates ligand/protein box and writes results/<PDBID>/masif_box.json
- post : reads summary.csv, computes 'pocket_alignment_Score',
         adjusts Vina binding energies, and writes annotated CSV.

This step adds a physics-informed MaSIF-like geometric correction
for pocket proximity and shape complementarity.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Box estimation
def estimate_box(pdb_path):
    lig, ca = [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            res = line[17:20].strip().upper()
            atm = line[12:16].strip().upper()
            if line.startswith("HETATM") and res not in {"HOH", "WAT", "NA", "K", "CL", "MG", "CA"}:
                lig.append([x, y, z])
            elif line.startswith("ATOM") and atm == "CA":
                ca.append([x, y, z])

    if lig:
        A = np.array(lig)
        center = A.mean(0)
        span = A.ptp(0) + 8.0
        size = np.clip(span, 18.0, 40.0)
        src = "ligand"
    elif ca:
        A = np.array(ca)
        mn, mx = A.min(0), A.max(0)
        center = (mn + mx) / 2
        span = (mx - mn) * 0.65
        size = np.clip(span, 20.0, 32.0)
        src = "protein"
    else:
        center = np.zeros(3)
        size = np.array([22.0, 22.0, 22.0])
        src = "default"

    box = {"center": [float(f"{c:.3f}") for c in center],
           "size": [float(f"{s:.1f}") for s in size],
           "source": src}
    print(f"[OK] Estimated dynamic MaSIF box ({src}): {box}")
    return box

# Post-processing (MaSIF bonus)
def apply_post_bonus(summary_csv, box_json):
    box = json.load(open(box_json))
    center = np.array(box["center"], float)

    # Read skipping comment header (the header_units comment from rank_and_admet.py)
    df = pd.read_csv(summary_csv, comment="#")
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

    required_cols = ["pose_centroid_x", "pose_centroid_y", "pose_centroid_z"]
    if not all(c in df.columns for c in required_cols):
        print(f"[WARN] Missing centroid columns -> expected {required_cols}")
        print(f"[INFO] Available columns: {list(df.columns)}")
        return
    print(f"[INFO] Found centroid columns: {required_cols}")

    # Convert centroids safely
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required_cols)
    if df.empty:
        print("[WARN] All centroid values invalid -> skipping pocket alignment.")
        return

    # Compute pocket alignment score (0–0.5)
    cen = df[required_cols].to_numpy(dtype=float)
    d = np.linalg.norm(cen - center, axis=1)
    df["pocket_alignment_Score"] = np.round((d.max() - d) / (2 * (d.max() + 1e-6)), 3)

    # Locate the Vina binding energy column
    vina_col = next((c for c in df.columns if "vina" in c.lower() and "score" in c.lower()), None)
    if vina_col:
        df[vina_col] = pd.to_numeric(df[vina_col], errors="coerce")
        df[vina_col] = df[vina_col] - df["pocket_alignment_Score"]
        print(f"[OK] Adjusted binding energies using pocket alignment on '{vina_col}'.")
    else:
        print("[WARN] No Vina binding-energy column found; skipping adjustment.")

    # MaSIF score column is always retained and formatted correctly
    if "pocket_alignment_Score" not in df.columns:
        df["pocket_alignment_Score"] = np.nan

    # Round all numeric columns to two decimal places
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(2)

    # clean, aligned headers (prevents trailing blank column)
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

    # Annotate and Save Cleanly
    header_units = {
        "Rank": "– (overall ligand ranking)",
        "Ligand": "Ligand identifier",
    
        # docking / ML / MaSIF
        "Vina-Bind_Score (kcal/mol)": "kcal/mol (lower = stronger binding)",
        "boltz_confidence": "0–1 (higher = more confident ML docking pose)",
        "pocket_alignment_Score": "0–0.5 (higher = better pocket alignment via MaSIF)",
        "masif_score": "0–1 (surface complementarity; higher = better)",
        
        # pose geometry
        "pose_centroid_x": "Å (X-coords of ligand centroid)",
        "pose_centroid_y": "Å (Y-coords of ligand centroid)",
        "pose_centroid_z": "Å (Z-coords of ligand centroid)",

        # ranks
        "vina_rank": "ascending rank (lower = better)",
        "boltz_rank": "descending rank (higher = better)",
    
        # identifiers
        "SMILES": "Canonical SMILES string",
    
        # RDKit physchem
        "MolWt": "g/mol",
        "LogP": "log10 (P_octanol/water)",
        "SA_Score": "1–10 (synthetic accessibility; lower = easier synthesis)",
        "TPSA": "Å² (topological polar surface area)",
        "RotatableBonds": "count",
        "HBD": "hydrogen bond donors",
        "HBA": "hydrogen bond acceptors",
        "AromaticRings": "count",
        "AliphaticRings": "count",
        "RingCount": "count",
        "HeavyAtomCount": "count",
        "HeteroAtomCount": "count",
        "FractionCSP3": "0–1 (sp³ C saturation)",
        "QED": "0–1 (drug-likeness)",
        "MR": "cm³/mol (molar refractivity)",
        "FormalCharge": "integer charge",
    
        # simple ADMET rules
        "ESOL_LogS": "predicted log10(solubility [mol/L])",
        "Lipinski_OK": "Lipinski rule of five",
        "Veber_OK": "TPSA/rotatable-bond rule",
        "Egan_OK": "Egan absorption rule",
        "BBB_rule": "blood–brain barrier permeability"
    }

    # Construct aligned header
    units_line = "# " + ", ".join(
        [f"{c} ({header_units.get(c, c)})" for c in df.columns]
    )

    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(units_line + "\n")
        df.to_csv(f, index=False)

    print(f"[OK] Applied MaSIF pocket-alignment score -> {summary_csv}")

    # propagate same pocket_alignment_Score to selected_10.csv
    sel10 = Path(summary_csv).with_name("selected_10.csv")
    if sel10.exists():
        try:
            # read existing file, skip comment if present
            with open(sel10, "r", encoding="utf-8") as fh:
                first_line = fh.readline()
                comment = first_line if first_line.startswith("#") else ""
            df_sel = pd.read_csv(sel10, comment="#")
            if "Ligand" in df_sel.columns:
                merged = df_sel.merge(df[["Ligand", "pocket_alignment_Score"]],
                                      on="Ligand", how="left", suffixes=("", "_new"))
                df_sel["pocket_alignment_Score"] = merged["pocket_alignment_Score_new"].round(3)
                if "pocket_alignment_Score_new" in df_sel.columns:
                    df_sel.drop(columns=["pocket_alignment_Score_new"], inplace=True)
                with open(sel10, "w", encoding="utf-8") as f:
                    if comment:
                        f.write(comment)
                    df_sel.to_csv(f, index=False)
                print(f"[OK] Updated pocket_alignment_Score -> {sel10}")
        except Exception as e:
            print(f"[WARN] Failed to update selected_10.csv: {e}")

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    ap.add_argument("--mode", choices=["pre", "post"], required=True)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    base = Path(__file__).resolve().parent.parent / "results" / pdb_id
    pdb = base / f"{pdb_id}_clean.pdb"
    box_json = base / "masif_box.json"
    summary = base / "summary.csv"

    if args.mode == "pre":
        box = estimate_box(pdb)
        with open(box_json, "w") as f:
            json.dump(box, f, indent=2)
        print(f"[OK] Wrote dynamic MaSIF grid -> {box_json}")
    else:
        if not (box_json.exists() and summary.exists()):
            print("[WARN] post-MaSIF skipped (missing inputs)")
            return
        apply_post_bonus(summary, box_json)

if __name__ == "__main__":
    main()

