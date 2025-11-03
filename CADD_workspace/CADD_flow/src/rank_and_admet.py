#!/usr/bin/env python3
"""
rank_and_admet.py
--------------------------------------------
Aggregate docking (Vina), ML (Boltz), and surface (MaSIF) results
with RDKit-derived ADMET-like descriptors.
Generates a ranked, unit-annotated summary table suitable for CADD reports.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rdmd, QED
from scipy.stats import spearmanr, kendalltau
from math import log
from rdkit.Chem import rdMolDescriptors


def sa_score(mol):
    """Approximate synthetic accessibility score."""
    if mol is None:
        return np.nan
    try:
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()
        n_atoms = mol.GetNumAtoms()
        size_penalty = 0.0 if n_atoms <= 10 else 0.1 * (n_atoms - 10)
        complexity_penalty = sum(log(v + 1) for v in bits.values())
        score = 2.5 + size_penalty + 0.05 * complexity_penalty
        return round(min(max(score, 1.0), 10.0), 3)
    except Exception:
        return np.nan


def load_smiles_map(smi_path: Path):
    """Read ligand_set.smi → {ligand_name: smiles}"""
    name2smi = {}
    for i, ln in enumerate(smi_path.read_text().splitlines()):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) == 1:
            name, smi = f"lig_{i:04d}", parts[0]
        else:
            if any(ch in parts[0] for ch in "[]()=#") or len(parts[0]) > 20:
                smi, name = parts[0], parts[-1]
            else:
                name, smi = parts[0], parts[-1]
        name2smi[name] = smi
    return name2smi


def compute_rdkit_props(smiles_list):
    """Compute molecular descriptors using RDKit safely."""
    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles_list]

    def safe(fn, default=None):
        out = []
        for m in mols:
            try:
                out.append(fn(m) if m is not None else default)
            except Exception:
                out.append(default)
        return out

    props = {}
    props["MolWt"] = safe(lambda m: Descriptors.MolWt(m))
    props["LogP"] = safe(lambda m: Crippen.MolLogP(m))
    props["TPSA"] = safe(lambda m: rdmd.CalcTPSA(m))
    props["RotatableBonds"] = safe(lambda m: rdmd.CalcNumRotatableBonds(m))
    props["HBD"] = safe(lambda m: rdmd.CalcNumHBD(m))
    props["HBA"] = safe(lambda m: rdmd.CalcNumHBA(m))
    props["AromaticRings"] = safe(lambda m: rdmd.CalcNumAromaticRings(m))
    props["FractionCSP3"] = safe(lambda m: rdmd.CalcFractionCSP3(m))
    props["QED"] = safe(lambda m: QED.qed(m))
    props["HeavyAtomCount"] = safe(lambda m: rdmd.CalcNumHeavyAtoms(m))
    props["HeteroAtomCount"] = safe(lambda m: rdmd.CalcNumHeteroatoms(m))
    props["RingCount"] = safe(lambda m: rdmd.CalcNumRings(m))
    props["AliphaticRings"] = safe(lambda m: rdmd.CalcNumAliphaticRings(m))
    props["MR"] = safe(lambda m: Descriptors.MolMR(m))
    props["SA_Score"] = safe(lambda m: sa_score(m))
    props["FormalCharge"] = safe(lambda m: Chem.GetFormalCharge(m))
    return props

def _read_csv_robust(path: Path):
    """Try reading with autodetected delimiter."""
    if not path.exists():
        return pd.DataFrame()
    for sep in (None, ",", "\t", ";"):
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
    return pd.DataFrame()



# Main pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    root = Path(__file__).resolve().parent.parent
    base = root / "results" / pdb_id

    vina_csv = base / "vina_scores.csv"
    boltz_csv = base / "boltz_results.csv"
    masif_csv = base / "masif_scores.csv"
    smi_file = base / "ligand_set.smi"
    out_csv = base / "summary.csv"
    out_sel10 = base / "selected_10.csv"
    out_corr = base / "rank_correlation.dat"

    # Load & normalize
    vina = _read_csv_robust(vina_csv)
    if not vina.empty:
        vina = vina.rename(columns={"name": "Ligand", "vina_score": "Vina-Bind_Score (kcal/mol)"})
    else:
        vina = pd.DataFrame(columns=["Ligand", "Vina-Bind_Score (kcal/mol)"])

    boltz = _read_csv_robust(boltz_csv)
    if not boltz.empty and "name" in boltz.columns:
        boltz = boltz.rename(columns={"name": "Ligand"})
    if boltz.empty:
        boltz = pd.DataFrame(columns=["Ligand", "boltz_confidence"])

    masif = _read_csv_robust(masif_csv)
    if not masif.empty and "Ligand" not in masif.columns and "name" in masif.columns:
        masif = masif.rename(columns={"name": "Ligand"})
    if not masif.empty and "masif_score" not in masif.columns:
        masif.columns = ["Ligand", "masif_score"]

    # Merge
    df = pd.merge(vina, boltz, on="Ligand", how="outer")
    if not masif.empty:
        df = pd.merge(df, masif, on="Ligand", how="left")

    # pocket_alignment_Score always exists for MaSIF post-step
    if "pocket_alignment_Score" not in df.columns:
        df["pocket_alignment_Score"] = ""

    # Add SMILES & RDKit props
    name2smi = load_smiles_map(smi_file) if smi_file.exists() else {}
    df["SMILES"] = df["Ligand"].map(name2smi).fillna("")
    props = compute_rdkit_props(df["SMILES"].tolist())
    for k, v in props.items():
        df[k] = v

    # Rule-based ADMET descriptors
    df["ESOL_LogS"] = -0.01 * df["MolWt"] + 0.54
    df["Lipinski_OK"] = (
        (df["MolWt"] <= 500)
        & (df["LogP"] <= 5)
        & (df["HBD"] <= 5)
        & (df["HBA"] <= 10)
    )
    df["Veber_OK"] = (df["TPSA"] <= 140) & (df["RotatableBonds"] <= 10)
    df["Egan_OK"] = (df["TPSA"] <= 131) & (df["LogP"] <= 5.88)
    df["BBB_rule"] = (
        (df["TPSA"] < 90)
        & (df["MolWt"] < 450)
        & (df["LogP"].between(0, 6, inclusive="both"))
    )

    # Rankings
    df["vina_rank"] = df["Vina-Bind_Score (kcal/mol)"].rank(ascending=True, method="min")
    df["boltz_rank"] = df["boltz_confidence"].rank(ascending=False, method="min")

    sort_keys = []
    if "Vina-Bind_Score (kcal/mol)" in df.columns:
        sort_keys.append(("Vina-Bind_Score (kcal/mol)", True))
    if "boltz_confidence" in df.columns:
        sort_keys.append(("boltz_confidence", False))
    if "masif_score" in df.columns:
        sort_keys.append(("masif_score", False))

    if sort_keys:
        df = df.sort_values(
            by=[k for k, _ in sort_keys],
            ascending=[asc for _, asc in sort_keys]
        ).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))

    # Clean formatting
    if "out_pdbqt" in df.columns:
        df = df.drop(columns=["out_pdbqt"])
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].round(2)

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
        "ESOL_LogS": "predicted log10(solubility [mol/L]) ",
        "Lipinski_OK": "Lipinski rule of five",
        "Veber_OK": "TPSA/rotatable-bond rule",
        "Egan_OK": "Egan absorption rule",
        "BBB_rule": "blood–brain barrier permeability"
    }

    # Clean up any old parentheses or spacing in headers
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

    # boolean formatting (TRUE/FALSE/blank)
    for col in ["Lipinski_OK", "Veber_OK", "Egan_OK", "BBB_rule"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "TRUE" if x is True else ("FALSE" if x is False else ""))

    # Prepare readable header comment with units
    units_line = "# " + ", ".join(
        [f"{c} ({header_units.get(c, c)})" for c in df.columns]
    )
    
    # Ensure all column names are stripped clean of old units before saving
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(units_line + "\n")
        df.to_csv(f, index=False)

    print(f"[OK] Physically interpretable ranked summary saved → {out_csv}")

    # Top-10
    if len(df) > 0:
        df.head(10).to_csv(out_sel10, index=False)
        print(f"[OK] Top-10 saved → {out_sel10}")

    # Correlation stats
    vina_col = next((c for c in df.columns if "vina" in c.lower() and "bind" in c.lower()), None)
    boltz_col = next((c for c in df.columns if "boltz" in c.lower() and "conf" in c.lower()), None)

    if vina_col and boltz_col:
        mask = df[vina_col].notna() & df[boltz_col].notna()
        if mask.sum() >= 3:
            rho, p_rho = spearmanr(df.loc[mask, vina_col], df.loc[mask, boltz_col])
            tau, p_tau = kendalltau(df.loc[mask, vina_col], df.loc[mask, boltz_col])
            with open(out_corr, "w") as f:
                f.write(f"Spearman_rho = {rho:.3f} (p={p_rho:.3g})\n")
                f.write(f"Kendall_tau  = {tau:.3f} (p={p_tau:.3g})\n")
                f.write(f"N = {mask.sum()}\n")
            print(f"[OK] Correlation statistics saved → {out_corr}")
        else:
            print("[INFO] Not enough valid pairs for correlation statistics.")
    else:
        print(f"[WARN] Could not find Vina or Boltz columns in {list(df.columns)}")

    print("[OK] Rank aggregation and ADMET annotation complete.")

if __name__ == "__main__":
    main()


