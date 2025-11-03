#!/usr/bin/env python3
"""
dock_vina.py
--------------------------------------------
AutoDock Vina docking for all ligands prepared
under results/<PDB_ID>/lig_batch/*.pdbqt using
the receptor pdbqt and the MaSIF box (if found).

Robust batch docking with dynamic MaSIF box
   Reads results/{PDB}/ligand_set.smi  (SMILES<TAB>NAME)
   Ensures results/{PDB}/lig_batch/ exists and contains NAME.pdbqt for each ligand
   Runs AutoDock Vina and writes outputs into results/{PDB}/vina_out/
   Writes results/{PDB}/vina_scores.csv with (name, vina_score, pose_centroid_x/y/z, out_pdbqt)
Resilient to RDKit/Meeko failures and Vina internal errors; continues on errors.
"""
import argparse, csv, json, os, sys, subprocess, shutil
from pathlib import Path

def sh(*args, check=True, capture=False):
    if capture:
        return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    else:
        return subprocess.run(args, check=check)

def have(cmd):
    return shutil.which(cmd) is not None

def meeko_smiles_to_pdbqt(smiles, out_pdbqt):
    # minimal meeko-based conversion if available
    try:
        from meeko import MoleculePreparation
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError("RDKit failed to parse SMILES")
        m = Chem.AddHs(m)
        mp = MoleculePreparation()
        mp.prepare(m)
        pdbqt_string, _ = mp.write_pdbqt_string()
        Path(out_pdbqt).write_text(pdbqt_string)
        return True
    except Exception as e:
        print(f"[WARN] Meeko failed: {e}; falling back to Open Babel.")
        return False

def obabel_smiles_to_pdbqt(smiles, out_pdbqt):
    # use Open Babel to make a 3D conformer and then pdbqt
    cmd = ["obabel", "-:", smiles, "--gen3d", "-opdbqt", "-O", str(out_pdbqt)]
    sh(*cmd)
    return True

def ensure_lig_batch(ligand_smi, lig_batch_dir):
    lig_batch_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with open(ligand_smi) as f:
        for line in f:
            if not line.strip(): continue
            smi, name = line.strip().split("\t")[:2]
            name = "".join(c if c.isalnum() or c in "_-." else "_" for c in name)
            out_pdbqt = lig_batch_dir / f"{name}.pdbqt"
            if not out_pdbqt.exists():
                ok = meeko_smiles_to_pdbqt(smi, out_pdbqt)
                if not ok:
                    obabel_smiles_to_pdbqt(smi, out_pdbqt)
            names.append((name, out_pdbqt))
    return names

def parse_masif_box(box_json):
    if not box_json.exists():
        # conservative default cube
        print("[WARN] masif_box.json missing → using default 22Å cube at origin.")
        return {"center_x":0.0,"center_y":0.0,"center_z":0.0,"size_x":22.0,"size_y":22.0,"size_z":22.0}
    box = json.load(open(box_json))
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    # Clip overly large boxes (Vina sometimes crashes with huge volumes)
    sx, sy, sz = [max(14.0, min(32.0, float(v))) for v in (sx, sy, sz)]
    return {"center_x":float(cx), "center_y":float(cy), "center_z":float(cz),
            "size_x":float(sx), "size_y":float(sy), "size_z":float(sz)}

def pose_centroid_from_pdbqt(pdbqt_path):
    try:
        xs= []; ys= []; zs= []
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
        if xs:
            return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    except Exception:
        pass
    return None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "results" / pdb_id
    receptor = data_dir / f"{pdb_id}_receptor.pdbqt"
    ligand_smi = data_dir / "ligand_set.smi"
    lig_dir = data_dir / "lig_batch"
    out_dir = data_dir / "vina_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not receptor.exists():
        raise FileNotFoundError(f"Missing receptor PDBQT: {receptor}")

    # Ensure lig_batch exists with pdbqt ligands
    names = ensure_lig_batch(ligand_smi, lig_dir)

    # Masif box
    box = parse_masif_box(data_dir / "masif_box.json")
    print(f"[INFO] Using MaSIF box: {box}")

    # Dock each ligand
    rows = []
    vina = shutil.which("vina") or shutil.which("qvina2") or shutil.which("vina_1.2.7")
    if not vina:
        raise RuntimeError("AutoDock Vina executable not found in PATH")

    for name, lig_pdbqt in names:
        out_pdbqt = out_dir / f"{name}_out.pdbqt"
        log_prefix = f"[VINA] {name}"
        cmd = [vina, "--receptor", str(receptor), "--ligand", str(lig_pdbqt),
               "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
               "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
               "--exhaustiveness", "8", "--num_modes", "9", "--out", str(out_pdbqt)]
        try:
            print(f"{log_prefix}: docking…")
            sh(*cmd)
        except subprocess.CalledProcessError as e:
            # Retry strategy: smaller box and/or --local_only to avoid tree.h crash
            print(f"{log_prefix}: first attempt failed ({e}); retrying with --local_only and reduced box.")
            try:
                rx, ry, rz =  max(14.0, box["size_x"]*0.8), max(14.0, box["size_y"]*0.8), max(14.0, box["size_z"]*0.8)
                cmd2 = [vina, "--receptor", str(receptor), "--ligand", str(lig_pdbqt),
                        "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
                        "--size_x", str(rx), "--size_y", str(ry), "--size_z", str(rz),
                        "--local_only", "--exhaustiveness", "8", "--num_modes", "1", "--out", str(out_pdbqt)]
                sh(*cmd2)
            except subprocess.CalledProcessError as e2:
                print(f"{log_prefix}: FAILED (skipping).")
                rows.append({"name": name, "vina_score": None, "pose_centroid_x": None, "pose_centroid_y": None, "pose_centroid_z": None, "out_pdbqt": ""})
                continue

        # Parse best score from output
        best = None
        try:
            with open(out_pdbqt) as f:
                for line in f:
                    if line.strip().startswith("REMARK VINA RESULT:"):
                        parts = line.split()
                        best = float(parts[3])
                        break
        except Exception:
            pass
        cx, cy, cz = pose_centroid_from_pdbqt(out_pdbqt)
        rows.append({"name": name, "vina_score": best, "pose_centroid_x": cx, "pose_centroid_y": cy, "pose_centroid_z": cz, "out_pdbqt": str(out_pdbqt)})

    # Write scores
    score_csv = data_dir / "vina_scores.csv"
    with open(score_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","vina_score","pose_centroid_x","pose_centroid_y","pose_centroid_z","out_pdbqt"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] Wrote docking summary: {score_csv}")

if __name__ == "__main__":
    main()
