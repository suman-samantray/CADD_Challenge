#!/usr/bin/env python3
"""
vendor_ligands.py
-----------------
Generates ligand_set.smi for the CADD assignment workflow.

  vendor_mode=0 → random 20-ligand subset (quick test run)
  vendor_mode=1 → full purchasable vendor library (from smiles_database.json)
  vendor_mode=2 → dynamically fetched vendor library from webserver endpoint

Each output:  results/{PDB}/ligand_set.smi   (format: SMILES<TAB>NAME)
"""
import argparse, json, random
from pathlib import Path
import subprocess
import requests

DATA = Path(__file__).resolve().parent.parent / "results"


def sh(*args):
    print(f"[RUN] {' '.join(args)}")
    subprocess.run(args, check=False)


def fetch_dynamic_vendor_set():
    """
    Fetch purchasable Enamine (and similar) ligands using the ChEMBL public API.
    Returns a list of [name, smiles] pairs for use in ligand_set.smi.
    """
    import requests

    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_supplier__icontains=Enamine&limit=100"
    print(f"[INFO] Fetching Enamine ligands from ChEMBL: {url}")

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        ligands = []
        for i, m in enumerate(data.get("molecules", [])):
            smi = m.get("molecule_structures", {}).get("canonical_smiles")
            chembl_id = m.get("molecule_chembl_id")
            if smi:
                ligands.append([f"{chembl_id or 'enamine_'+str(i)}", smi])

        if not ligands:
            raise ValueError("No valid SMILES in ChEMBL response")

        print(f"[OK] Retrieved {len(ligands)} Enamine-like ligands from ChEMBL.")
        return ligands

    except Exception as e:
        print(f"[WARN] Could not fetch ChEMBL Enamine ligands: {e}")
        print("[INFO] Falling back to smiles_database.json instead.")
        ligBase = Path(__file__).resolve().parent.parent / "vendor" / "smiles_database.json"
        return json.load(open(ligBase))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    ap.add_argument("--vendor_mode", type=int, default=0)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    outdir = DATA / pdb_id
    outdir.mkdir(parents=True, exist_ok=True)
    lig_batch = outdir / "lig_batch"
    lig_batch.mkdir(exist_ok=True)

    ligand_set = outdir / "ligand_set.smi"
    ligBase = Path(__file__).resolve().parent.parent / "vendor" / "smiles_database.json"
    lig = json.load(open(ligBase))

    # choose vendor modes 
    if args.vendor_mode == 0:
        print("[INFO] Using random subset of 20 ligands (quick test mode).")
        random.shuffle(lig)
        lig = lig[:20]
    elif args.vendor_mode == 1:
        print(f"[INFO] Using full dataset of ({len(lig)} ligands).")
    elif args.vendor_mode == 2:
        lig = fetch_dynamic_vendor_set()
    else:
        raise ValueError("vendor_mode must be 0, 1, or 2")

    # write SMILES list and generate PDB/PDBQT 
    with open(ligand_set, "w") as f:
        for name, smi in lig:
            f.write(f"{smi}\t{name}\n")
            lig_name = name.replace(" ", "_")
            pdb_path = lig_batch / f"{lig_name}.pdb"
            pdbqt_path = lig_batch / f"{lig_name}.pdbqt"

            # Generate 3D coordinates and hydrogens
            sh("obabel", f"-:{smi}", "-opdb", "--gen3d", "-h", "-O", str(pdb_path))
            sh("obabel", "-ipdb", str(pdb_path), "-opdbqt", "-h", "-O", str(pdbqt_path))

    print(f"[OK] Wrote ligand set ({len(lig)} mols): {ligand_set}")
    print(f"[INFO] Ligand batch stored in: {lig_batch}\n")


if __name__ == "__main__":
    main()
