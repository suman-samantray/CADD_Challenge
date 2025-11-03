#!/usr/bin/env python3
"""
prepare_target.py
-----------------
   Fetches PDB if needed
   Writes {PDB}_clean.pdb (filtered protein only)
   Extracts non-water HETATM as {PDB}_ligand.pdb (if any)
   Generates rigid receptor PDBQT via Open Babel (-xr)
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    MEEKO_AVAILABLE = True
except Exception:
    MEEKO_AVAILABLE = False


def run(cmd, check=True):
    """Run a shell command with printed output."""
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    root = Path(__file__).resolve().parent.parent
    outdir = root / "results" / pdb_id
    outdir.mkdir(parents=True, exist_ok=True)

    clean_pdb = outdir / f"{pdb_id}_clean.pdb"
    receptor_pdbqt = outdir / f"{pdb_id}_receptor.pdbqt"
    ligand_pdb = outdir / f"{pdb_id}_ligand.pdb"

    print(f"[OK] Downloading PDB: {pdb_id}")
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_text = urllib.request.urlopen(url).read().decode("utf-8")

    # Split into receptor and ligand components
    (outdir / f"{pdb_id}.pdb").write_text(pdb_text)
    with open(clean_pdb, "w") as rec_f, open(ligand_pdb, "w") as lig_f:
        for line in pdb_text.splitlines():
            if line.startswith("HETATM") and ("HOH" not in line):
                lig_f.write(line + "\n")
            elif line.startswith("ATOM"):
                rec_f.write(line + "\n")

    print(f"[OK] Wrote cleaned receptor: {clean_pdb}")
    print(f"[OK] Wrote co-ligand (if any): {ligand_pdb}")

    # Rigid receptor conversion
    if MEEKO_AVAILABLE:
        try:
            print("[INFO] Using Meeko for receptor conversion.")
            mol = Chem.MolFromPDBFile(str(clean_pdb), removeHs=False)
            if mol is None:
                raise ValueError("RDKit failed to read receptor PDB")
            mol = Chem.AddHs(mol)

            # Correct Meeko v0.5+ API for rigid receptor
            prep = MoleculePreparation()
            setups = prep.prepare(mol)        # list of MoleculeSetup objects
            writer = PDBQTWriterLegacy()
            pdbqt_str = writer.write_string(setups[0])

            receptor_pdbqt.write_text(pdbqt_str)
            print(f"[OK] Wrote rigid receptor PDBQT: {receptor_pdbqt}")

        except Exception as e:
            print(f"[WARN] Meeko failed: {e}\n[INFO] Falling back to Open Babel.")
            if shutil.which("obabel"):
                run([
                    "obabel", "-ipdb", str(clean_pdb),
                    "-opdbqt", "-xr",
                    "-O", str(receptor_pdbqt)
                ])
                print(f"[OK] Wrote rigid receptor PDBQT: {receptor_pdbqt}")
            else:
                raise RuntimeError("Neither Meeko nor Open Babel available.")
    else:
        print("[INFO] Meeko not available → using Open Babel fallback.")
        run([
            "obabel", "-ipdb", str(clean_pdb),
            "-opdbqt", "-xr",
            "-O", str(receptor_pdbqt)
        ])
        print(f"[OK] Wrote rigid receptor PDBQT: {receptor_pdbqt}")

    # Post-check for invalid tags
    txt = receptor_pdbqt.read_text()
    if "ROOT" in txt or "ENDROOT" in txt:
        print("[WARN] Receptor file contains ROOT tags; rewriting via Open Babel rigid mode.")
        run([
            "obabel", "-ipdb", str(clean_pdb),
            "-opdbqt", "-xr",
            "-O", str(receptor_pdbqt)
        ])
        print(f"[OK] Rewrote rigid receptor PDBQT (ROOT tags removed): {receptor_pdbqt}")

    print(f"{receptor_pdbqt} <-- static rigid receptor")
    print("[OK] Target preparation complete.\n")


if __name__ == "__main__":
    main()
