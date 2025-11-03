#!/usr/bin/env python3
import os, sys, argparse, subprocess

HERE = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = HERE
DATA_DIR = os.path.join(HERE, "..", "results")

def sh(*args):
    print("[RUN]", " ".join(args))
    subprocess.run(args, check=True)

def main():
    parser = argparse.ArgumentParser(description="CADD-Flow Unified Pipeline (Vina + Boltz)")
    parser.add_argument("--pdb_id", required=True)
    parser.add_argument("--vendor_mode", type=int, default=0)
    parser.add_argument("--do_masif_pre", action="store_true")
    parser.add_argument("--do_vina", action="store_true")
    parser.add_argument("--do_boltz", action="store_true")
    parser.add_argument("--do_rank_admet", action="store_true")
    parser.add_argument("--do_masif_post", action="store_true")
    args = parser.parse_args()

    pdb_id = args.pdb_id
    print(f"\n=== [PIPELINE START] {pdb_id} ===\n")

    # --- [1] Target Preparation ---
    print("\n--- [1] Target Preparation ---\n")
    sh(sys.executable, os.path.join(SCRIPTS_DIR, "prepare_target.py"), "--pdb_id", pdb_id)

    # --- [2] Ligand Vendor Set ---
    print("\n--- [2] Ligand Vendor Set ---\n")
    sh(sys.executable, os.path.join(SCRIPTS_DIR, "vendor_ligands.py"),
       "--pdb_id", pdb_id, "--vendor_mode", str(args.vendor_mode))

    # --- [3] MaSIF Preprocessing ---
    if args.do_masif_pre:
        print("\n--- [3] MaSIF Pre ---\n")
        sh(sys.executable, os.path.join(SCRIPTS_DIR, "../models/masif_torch.py"),
           "--pdb_id", pdb_id, "--mode", "pre")

    # --- [4] AutoDock Vina Docking ---
    if args.do_vina:
        print("\n--- [4] AutoDock Vina Docking ---\n")
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "dock_vina.py"), "--pdb_id", pdb_id]
        subprocess.run(cmd, check=True)

    # --- [5] Boltz Docking ---
    if args.do_boltz:
        print("\n--- [5] Boltz Docking ---\n")
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "dock_boltz.py"), "--pdb_id", pdb_id]
        subprocess.run(cmd, check=True)

    # --- [6] ADMET + Ranking ---
    if args.do_rank_admet:
        print("\n--- [6] ADMET Ranking ---\n")
        sh(sys.executable, os.path.join(SCRIPTS_DIR, "rank_and_admet.py"), "--pdb_id", pdb_id)

    # --- [7] MaSIF Postprocessing ---
    if args.do_masif_post:
        print("\n--- [7] MaSIF Post ---\n")
        sh(sys.executable, os.path.join(SCRIPTS_DIR, "../models/masif_torch.py"),
           "--pdb_id", pdb_id, "--mode", "post")

    print(f"\n=== [PIPELINE COMPLETE] {pdb_id} ===\n")

if __name__ == "__main__":
    main()
