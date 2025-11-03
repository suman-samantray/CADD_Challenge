#!/usr/bin/env python3
"""
dock_boltz.py — Robust Boltz docking with RDKit SMILES validation,
dual-path confidence JSON search, and Apple→HPC offload support.
---------------------------------------------------------------
✅ Compatible with boltz>=2.2.1 (confidence_*.json under predictions/)
"""

import argparse, json, subprocess, shutil, yaml, platform, os
from pathlib import Path
import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results"
REMOTE = "sumansam@perlmutter.nersc.gov"
REMOTE_ENV = "source ~/.bashrc && conda activate caddenv"

# ---------------------------------------------------------------------
def extract_sequence_from_pdb(pdb_path):
    seq, seen = [], set()
    resmap = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
        "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
        "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
        "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
    }
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                res, resi = line[17:20].strip(), line[22:27].strip()
                if resi not in seen and res in resmap:
                    seq.append(resmap[res]); seen.add(resi)
    return "".join(seq)

# ---------------------------------------------------------------------
def patch_diffusion_steps(model_cfg_path, steps=50):
    if not model_cfg_path.exists():
        return
    try:
        with open(model_cfg_path) as f:
            data = yaml.safe_load(f)
        if "num_diffusion_steps" in data and data["num_diffusion_steps"] != steps:
            data["num_diffusion_steps"] = steps
            with open(model_cfg_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"[INFO] num_diffusion_steps set → {steps} in {model_cfg_path.name}")
    except Exception as e:
        print(f"[WARN] Could not patch diffusion steps: {e}")

# ---------------------------------------------------------------------
def run_local(cmd):
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    print(f"[RUN-local] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def run_remote(yaml_relpath):
    remote_cmd = (
        f"{REMOTE_ENV} && cd {yaml_relpath.parent.parent} && "
        f"boltz predict {yaml_relpath} --no_msa_server "
        f"--accelerator gpu"
    )
    print(f"[SSH] Offloading Boltz inference to {REMOTE}")
    subprocess.run(["ssh", REMOTE, remote_cmd], check=True)

# ---------------------------------------------------------------------
def safe_name_from_smiles(smi):
    """Generate filesystem-safe name from SMILES."""
    return smi.replace("=", "_").replace("(", "_").replace(")", "_")\
              .replace("/", "_").replace("\\", "_").replace(":", "_")[:24]

# ---------------------------------------------------------------------
def parse_conf_json(name, out_dir):
    """Search multiple locations for Boltz confidence JSON."""
    aff = conf = ""
    json_path = None
    source = None

    # Sanitized fallback alias (when Boltz folder uses SMILES-like names)
    safe_alias = safe_name_from_smiles(name)

    # (1) Pipeline layout (results/.../boltz_results/<ligand>/predictions/*)
    for cand in out_dir.glob("predictions/*/confidence_*.json"):
        if cand.exists():
            json_path = cand
            source = "pipeline"
            break

    # (2) Boltz default folder using ligand name
    if not json_path or not json_path.exists():
        alt_dir = ROOT / f"boltz_results_{name}/predictions/{name}"
        if alt_dir.exists():
            for cand in alt_dir.glob("confidence_*.json"):
                if cand.exists():
                    json_path = cand
                    source = f"boltz_default:{name}"
                    break

    # (3) Boltz default folder using SMILES alias
    if (not json_path or not json_path.exists()) and safe_alias != name:
        alt_dir = ROOT / f"boltz_results_{safe_alias}/predictions/{safe_alias}"
        if alt_dir.exists():
            for cand in alt_dir.glob("confidence_*.json"):
                if cand.exists():
                    json_path = cand
                    source = f"boltz_default:{safe_alias}"
                    break

    # (4) Parse whichever JSON we found
    if json_path and json_path.exists():
        try:
            data = json.load(open(json_path))
            aff  = round(float(data.get("ligand_iptm", 0.0)), 3)
            conf = round(float(data.get("confidence_score", 0.0)), 3)
            print(f"[OK] Parsed {json_path.name} from {source}: ligand_iptm={aff}, confidence={conf}")

            # Mirror JSON to organized pipeline directory
            dest = out_dir / f"predictions/{name}/"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(json_path, dest / json_path.name)

        except Exception as e:
            print(f"[WARN] Could not parse {json_path.name} for {name}: {e}")
    else:
        print(f"[WARN] No confidence JSON found for {name} or alias {safe_alias}")

    return aff, conf

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_id", required=True)
    ap.add_argument("--cleanup", action="store_true", help="Remove stray boltz_results_* folders after parsing")
    args = ap.parse_args()
    pdb_id = args.pdb_id.upper()

    workdir = DATA / pdb_id
    receptor = workdir / f"{pdb_id}_clean.pdb"
    ligfile  = workdir / "ligand_set.smi"
    outcsv   = workdir / "boltz_results.csv"

    if not receptor.exists(): raise FileNotFoundError(receptor)
    if not ligfile.exists(): raise FileNotFoundError(ligfile)

    seq = extract_sequence_from_pdb(receptor)
    print(f"[INFO] Extracted sequence length {len(seq)} from {receptor.name}")

    # Load ligands and validate SMILES
    ligands = []
    for line in open(ligfile):
        if line.strip():
            parts = line.strip().split()
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else safe_name_from_smiles(smi)

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"[WARN] Invalid SMILES skipped: {smi}")
                continue
            ligands.append((name, smi))
    print(f"[INFO] {len(ligands)} valid ligands loaded.")

    boltz_exec = shutil.which("boltz")
    if not boltz_exec:
        raise RuntimeError("Boltz not found. Install via `pip install git+https://github.com/jwohlwend/boltz.git`.")

    on_apple = "arm" in platform.processor().lower() or "Apple" in platform.platform()
    print(f"[INFO] Platform = {platform.platform()}  →  Apple Silicon = {on_apple}")

    # macOS diffusion patch
    if on_apple:
        model_cfg = Path(shutil.which("boltz")).resolve().parent / "../lib/python3.12/site-packages/boltz/data/configs/model/boltz-1.yaml"
        model_cfg = model_cfg.resolve()
        patch_diffusion_steps(model_cfg, steps=50)

    results = []

    # -----------------------------------------------------------------
    for name, smi in ligands:
        out_dir = workdir / "boltz_results" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create YAML input
        yaml_path = out_dir / f"{name}.yaml"
        yaml_content = {
            "name": name,
            "sequences": [
                {"protein": {"id": "A", "sequence": seq}},
                {"ligand": {"id": "L", "smiles": smi}},
            ],
            "targets": [{"receptor": 0, "ligand": 1}],
            "output": str(out_dir / "output.json"),
            "model": "boltz-1",
            "accelerator": "gpu" if not on_apple else "cpu",
        }
        yaml.safe_dump(yaml_content, open(yaml_path, "w"), sort_keys=False)

        # Run Boltz
        try:
            cmd = [boltz_exec, "predict", str(yaml_path),
                   "--use_msa_server", "--accelerator", "gpu" if not on_apple else "cpu"]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[WARN] Boltz failed for {name}: {e}")

        # Parse Boltz results
        aff, conf = parse_conf_json(name, out_dir)
        results.append({
            "Ligand": name,
            "boltz_affinity": aff,
            "boltz_confidence": conf
        })

    # -----------------------------------------------------------------
    pd.DataFrame(results).to_csv(outcsv, index=False)
    print(f"[OK] Boltz results saved → {outcsv}")

    # Optional cleanup of stray top-level Boltz folders
    if args.cleanup:
        for p in ROOT.glob("boltz_results_*"):
            try:
                shutil.rmtree(p)
                print(f"[CLEAN] Removed stray {p}")
            except Exception:
                pass

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()

