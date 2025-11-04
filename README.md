# CADD_Challenge
End-to-end Hybrid CADD Pipeline for Drug Discovery — integrates AutoDock Vina, Boltz (ML docking), MaSIF-torch, and RDKit-based ADMET analysis in a reproducible, HPC-ready workflow.

# 🧬 CADD_flow: Hybrid ML + Physics Docking and ADMET Pipeline

## 📘 Overview
**CADD_flow** is a fully automated **Computer-Aided Drug Discovery (CADD)** pipeline that integrates classical and machine-learning docking with molecular property analysis and ADMET filtering.  
It was designed and validated on **NERSC Perlmutter** (A100 GPU nodes) for large-scale screening and can run locally in a reduced configuration.

This framework combines:
- **Physics-based docking** — AutoDock Vina  
- **ML docking (DiffDock-class)** — Boltz (PyTorch-Lightning)  
- **Protein surface analysis & validation** — MaSIF-torch (pre- and post-docking)  
- **Molecular descriptors & ADMET rules** — RDKit + ADMET-AI  
- **Visualization & statistics**

---

## 🧩 Workflow Summary

| Stage | Script | Description |
|-------|---------|-------------|
| **1. Target Preparation** | `src/prepare_target.py` | Downloads PDB, cleans receptor & co-ligand, generates rigid receptor PDBQT (via Meeko or OpenBabel). |
| **2. Ligand Vendor Generation** | `src/vendor_ligands.py` | Builds per-ligand PDBQT batch from the SMILES library. |
| **3. Surface Localization (Pre-Docking)** | `models/masif_torch.py` | Uses MaSIF-Torch to predict likely binding pockets and generate `masif_box.json` (center and box size for docking). |
| **4a. Docking (AutoDock Vina)** | `src/dock_vina.py` | Performs classical docking within MaSIF-defined box to compute binding energies and centroids. |
| **4b. Docking (Boltz ML)** | `src/dock_boltz.py` | Performs ML-based docking using pretrained Boltz model, yielding predicted affinity and pose confidence (0–1). |
| **5. Ranking + ADMET Integration** | `src/rank_and_admet.py` | Merges Vina, Boltz, and SMILES data; computes RDKit descriptors and rule-based ADMET filters; generates ranked summary tables. |
| **6. Surface Validation (Post-Docking)** | `models/masif_torch.py` | Reuses MaSIF-Torch to score pocket–ligand surface complementarity (`pocket_alignment_Score`) and update rankings. |
| **7. Visualization** | `src/plot_rank_summary.py` | Generates scatter, histogram, bar, and heatmap plots from `summary.csv` or `selected_10.csv`. |

---

## 🧠 Conceptual Flow

```
                                             [Target PDB] 
                                                 ↓
                                           [Receptor Prep] 
                                                 ↓
                                     [Ligand Batch Generation] 
                                                 ↓
                                  [MaSIF-Pre: Pocket Localization] 
                                                 ↓
                                 ┌───────────────┬──────────────────┐
                                 │ AutoDock Vina │ Boltz (ML Dock)  │
                                 └───────────────┬──────────────────┘
                                                 ↓       
                                    [Ranking + ADMET Integration]
                                                 ↓
                                   [MaSIF-Post: Pocket Validation]
                                                 ↓
                                   [Plots + Correlation Analysis]
```

---

## ⚙️ Environment Setup (HPC / Perlmutter)

Two **micromamba environments** are created for reproducibility.
# One-time setup
```bash
git clone https://github.com/suman-samantray/CADD_Challenge.git
cd CADD_Challenge/
bash install_CADDflow_env.sh
```

# After each login
```bash
cd $WORKSPACE/CADD_flow
source ../init_CADDflow_env.sh
```

### 🧩 1️⃣ Boltz + Docking Environment
>Check CADD_Challenge/set_env/boltz_env.yml for complete dependencies
```bash
module purge
module load python/3.10 cuda/12.4 gcc/11.2.0

export WORKSPACE=<path_to_CADD_workspace>
cd $WORKSPACE

micromamba create -y -p $WORKSPACE/boltz_env python=3.10 -c conda-forge
$WORKSPACE/boltz_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$WORKSPACE/boltz_env/bin/pip install biopython pytorch-lightning==2.5.0.post0 git+https://github.com/jwohlwend/boltz.git
micromamba install -y -p $WORKSPACE/boltz_env openbabel autodock-vina -c bioconda -c conda-forge
$WORKSPACE/boltz_env/bin/pip install meeko==0.3.0 matplotlib seaborn tqdm
```

### 🧩 2️⃣ ADMET + Descriptor Environment
>Check CADD_Challenge/set_env/admet_env.yml for complete dependencies
```bash
micromamba create -y -p $WORKSPACE/admet_env python=3.10 -c conda-forge
$WORKSPACE/admet_env/bin/pip install admet-ai pandas numpy torch torchvision torchaudio scikit-learn
```

Export environment files:
```bash
micromamba env export -p $WORKSPACE/boltz_env > boltz_env.yml
micromamba env export -p $WORKSPACE/admet_env > admet_env.yml
```

---

## 🧮 Software Requirements and Notes

| Component | Requirement | Description |
|------------|--------------|-------------|
| **AutoDock Vina (≥1.2.7)** | CPU-compatible | Used for physics-based docking. |
| **Boltz (≥2.2.1)** | **GPU required** (CUDA ≥12.1, PyTorch ≥2.1) | DiffDock-class ML docking; *cannot run on CPU*. |
| **MaSIF-torch** | GPU recommended | Used twice: pre-docking (pocket localization) and post-docking (pose validation). |
| **RDKit** | CPU | Descriptor & ADMET rule calculation. |
| **ADMET-AI** | CPU / GPU | ML-based ADMET property prediction. |
| **Meeko (0.3.0)** | CPU | Ligand preparation (PDBQT). |
| **OpenBabel (≥3.1.1)** | CPU | Receptor / ligand conversion fallback. |
| **Python** | 3.10+ | Base interpreter for all modules. |

> 💡 **Note:** On local macOS / Windows, Boltz and MaSIF will be skipped automatically (CPU-only mode).  
> The pipeline will still perform ligand preparation, Vina docking, and ADMET ranking.

---

## 🚀 Running the Pipeline

### Example: Thrombin (PDB ID: 1PPB)
```bash
  vendor_mode=0 → random 20-ligand subset (quick test run)
  vendor_mode=1 → full purchasable vendor library (from smiles_database.json)
  vendor_mode=2 → dynamically fetched vendor library from webserver endpoint

Each output:  results/{PDB ID}/ligand_set.smi
```

```bash
cd $WORKSPACE/CADD_flow
../boltz_env/bin/python -m src.run_pipeline  --pdb_id 1PPB  --vendor_mode 1  --do_masif_pre  --do_vina  --do_boltz  --do_rank_admet  --do_masif_post
```

### Example: CDK2 (PDB ID: 1H1Q)
```bash
../boltz_env/bin/python -m src.run_pipeline  --pdb_id 1H1Q  --vendor_mode 2  --do_masif_pre  --do_vina  --do_boltz  --do_rank_admet  --do_masif_post
```

---

## 📊 Visualization

After completion, generate plots separately:
```bash
../boltz_env/bin/python src/plot_rank_summary.py --pdb_id 1PPB
```

**Output:**
```
results/1PPB/vina_vs_boltz.png
results/1PPB/vina_hist.png
results/1PPB/admet_heatmap.png
results/1PPB/top10_bar.png
```

---

## 💻 Quick Start (Colab / Local Mac)

A CPU-only subset of this pipeline can be tested **without HPC access** using AutoDock Vina and RDKit.

### ⚙️ Requirements
- macOS / Windows / Linux with Python ≥3.10  
- [Miniforge / Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)  
- Internet access to download PDB and ligand SMILES  

### 🧩 Environment Setup
```bash
micromamba create -y -n caddflow_local python=3.10 -c conda-forge
micromamba activate caddflow_local
pip install rdkit-pypi openbabel meeko admet-ai pandas numpy matplotlib seaborn tqdm
```

### 🚀 Run a minimal example
```bash
git clone https://github.com/suman-samantray/CADD_Challenge.git
cd CADD_Challenge/CADD_workspace/CADD_flow

python src/prepare_target.py --pdb_id 1PPB
python src/dock_vina.py --pdb_id 1PPB
python src/rank_and_admet.py --pdb_id 1PPB
python src/plot_rank_summary.py --pdb_id 1PPB
```

This will generate receptor + ligand preparation, perform Vina docking (CPU), MaSIF surface complementary, compute ADMET descriptors, and produce all plots.  
*Boltz stages will be automatically skipped if CUDA is unavailable.*

---

## 🧾 References
- Eberhardt et al., *J. Chem. Inf. Model.* (2021) — AutoDock Vina  
- Wohlwend et al., *ICLR* (2023) — Boltz  
- Gainza et al., *Nat. Methods* (2020) — MaSIF  
- Landrum et al., *RDKit* (2006–)  
- Davis et al., *Nat. Mach. Intell.* (2024) — ADMET-AI  

---

## 👤 Author

**Dr. Suman Samantray**  
Computational Chemistry | ML for Molecular Design  
🔗 [https://suman-samantray.github.io](https://suman-samantray.github.io)
