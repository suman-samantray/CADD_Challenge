#!/usr/bin/env bash
# ============================================================
#  🚀 Full Perlmutter Setup for CADD_flow + Boltz + ADMET-AI
#  Reproducible one-shot environment initialization script
#  (no lines skipped or omitted from your working setup)
# ============================================================

# --[ 0. Clear Mamba lock ]-----------------------------------
rm -f /global/u1/s/sumansam/.local/share/mamba/pkgs/cache/lock

# --[ 1. Load required modules ]-------------------------------
module purge
module load python/3.10
module load cuda/12.4
module load gcc/11.2.0
which python
python --version

# --[ 2. Install micromamba binary locally ]-------------------
mkdir -p $HOME/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C $HOME/bin --strip-components=1 bin/micromamba
export PATH="$HOME/bin:$PATH"
micromamba --version
ls -l $HOME/bin/micromamba
module list
micromamba --version

# --[ 3. Organize workspace ]----------------------------------
mv /global/cfs/cdirs/m3288/suman/CADD_flow \
   /global/cfs/cdirs/m3288/suman/CADD_workspace/
export WORKSPACE=/global/cfs/cdirs/m3288/suman/CADD_workspace
cd $WORKSPACE

# --[ 4. Create boltz_env environment ]------------------------
micromamba create -y -p $WORKSPACE/boltz_env python=3.10 -c conda-forge
ls -l $WORKSPACE/boltz_env/bin/python
$WORKSPACE/boltz_env/bin/python --version

# --[ 5. Core Boltz + PyTorch stack ]--------------------------
$WORKSPACE/boltz_env/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
$WORKSPACE/boltz_env/bin/pip install biopython pytorch-lightning==2.5.0.post0
$WORKSPACE/boltz_env/bin/pip install git+https://github.com/jwohlwend/boltz.git

# --[ 6. Clone ProteinMPNN ]----------------------------------
cd $WORKSPACE
git clone https://github.com/dauparas/ProteinMPNN.git
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/ProteinMPNN
$WORKSPACE/boltz_env/bin/python -c "import boltz; print('✅ Boltz ready, version', boltz.__version__)"

# --[ 7. Chemistry stack: OpenBabel + Vina + Meeko ]----------
cd $WORKSPACE
micromamba install -y -p $WORKSPACE/boltz_env openbabel -c conda-forge
export PATH="$WORKSPACE/boltz_env/bin:$PATH"
micromamba install -y -p $WORKSPACE/boltz_env autodock-vina -c bioconda -c conda-forge
export PATH="$WORKSPACE/boltz_env/bin:$PATH"
$WORKSPACE/boltz_env/bin/obabel -V
$WORKSPACE/boltz_env/bin/pip install meeko==0.3.0

# --[ 8. Check workspace layout ]------------------------------
tree -L 2 $WORKSPACE

# --[ 9. Test OpenBabel ]-------------------------------------
$WORKSPACE/boltz_env/bin/python - <<'EOF'
import openbabel
obVersion = openbabel.OBReleaseVersion()
print("✅ OpenBabel:", obVersion)
EOF

# --[ 10. Test Meeko ligand prep ]-----------------------------
$WORKSPACE/boltz_env/bin/python - <<'EOF'
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
prep = MoleculePreparation()
prep.prepare(mol)
pdbqt_string = prep.write_pdbqt_string()
print("✅ Meeko prepared ligand successfully | length:", len(pdbqt_string))
EOF

# --[ 11. Create admet_env environment ]-----------------------
micromamba create -y -p $WORKSPACE/admet_env python=3.10 -c conda-forge
$WORKSPACE/admet_env/bin/pip install admet-ai pandas numpy torch torchvision torchaudio scikit-learn
$WORKSPACE/admet_env/bin/python -c "import admet_ai; print('✅ ADMET-AI ready')"

# --[ 12. Extend PATH and finalize Boltz dependencies ]-------
export PATH="$WORKSPACE/boltz_env/bin:$PATH"
cd /global/cfs/cdirs/m3288/suman/CADD_workspace/CADD_flow
$WORKSPACE/boltz_env/bin/pip install ninja
$WORKSPACE/boltz_env/bin/pip install cuequivariance-torch
micromamba install -y -p $WORKSPACE/boltz_env -c conda-forge cuequivariance
micromamba install -y -p $WORKSPACE/boltz_env -c conda-forge "numpy<=2.1.0"
micromamba search -c conda-forge cuequivariance

# --[ 13. Build cuEquivariance from source ]------------------
cd $WORKSPACE
git clone https://github.com/NVIDIA/cuEquivariance.git
find $WORKSPACE/cuEquivariance -type f -name "pyproject.toml" | grep -i torch

cd $WORKSPACE/cuEquivariance/cuequivariance_torch
module purge
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load cmake
$WORKSPACE/boltz_env/bin/pip install -v .
$WORKSPACE/boltz_env/bin/pip install build setuptools wheel ninja cmake
$WORKSPACE/boltz_env/bin/python setup.py build_ext --inplace
$WORKSPACE/boltz_env/bin/python -m build --wheel
$WORKSPACE/boltz_env/bin/pip install dist/cuequivariance_torch-0.7.0rc1-*.whl --force-reinstall
find $WORKSPACE/boltz_env/lib/python3.10/site-packages -name "cuequivariance_ops_torch*.so"

$WORKSPACE/boltz_env/bin/pip install "numpy<2.0,>=1.26" "scipy==1.13.1" "sympy==1.13.1" --force-reinstall
$WORKSPACE/boltz_env/bin/pip install hatchling hatch-vcs
$WORKSPACE/boltz_env/bin/python -m pip install --no-cache-dir --no-build-isolation --force-reinstall .

# --[ 14. Verify build and re-install core packages ]---------
$WORKSPACE/boltz_env/bin/python -m cuequivariance_torch.build
cd /global/cfs/cdirs/m3288/suman/CADD_workspace
micromamba create -y -p boltz_env python=3.10
eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz_env
pip install boltz==2.2.1
pip install "numpy<=2.1" "scipy==1.13.1" "sympy==1.13.1" "numba==0.61.0"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install boltz==2.2.1 meeko openbabel

cd /global/cfs/cdirs/m3288/suman/CADD_workspace/cuEquivariance/cuequivariance_torch
pip install hatchling build ninja packaging
\rm -rf build/ dist/ *.egg-info
pip install --no-cache-dir --no-build-isolation .
pip install --no-cache-dir cuequivariance-ops-torch-cu12==0.7.0

$WORKSPACE/boltz_env/bin/pip install matplotlib seaborn tqdm

# --[ 15. Verify site packages & .so presence ]---------------
python -m site
find /global/u1/s/sumansam/.local/share/mamba/envs/boltz_env/lib/python3.10/site-packages -type f -name "cuequivariance*.so"

# --[ 16. Run full Boltz-based pipeline ]---------------------
cd /global/cfs/cdirs/m3288/suman/CADD_workspace/CADD_flow
../boltz_env/bin/python -m src.run_pipeline \
  --pdb_id 1PPB \
  --vendor_mode 1 \
  --do_masif_pre \
  --do_vina \
  --do_boltz \
  --do_rank_admet \
  --do_masif_post
