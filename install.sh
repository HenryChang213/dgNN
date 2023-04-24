module purge
module load anaconda3/2023.3
module load cudatoolkit/11.4
module load nvhpc-nocompiler/22.5
conda activate cos598d
pip uninstall dgNN
python -W ignore setup.py build
python -W ignore setup.py install