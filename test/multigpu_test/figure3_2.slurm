#!/bin/bash
#SBATCH --job-name=py-job        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=100G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80

module purge
module load cudatoolkit/11.7
module load nvhpc/22.5
module load anaconda3/2023.3
conda activate cos598d


for dataset in "reddit" "ogbn-arxiv" "ogbn-products"
do
    for partion_alg in "equal" "binary_search"
    do
    python inference_multigpu_gcn.py --dataset=$dataset --partition-alg=$partion_alg --output="figure3_2.csv"
    done
done