#!/bin/bash
#SBATCH --job-name=py-job        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=500G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)

module purge
module load cudatoolkit/11.7
module load nvhpc/22.5
module load anaconda3/2023.3
conda activate cos598d

# python inference_multigpu_gcn.py --dataset="ogbn-papers100M" --partition-alg="binary_search" --output="figure1_both.csv"

for dataset in   "reddit" "ogbn-arxiv" "ogbn-products" "ogbn-papers100M" 
do
    for memory_manage in 0 #1 2
    do
        for partion_alg in "equal" "binary_search"
        do
        python inference_multigpu_gcn.py --dataset=$dataset --memory-manage=$memory_manage --partition-alg=$partion_alg --output="figure1_new_new.csv" --trace=0
        done
    done
    
done
