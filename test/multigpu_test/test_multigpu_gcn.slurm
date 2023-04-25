#!/bin/bash
#SBATCH --job-name=py-job        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=100G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:29:00          # total run time limit (HH:MM:SS)


module purge
module load cudatoolkit/11.7
module load nvhpc/22.5
module load anaconda3/2023.3
conda activate cos598d

# nsys profile --trace=cuda,nvtx,osrt -o myprofile_${SLURM_JOBID}.qdrep 
python test_multigpu_gcn.py