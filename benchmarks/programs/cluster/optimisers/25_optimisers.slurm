#!/bin/bash -l
#SBATCH --job-name=quop_optimisers
#SBATCH --partition=workq
#SBATCH --nodes=25
#SBATCH --ntasks=600
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qwoa_execute_global/log/16_qwoa_execute.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 25 -n 600 singularity exec ../container/quop_mpi.sif python3 -m quop_bench optimisers results/cluster/optimisers
