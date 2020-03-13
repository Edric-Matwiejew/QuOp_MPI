#!/bin/bash -l
#SBATCH --job-name=QuOp_MPI_Container_Build
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=build.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 1 bash remote_build.sh
