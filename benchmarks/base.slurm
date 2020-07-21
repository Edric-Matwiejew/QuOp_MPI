#!/bin/bash -l
#SBATCH --job-name=NAME_PLACEHOLDER
#SBATCH --partition=workq
#SBATCH --nodes=NODES_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --account=
#SBATCH --output=OUTPUT_PLACEHOLDER

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N NODES_PLACEHOLDER -n MPI_PROCS_PLACEHOLDER --cpu_bind=sockets singularity exec ../container/quop_mpi.sif python3 ../benchmark.py
