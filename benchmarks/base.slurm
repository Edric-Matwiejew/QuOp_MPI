#!/bin/bash -l
#SBATCH --job-name=NAME_PLACEHOLDER
#SBATCH --partition=workq
#SBATCH --nodes=NODES_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --output=OUTPUT_PLACEHOLDER

export OMP_NUM_THREADS=THREADS_PLACEHOLDER

module load singularity
echo "Starting evolution..."
command time -v srun --export=all -N NODES_PLACEHOLDER -n MPI_PROCS_PLACEHOLDER --cpu_bind=sockets -c THREADS_PLACEHOLDER singularity exec ../quop_mpi.sif python3 ../evolution_benchmark.py TIME_PLACEHOLDER CSV_PLACEHOLDER TEST_PROGRAM_PLACEHOLDER
echo "...done."
