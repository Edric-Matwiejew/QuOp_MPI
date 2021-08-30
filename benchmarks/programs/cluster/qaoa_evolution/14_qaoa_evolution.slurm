#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_14
#SBATCH --partition=workq
#SBATCH --nodes=14
#SBATCH --ntasks=336
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qaoa_evolution/log/14_qaoa_evolution.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 14 -n 336 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution 3600 results/cluster/qaoa_evolution/csv/14_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
