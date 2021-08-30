#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_4
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qaoa_evolution/log/4_qaoa_evolution.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 4 -n 96 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution 3600 results/cluster/qaoa_evolution/csv/4_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
