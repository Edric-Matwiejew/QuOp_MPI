#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_15
#SBATCH --partition=workq
#SBATCH --nodes=15
#SBATCH --ntasks=360
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qaoa_evolution/log/15_qaoa_evolution.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 15 -n 360 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution 3600 results/cluster/qaoa_evolution/csv/15_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
