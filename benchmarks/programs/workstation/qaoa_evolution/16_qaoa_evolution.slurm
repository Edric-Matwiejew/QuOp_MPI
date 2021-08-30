#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_16
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/workstation/qaoa_evolution/log/16_qaoa_evolution.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 1 -n 16 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution 3600 results/workstation/qaoa_evolution/csv/16_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
