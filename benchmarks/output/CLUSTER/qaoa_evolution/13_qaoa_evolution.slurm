#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_13
#SBATCH --partition=workq
#SBATCH --nodes=13
#SBATCH --ntasks=312
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=out/13_qaoa_evolution.out

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 13 -n 312 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution $(date +'%s' -d "01:00:00") output/CLUSTER/qaoa_evolution/csv/13_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
