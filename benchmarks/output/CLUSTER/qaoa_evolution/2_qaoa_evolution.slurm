#!/bin/bash -l
#SBATCH --job-name=qaoa_evolution_2
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=out/2_qaoa_evolution.out

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 2 -n 48 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution $(date +'%s' -d "01:00:00") output/CLUSTER/qaoa_evolution/csv/2_qaoa_evolution.csv quop_bench.evolution.qaoa_evolution
