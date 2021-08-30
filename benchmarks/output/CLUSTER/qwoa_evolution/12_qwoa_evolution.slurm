#!/bin/bash -l
#SBATCH --job-name=qwoa_evolution_12
#SBATCH --partition=workq
#SBATCH --nodes=12
#SBATCH --ntasks=288
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=out/12_qwoa_evolution.out

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 12 -n 288 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution $(date +'%s' -d "01:00:00") output/CLUSTER/qwoa_evolution/csv/12_qwoa_evolution.csv quop_bench.evolution.qwoa_evolution
