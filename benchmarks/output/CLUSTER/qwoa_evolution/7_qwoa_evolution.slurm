#!/bin/bash -l
#SBATCH --job-name=qwoa_evolution_7
#SBATCH --partition=workq
#SBATCH --nodes=7
#SBATCH --ntasks=168
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=out/7_qwoa_evolution.out

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 7 -n 168 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution $(date +'%s' -d "01:00:00") output/CLUSTER/qwoa_evolution/csv/7_qwoa_evolution.csv quop_bench.evolution.qwoa_evolution
