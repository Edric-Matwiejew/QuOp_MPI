#!/bin/bash -l
#SBATCH --job-name=qwoa_evolution_9
#SBATCH --partition=workq
#SBATCH --nodes=9
#SBATCH --ntasks=216
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qwoa_evolution/log/9_qwoa_evolution.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 9 -n 216 singularity exec ../container/quop_mpi.sif python3 -m quop_bench evolution 3600 results/cluster/qwoa_evolution/csv/9_qwoa_evolution.csv quop_bench.evolution.qwoa_evolution
