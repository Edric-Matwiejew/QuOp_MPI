#!/bin/bash -l
#SBATCH --job-name=qwoa_execute_global
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=../../../results/cluster/qwoa_execute_global/log/4_qwoa_execute.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 4 -n 96 singularity exec ../container/quop_mpi.sif python3 -m quop_bench execute_depth 3600 18 results/cluster/qwoa_execute_global/csv 4_qwoa_execute_bench_log.csv 4_qwoa_execute_quop_log quop_bench.execute.qwoa_parallel_jacobian
