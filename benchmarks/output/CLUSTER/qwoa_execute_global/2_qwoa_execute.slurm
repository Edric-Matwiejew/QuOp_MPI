#!/bin/bash -l
#SBATCH --job-name=qwoa_execute
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=2_qwoa_execute.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 2 -n 48 singularity exec ../container/quop_mpi.sif python3 -m quop_bench execute_depth 3600 18 output/CLUSTER/qwoa_execute_global 2_qwoa_execute_bench_log.csv 2_qwoa_execute_quop_log quop_bench.execute.qwoa_parallel_jacobian
