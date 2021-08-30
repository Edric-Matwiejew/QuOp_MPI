#!/bin/bash -l
#SBATCH --job-name=qwoa_execute
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=1_qwoa_execute.log

module load singularity

export OMP_NUM_THREADS=1

cd ../../../

command time -v srun  -N 1 -n 1 singularity exec ../container/quop_mpi.sif python3 -m quop_bench execute 3600 18 output/WORKSTATION/qwoa_execute 1_qwoa_execute_bench_log.csv 1_qwoa_execute_quop_log quop_bench.execute.qwoa
