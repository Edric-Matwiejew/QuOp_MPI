#!/bin/bash -l
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=
#SBATCH --nodes=NODES
#SBATCH --time=TIME
#SBATCH --account=
#SBATCH --output=log/LOG_NAME

module load singularity
export PYTHONPATH=$PYTHONPATH:../../../
export OMP_NUM_THREADS=THREADS

command time -v srun --export=all -N NODES -n TOTAL_MPI_PROCESSES --cpu_bind=sockets -c THREADS singularity exec ../container/quop_mpi.sif python3 -m quop_bench BENCH_TYPE $(date +'%s' -d "TIME") csv/CSV_NAME TEST_MODULE_IMPORT_PATH
