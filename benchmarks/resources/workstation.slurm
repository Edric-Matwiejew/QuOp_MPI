#!/bin/bash -l
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=coursework
#SBATCH --nodes=NODES
#SBATCH --ntasks=TOTAL_MPI_PROCESSES
#SBATCH --cpus-per-task=1
#SBATCH --time=TIME
#SBATCH --mem=8000MB
#SBATCH --output=OUT_DIR/LOG_NAME

module load python3.9/3.9.2 gcc/9.3.1 open-mpi/4.1.0 fftw/3.3.9 hdf5/1.12.0

export OMP_NUM_THREADS=THREADS

( cd ../../../ ; command time -v mpiexec -n TOTAL_MPI_PROCESSES python3 -m quop_bench BENCH_TYPE $(date +'%s' -d "TIME") CSV_DIR/CSV_NAME TEST_MODULE_IMPORT_PATH )
