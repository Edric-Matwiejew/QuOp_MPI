#!/bin/bash -l
#SBATCH --job-name=QuOp_Build
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=500MB
#SBATCH --account=research
#SBATCH --output=build.out

module load python3.9/3.9.2 gcc/9.3.1 open-mpi/4.1.0 fftw/3.3.9 hdf5/1.12.0

srun -N 1 -n 1 make -j4
