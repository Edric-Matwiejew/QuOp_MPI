#!/bin/bash
#
# An example slurm script for a job using MPI.
#
#SBATCH --job-name=mpi_job             # the job's name
#SBATCH --nodes=1                      # the workstation is the node, always set to 1
#SBATCH --ntasks=1                     # number of tasks (or 'programs')
#SBATCH --cpus-per-task=1              # number of threads per task
#SBATCH --mem=500MB                    # RAM requested for the task
#SBATCH --time=00:10:00                # time limit hrs:min:sec
#SBATCH --output=mpi_example.log       # redirect the task terminal output to a *.log
#SBATCH --partition=research         # partition to which the job is submitted

export OMP_NUM_THREADS=1 # set the number of openMP threads

module load singularity

singularity pull --disable-cache --dir /group/quisa/00100880/QuOp_MPI/container --arch amd64 library://edric_matwiejew/default/image:latest
