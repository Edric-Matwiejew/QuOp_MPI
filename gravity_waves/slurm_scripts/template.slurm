#!/bin/bash --login
#SBATCH --job-name=gravity_qmoa_example	# the job's name
#SBATCH --account=pawsey0309
#SBATCH --nodes=1                    # number of tasks (or 'programs')
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128                      # the workstation is the node, always set to 1
#SBATCH --cpus-per-task=1             # number of threads per task
#SBATCH --exclusive
#SBATCH --time=12:00:00                # time limit hrs:min:sec
#SBATCH --partition=work # partition to which the job is submitted
#SBATCH --output=test_10.out
#SBATCH --qos=high

export OMP_NUM_THREADS=1
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1

# SUBMIT THIS FROM THE gravity_waves folder.
# sbatch experiments/test_10.slurm

# Set PYTHONPATH and experiment environment variables.
. ../../env_setonix.sh

# For QMOA simulations the number of MPI processes (--ntasks above) must be a divisor of the 
# size of the first dimension (a limitation of FFTW3).
time srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS python3 experiments/test_10.py
