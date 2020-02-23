# Singularity Containerisation

quop_mpi/container/quop_mpi.def, can be used to build a singularity container in which to run quop_mpi on the Pawsey Supercomputing Center's Magnus system.

The requires that the container is built on a system on which you have adminstraive rights:

    sudo singularity build quop_mpi.sif <path to>/quop_mpi.def

quop_mpi.sif should then be copied to your group directory in the Pawsey file system.

And to run a python script using quop_mpi, the singularity module whould first be imported:

    module load singularity

Then:

	srun -N <number of processes> singularity exec <path to>/quop_mpi.sif python3 <pyhton script>.py
