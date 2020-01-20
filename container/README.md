# Singularity Containerisation

qwao_mpi/container/qwao_mpi.def, can be used to build a singularity container in which to run qwao_mpi. It is based on the docker file supported by the Pawsey Supercomputing Center and requires that MPICH version 3.1.4 is installed on the host system.

To build the container:

    sudo singularity build qwao_mpi.sif <path to>/qwao_mpi.def

And to run a python script using qwao_mpi:

	mpiexec -N <number of processes> singularity exec <path to>/qwao_mpi.sif python3 <pyhton script>.py
