# Singularity Containerisation

On Pawsey an up to date container for QuOp_MPI should be located in $MYGROUP/QuOp_MPI. Please check that your scripts point the name of the most recent container. These follow the naming convention quop_mpi_DDMMYYYY.sif.

## Building Containers

quop_mpi/container/quop_mpi.def, can be used to build a singularity container in which to run quop_mpi on the Pawsey Supercomputing Center's Magnus system.

The requires that the container is built on a system on which you have adminstraive rights:

    sudo singularity build quop_mpi.sif <path to>/quop_mpi.def

quop_mpi.sif should then be copied to your group directory in the Pawsey file system.

Alternatively, build the container remotely on Magnus using the example slurm and remote_build.sh scripts (remember to modify the slurm script to save the conatiner to an appropriate file path and name). These require a sylabs-token to be saved into the same folder. Tokens are generated at https://cloud.sylabs.io/builder.

## Using Containers

And to run a python script using quop_mpi, the singularity module whould first be imported:

    module load singularity

Then:

	srun -N <number of processes> singularity exec <path to>/quop_mpi.sif python3 <pyhton script>.py

