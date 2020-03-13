#!/bin/bash
singularity remote login --tokenfile sylabs-token
singularity build -r ../QuOp_MPI/quop_mpi_13032020.sif quop_mpi.def
