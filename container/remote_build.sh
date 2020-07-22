#!/bin/bash
singularity remote login --tokenfile sylabs-token
singularity build -r quop_mpi.sif quop_mpi.def
