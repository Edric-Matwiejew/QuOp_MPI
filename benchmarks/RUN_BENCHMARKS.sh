#!/bin/bash

rm qwoa_evolution/*slurm
rm qaoa_evolution/*slurm


bash slurm_gen.sh "quop_mpi_evolution" 24 1 01:00:00 3600 "qwoa_evolution" "base.slurm" "qwoa_evolution.py" "pawsey0309"
bash slurm_gen.sh "quop_mpi_evolution" 24 1 01:00:00 3600 "qaoa_evolution" "base.slurm" "qaoa_evolution.py" "pawsey0309"

#( cd qwoa_evolution ; bash launch.sh )
#( cd qaoa_evolution ; bash launch.sh )
