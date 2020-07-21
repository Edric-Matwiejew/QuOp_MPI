#!/bin/bash

bash slurm_gen.sh 12 12:00:00 "float" "base.slurm" "QuOp_MPI"

( cd float; bash launch.sh )
