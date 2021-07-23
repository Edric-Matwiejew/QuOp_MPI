#!/bin/bash

# Commandline argument either 'CLUSTER' or 'WORKSTATION'.

bash resources/slurm_gen.sh "qwoa_evolution" 01:00:00 1 24 "evolution" "quop_bench.evolution.qwoa_evolution" $1
bash resources/slurm_gen.sh "qaoa_evolution" 01:00:00 1 24 "evolution" "quop_bench.evolution.qaoa_evolution" $1
