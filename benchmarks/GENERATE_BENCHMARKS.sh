#!/bin/bash

# Commandline argument either 'CLUSTER' or 'WORKSTATION'.

if ! ( [ "$1" = "WORKSTATION" ] || [ "$1" = "CLUSTER" ] ); then
	echo "Benchmark system type not defined, aborting."
	exit 
fi

if [ "$1" = "WORKSTATION" ]; then
	BASE_SLURM=resources/workstation.slurm
else
	BASE_SLURM=resources/cluster.slurm
fi

bash resources/slurm_gen.sh "$BASE_SLURM" "qwoa_evolution" 01:00:00 1 24 "evolution" "quop_bench.evolution.qwoa_evolution" $1
bash resources/slurm_gen.sh "$BASE_SLURM" "qaoa_evolution" 01:00:00 1 24 "evolution" "quop_bench.evolution.qaoa_evolution" $1
