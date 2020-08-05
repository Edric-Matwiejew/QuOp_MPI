#!/bin/bash

bash slurm_gen.sh 24 12:00:00 "float" "base.slurm" "qaoa" "..\/qaoa_benchmark.py"
bash slurm_gen.sh 24 12:00:00 "float" "base.slurm" "qwoa" "..\/qwoa_benchmark.py"

( cd float ; bash launch.sh )
