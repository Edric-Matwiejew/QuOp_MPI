#!/bin/bash

mkdir -p plots

rm *.h5 *.out plots/*

qubits=(5 6 7 8 9 10 11 12)

for qubit in ${qubits[@]}; do
	mpiexec -N 2 python3 experiment_1.py $qubit "experiment_1" >> experiment_1.out
done
