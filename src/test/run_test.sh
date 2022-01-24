#!/bin/bash

MPIEXEC="${MPIEXEC:=mpiexec}"

for i in 1 2 3 4
do
	$MPIEXEC -N $i ./$1
done
