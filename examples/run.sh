#!/bin/bash
for file in `ls -v *.py`; do
	mpiexec -N 2 python3 $file
done
