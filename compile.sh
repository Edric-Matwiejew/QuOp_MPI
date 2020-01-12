#!/bin/bash
python3 -m numpy.f2py --f90exec="mpifort" -c -L/usr/local/lib -lfftw3 -lfftw3_mpi -I/usr/local/include -m fqwao_mpi fqwao_mpi.f90
