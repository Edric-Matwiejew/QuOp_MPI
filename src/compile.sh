#!/bin/bash
#python3 -m numpy.f2py --f90exec="mpifort" --f90flags="-Wall -Og -fcheck=all -fbacktrace" -c -L/usr/local/lib -lfftw3 -lfftw3_mpi -I/usr/local/include -m fqwao_mpi fqwao_mpi.f90
python3 -m numpy.f2py --f90exec="mpifort" --f90flags="-Wall -Og -fcheck=all -fbacktrace" -c -L$LIB -lhdf5_fortran -lfftw3 -lfftw3_mpi -I$INCLUDE -m fqwao_mpi fqwao_mpi.f90
