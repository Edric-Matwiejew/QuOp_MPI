#!/bin/bash
mpifort -O0 -g -I/usr/include/ fqwoa_mpi_test.f90 -L/usr/lib -lhdf5_fortran -lfftw3 -lfftw3_mpi -o test
