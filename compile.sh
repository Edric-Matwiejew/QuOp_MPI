#!/bin/bash
python3 -m numpy.f2py --f90exec="mpifort" -c -L/usr/local/lib -lfftw3 -lfftw3_mpi -I/usr/local/include -m fqwao_mpi fqwao_mpi.f90
#mpifort -w -pedantic -shared -fPIC -I/usr/local/include -o fqwao.so fqwao.f90 -L/usr/local/lib/ -lfftw3_mpi -lfftw3
