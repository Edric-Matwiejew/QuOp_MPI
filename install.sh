apt-get update -qq && apt-get -y  --no-install-recommends install build-essential pkg-config python3-pip python3-dev cython git mpich libhdf5-mpich-dev libfftw3-dev libfftw3-mpi-dev
python3 -m pip install setuptools
python3 -m pip install wheel numpy pandas pandas-datareader scipy networkx nlopt mpi4py
CC=mpicc HDF5_MPI="ON" python3 -m pip -v install --no-binary=h5py h5py
python3 setup.py sdist bdist_wheel
cd dist
python3 -m pip install quop_mpi-1.0.0.tar.gz
cd ../
cd examples/maxcut
python3 maxcut.py
cd ../../




