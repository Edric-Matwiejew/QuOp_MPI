apt-get update -qq && apt-get -y  --no-install-recommends install build-essential pkg-config python3-pip cython git mpich libhdf5-mpich-dev libfftw3-dev libfftw3-mpi-dev
python3 -m pip install wheel numpy pandas pandas-datareader scipy networkx nlopt mpi4py
CC=mpicc HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
cd ~/
git clone https://github.com/Edric-Matwiejew/QuOp_MPI
cd QuOp_MPI
python3 setup.py sdist bdist_wheel
cd dist
python3 -m pip install quop_mpi-1.0.0.tar.gz
cd ../
cd examples/maxcut
python3 maxcut.py




