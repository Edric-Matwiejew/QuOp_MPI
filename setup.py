import os
os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"  # Force single-thread build due to CMake Fortran dependency bugs

from skbuild import setup
from setuptools import find_packages

setup(
    name="QuOp_MPI",
    version="1.3.0",
    description="A parallel framework for the design and simulation of quantum variational algorithms.",
    packages=find_packages(where='.'),
    cmake_languages=["Fortran", "C"],
    cmake_args=["-DSKBUILD=TRUE"],
    cmake_install_dir=".",
)

