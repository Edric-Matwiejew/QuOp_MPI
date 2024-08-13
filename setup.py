from skbuild import setup
from setuptools import find_packages

setup(
    name="QuOp MPI",
    version="1.2.0",
    description="A parallel framework for the design and simulation of quantum variational algorithms.",
    packages=find_packages(where='.'),
    cmake_languages = ["Fortran", "C"],
    cmake_args = ["-DSKBUILD=TRUE -G \"Unix Makefiles\""],
)
