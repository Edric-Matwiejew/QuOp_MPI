#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import subprocess

from setuptools import find_packages, setup, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from setuptools import Command

class Build_Ext(build_ext):

    def find_pkg(ENV, lib, prefix, pk_option):

        if ENV in os.environ:

            env = f"-{prefix}/{os.environ[ENV]}"

        else:

            pkgconfig_exists = subprocess.getoutput("which pkg-config")

            if pkgconfig_exists == "":
                raise RuntimeError(
                "Package pkg-config not found.\n\
                 If environment variables 'FFTW3_LIB',\
                 'FFTW3_INCLUDE', 'HDF5_LIB' and 'HDF5_INCLUDE'\
                 are not set, installation of QuOp_MPI requires\
                 pkg-config."
                 )

            env = subprocess.getoutput(f"pkg-config {pk_option} {lib}")

            if f"No package '{lib}' found" in env:

                raise ValueError(
                    f"Package {lib} not found in the pkg-config search path.\
                    \nIf {lib} is installed try specifying the path using the '{ENV}' enviroment variable."
                )

        return env

    Lfftw3 = find_pkg("FFTW3_LIB", "fftw3", "L", "--libs-only-L")
    Ifftw3 = find_pkg("FFTW3_INCLUDE", "fftw3", "I", "--cflags-only-I")
    Lhdf5 = find_pkg("HDF5_LIB", "hdf5", "L", "--libs-only-L")
    Ihdf5 = find_pkg("HDF5_INCLUDE", "hdf5", "I", "--cflags-only-I")

    Lpaths = [Lfftw3, Lhdf5]
    Ipaths = [Ifftw3, Ihdf5]

    # pkg-config may return an empty string if the libraries are present
    # in a deafult search path. If that happens, set deafults:

    if "" in Lpaths:
        Lpaths.append("-L/usr/lib")
    if "" in Ipaths:
       ILpaths.append("-I/usr/include")

    if (
        subprocess.call(
            f"make -C src LIB='{Lfftw3} {Lhdf5}' INCLUDE='{Ifftw3} {Ihdf5}'", shell=True
        )
        != 0
    ):
        sys.exit(-1)

        build_ext.run(self)

class Sdist(sdist):
    def run(self):
        self.run_command("build_ext")
        return super().run()

class Clean(Command):
    """ Run 'make clean' in source directory.
    """
    description = 'delete built extension modules'

    user_options = [
            ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        if (
            subprocess.call(
                'make -C src clean', shell=True
            )
            != 0
        ):
            sys.exit(-1)


# Package meta-data.
NAME = "quop_mpi"
DESCRIPTION = "A framework for simulation of quantum variational algorithms."
URL = "https://github.com/Edric-Matwiejew/QuOp_MPI"
EMAIL = "edric.matwiejew@research.uwa.au"
AUTHOR = "Edric Matwiejew"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "1.0.0"

# What packages are required for this module to be exeuted?
REQUIRED = [
    "numpy",
    "scipy",
    "mpi4py",
    "h5py",
    "nlopt",
    "networkx",
    "pandas",
    "pandas-datareader",
]

# What packages are optional?
EXTRAS = {}

EXTENSION = (
    subprocess.run(
        ["python3-config --extension-suffix"], shell=True, stdout=subprocess.PIPE
    )
    .stdout.decode("utf-8")
    .strip()
)

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={
        "quop_mpi": ["__lib/*" + EXTENSION],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,

    license="GPLv3",
    cmdclass={
        "build_ext": Build_Ext,
        "sdist":Sdist,
        "clean":Clean
    },
)
