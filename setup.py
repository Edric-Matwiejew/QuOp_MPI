#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import subprocess

from setuptools import find_packages, setup, Command
from setuptools.command.build_ext import build_ext


class Build(build_ext):

    def run(self):

        def parse_paths(paths, prefix):

            existant = []
            for path in paths:
                if os.path.exists(path):
                    existant.append(path)

            for i in range(len(existant)):
                existant[i] = " {}{}".format(prefix, existant[i])

            return "".join(existant)

        lib = parse_paths(self.library_dirs, "-L")
        include = parse_paths(self.include_dirs, "-I")

        if (
            subprocess.call(
                'make -C src LIB="{}" INCLUDE="{}"'.format(lib, include), shell=True
            )
            != 0
        ):
            sys.exit(-1)

        os.environ.clear()
        os.environ.update(old_env)

        build_ext.run(self)


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
        "quop_mpi": ["*" + EXTENSION],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    license="GPLv3",
    cmdclass={
        "build_ext": Build,
    },
)
