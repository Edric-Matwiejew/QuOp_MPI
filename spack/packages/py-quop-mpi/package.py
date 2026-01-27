# Copyright 2026 Edric Matwiejew
# SPDX-License-Identifier: GPL-3.0-only

from spack.package import *


class PyQuopMpi(PythonPackage):
    """QuOp_MPI: A parallel framework for the design and simulation of
    quantum variational algorithms."""

    homepage = "https://github.com/Edric-Matwiejew/QuOp_MPI"
    url = "https://github.com/Edric-Matwiejew/QuOp_MPI/archive/refs/tags/v1.4.0.tar.gz"
    git = "https://github.com/Edric-Matwiejew/QuOp_MPI.git"

    maintainers("Edric-Matwiejew")

    license("GPL-3.0-only")

    version("main", branch="main")
    version("1.4.0", sha256="d999bf16187a0300b0bff0fe56c5d3a5966bc9e93381935e47cbbbc299d2ed56")

    # Python version requirement
    depends_on("python@3.11:", type=("build", "link", "run"))

    # Build system dependencies
    depends_on("py-setuptools@42:68", type="build")
    depends_on("py-scikit-build@0.13:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-ninja", type="build")
    depends_on("cmake@3.18:", type="build")

    # Runtime dependencies
    depends_on("py-numpy@:1.25", type=("build", "run"))
    depends_on("py-mpi4py@3.1.2:", type=("build", "run"))
    depends_on("py-scipy@1.9.1:", type=("build", "run"))
    depends_on("py-h5py@3:", type=("build", "run"))
    depends_on("py-pandas@1.3.4:", type=("build", "run"))
    depends_on("py-networkx@2.8.6:", type=("build", "run"))

    # External library dependencies
    depends_on("mpi", type=("build", "link", "run"))
    depends_on("fftw@3:+mpi+fortran", type=("build", "link", "run"))
    depends_on("hdf5@1.10:+fortran+shared+mpi", type=("build", "link", "run"))

    # Optional dependencies for examples
    # py-yfinance and py-seaborn are provided by this repo (not in Spack builtin)
    variant("examples", default=False, description="Install dependencies for examples")
    depends_on("py-yfinance@0.2:", when="+examples", type="run")
    depends_on("py-matplotlib@3.6:", when="+examples", type="run")
    depends_on("py-seaborn@0.11.2:", when="+examples", type="run")
    depends_on("py-jupyterlab", when="+examples", type="run")

    # Optional dependencies for documentation (these are in Spack builtin)
    variant("docs", default=False, description="Install dependencies for documentation")
    depends_on("py-numpydoc@1.5:", when="+docs", type="run")
    depends_on("py-sphinxcontrib-bibtex@2.5:", when="+docs", type="run")
    depends_on("py-sphinx-rtd-theme@1.2:", when="+docs", type="run")

    def setup_build_environment(self, env):
        """Set up environment variables for the build."""
        # Ensure MPI compilers are used
        env.set("CC", self.spec["mpi"].mpicc)
        env.set("FC", self.spec["mpi"].mpifc)

        # Set FFTW paths for FindFFTW.cmake
        env.set("FFTW_DIR", self.spec["fftw"].prefix)
        env.prepend_path("LD_LIBRARY_PATH", self.spec["fftw"].prefix.lib)

        # Set HDF5 paths - CMake's FindHDF5 uses HDF5_ROOT
        env.set("HDF5_ROOT", self.spec["hdf5"].prefix)
        env.prepend_path("LD_LIBRARY_PATH", self.spec["hdf5"].prefix.lib)

        # Pass CMake arguments via CMAKE_ARGS for scikit-build
        # scikit-build reads CMAKE_ARGS from environment
        cmake_args = [
            f"-DHDF5_Fortran_INCLUDE_DIR={self.spec['hdf5'].prefix.include}",
            f"-DHDF5_C_INCLUDE_DIR={self.spec['hdf5'].prefix.include}",
            f"-DFFTW_ROOT={self.spec['fftw'].prefix}",
        ]
        env.set("CMAKE_ARGS", " ".join(cmake_args))

        # MPI backend enabled, wavefront (GPU) disabled
        env.set("MPI_BACKEND", "ON")
        env.set("WAVEFRONT_BACKEND", "OFF")

    def setup_run_environment(self, env):
        """Set up runtime environment."""
        env.prepend_path("LD_LIBRARY_PATH", self.spec["fftw"].prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.spec["hdf5"].prefix.lib)
