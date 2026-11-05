Ella GH200
==========

Run the commands below from the repository root.

The ``pawsey-ella`` profile uses GCC/GFortran for host code, NVCC for HIP device
code, and Open MPI with UCX. It targets Hopper architecture ``90`` and requires
CMake 3.28 or newer for the NVIDIA HIP language support.
Load a compatible native aarch64 stack: GCC with GFortran, Python 3.11, CUDA,
HIP configured for NVIDIA, hipFFT, hipfort, parallel HDF5 with C and Fortran
interfaces, FFTW3 MPI, and numactl. The MPI libraries used by SHAFFT, QuOp_MPI,
mpi4py and h5py must match; the installer builds mpi4py and h5py from source.

Copy the supplied config and fill its ``[modules]`` lists with the exact names
reported by ``module avail``. Ella's Spack module names can include hashes, so
the supplied lists are empty and use the already loaded environment. Configured
module lists are reloaded by the generated activation script. Module-provided
``PAWSEY_*_HOME`` and ``CMAKE_PREFIX_PATH`` values are used for discovery,
including the NVIDIA HPC SDK's separate CUDA math-library prefix.
When using preloaded modules, load the same stack before sourcing the activation
script in a new shell or batch job.

.. code-block:: bash

    cp environments/profiles/pawsey-ella/wavefront/config.toml ./ella.toml
    # Edit ella.toml with the site's exact module names and any [paths] overrides.
    bash environments/install.sh -p pawsey-ella -b wavefront -c ./ella.toml \
        --prefix /scratch/$USER/quop-ella
    source /scratch/$USER/quop-ella/activate-pawsey-ella-wavefront.sh

To use an existing CUDA SHAFFT build, set ``SHAFFT_PATH`` to its installation
prefix, or ``paths.shafft_path`` in the config. Otherwise SHAFFT is fetched and
built with the same NVIDIA platform and MPI setting. An existing SHAFFT build's
GPU-aware MPI setting must match QuOp's setting. hipfort's Fortran modules must
have been built with the selected GFortran compiler. ``CUDA_PATH``,
``HIP_PATH``, ``HIPFORT_PATH``, ``HDF5_DIR`` and ``FFTW_ROOT`` can be set explicitly
or through the corresponding ``[paths]`` config keys. ``paths.cmake_prefix_path``
accepts additional semicolon-separated prefixes. The default uses host-staged
MPI for initial validation. After confirming the loaded Open MPI/UCX supports
CUDA device buffers, set ``build.gpu_aware_mpi = true`` (or
``GPU_AWARE_MPI=ON`` in the environment) and use a separate installation prefix.

Run an import check and the MPI tests inside an allocated GPU job, with the
launcher and GPU resource flags required by the site, for example:

.. code-block:: bash

    srun -N 1 -n 1 --gpus=1 python tests/test_import_extensions.py
    srun -N 1 -n 1 --gpus=1 python -m pytest tests/mpi/ --with-mpi --backend wavefront

Then validate multiple GPUs and multiple nodes before scaling production runs.
``run_tests.sh`` accepts ``QUOP_MPI_LAUNCHER=srun``, extra launcher flags through
``QUOP_MPI_LAUNCHER_ARGS``, and ``QUOP_TEST_GPUS_PER_NODE`` when allocation
discovery needs an override. The topology runner also accepts
``--gpus-per-node N`` and ``--visibility all|masked`` for the site's GPU binding.
Set the benchmark config's partition, GPU counts and rank counts for the
allocated node type. The ``mpi`` backend is available with the same profile
without the GPU dependencies.
