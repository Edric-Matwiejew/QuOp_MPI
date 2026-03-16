QuOp_MPI Build System
=====================

This page documents the current CMake build flow for QuOp_MPI, with focus on
the wavefront (GPU) backend.

Overview
--------

QuOp_MPI uses CMake with scikit-build for Python extension builds.

- MPI backend (CPU): distributed execution on ``MPI::MPI_Fortran``
- Wavefront backend (GPU): HIP/ROCm, hipfort, and wavefront Fortran modules

Key build files:

- ``CMakeLists.txt``
- ``src/add_f2py_library.cmake``
- ``cmake/HipfortDependency.cmake``

Build Backends
--------------

MPI backend:

.. code-block:: bash

   cmake -DMPI_BACKEND=ON -DWAVEFRONT_BACKEND=OFF ..

Wavefront backend:

.. code-block:: bash

   cmake -DWAVEFRONT_BACKEND=ON -DROCM_PATH=/opt/rocm -DOFFLOAD_ARCH=gfx90a ..

Dependency Management
---------------------

Dependencies fetched via ``FetchContent`` are stored under ``.deps/`` at the
project root.

- ``.deps`` survives ``rm -rf build``
- delete ``.deps`` when switching compilers or ROCm versions

Full clean rebuild:

.. code-block:: bash

   rm -rf build .deps
   cmake -S . -B build -DWAVEFRONT_BACKEND=ON -DROCM_PATH=/opt/rocm -DOFFLOAD_ARCH=gfx90a
   cmake --build build

hipfort
~~~~~~~

hipfort is auto-fetched if not found. hipfort ``.mod`` files are
compiler-version specific.

If you see incompatible ``.mod`` errors:

.. code-block:: bash

   rm -rf .deps/hipfort-install .deps/hipfort-ep

Fortran Preprocessor Requirements
---------------------------------

Files using ``#ifdef/#endif`` must be compiled with ``-cpp``.

Files that require preprocessing:

.. list-table::
   :header-rows: 1

   * - File
     - Directives
   * - ``src/wavefront/context/gpu_transfer.f90``
     - ``QUOP_GPU_AWARE_MPI``
   * - ``src/comm_info/comm_info_module.f90``
     - ``WAVEFRONT_BACKEND``
   * - ``src/sparse_propagators/src/sparse.f90``
     - ``USE_HIP``, ``QUOP_GPU_AWARE_MPI``
   * - ``src/sparse_propagators/src/chebyshev.f90``
     - ``USE_HIP``, ``QUOP_GPU_AWARE_MPI``

Preferred target-local setup:

.. code-block:: cmake

   target_compile_options(my_target PRIVATE -cpp)

Avoid global flags unless necessary:

.. code-block:: cmake

   set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

F2py Integration
----------------

``add_f2py_library`` in ``src/add_f2py_library.cmake`` wraps extension
generation and linking.

Example:

.. code-block:: cmake

   add_f2py_library(
       MODULE_NAME my_module
       DEPENDS target1 target2
       DEFINITIONS define1=value1 define2=value2
       SRC ${SOURCE_FILE}
       INSTALL_SUBDIR subdir
       LIBRARIES ${LIBS}
   )

Target naming pattern for module ``foo``:

- ``foomodule``: preprocessed Fortran object library
- ``foowrapper``: f2py wrapper object library
- ``foo_f2py``: final Python extension

When linking f2py targets, always use keyword signature:

.. code-block:: cmake

   target_link_libraries(foo_f2py PRIVATE bar)

Do not mix plain and keyword signatures on the same target.

Custom wrapper compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Propagator wrappers use ``add_custom_command`` to avoid CMake Fortran module
tracking issues when preprocessor renames module symbols.

HIP and hipfort Integration
---------------------------

HIP object libraries should be position-independent and linked against
``hip::device``:

.. code-block:: cmake

   add_library(my_hip_lib OBJECT my_hip_code.cpp)
   target_link_libraries(my_hip_lib hip::device)
   set_target_properties(my_hip_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

HIP package discovery must occur before HIP targets are created:

.. code-block:: cmake

   find_package(hip REQUIRED CONFIG PATHS "${ROCM_PATH}")

Common Issues
-------------

Cannot open module file ``xxx.mod``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cause: missing dependency ordering between producer and consumer targets.

Fix:

.. code-block:: cmake

   add_dependencies(consumer_target producer_target)

Signature mismatch in ``target_link_libraries``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error pattern:

::

   The keyword signature for target_link_libraries has already been used...

Fix: use ``PRIVATE/PUBLIC/INTERFACE`` consistently.

Preprocessor directives ignored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom: wrong backend paths compiled or runtime behavior mismatch.

Fix: ensure ``-cpp`` is present for the target.

hipfort ``.mod`` incompatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fix:

.. code-block:: bash

   rm -rf .deps/hipfort-install .deps/hipfort-ep

Parallel build flakiness with ``make -j``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom: intermittent missing ``.mod`` files.

Fix: use sequential ``make`` for module-heavy builds.

Troubleshooting
---------------

Full clean rebuild:

.. code-block:: bash

   rm -rf build .deps
   cmake -S . -B build -DWAVEFRONT_BACKEND=ON -DROCM_PATH=/opt/rocm -DOFFLOAD_ARCH=gfx90a
   cmake --build build

Inspect effective flags for a target:

.. code-block:: bash

   cat build/src/wavefront/context/CMakeFiles/wf_gpu_transfer.dir/flags.make

Verbose build commands:

.. code-block:: bash

   make VERBOSE=1

CMake configure log:

::

   build/CMakeFiles/CMakeConfigureLog.yaml

Useful environment variables:

- ``ROCM_PATH`` (default ``/opt/rocm``)
- ``OFFLOAD_ARCH`` (for example ``gfx90a`` or ``gfx1100``)
- ``QUOP_RANKS_PER_GPU``

Contributing Notes
------------------

When adding Fortran files:

1. Check for preprocessor directives.
2. Add ``-cpp`` on the owning target if required.
3. Add explicit dependencies for module producers.
4. Keep ``target_link_libraries`` signatures consistent.
