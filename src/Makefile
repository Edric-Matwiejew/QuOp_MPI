PYTHON_CALL ?= python3

FC = mpifort

FCFLAGS = -O3 -ffast-math -funroll-loops

EXT_SUFFIX := $(shell $(PYTHON_CALL)-config --extension-suffix)

LIB = /usr/lib/
INCLUDE = /usr/include/

OBJDIR := obj
PYDIR := ../qwao_mpi

$(shell mkdir -p $(OBJDIR))

%.o: %.mod

$(PYDIR)/fqwao_mpi$(EXT_SUFFIX): fqwao_mpi.f90
	$(PYTHON_CALL) -m numpy.f2py --f90exec="$(FC)" --f90flags="$(FCFLAGS)" -c -L$(LIB) -lhdf5_fortran -lfftw3 -lfftw3_mpi -I$(INCLUDE) -m fqwao_mpi fqwao_mpi.f90
	mv fqwao_mpi$(EXT_SUFFIX) $(PYDIR)



