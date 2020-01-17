PYTHON_CALL ?= python3

DEBUG ?= FALSE

FC = mpifort

ifeq ($(DEBUG),TRUE)
	FCFLAGS = -O0 -g -fcheck=array-temps,bounds,do,mem,pointer,recursion -Wall -Wextra -Wconversion -pedantic
else
	FCFLAGS = -O3 -ffast-math -funroll-loops
endif

EXT_SUFFIX := $(shell $(PYTHON_CALL)-config --extension-suffix)

LIB = -L/usr/
INCLUDE = -I/usr/

PYDIR := ../qwao_mpi

%.o: %.mod

$(PYDIR)/fqwao_mpi$(EXT_SUFFIX): fqwao_mpi.f90
	$(PYTHON_CALL) -m numpy.f2py --f90exec="$(FC)" --f90flags="$(FCFLAGS)" -c $(LIB) -lhdf5_fortran -lfftw3 -lfftw3_mpi $(INCLUDE) -m fqwao_mpi fqwao_mpi.f90
	mv fqwao_mpi$(EXT_SUFFIX) $(PYDIR)

clean:
	rm $(PYDIR)/*.so
