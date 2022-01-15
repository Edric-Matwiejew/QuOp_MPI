PYTHON_CALL ?= python3

DEBUG ?= FALSE

FC = mpifort

ifeq ($(DEBUG),TRUE)
	FCFLAGS += -O0 -g -fcheck=array-temps,bounds,do,mem,pointer,recursion -Wall -Wextra -Wconversion -pedantic
else
	FCFLAGS += -O3 -ffast-math -funroll-loops
endif

EXT_SUFFIX := $(shell $(PYTHON_CALL)-config --extension-suffix)

OBJDIR := obj
PYDIR := ../quop_mpi/__lib

$(shell mkdir -p $(OBJDIR))

$(OBJDIR)/%.o: %.f90
	$(FC) $(FCFLAGS) -I$(OBJDIR) -J$(OBJDIR) -c $< -o $@ -fPIC

%.o: %.mod

$(PYDIR)/mixers_mpi$(EXT_SUFFIX):  mixers_mpi.f90 $(PYDIR)/fqwoa_mpi$(EXT_SUFFIX)
	$(PYTHON_CALL) -m numpy.f2py --f90exec="$(FC)" --f90flags="$(FCFLAGS)" -c -m mixers_mpi mixers_mpi.f90
	mv mixers_mpi$(EXT_SUFFIX) $(PYDIR)

$(PYDIR)/fqwoa_mpi$(EXT_SUFFIX): fqwoa_mpi.f90 $(PYDIR)/fMPI$(EXT_SUFFIX)
	$(PYTHON_CALL) -m numpy.f2py --f90exec="$(FC)" --f90flags="$(FCFLAGS)" -c $(LIB) -lhdf5_fortran -lfftw3 -lfftw3_mpi $(INCLUDE) -m fqwoa_mpi fqwoa_mpi.f90
	mv fqwoa_mpi$(EXT_SUFFIX) $(PYDIR)

$(PYDIR)/fMPI$(EXT_SUFFIX):  $(addprefix $(OBJDIR)/, iso_precisions.o sparse.o one_norms.o expm.o) fMPI.f90
	$(PYTHON_CALL) -m numpy.f2py --f90exec="$(FC)" --f90flags="-I$(OBJDIR) $(FCFLAGS)" -c fMPI.pyf iso_precisions.f90 sparse.f90 one_norms.f90 expm.f90 -m fMPI fMPI.f90
	mv fMPI$(EXT_SUFFIX) $(PYDIR)

$(OBJDIR)/expm.o: $(addprefix $(OBJDIR)/, iso_precisions.o sparse.o one_norms.o) expm.f90
$(OBJDIR)/one_norms.o: $(addprefix $(OBJDIR)/, iso_precisions.o sparse.o) one_norms.f90
$(OBJDIR)/sparse.o: $(addprefix $(OBJDIR)/, iso_precisions.o) sparse.f90
$(OBJDIR)/iso_precisions.o: iso_precisions.f90

clean:
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.mod $(PYDIR)/*.so
