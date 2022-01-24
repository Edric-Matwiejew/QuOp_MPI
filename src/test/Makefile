FC = mpifort

TEST ?= TRUE
DEBUG ?= FALSE

ifeq ($(DEBUG),TRUE)
	FCFLAGS += -O0 -g -fcheck=array-temps,bounds,do,mem,pointer,recursion -Wall -Wextra -Wconversion -pedantic
else
	FCFLAGS += -O3 -ffast-math -funroll-loops
endif

SOURCE = ../iso_precisions.f90 ../sparse.f90 ../sparse_graphs.f90 

.PHONY:
all: dagger eigen tidy

%: $(SOURCE) %.f90
	$(FC) $(FCFLAGS) $^ -o $@; bash run_test.sh $@

tidy:
	@rm -f *.mod

clean:
	rm -f eigen dagger *.mod
