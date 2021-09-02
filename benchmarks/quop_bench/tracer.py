import trace
import sys
from mpi4py import MPI
# define Trace object: trace line numbers at runtime, exclude some modules

class traced_function(object):

    def __init__(self, function):

        self.function = function

        self.tracer = trace.Trace(
                ignoredirs=[sys.prefix, sys.exec_prefix],
                ignoremods=[
                    'inspect',
                    'contextlib',
                    '_bootstrap',
                    '_weakrefset',
                    'abc',
                    'posixpath',
                    'genericpath',
                    'textwrap',
                    'arrayprint',
                    '_ufunc_config',
                    'fromnumeric',
                    ],
                    trace=1,
                    count=0)

        # by default trace goes to stdout
        # redirect to a different file for each processes
        sys.stdout = open('trace_{:04d}.txt'.format(MPI.COMM_WORLD.rank), 'w')
        
    def trace(self):
        self.tracer.runfunc(self.function)



