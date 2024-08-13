from os import environ
from mpi4py import MPI

backends = ["mpi"]
env = environ.get("QUOP_BACKEND")
backend = "mpi" if not env in backends else env
