import sys
sys.path.append('../quop_mpi/__utils')
from mpi4py import MPI
from __mpi import subcomms

COMM = MPI.COMM_WORLD

scomm = subcomms(1, 4, 4, COMM)
#print(scomm.get_index(), flush = True)

if COMM.Get_rank() == 1:
    local_i = 0
else:
    local_i = 1

#print(scomm.shrink_subcomms(local_i), flush = True)
#
#print(scomm.shrink_subcomms(1), flush = True)
#print(scomm.in_subcomm(), flush = True)
#print(scomm.get_size())

scomm.create_jaccomm(range(1, scomm.get_size()))


if scomm.in_jaccomm():
    print(scomm.JACCOMM.Get_size())
