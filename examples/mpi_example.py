"""
We will now implement a simple MPI program, which adds 1s to 50000000 in parallel. This is not a 'useful' program, but it provides an example of how MPI can speed up computation by dividing the computational workload between multiple processors. 
"""

from mpi4py import MPI

mpi_communicator = MPI.COMM_WORLD

rank = mpi_communicator.Get_rank()
size = mpi_communicator.size

"""
Each rank will be responsible for summing an equal portion of the final number.
"""
local_number = int(50000000/size)

local_total = 0
for i in range(local_number):
    local_total += 1

"""
Once each node has finished summing up its local total,
we sum together each local total and send that result to
rank 0.
"""
global_total = mpi_communicator.reduce(local_total, MPI.SUM, root = 0)
"""
The final result is now present at rank 0, but global_total will
not be defined at the other ranks. To print the final result at only 
rank 0, we wrap the print command in a conditional statement.
"""
if rank == 0:
	print(global_total)
"""
Try running this program with a different number of nodes, measuring
the program run-time using the command 'time':

    time mpiexec -N 1 python3 mpi_tutorial_2.py

The time command will provide three times. Of these, we care about the 'real' time. The 'real' is the total time the program took to run as measured by a 'clock on the wall'.

If your computer has more than 1 CPU core, you should notice a speed-up as the number of MPI nodes is increased, up to your computer's total number of CPU cores.

You may also notice that the program does not give the 'correct' output if the number of MPI nodes is not a multiple of 50000000. When writing MPI programs, we must consider carefully how the tasks are divided between the different MPI nodes.

"""
