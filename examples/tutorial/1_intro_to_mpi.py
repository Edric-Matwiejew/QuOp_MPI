"""
This tutorial provides an introduction to MPI in python.

MPI (Message Passing Interface) is a protocol used for parallel computation. 
It is designed to allow multiple CPUs with isolated memory to perform tasks in 
parallel, communicating with each other through 'message passing'. These CPUs 
could be a network of desktop computers or a supercomputer cluster (like Magnus). 
MPI is highly portable and scalable, making it one of the most commonly used 
forms of parallelisation in scientific computing.

You will not need to do much MPI programming to use QuOp_MPI, but understanding 
how QuOp_MPI achieves parallelisation will assist you in defining your quality functions.

MPI has implementations in the  C, Fortran, Java and Python programming languages. 
Here we will be using the python implementation of MPI, 'mpi4py'.
"""

from mpi4py import MPI

"""
MPI programs work by running multiple copies of the same program simultaneously, 
these programs work independently, except when they send messages to each other 
using calls to MPI functions. Each copy of the program is referred to as an MPI 'node', 
and an 'MPI communicator' groups these nodes. In an MPI communicator, the nodes 
are individually identified by a 'rank' number.

The number of nodes is chosen when the MPI program is run, the terminal command:

    mpiexec -N 1 python3 mpi_tutorial.py

Will run 1 MPI node with rank number 0. While:


    mpiexec -N 4 python3 mpi_tutorial.py

Will run 4 MPI nodes with ranks numbers 0, 1, 2 and 3.

So, the first thing we must do for all MPI programs is to create an MPI communicator:
"""

mpi_communicator = MPI.COMM_WORLD

"""
We can use this 'MPI communicator' object to perform MPI operations. 
For example, we can obtain the MPI node rank.
"""

rank = mpi_communicator.Get_rank()
print('rank:', rank)

"""
Or the total number of nodes in the MPI communicator:
"""

comm_size = mpi_communicator.size
print('communicator size:', comm_size)

"""
Try running this code with different numbers of nodes and examine the output. 
Do the nodes always execute in the same order?
"""
