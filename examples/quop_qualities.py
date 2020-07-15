"""
This tutorial uses the python package 'pandas'. This allows us to work with .csv file in python.
Install pandas using pip:

    pip3 install pandas

The main task in creating a quop_mpi simulation is defining a function
to create a list of qualities relating to the specific problem you would
like to optimise. This function requires specific inputs for it
to be compatible with the MPI operations used by QuOp_MPI.

To start, we will need to consider how QuOp_MPI distributes the qualities over an MPI communicator. For example, if we have a list of integer qualities:

   q_global = [0, 1, 2, 3, 4, 5, 6, 7, 8]

And an MPI communicator of size 3, QuOp_MPI will distribute q_global as:

    rank 0: q_local = [0, 1, 2]
    rank 1: q_local = [3, 4, 5]
    rank 2: q_local = [6, 7, 8]

Each rank has 3 local qualities (local_i = 3). At rank 0 the offset of the
local qualities relative to q_global is 0, whereas at
rank 1 the offset is 3, and at rank 2 the offset is 6.

With that in mind, all QuOp_MPI quality methods must have the following signature:

    def quality_function_name(N, local_i, local_i_offset, seed = None):

* 'N' is the size of q_global (the total size of the simulated system).

* 'local_i' is the number of quality elements at a given MPI rank.

* 'local_i_offset' is the offset of the local quality array indexes, relative to 'q_global'.
 
* 'seed' is a number used to set the seed for any random number generation used by the quality function. If the quality function does not use random number generation, this is euqal to 'None'.

For example, consider the following quality method, which produces a list of ordered integers, following the pattern shown above.
"""

import numpy as np
import pandas as pd

def ordered_integers_A(N, local_i, local_i_offset, seed = None):

    q_global = np.asarray(range(N), dtype = np.float64)

    return q_global[local_i_offset:local_i_offset + local_i]

"""
The above function creates q_global at each MPI rank and then returns the slice needed at that rank. 

However, this approach is computationally inefficient as each rank creates all of q_global, when it only needs a slice of the array. The function shown below creates only the qualities needed at each rank, such that q_global only ever exists  over the
MPI communicator.

"""

def ordered_integers_B(N, local_i, local_i_offset, seed = None):

    return np.asarray(range(local_i_offset, local_i_offset + local_i), dtype = np.float64)

"""
It may also be the case that creating the qualities at run-time is too time-consuming. In that case, we can create the qualities ahead of time and save them to disk.
"""

def create_and_save_integer_qualities(N):

    # Note: This function should only be called at a single MPI rank.

    q_global = np.asarray(range(N))
    
    pd.DataFrame(q_global).to_csv('q_global.csv')    
    
"""
Then have the quality function load those values from disk.
"""

def ordered_integers_C(N, local_i, local_i_offset, seed = None):

    q_global = pd.read_csv('q_global.csv')['0'].values

    return q_global[local_i_offset:local_i_offset + local_i]

"""
The specific method you choose will depend on the computational requirements of your simulation. For instance, while 'ordered_integers_B' is the most efficient approach here,
creating the qualities as independent slices might not always be possible. 

Try implementing a QWOA simulation using these quality functions; the results should be the same in each instance. After this, you can use these functions as a starting point for your quality functions.

Remember to import the required python modules (e.g. quop_mpi). If you would like to use the quality functions you've written in another python script you can import them as a module via:

    import 3_quop_qualities as quop_qualities

The quality functions can then be called from that file:

    qwoa.set_qualities(quop_qualities.ordered_integers_A)

Good luck!
"""
