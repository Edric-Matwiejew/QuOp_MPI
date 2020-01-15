import numpy as np

"""
Functions for parallel distributed memory generation of the QWAO quality array.
"""
def integer(N, local_i, local_i_offset):
    """
    Produces the array [1, ..., N - 1] distributed of the active MPI communicator.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the \ 
    zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer.
    """
    return np.asarray(range(local_i_offset, local_i_offset + local_i), dtype = np.float64)
