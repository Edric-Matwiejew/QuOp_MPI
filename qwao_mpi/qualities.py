import numpy as np

def ordered_integers(N, local_i, local_i_offset):
    """
    The array [1, ..., N].

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer.
    """

    return np.asarray(range(local_i_offset, local_i_offset + local_i), dtype = np.float64)

def random_integers(N, local_i, local_i_offset):
    """
    Random integers evenly distributed between :math:`(1, N)`.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer
    """

    np.random.seed(local_i_offset)
    return np.random.randint(1, N + 1, size = local_i, dtype = np.float64)
