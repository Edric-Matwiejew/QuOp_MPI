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

def random_integers(N, local_i, local_i_offset, seed = 0):
    """
    Random integers evenly distributed between :math:`(1, N)`.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer

    :param seed: Integer to pass to np.random.seed(local_i_offset + seed).
    :type seed: integer, optional, default = 0
    """
    np.random.seed(local_i_offset + seed)
    return np.random.randint(1, N + 1, size = local_i).astype(np.float64)

def random_floats(N, local_i, local_i_offset, seed = 0, low = 0.0, high = 1.0):
    """
    Random floats evenly distributed between :math:`[low, high]`.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer

    :param seed: Integer to pass to np.random.seed(local_i_offset + seed).
    :type seed: integer, optional, default = 0

    :param low: Lower bound.
    :type low: float, optional, default = 0.0

    :param high: Upper bound.
    :type low: float, optional, default = 1.0

    """
    np.random.seed(local_i_offset + seed)
    return np.random.uniform(low = low, high = high, size = local_i)
