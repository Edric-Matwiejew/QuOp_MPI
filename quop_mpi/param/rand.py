import numpy as np

def uniform(
        n_params,
        seed,
        low = 0,
        high = 2*np.pi):
    """Generate variational parameters :math:`\\theta` randomly from a uniform
    distribution between (`low`, `high`].

    :param n_params: Number of variational parameters.
    :type n_params: integer

    :param seed: Sets the state of the random number generator.
    :type seed: integer

    :param low: Lower bound of the distribution.
    :type low: optional, float, default = 0.

    :param high: Upper bound of the distribution.
    :type high: optional, float, default = :math:`2 \pi`
    """

    np.random.seed(seed)

    return np.random.uniform(low = low, high = high, size = n_params)
