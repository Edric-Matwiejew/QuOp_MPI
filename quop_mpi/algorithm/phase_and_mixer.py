from .. import Ansatz


class phase_and_mixer(Ansatz):

    """An :class:`Ansatz` subclass that used to define a quantum variational
    algorithm consisting of a phase-shift unitary and mixing unitary.

    It also implements methods that mimic the interface of QuOp_MPI v0.0.x.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.operator_function = None
        self.param_function = None

    def set_qualities(self, function, kwargs=None):
        """Define the quality operator :math:`\\text{diag}(\hat{Q})`.

        :param function: Function returning a parition of :math:`\\text{diag}(\hat{Q})` with `local_i` elements and a global positional offset of `local_i_offset`.
        :type function: callable

        :param kwargs: Keyword arguments associated `function`.
        :type kwargs: optional, dictionary, default = None
        """
        if kwargs is None:
            kwargs = {}

        self.operator_function = function
        self.operator_function_kwargs = kwargs

        self.set_observables(0)

    def set_params(self, param_function, kwargs=None):
        """
        Define the initial parameters :math:`\\boldsymbol{\\theta} = \{\\vec{\gamma}, \\vec{t}\}`.

        :param param_function: Function that accepts `n_params` as one of its positional arguments and returns an array of size `n_params`.
        :type param_function: array, float

        :param kwargs: Keyword arguments assocaited with `param_function`.
        :type kwargs: optional, dictionary, default = None
        """
        if kwargs is None:
            kwargs = {}

        self.param_function
        self.param_kwargs = kwargs
