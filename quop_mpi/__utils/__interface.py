from mpi4py import MPI
from inspect import signature
from functools import partial
import numpy as np


class interface:
    """
    This class takes an user-input function, instance of a class and
    list of class attributes. It binds the function's positional parameters
    to corresponding class attributes where matches are found; creating
    a partially bound function.

    Function keyword parameters are not bound. They are expected to be defined
    when calling the partially bound function or to have  appropriate default values.

    The bound function is accessible through the 'call' class attribute, e.g.:

        interface.call(**kwargs)

    To update the bound parameters:

        interface.update_parameters()

    """

    def __init__(self, objs, function, function_name, MPI_COMM):

        self.function_name = function_name

        self.rank = MPI_COMM.Get_rank()

        function_signature = signature(function)
        function_parameters = function_signature.parameters

        positional_params = [
            str(param)
            for param in function_parameters.values()
            if param.default == param.empty
        ]

        self.function = function
        self.positional_params = positional_params
        self.objs = objs

        self.update_parameters()

    def update_parameters(self):

        self.args = []
        for positional_param in self.positional_params:
            for obj in self.objs:
                param_value = getattr(obj, positional_param.split(":")[0], None)
                if param_value is not None:
                    self.args.append(param_value)
                    break
        self.call = partial(self.function, *self.args)
