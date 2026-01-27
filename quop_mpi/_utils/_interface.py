from mpi4py import MPI
from inspect import signature
from functools import partial
import warnings
import numpy as np


class interface:
    """
    This class takes a user-input function, instance of a class and
    list of class attributes. It binds the function's positional parameters
    to corresponding class attributes where matches are found; creating
    a partially bound function.

    Function keyword parameters are not bound. They are expected to be defined
    when calling the partially bound function or to have appropriate default values.

    The bound function is accessible through the 'call' class attribute, e.g.:

        interface.call(**kwargs)

    To update the bound parameters:

        interface.update_parameters()

    Parameter Binding
    -----------------
    Positional parameters in the function signature are matched by name to
    attributes of the provided objects (typically an Ansatz instance). If a
    parameter name matches an attribute name, that attribute's value is bound.

    Common bindable attributes from Ansatz:
        - system_size: Total number of quantum basis states
        - local_i: Number of elements in this rank's partition
        - local_i_offset: Global index offset for this rank
        - partition_table: Array describing global partitioning
        - observables: Local partition of observable values
        - ansatz_depth: Number of ansatz iterations
        - total_params: Parameters per ansatz iteration
        - MPI_COMM: MPI subcommunicator

    See Ansatz.get_bindable_attributes() for a complete list.
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
        self.unbound_params = []  # Track which params couldn't be bound

        self.update_parameters()

    def update_parameters(self):

        self.args = []
        self.unbound_params = []
        
        for positional_param in self.positional_params:
            param_name = positional_param.split(":")[0]
            bound = False
            for obj in self.objs:
                param_value = getattr(obj, param_name, None)
                if param_value is not None:
                    self.args.append(param_value)
                    bound = True
                    break
            if not bound:
                self.unbound_params.append(param_name)
        
        # Warn on rank 0 if there are unbound parameters that aren't clearly
        # meant to come from FunctionDict args/kwargs
        if self.rank == 0 and self.unbound_params:
            # Only warn if the first few expected params weren't bound
            # (later params are likely from FunctionDict)
            n_bound = len(self.args)
            n_total = len(self.positional_params)
            if n_bound < n_total:
                warnings.warn(
                    f"{self.function_name} function: {len(self.unbound_params)} "
                    f"positional parameter(s) not bound to Ansatz attributes: "
                    f"{self.unbound_params}. These must be provided via FunctionDict['args']. "
                    f"Use Ansatz.get_bindable_attributes() to see available bindings.",
                    stacklevel=4
                )
        
        self.call = partial(self.function, *self.args)
