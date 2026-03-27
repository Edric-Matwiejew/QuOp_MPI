from functools import partial
from inspect import Parameter, signature
from warnings import warn

_BINDABLE_MISSING = object()
_BINDABLE_NONE = object()


class Interface:
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
    Any additional positional arguments supplied via a FunctionDict must trail
    all bindable positional parameters in the signature. FunctionDict keyword
    arguments must not target auto-bound parameters or positional-only
    parameters. Bindable keyword-only parameters are not supported.

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

    def __init__(
        self,
        objs,
        function,
        function_name,
        MPI_COMM,
        call_args=None,
        call_kwargs=None,
    ):  # noqa: N803

        self.function_name = function_name
        self.rank = MPI_COMM.Get_rank()

        self.function_signature = signature(function)
        function_parameters = self.function_signature.parameters

        # Only consider regular positional parameters (no defaults)
        # Skip *args and **kwargs (VAR_POSITIONAL and VAR_KEYWORD)
        positional_params = [
            param.name
            for param in function_parameters.values()
            if param.default == param.empty
            and param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        ]

        self.function = function
        self.positional_params = positional_params
        self.objs = objs
        self.call_args = [] if call_args is None else call_args
        self.call_kwargs = {} if call_kwargs is None else call_kwargs

        self._validate_explicit_args_trail_bindable_params()
        self._validate_bindable_keyword_only_params()
        self._validate_explicit_kwargs()
        self.update_parameters(strict_none=True)

    def _has_bindable_attribute(self, param_name):
        return self._get_bindable_value(param_name) is not _BINDABLE_MISSING

    def _get_bindable_value(self, param_name):
        found_none = False

        for obj in self.objs:
            value = getattr(obj, param_name, _BINDABLE_MISSING)

            if value is _BINDABLE_MISSING:
                continue

            if value is None:
                found_none = True
                continue

            return value

        if found_none:
            return _BINDABLE_NONE

        return _BINDABLE_MISSING

    def _validate_explicit_args_trail_bindable_params(self):
        first_explicit = None

        for param in self.function_signature.parameters.values():
            if param.default is not param.empty:
                continue

            if param.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                continue

            if self._has_bindable_attribute(param.name):
                if first_explicit is not None:
                    raise TypeError(
                        f"Interface '{self.function_name}': bindable parameter "
                        f"'{param.name}' appears after explicit positional parameter "
                        f"'{first_explicit}'. FunctionDict['args'] must trail all "
                        f"bindable positional parameters."
                    )
            elif first_explicit is None:
                first_explicit = param.name

    def _validate_bindable_keyword_only_params(self):
        for param in self.function_signature.parameters.values():
            if param.kind == Parameter.KEYWORD_ONLY and self._has_bindable_attribute(param.name):
                raise TypeError(
                    f"Interface '{self.function_name}': bindable keyword-only parameter "
                    f"'{param.name}' is not supported. Bindable parameters must not be "
                    f"keyword-only."
                )

    def _validate_explicit_kwargs(self):
        for kwarg_name in self.call_kwargs:
            param = self.function_signature.parameters.get(kwarg_name)

            if param is None:
                continue

            if param.kind == Parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"Interface '{self.function_name}': FunctionDict['kwargs'] cannot "
                    f"target positional-only parameter '{kwarg_name}'."
                )

            if self._has_bindable_attribute(kwarg_name):
                raise TypeError(
                    f"Interface '{self.function_name}': FunctionDict['kwargs'] cannot "
                    f"target bindable parameter '{kwarg_name}'."
                )

    def update_parameters(self, strict_none=True):

        self.args = []
        unresolved_none_params = set()

        for positional_param in self.positional_params:
            param_value = self._get_bindable_value(positional_param)

            if param_value is _BINDABLE_MISSING:
                continue

            if param_value is _BINDABLE_NONE:
                if strict_none:
                    raise TypeError(
                        f"Interface '{self.function_name}': bindable parameter "
                        f"'{positional_param}' exists on a bound object but is None. "
                        f"Bindable parameters must be populated before call."
                    )
                unresolved_none_params.add(positional_param)
                continue

            self.args.append(param_value)

        try:
            bound = self.function_signature.bind_partial(
                *self.args,
                *self.call_args,
                **self.call_kwargs,
            )
            bound_arguments = bound.arguments
        except TypeError:
            bound_arguments = self.function_signature.bind_partial(*self.args).arguments

        for positional_param in self.positional_params:
            if positional_param in unresolved_none_params:
                continue
            if positional_param not in bound_arguments:
                warn(
                    f"Interface '{self.function_name}': parameter '{positional_param}' "
                    f"not found on any bound object (rank {self.rank}). "
                    f"Positional argument binding may be incorrect.",
                    stacklevel=2,
                )

        self.call = partial(self.function, *self.args)
