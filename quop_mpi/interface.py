from mpi4py import MPI
from inspect import signature
from functools import partial
import numpy as np

class interface(object):

    def __init__(
            self,
            obj_self,
            signatures,
            returns,
            function_name,
            function,
            variational,
            MPI_communicator,
            **kwargs):

        self.function_name = function_name
        self.returns = returns

        self.rank = MPI_communicator.Get_rank()

        function_signature = signature(function)
        parameters = function_signature.parameters
        params = [param for param in parameters.values() if (param.default == param.empty)]
        n_params = len(params)

        for key in signatures.keys():

            n_check = len(signatures[key])

            if ((n_params > n_check) and variational) or (n_params == n_check):

                arg_zip = zip(params, signatures[key])
                match = self.compare_args(arg_zip)

            if match:
                self.signature_type = key
                break

        else:

            raise RuntimeError("Not found!")

        args = []
        for param_name in signatures[self.signature_type]:
            args.append(getattr(obj_self, param_name))

        self.function = partial(function, *args, **kwargs)

        if variational:
            self.n_variational_params = n_params - n_check
        else:
            self.n_variational_params = 0

    def compare_args(self, compare_args):

        for arg_1, arg_2 in compare_args:
            print(str(arg_1), str(arg_2), flush = True)
            if str(arg_1) != str(arg_2):
                return False
        else:
            return True

    def call(self, *args):

        self.outputs = self.function(*args)

        self.__validate_output()

        return self.outputs

    def __validate_output(self):

        expected_output_types = self.returns[self.signature_type]

        if isinstance(self.outputs, tuple):
            for i, (output, expected) in enumerate(zip(self.outputs, expected_output_types)):
                self.__check_element(output, expected, i)
        else:
            self.__check_element(self.outputs, expected_output_types, 0)

    def __check_element(self, obj, e_obj, indx):

        try:
            iter(obj)
        except:
            raise TypeError("Output of {} index {} is not iterable.".format(self.function_name, indx))

        o_type = type(obj)
        e_type = type(e_obj)

        o_el_type = type(obj[0])
        e_el_type = type(e_obj[0])

        # attempt type conversion if there is a mismatch
        if (o_type != e_type) or (o_el_type != e_el_type):
            obj = np.array(obj, e_el_type)

        assert ((o_type == e_type) and (o_el_type == e_el_type)), \
                "Mismatch in {} at output element {}: cannot cast from a {} of {} to {} of {}".format(name, indx, o_type, o_el_type, e_type, e_el_type)

        assert (obj.ndim == e_obj.ndim), \
                "Mismatch in {} at output element {}: expect ndim {}, received ndim {}.".format(name, indx, o_obj.ndim, e_obj.ndim)
