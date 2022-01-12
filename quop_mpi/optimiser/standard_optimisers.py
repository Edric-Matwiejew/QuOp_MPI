from scipy.optimize import minimize as sp_minimize
from quop_mpi.__utils.__nlopt_wrap import minimize as nlopt_minimize




class ScipyOptimiser():
    def __init__(self, method="BFGS", **kwargs):
        self.objective=None
        self.method=method
        self.kwargs=kwargs

        
    def __call__(self, func, param, **kwargs):
        self.objective = func
        return(
            sp_minimize(self.objective,
                     param,
                     method=self.method,
                     **self.kwargs))


class NloptOptimiser():
    def __init__(self, **kwargs):
        self.objective=None
        self.kwargs=kwargs

        
    def __call__(self, func, param, **kwargs):
        self.objective = func
        return(
            nlopt_minimize(self.objective,
                           param,
                           **self.kwargs))

