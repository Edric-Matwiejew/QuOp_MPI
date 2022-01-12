from scipy.optimize import minimize


class ScipyOptimiser():
    def __init__(self, method="BFGS"):
        self.objective=None
        self.method=method

        
    def __call__(self, func, param, **kwargs):
        self.objective = func
        return(
            minimize(self.objective,
                     param,
                     method=self.method))

