from dataclasses import dataclass
import warnings
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


@dataclass
class test_function:

    """Data-class for single-objective test functions.
    
    A callable test function (:math:`f(\\rm{x}) = \mathbb{R}^d \\rightarrow \mathbb{R}`) class
    for benchmarking single-objective optimisers, and storing search domain and function 
    minima information. Callable arguments may be defined as a lambda functions. 

    The ``test_function`` module contains functions defined in: 
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    Parameters
    ----------
    name : str
        Function name.
    required_dimension : int
        Dimension (:math:`d`) for which the function and minima information is defined. 
        ``None`` if defined for all :math:`d \geq 1`.
    search_domain : callable 
        ``search_domain(int d) -> array_like, shape ((2), ...)``. Returns the search domain 
        of :math:`f(\\rm{x})` in :math:`d` as a list of [minimum, maximum] pairs for each
        dimension.
    minimum_point : callable 
        ``minimum_point(int d) -> array_like, shape ((d), ...)``. Returns an array of minima
        coordinates for :math:`f(\\rm{x})` in :math:`d`. 
    function : callable
        ``functiona(array_like d) -> float``. Returns :math:`f(\\rm{x})` for 
        :math:`\\rm{x} \in \mathbb{R}^d`.
    log: bool, optional 
        Set to ``True`` in order to use a logarithmic scale on the Z-axis when plotting using 
        the ``plot`` method.

    """

    name: str
    required_dimension: int
    search_domain: callable
    minimum_point: callable
    function: callable
    log: bool = False

    def __call__(self, x):
        return self.function(np.array(x, dtype=np.float32))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def minimum(self, dimension):
        """Value of  :math:`f(\\rm{x})` at minima in :math:`d`.

        Parameters
        ----------
        dimension : int 
            :math:`d`

        Returns
        -------
        array_like[float]
        """
        return [np.round(self(point), 8) for point in self.minimum_point(dimension)]

    def valid_dimension(self, dimension):
        """Check if the function, global minimum and search domains are defined
        for a particular dimension.

        Parameters
        ----------
        dimension : int
            Target :math:`d`.

        Returns
        -------
        bool
            Validity of class instance at :math:`d`. `True` if valid.
        """
        if self.required_dimension is None:
            return True
        else:
            return dimension == self.required_dimension

    def plot(self, xlim=None, ylim=None):

        """Generate a contour plot of the test function :math:`f(\\rm{x})` at :math:`d = 2`,
        where :math:`\\rm{x} = (x, y)`. A logarithmic scale is used for the Z-axis
        if then ``test_function`` instance was created with `log = True`. Function is
        evaluated on a grid of size (100, 100).

        Parameters
        ----------
        xlim : array_like[float], optional
            X-axis minimum and maximum values, default = ``self.minimum(2)[0]``.

        ylim : array_like[float], optional
            Y-axis minimum and maximum values, default = ``self.minimum(2)[1]``.

        Returns
        -------
        None
            Plot saved to working directory under the function name.
        """

        if xlim is None:
            xlim = self.search_domain(2)[0]
        if ylim is None:
            ylim = self.search_domain(2)[1]

        xlist = np.linspace(xlim[0], xlim[1], 100)
        ylist = np.linspace(ylim[0], ylim[1], 100)

        X, Y = np.meshgrid(xlist, ylist)

        Z = np.array(
            [self(point) for point in zip(X.flatten(), Y.flatten())], dtype=np.float64
        )
        Z = Z.reshape((100, 100))

        fig, ax = plt.subplots(1, 1)

        if self.log == False:
            cp = ax.contourf(X, Y, Z)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cp = ax.contourf(X, Y, Z, locator=ticker.LogLocator())

        fig.colorbar(cp)
        ax.set_title(self.name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        plt.savefig(f"{self.name}.png")

    def report(self):
        """Print the:

            * Function name
            * Minimum/minima points at :math:`d = 2`
            * Mini/minima values at :math:`d = 2`

        And call the ``plot`` method.


        """
        print(
            f"Name: {self.name}\n"
            f"Minimum/minima points (d = 2) at: {self.minimum_point(2)}\n"
            f"Minimum/minima values (d = 2): {self.minimum(2)}\n"
            f'Contour plot saved to: "{self.name}.png".\n'
        )
        self.plot()


rastrigin = test_function(
    "Rastrigin function",
    None,
    lambda dimension: dimension * [[-5.12, 5.12]],
    lambda dimension: [dimension * [0]],
    lambda x: len(x) * 10 + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(),
)

ackley = test_function(
    "Ackley function",
    None,
    lambda dimension: dimension * [[-32.768, 32.768]],
    lambda dimension: [dimension * [0]],
    lambda x: -20 * np.exp(-0.2 * np.sqrt(1/len(x) * (x ** 2).sum()))
    - np.exp(1/len(x) * np.cos(2 * np.pi * x).sum())
    + np.e
    + 20,
)

rosenbrock = test_function(
    "Rosenbrock function",
    None,
    lambda dimension: dimension * [[-3, 3]],
    lambda dimension: [dimension * [1]],
    lambda x: (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
    log=True,
)

# 2D only
beale = test_function(
    "Beale function",
    2,
    lambda *args: 2 * [[-4.5, 4.5]],
    lambda *args: [[3, 0.5]],
    lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2
    + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
    + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2,
    log=True,
)

sphere = test_function(
    "Sphere function",
    None,
    lambda dimension: dimension * [[-2, 2]],
    lambda dimension: [dimension * [0]],
    lambda x: (x**2).sum(),
)

# 2D only
goldstein_price = test_function(
    "Goldstein Price function",
    2,
    lambda *args: 2 * [[-2, 2]],
    lambda *args: [[0, -1]],
    lambda x: (
        1
        + (x[0] + x[1] + 1) ** 2
        * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    )
    * (
        30
        + (2 * x[0] - 3 * x[1]) ** 2
        * (
            18
            - 32 * x[0]
            + 12 * x[0] ** 2
            + 48 * x[1]
            - 36 * x[0] * x[1]
            + 27 * x[1] ** 2
        )
    ),
    log=True,
)

# 2D only
booth = test_function(
    "Booth function",
    2,
    lambda *args: 2 * [[-10, 10]],
    lambda *args: [[1, 3]],
    lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
    log=True,

)

bukin = test_function(
    "Bukin function N. 6",
    2,
    lambda *args: [[-15, -5], [-3, 3]],
    lambda *args: [[-10, 1]],
    lambda x: 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10),
)

matyas = test_function(
    "Matyas function",
    2,
    lambda *args: 2 * [[-10, 10]],
    lambda *args: [[0, 0]],
    lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1],
    log=True,
)

# removed unicode
levi = test_function(
    "Levi function N.13",
    2,
    lambda *args: 2 * [[-10, 10]],
    lambda *args: [[1, 1]],
    lambda x: np.sin(3 * np.pi * x[0]) ** 2
    + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
    + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2),
)

himmelblau = test_function(
    "Himmelblaus function",
    2,
    lambda *args: 2 * [[-5, 5]],
    lambda *args: [
        [3, 2],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ],
    lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
)

three_hump = test_function(
    "Three-hump camel function",
    2,
    lambda *args: 2 * [[-5, 5]],
    lambda *args: [[0, 0]],
    lambda x: 2 * x[0] ** 2
    - 1.05 * x[0] ** 4
    + x[0] ** 6 / 6
    + x[0] * x[1]
    + x[1] ** 2,
    log=True,
)

easom = test_function(
    "Easom function",
    2,
    lambda *args: 2 * [[-100, 100]],
    lambda *args: [[np.pi, np.pi]],
    lambda x: -np.cos(x[0])
    * np.cos(x[1])
    * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)),
)

cross = test_function(
    "Cross-in-tray function",
    2,
    lambda *args: 2 * [[-10, 10]],
    lambda *args: [
        [1.34941, -1.34941],
        [1.34941, 1.34941],
        [-1.34941, 1.34941],
        [-1.34941, -1.34941],
    ],
    lambda x: -0.0001
    * (
        np.abs(
            np.sin(x[0])
            * np.sin(x[1])
            * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        )
        + 1
    )
    ** 0.1,
)

eggholder = test_function(
    "Eggholder function",
    2,
    lambda *args: 2 * [[-512, 512]],
    lambda *args: [[512, 404.2319]],
    lambda x: -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47))))
    - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47)))),
)

# removed unicode
holder = test_function(
    'Holder table functon',
    2,
    lambda *args: 2 * [[-10, 10]],
    lambda *args: [
        [8.05502, 9.66459],
        [-8.05502, 9.66459],
        [8.05502, -9.66459],
        [-8.05502, -9.66459],
    ],
    lambda x: -np.abs(
        np.sin(x[0])
        * np.cos(x[1])
        * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
    ),
)

mccormick = test_function(
    "McCormick function",
    2,
    lambda *args: [[-1.5, 4], [-3, 4]],
    lambda *args: [[-0.54719, -1.54719]],
    lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
)

schaffer_2 = test_function(
    "Schaffer function N. 2",
    2,
    lambda *args: 2 * [[-100, 100]],
    lambda *args: [[0, 0]],
    lambda x: 0.5
    + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5)
    / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2,
)

schaffer_4 = test_function(
    "Schaffer function N. 4",
    2,
    lambda *args: 2 * [[-100, 100]],
    lambda *args: [[0, 1.25313], [0, -1.25313], [1.25313, 0], [-1.25313, 0]],
    lambda x: 0.5
    + (np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5)
    / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2,
)

styblinski_tang = test_function(
    "Styblinski-Tang function",
    None,
    lambda dimension: dimension * [[-5, 5]],
    lambda dimension: [dimension * [-2.903534]],
    lambda x: (x**4 - 16 * x**2 + 5 * x).sum() / 2,
)

functions = [func for func in gc.get_objects() if isinstance(func, test_function)]

functions_d3 = [func for func in gc.get_objects() if (isinstance(func, test_function) and (func.required_dimension is None))]

names = [f"\"{func.name}\"" for func in functions]
get = {func.name:func for func in functions}

def main():

    for func in functions:
        func.report()


if __name__ == "__main__":
    main()
