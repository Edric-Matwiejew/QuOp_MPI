The 'experiments' folder contains the code used to obtain results detailed in "Quantum Optimisation for Continuous Multivariable Functions by a Structured Search". Optimisation test functions are implement in 'experiments/test_function'.

All of the experiments expect to be ran from the parent folder of 'experiments'. They depend on the following environment variables:

* D			- the simulation dimension.
* REPEATS		- number of simulation repeats at each ansatz depth (p)
* TIMELIMITSECONDS	- Total allowed in-program time auto-suspend
* N			- the number of QUBITS in each dimension (2^N total points)
* PMAX			- the maximum ansatz depth (p)
* MAXITER		- the maximum number of Nelder-Mead iterations
* FUNCINT		- index of a test function (0 to 19)
* REPEATMIN		- minimum repeat index (if REPEAT not used)
* REPEATMAX		- maximum repeat index (if REPEAT not used)
* DMIN			- minimum simulation dimension (if D not used)
* DMAX			- maximum simulation dimension (if D not used)

Example values appropriate for a PC (i.e. not a cluster) are given in 'env/sh'. 
