"""
Welcome to this QuOp_MPI tutorial. Before reading through this program,
it might be useful to read through the "Mathematical Framework" section of
the QuOp_MPI documentation:

https://quop-mpi.readthedocs.io/en/latest/symbols.html


If you are using the Windows Subsystem for Linux (WSL), we will first create a folder in Windows
from which you will be able to modify the code you will be running on Linux. After opening WSL,
chnage to your user folder in windows:

    cd /mnt/c/Users/<your username>

And then create a folder to store this file and its results. And chnage to that directory.

   mkdir QuOp_MPI
   cd QuOp_MPI

In Windows, save or copy this file to that directory. Afterwards, if you list the contents of the 
folder in WSL

   ls

You should see a file called "quop_tutorial.py".

If you are using stand-alone Linux, you can create a folder using the Linux graphical file manager and move
or save "quop_tutorial.py" to that folder. Next, open a terminal in that folder or 'cd' to that folder.

To start with, we will run our simulations as serial-code (using 1 MPI process). The command to do
so is:

    mpiexec -N 1 python3 quop_tutorial.py

"""

"""
First we will import the required python modules
"""

from mpi4py import MPI # Provides for parallel MPI computation.
import numpy as np # NumPy makes numerical computation easier and faster in python.
import quop_mpi as qu # Our QWOA simulation package
import matplotlib.pyplot as plt # MatPlotLib is used to visualise our results.
import h5py as h5 # h5py supports saving and accessing HDF5 (.h5) files in python, QuOp_MPI uses this format.


"""
Becuase QuOp_MPI uses MPI we have to create an MPI communicator.
But don't worry about this for know, we will disucss MPI another time.
"""
comm = MPI.COMM_WORLD


"""
We will know define our QAOA simulation parameters.

'p' is the number of 'steps' used, each step consists of a continuous-time quantum walk (CTQW)
followed by a phase shift. Each CTQW and phase shift as a scalar parameter (as described by
Step 4 of the Mathematical Framework) meaning that we have 2p parameters in total.

'n_qubits' defines the number of qubits we will be simulating. Due to the quantum property of superposition,
n qubits can represent 2^n possible solutions. So with 'n_qubits = 3', we can optimise over a set of 8 possible
solutions.
"""
n_qubits = 3
p = 3


"""
The goal of the QWOA algorithm is to use a classical optimiser to arrive at a set of 2p parameters such that
the 2p pairs of CTQWs and phase shifts result in a quantum system state with a high probability of being
measured in a state corresponding to a 'high quality solution'. 

Below we define a method 'x0' which we will use to generate random starting parameters.
"""

np.random.seed(1) # This ensures that the same 'random' numbers are generated whenever we run this program.
def x0(p):
    return np.random.uniform(low = 0, high = 1, size = 2 * p)

"""
We can now create a qwoa object. To do so we must initialise it with the number of qubits and the
MPI communicator object.
"""

qwoa = qu.MPI.qwoa(n_qubits, comm)

"""
To keep track of our results over repeated runs of this program (prehaps with different parameters), we
now instruct QuOp_MPI to save information to "log.csv", in which this simulation with be labelled as "qwoa".
By choosing "action = "a" we tell QuOp_MPI to append to "log.csv", instead over overwritting it (action = "w").
"""
qwoa.log_results("log", "qwoa", action = "a")
"""
Next we will define the topology of the CTQW. For reasons that Jingbo wilil soon explain we require that 
the graph be circulant. A circulant graph can be represented by a circulant adjacency matrix. For example,
consider a complete graph with 3 vertices, it will have the adjacency matrix:

        0 1 1
    G = 1 0 1
        1 1 0

Where the row i corresponds to vertex i of the graph and a '1' indicates an edge between vertex i and j. 

Importly for our simulations, is that the first row of the adjacency matrix completely specifies the graph:

    g = [0 1 1]

As we can see that each row after is simply this vector 'shifted' by one place. 

So, to define the adjacency matrix of the CTQW, we create an array describing the 1st row of our graph,
passing it to 'qwoa.set_graph'. This array must of size 1 x 2^n_qubits.
"""

complete_graph = np.ones(n_qubits**2)
complete_graph[0] = 0 

qwoa.set_graph(complete_graph)
"""
Next we define the stating state of the quantum system, Psi(0). 

To start the system as a equal superposition, we pass the argument "name = 'equal'" to the 
"set_initial_state" method. 

Otherwise, if we wish to start in a localised state at the first graph vertex, we can instead pass
"name = 'localized'".
"""
qwoa.set_initial_state(name="equal")

"""
We will now set the qualities over which we are optimizing. In actual application, these qualities are
scalar variables, each corresponding to a solution. These qualties are computed in 'quantum parallel',
meaning that we can calculate every possible solution in a single step! However, if we measure the quantum
system it will collapse into a single (maybe not very good) state. QWOA is a method for increasing our chances
of measuring a good solution.

For our simulations we can see the best quality solution simply by looking at the array, or by using:

    np.min(qualities)

If we want to minimise, or:

    np.max(qualities)

If we want to maximise.

But for the problems we want QWOA to solve, the array of qualities would too large to fit in a classical computer's memory!

For this example we will create an array of random floats, if QWOA works it should be able to pick out one of the smaller
or larger values.
"""

qwoa.set_qualities(qu.qualities.random_floats)

"""
Note: By default, QuOp_MPI seeks to minimise the problem. To find the maximum quality we could use:

    qualities = -np.random.uniform(low = 0, high = 1, size = 2 * n_qubits)
"""

"""
Now we can run the simulation. The command 'qwoa.size()' optimises the numerical methods used in the simulation. Next, 
'qwoa.execute' runs the qaoa simulation with intial parameters as defined by x0(p). The repeatedly applies the 2p
CTQWs and phase shifts, each time varying the t and gamma parameters to minimise the expectation value of the quantum system.
Lower expectation value corresponds to a higher chance of measuring a better quality solution.

We then save the simulation results to a HDF5 file (.h5) called "qwoa", under the heading "example_config". We can save
multiple simulation results to same .h5 file, but they must have unique names. e.g. "example_config_1", "example_config_2".
"""
qwoa.plan()
qwoa.execute(x0(p))
qwoa.save("qwoa", "example_config", action = "w")
"""
Once the smulation as complete, "qwoa.destroy_plan()" frees up computer memory. We can then see the results of the optimisation 
process by calling "qwoa.print_result". The provides a lot of information relating to optimisation process. Of most interest are

    fun: The final expectation value of the system (lower is better).
    x: The optimal values of gamma and t.
"""
qwoa.destroy_plan()
qwoa.print_result()

"""
Now we have carried out a simulation let's look at the results!

To open a .h5 file in python we will use h5py.
Below we are opening the "qwoa.h5" as a read-only file and loading from it the final state of the system and the 
qualities we used for the simulation.
"""
f = h5.File('qwoa.h5', 'r')
final_state = np.array(f["example_config"]["final_state"]).view(np.complex128)
qualities = np.array(f["example_config"]["qualities"]).view(np.float64)


"""
To plot the results we are using Matplotlib. 
"""

ax1 = plt.gca()
ax1.plot(np.abs(final_state)**2, label = r'Final quantum state, $<\vec{t}, \vec{\gamma}|q_i|\vec{t}, \vec{\gamma} >$')
ax1.plot(qualities,'*', color = 'red', label = r'Qualities, $\vec{q} = q_i$.')
plt.legend()
ax1.set_ylabel("Probability")
ax2 = ax1.twinx()
ax2.set_ylabel("quality")
ax1.set_xlabel("Quantum State/Possible Solution")
plt.savefig("qwoa_final_state")
plt.close()
"""
A plot called "qwoa_final_state" should be saved to the same folder as this file, showing
the probabilities of the final quantum state and the qualities. It shows that that the system
will most like be measured in a state corresponding to the lowest, or second lowest, quality
value.

Once you have ran this program and feel comfortable with the code try to modifiy the simulation.
Some things to consider are:

    1. What happens as we increase p?
    2. WHat happens if we change the initial quantum state from an equal superposition?
    3. Can you make the simulation converge to the *highest* quality value?
    4. Does it matter if we change the graph used for the CTQW?
            
            e.g. An edge can be removed from each graph graph vertex of the complete graph like so:
                complete_graph[1] = 0             

Good luck!                
"""
