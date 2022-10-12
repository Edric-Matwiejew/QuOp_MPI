from mpi4py import MPI
import numpy as np
import fCQAOA

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

'''
These functions return the value of the momentum mixer, initial wavefunction and
cost function at grid point x = (x_1, x_2, ..., x_N), where N is the total number
of points.
'''

def mixer(x):
    return np.sum(x**2)/len(x)

def initial_state(x):

    n_dim = len(x)
    mean = np.array(n_dim*[0], dtype = np.float64)
    std = np.array(n_dim*[np.exp(1)/np.sqrt(2)], dtype = np.float64)
    velocity = np.array(n_dim*[0], dtype = np.float64)

    state = 0

    for x_i, m, s, v in zip(x, mean, std, velocity): 
        state += np.exp(1j*v*x_i)*np.exp((-(x_i - m)**2)/(2*s**2))/(np.sqrt(np.pi)*s)
    return state

def cost_function(x):
    return np.sum(x**4 - 16*x**2 + 5*x)/len(x)

'''
Simulation parameters, same as Octave example.
'''
p = 3 # anzatz repeats
tM = 0.1 # momentum mixer time
tC = 0.1 # cost function time

M = 20 # minimum number of grid points in fastest mixer oscillation
L = 4 # box size in position space

# estimate number of points needed for good approximation
#n = np.ceil(np.max([np.pi*M*tM/dq**2, 2*L/dq])) 

d = 2
n = 2**5
dq = 2*L/n # position space grid spacing


dk = 2*np.pi/(n*dq) # momentum space grid spacing

'''
Simulation parameters, particular to this program.

The number of grid points, and grid size is specified for each
dimension. The flexibility may be useful at some point.
'''

Ns = d*[n] # number of grid points

N = 1
for dim in Ns:
    N *= dim

# grid sizes
deltasq = np.array(d*[dq], dtype = np.float64)
deltask = np.array(d*[dk], dtype = np.float64)
# minimum grid values
minsq = np.array(d*[-(n/2)*dq], dtype = np.float64)
minsk = np.array(d*[-(n/2)*dk], dtype = np.float64)

'''
MPI calls start here.

The multidimensional grid is partitioned along its first
dimension (local_n0, local_n0_offset). 

At all stages, the state vector and matrix operators are 
vectorised (no reshaping). Local partitions are realted
to the global array via local_i and local_i_offset.

'''

def phase_k(x):
    return np.exp(-1.0j*np.sum(x*minsq))


def phase_q(x):
    return np.exp(1.0j*np.sum(x*minsk))

# initialise the default MPI communicator

def pprint(message, rank = rank):
    if rank == 0:
        print(message, flush = True)

if rank == 0:
    print(f'Running on {size} MPI processes.')
    print(f'Position grid spacing: {dq}, Dimension: {d}.')

# determine parallel partitioning 
part = fCQAOA.continuous.plan_partition(Ns, COMM.py2f())

alloc_local = part[0]
local_i = part[1]
local_i_offset = part[2]
local_n0 = part[3] 
local_n0_offset = part[4]
strides = part[5]
pprint('Planned parallel Partitions.')

'''
For vectorised index in [local_i_offset, local_i_offset + local_i]
'dist_vector' computes the corresponding grid point and passes
the point to a function (first argument) whose return value defines 
a single point in a vectorised operator or state vector array (last argument).

On-the-fly computation of the grid points is done for memory efficiency
at the expense of a constant-time computational overhead.
'''


pk = np.empty(shape= [local_i], dtype = np.complex128)
pq = np.empty(shape= [local_i], dtype = np.complex128)

fCQAOA.continuous.dist_vector(
        phase_k,
        Ns,
        strides,
        deltask,
        minsk,
        local_i_offset,
        pk)

fCQAOA.continuous.dist_vector(
        phase_q,
        Ns,
        strides,
        deltasq,
        minsq,
        local_i_offset,
        pq)

# initial squeezed gaussian state
state = np.empty(shape= [alloc_local], dtype = np.complex128)
# cost operator
costs = np.empty(shape = [local_i], dtype = np.complex128)

fCQAOA.continuous.dist_vector(
        cost_function,
        Ns,
        strides,
        deltasq,
        minsq,
        local_i_offset,
        costs)

pprint('Generated costs.')

# momentum space mixer
mix = np.empty(shape= [local_i], dtype = np.complex128)

fCQAOA.continuous.dist_vector(
        mixer,
        Ns,
        strides,
        deltask,
        minsk,
        local_i_offset,
        mix)

pprint('Generated momentum mixer.')

'''
Set up ancilla data-sctructures used by FFTW.
'''

fCQAOA.continuous.evolve(
        N,
        Ns,
        local_i_offset,
        local_n0,
        strides,
        1,
        mix,
        pk,
        pq,
        state,
        COMM.py2f(),
        1)

pprint('Planned FFT.')

fCQAOA.continuous.dist_vector(
        initial_state,
        Ns,
        strides,
        deltasq,
        minsq,
        local_i_offset,
        state[:local_i])

# normalize the wavefunction
norm = np.sum(np.abs(state[:local_i])**2)
norm = COMM.allreduce(norm, op = MPI.SUM)
state = state/np.sqrt(norm)


'''
Ansatz iterations
'''
print('start')
for s in state:
    print(s)
for _ in range(p):

    print(state[0])
    print(costs[0])
    print(strides)
  
    # phase shift
    state[:local_i] = np.multiply(np.exp(-1.0j* tC * costs),state[:local_i])

    print(state[0])
    print(mix[0])
    print(strides)
            # Forward FFT, mix in momentum basis, backward FFT.
    fCQAOA.continuous.evolve(
            N,
            Ns,
            local_i_offset,
            local_n0,
            strides,
            tM,
            mix,
            pk,
            pq,
            state,
            COMM.py2f(),
            0)
    
pprint('Computed ansatz iterations.')

probs = np.abs(state)**2
indx = np.argmax(probs)
indxs = COMM.gather(indx, root = 0)
maxes = COMM.gather(probs[indx], root = 0)

if rank == 0:
    best = np.argmax(maxes)
    inds = fCQAOA.continuous.get_index(indxs[best]+ 1, Ns, strides)
    grid_points = inds*deltasq
    grid_points += minsq
    best_value = cost_function(grid_points)    
    global_min = cost_function(np.array(d*[-2.90353]))
    print(f'Most amplified solution: {best_value} at x={grid_points}, Global minimum:{global_min}')


'''
Destroy ancilla data-sctructures used by FFTW.
'''

fCQAOA.continuous.evolve(
        N,
        Ns,
        local_i_offset,
        local_n0,
        strides,
        1,
        mix,
        pk,
        pq,
        state,
        COMM.py2f(), 
        -1)

pprint('destroy')
