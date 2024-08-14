import os
from pathlib import Path
import numpy as np
from quop_mpi.algorithm.multivariable import qmoa
from mpi4py import MPI

import sys
sys.path.append('.')
from grav_waves import *


alg_name = 'qmoa_cycle'

# QVA parameters

pmax = 15
repeats = 10

time_limit = 60 * 60 * 12


# GW data info
event_name = 'GW170817'
ifo = 'L1'
input_path_strain = f'gw_data/txt_data/{event_name}_strain_{ifo}.txt'
input_path_psd = f'gw_data/txt_data/{event_name}_psd_{ifo}.txt'

m1_signal = 1.3758 # Msol
m2_signal = 1.3758 # Msol
tc_signal = 170.710205078125 # seconds

# Mass grid
cds = 'theta1eta'
Ns = 2 * [512]
min_mass = 2*[1] # Msol
max_mass = 2*[5] # Msol

exp_name = f'{event_name}__{ifo}__{cds}__Ns_{Ns[0]}_{Ns[1]}'
results_path = f'results/{exp_name}'
suspend_path = f'suspend_data/{exp_name}'

# Create the output directories if they don't exist
if MPI.COMM_WORLD.Get_rank() == 0:
    Path(results_path).mkdir(parents=True, exist_ok=True)
    Path(suspend_path).mkdir(parents=True, exist_ok=True)
    Path('cost_values').mkdir(parents=True, exist_ok=True)

cost_file = f'cost_values/cost_values__{event_name}__{ifo}__{cds}__Ns_{Ns[0]}_{Ns[1]}.h5'

if MPI.COMM_WORLD.Get_rank() == 0:
    if not os.path.exists(cost_file):
        print('Cost file does not exist.')
        print('Generating cost function values...')
        write_cost_values(cost_file, input_path_strain, input_path_psd, cds, 
                          Ns, m1_signal, m2_signal, tc_signal, min_mass, max_mass)
        print('Done.')


# Set up and run algorithm

d = len(Ns)
theta1_grid, eta_grid = grid_setup_theta1eta(Ns, min_mass, max_mass, m1_signal, m2_signal, 4096.0)
deltas = np.array( [ (max(theta1_grid)-min(theta1_grid))/(Ns[0]-1), (max(eta_grid)- min(eta_grid))/(Ns[1]-1) ] )
mins = np.array( [ min(theta1_grid), min(eta_grid) ] )

strides = np.empty(len(Ns), dtype=int)
strides[-1] = 1
for i in range(len(Ns)-2, -1, -1):
    strides[i] = strides[i+1] * Ns[i+1]

Nqubits = [ int(np.log2(n)) for n in Ns ]

alg = qmoa(Nqubits)
alg.set_mixer(len(Ns)*[1])
alg.set_qualities(cost_values_from_file, {'args':[strides, deltas, mins, d, cost_file]})
alg.benchmark(
    range(1, pmax+1),
    repeats,
    param_persist=True,
    filename=f'{results_path}/{alg_name}_p{pmax}_r{repeats}',
    save_action='w',
    time_limit=time_limit,
    suspend_path=f'{suspend_path}/{alg_name}_p{pmax}_r{repeats}'
    )

