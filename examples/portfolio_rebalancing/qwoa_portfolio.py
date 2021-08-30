from quop_mpi.algorithms import qwoa
from quop_mpi.operators.diagonal import serial_array
import pandas as pd

qualities_df = pd.read_csv('qwoa_qualities.csv')
qualities = qualities_df.values[:,1]

system_size = len(qualities)

alg = qwoa(system_size)

alg.set_qualities(
        serial_array,
        {'array':qualities})


alg.set_log(
        'qwoa_portfolio_log',
        'qwoa',
        action = 'w')

alg.benchmark(range(1,6),
        1,
        filename = 'qwoa_portfolio',
        save_action = 'w')
