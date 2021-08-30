import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd

figure_size = (4,3)

plt.rcParams.update({'font.size': 9})

figure_size = (4,3)

qwoa_benchmark_df = pd.read_csv('qwoa_portfolio_log.csv')
qaoaz_benchmark_df = pd.read_csv('qaoaz_portfolio_log.csv')

qwoa_depth_min = qwoa_benchmark_df['ansatz_depth'].min()
qwoa_depth_max = qwoa_benchmark_df['ansatz_depth'].max()

plt.figure(figsize = figure_size)
plt.plot(qwoa_benchmark_df['ansatz_depth'], qwoa_benchmark_df['fun'], 'o--', label = "QWOA")
plt.plot(qaoaz_benchmark_df['ansatz_depth'], qaoaz_benchmark_df['fun'], 'o--', label = "QAOAZ")
plt.xticks([i for i in range(qwoa_depth_min, qwoa_depth_max + 1)])
plt.xlabel('depth')
plt.ylabel(r'$\langle \vec{\gamma}, \vec{t} | Q | \vec{\gamma}, \vec{t} \rangle$')
plt.legend()

plt.tight_layout()
plt.savefig('portfolio_rebalancing', dpi = 200)
