import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd

figure_size = (4,3)

plt.rcParams.update({'font.size': 9})

figure_size = (4,3)

benchmark_df = pd.read_csv('maxcut_log.csv')

depth_min = benchmark_df['ansatz_depth'].min()
depth_max = benchmark_df['ansatz_depth'].max()

plt.figure(figsize = figure_size)
plt.plot(benchmark_df['ansatz_depth'], benchmark_df['fun'], 'o--')
plt.xticks([i for i in range(depth_min, depth_max + 1)])
plt.xlabel('depth')
plt.ylabel(r'$\langle \vec{\gamma}, \vec{t} | Q | \vec{\gamma}, \vec{t} \rangle$')


plt.tight_layout()
plt.savefig('maxcut_benchmark')
plt.savefig('maxcut_solution', dpi = 200)
