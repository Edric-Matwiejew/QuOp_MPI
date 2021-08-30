import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import networkx as nx

plt.rcParams.update({'font.size': 9})

figure_size = (4,3)

nodes = 8
system_size = 2**nodes

Graph = nx.circular_ladder_graph(4)

G = nx.to_scipy_sparse_matrix(Graph)

maxcut = h5.File('maxcut.h5', 'r')
final_state = np.array(maxcut['depth 5/final_state']).view(np.complex128)
qualities = np.abs(np.array(maxcut['depth 5/observables']).view(np.float64))
unique, counts = np.unique(qualities, return_counts = True)

plt.figure(figsize = figure_size)
plt.bar(unique, counts*(1/len(qualities)))
plt.xticks(unique)
plt.xlabel(r'$q_i$')
plt.ylabel(r'$\langle \psi_0 | q_i | \psi_0 \rangle$')
plt.tight_layout()
plt.savefig('maxcut_starting_probabilities', dpi = 200)
plt.clf()

probability = np.abs(final_state)**2
bins = np.zeros(len(unique), dtype = np.float64)
for i in range(system_size):
    quality_index = np.where(unique == qualities[i])
    bins[quality_index] += probability[i]

plt.figure(figsize = figure_size)
plt.bar(unique, bins, color = 'orange')
plt.xticks(unique)
plt.xlabel(r'$|q_i|$')
plt.ylabel(r'$\langle \vec{\gamma}, \vec{t} | q_i | \vec{\gamma}, \vec{t} \rangle$')
plt.tight_layout()
plt.savefig('maxcut_qaoa_probabilities', dpi = 200)
plt.clf()

best_index = np.argmax(probability)
binary =np.binary_repr(best_index, width = 8)

colours = []
for bit in binary:
    if int(bit) == 0:
        colours.append('Purple')
    else:
        colours.append('green')

edge_width = []
edge_colours = []
for edge in Graph.edges:
    if binary[edge[0]] != binary[edge[1]]:
        edge_colours.append('orange')
        edge_width.append(5)
    else:
        edge_colours.append('black')
        edge_width.append(1)

nx.draw(
        Graph,
        node_color = colours,
        edge_color = edge_colours,
        width = edge_width,
        with_labels = True)

plt.savefig('maxcut_solution', dpi = 200)
