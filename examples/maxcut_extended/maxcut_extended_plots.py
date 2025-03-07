import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import networkx as nx

plt.rcParams["font.size"] = 16

figure_size = (5, 4)

nodes = 8
system_size = 2 ** nodes

Graph = nx.circular_ladder_graph(4)

G = nx.to_scipy_sparse_array(Graph)

maxcut = h5.File("maxcut_extended.h5", "r")
final_state = np.array(maxcut["depth 2/final_state"]).view(np.complex128)
qualities = np.abs(np.array(maxcut["depth 2/observables"]).view(np.float64))
unique, counts = np.unique(qualities, return_counts=True)

print("Starting expectation value: {}".format(-np.mean(qualities)))
print("Minimum quality: {}".format(np.min(-qualities)))

probability = np.abs(final_state) ** 2
bins = np.zeros(len(unique), dtype=np.float64)
for i in range(system_size):
    quality_index = np.where(unique == qualities[i])
    bins[quality_index] += probability[i]

plt.figure(figsize=figure_size)
plt.bar(unique, bins, hatch=".", color="tab:green")
plt.xticks(unique)
plt.xlabel("solution quality (absolute value)")
plt.ylabel(
     "probability"
)
plt.tight_layout()
plt.savefig("maxcut_extended_qaoa_probabilities", dpi=200)
plt.clf()

best_index = np.argmax(probability)
binary = np.binary_repr(best_index, width=8)

print(
    "Most probable solution: \n \t Index: {} \n \t Bit-String: {} \n \t Quality: {}".format(
        best_index, binary, -qualities[best_index]
    )
)

colours = []
for bit in binary:
    if int(bit) == 0:
        colours.append("Purple")
    else:
        colours.append("green")

edge_width = []
edge_colours = []
for edge in Graph.edges:
    if binary[edge[0]] != binary[edge[1]]:
        edge_colours.append("orange")
        edge_width.append(5)
    else:
        edge_colours.append("black")
        edge_width.append(1)

nx.draw(
    Graph,
    node_color=colours,
    edge_color=edge_colours,
    font_color="white",
    width=edge_width,
    with_labels=True,
)

plt.savefig("maxcut_extended_solution", dpi=200)
