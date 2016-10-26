import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community
import colorsys

def assign_colors(nodes):
    nodes_color = [[] for i in range(len(nodes))]
    n_clusters = len(set(nodes))

    h = 0
    for i in range(n_clusters):
        h += 360.0/n_clusters
        l = np.random.random()*10 + 50
        s = np.random.random()*10 + 90

        r, g, b = colorsys.hls_to_rgb(h / 360.0, l/100.0, s/100.0)

        for j in range(len(nodes)):
            if nodes[j] == i:
                nodes_color[j] = [r, g, b]

    return nodes_color


# g = nx.read_gml('datasets/karate.gml')
# g = nx.read_gml('datasets/football.gml')
g = nx.read_gml('datasets/dolphins.gml')


# Compute the partition of the graph nodes which maximises the modularity
# (or try..) using the Louvain heuristices
p = community.best_partition(g)
clusters = [x[1] for x in sorted(list(p.iteritems()))]
print(clusters)

colors = assign_colors(clusters)

nx.draw_circular(g, node_color=colors, with_labels=True)
plt.show()
