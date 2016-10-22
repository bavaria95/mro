import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import networkx as nx

def read_graph(path):
    return nx.read_gml(path)

def display_dendrogram(Z):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.
    )
    plt.show()


def main():
    global g
    global X
    g = read_graph('datasets/karate.gml')
    # g = read_graph('datasets/football.gml')
    # g = read_graph('datasets/dolphins.gml')
    X = nx.adjacency_matrix(g).toarray()

    Z = linkage(X, method='single', metric='euclidean')

    display_dendrogram(Z)

main()
