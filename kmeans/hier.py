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

def pearson_correlation(x, y):
    return abs(pearsonr(x, y)[0])

def shortest_path(x, y):
    Xp = map(list, list(X))
    xi = g.nodes()[Xp.index(list(x))]
    yi = g.nodes()[Xp.index(list(y))]
    return len(nx.shortest_path(g, xi, yi))

def main():
    global g
    global X
    g = read_graph('datasets/karate.gml')
    # g = read_graph('datasets/football.gml')
    # g = read_graph('datasets/dolphins.gml')
    X = nx.adjacency_matrix(g).toarray()

    # Z = linkage(X, method='single', metric='euclidean')
    # Z = linkage(X, method='single', metric=pearson_correlation)
    Z = linkage(X, method='single', metric=shortest_path)
    display_dendrogram(Z)

main()
