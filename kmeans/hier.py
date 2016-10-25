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

def select_clusters(lnkg):
    Z = map(lambda x: [int(x[0]), int(x[1]), x[2], x[3]], lnkg)
    N = len(X)
    clusters_on_step = []
    c = [set([i]) for i in range(N)]

    for j in range(len(Z)):
        z = Z[j]
        x, y = z[ :2]

        if x >= N:
            if c[Z[x - N][0]]:
                x = Z[x - N][0]
            else:
                x = Z[x - N][1]
            
        if y >= N:
            if c[Z[y - N][0]]:
                y = Z[y - N][0]
            else:
                y = Z[y - N][1]

        Z[j][ :2] = x, y

        for i in range(len(c)):
            if x in c[i]:
                x_i = i
            if y in c[i]:
                y_i = i

        c[x_i] |= c[y_i]
        c[y_i] = set([])

        clusters_on_step.append(map(list, filter(None, c)))

    return clusters_on_step

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
    # display_dendrogram(Z)

    c = select_clusters(Z)


if __name__ == "__main__":
    main()
