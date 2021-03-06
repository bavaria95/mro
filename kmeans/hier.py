import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import networkx as nx
import colorsys
import cv2

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
    # plt.show()
    plt.savefig("dendrogram.jpg", format="JPG")

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

def assign_colors(clusters):
    nodes_color = [[] for i in range(len(X))]
    n_clusters = len(clusters)

    h = 0
    for i in range(n_clusters):
        h += 360.0/n_clusters
        l = np.random.random()*10 + 50
        s = np.random.random()*10 + 90

        r, g, b = colorsys.hls_to_rgb(h / 360.0, l/100.0, s/100.0)

        for node in clusters[i]:
            nodes_color[node] = [r, g, b]

    return nodes_color


def main():
    global g
    global X
    g = read_graph('datasets/karate.gml')
    # g = read_graph('datasets/football.gml')
    # g = read_graph('datasets/dolphins.gml')

    X = nx.adjacency_matrix(g).toarray()

    Z = linkage(X, method='single', metric='euclidean')
    # Z = linkage(X, method='single', metric=pearson_correlation)
    # Z = linkage(X, method='single', metric=shortest_path)

    # Z = linkage(X, method='complete', metric='euclidean')
    # Z = linkage(X, method='complete', metric=pearson_correlation)
    # Z = linkage(X, method='complete', metric=shortest_path)

    # Z = linkage(X, method='average', metric='euclidean')
    # Z = linkage(X, method='average', metric=pearson_correlation)
    # Z = linkage(X, method='average', metric=shortest_path)


    clusters = select_clusters(Z)

    for c in range(len(clusters)):
        colors = assign_colors(clusters[c])

        nx.draw_circular(g, node_color=colors, with_labels=True)
        plt.savefig("%s.png" % c, format="PNG")

    display_dendrogram(Z)

if __name__ == "__main__":
    main()
