import numpy as np

def make_kd_tree(points, dim, i=0):
    if points.shape[0] > 1:
        points = np.array(sorted(list(points), key=lambda x: x[i]))
        i = (i + 1) % dim
        half = len(points) >> 1
        return (
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half])
    elif len(points) == 1:
        return (None, None, points[0])

def get_nearest(kd_node, point, dim, dist_func, i=0, best=None):
    if kd_node:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        get_nearest(kd_node[dx < 0], point, dim, dist_func, i, best)
        if dx * dx < best[0]:
            get_nearest(kd_node[dx >= 0], point, dim, dist_func, i, best)
    return best[1]
