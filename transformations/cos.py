import numpy as np
import matplotlib.pyplot as plt

def gen_add_signal():
    n_samples = 20000
    x = np.linspace(-np.pi, np.pi, n_samples)
    f = np.zeros(x.shape)
    for w in [5, 10, 25, 50, 100]:
        f += np.cos(x*w)

    return f

def gen_segment_signal():
    n_samples = 20000
    x = np.linspace(-np.pi, np.pi, n_samples)
    w = [5, 10, 25, 50, 100]
    f = np.zeros(x.shape)
    for i in range(5):
        index_1 = i * (n_samples / 5)
        index_2 = (i + 1) * (n_samples / 5)
        f[index_1:index_2] += np.cos(x[index_1:index_2]*w[i])

    return f

def main():
    pass

if __name__ == "__main__":
    main()
    # gen_add_signal()
    gen_segment_signal()