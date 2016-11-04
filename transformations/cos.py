import numpy as np
import matplotlib.pyplot as plt

def gen_signal():
    x = np.linspace(-np.pi, np.pi, 20001)
    f = np.zeros(x.shape)
    for w in [5, 10, 25, 50, 100]:
        f += np.cos(x*w)

    return f

def main():
    pass

if __name__ == "__main__":
    main()
    gen_signal()