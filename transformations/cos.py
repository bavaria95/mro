import numpy as np
import matplotlib.pyplot as plt

def generate_cosine(ff):
    Fs = 2000                        # sampling rate
    Ts = 1.0/Fs                      # sampling interval
    t = np.arange(0, 1, Ts)          # time vector
    y = np.cos(2 * np.pi * ff * t)

    return (t, y)

def gen_add_signal():
    f = np.zeros(2000)
    for w in [5, 10, 25, 50, 100]:
        t, y = generate_cosine(w)
        f += y

    return (t, f)

def gen_segment_signal():
    Fs = 2000
    w = [5, 10, 25, 50, 100]
    f = np.zeros(Fs)
    for i in range(5):
        index_1 = i * (Fs / 5)
        index_2 = (i + 1) * (Fs / 5)
        t, y = generate_cosine(w[i])
        f[index_1:index_2] += y[index_1:index_2]

    return (t, f)

def main():
    pass

if __name__ == "__main__":
    main()
    # gen_add_signal()
    gen_segment_signal()