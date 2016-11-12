from cos import *
import pywt

Fs = 2048
t, y = gen_add_signal()
# t, y = gen_segment_signal()


plt.subplot(2, 1, 1)
plt.plot(t, y,'k-')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
n = len(y)                       # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T                        # two sides frequency range
freq = frq[range(n/2)]           # one side frequency range

Y = np.copy(y)
for _ in range(int(np.log2(n))):
    cA, cD = pywt.dwt(Y[ :n], 'haar')
    Y[0:n/2] = cA
    Y[n/2:n] = cD
    n /= 2

Y = Y[range(len(Y)/2)]

plt.plot(freq, Y, 'r-')
# plt.xlabel('freq (Hz)')
# plt.ylabel('|Y(freq)|')

plt.show()
