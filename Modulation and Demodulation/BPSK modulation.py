import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy import signal

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.bool_)

# import 24 bit digital data
id_num = 2802461
Nbits = 24
tx_bin = bin_array(id_num, Nbits)

plt.figure()
plt.plot(tx_bin)
plt.show()

# initialise constants and variables
fc = 0.125
bit_period = 16
tx_mod = np.empty(0)

# BPSK modulation
for i in range(Nbits):
    for j in range(bit_period):
        tx_mod = np.append(tx_mod, (2 * tx_bin[i] - 1) *
                           np.cos(2 * np.pi * fc * (i * bit_period + j)))

plt.figure()
plt.plot(tx_mod)
plt.show()
plt.figure()
plt.plot(np.abs(fft.fft(tx_mod)))
plt.show()

# low-pass filter
numtaps = 32
cutoff = 0.1
b1 = signal.firwin(numtaps, cutoff)

# Digital filter frequency response
mixed = np.zeros(numtaps)
w1, h1 = signal.freqz(b1)

plt.title("Digital filter frequency response")
plt.plot(w1 / 2 / np.pi, 20 * np.log10(np.abs(h1)))
plt.ylabel("Amplitude Response/dB")
plt.xlabel("Frequency/sample rate")
plt.grid()
plt.show()

# Demodulation
rx_mixed = np.empty(0)
for i in range(Nbits):
    for j in range(bit_period):
        rx_mixed = np.append(rx_mixed, tx_mod[i * bit_period + j] *
                           np.cos(2 * np.pi * fc * (i * bit_period + j)))

rx_lpf = signal.lfilter(b1, 1, rx_mixed)
rx_lpf = np.append(rx_lpf, np.ones(numtaps // 2))

plt.figure()
plt.plot(rx_lpf)
plt.show()

rx_bin = np.empty(0)
for i in range(Nbits):
    t = (2 * i + 1) * bit_period // 2 + numtaps // 2
    # rx_bin = np.append(rx_bin, np.heaviside(rx_lpf[t], 0))
    rx_bin = np.append(rx_bin, rx_lpf[t] > 0.1)

print(tx_bin, "\n", rx_bin)
print(np.sum(rx_bin != tx_bin))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(rx_bin, color="blue", label="rx_bin")
ax2.plot(tx_bin, color="red", label="tx_bin")
fig.tight_layout()
plt.show()