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

# QPSK modulation
for i in range(0, Nbits, 2):
    for j in range(bit_period):
        tx_mod = np.append(tx_mod, (2 * tx_bin[i] - 1) *
                           np.cos(2 * np.pi * fc * (i * bit_period + j)) +
                           (2 * tx_bin[i + 1] - 1) * np.sin(2 * np.pi * fc * (i * bit_period + j)))

plt.figure()
plt.plot(tx_mod)
plt.show()
plt.figure()
plt.plot(np.abs(fft.fft(tx_mod)))
plt.show()

# low-pass filter
numtaps = 4
cutoff = 0.1
b1 = signal.firwin(numtaps, cutoff)

# Demodulation
rx_mixed_i = np.empty(0)
rx_mixed_q = np.empty(0)

for i in range(0, Nbits, 2):
    for j in range(bit_period):
        rx_mixed_i = np.append(rx_mixed_i, tx_mod[(i // 2) * bit_period + j] *
                           np.cos(2 * np.pi * fc * (i * bit_period + j)))
        rx_mixed_q = np.append(rx_mixed_q, tx_mod[(i // 2) * bit_period + j] *
                               np.sin(2 * np.pi * fc * (i * bit_period + j)))

rx_filt_i = signal.lfilter(b1, 1, rx_mixed_i)
rx_filt_i = np.append(rx_filt_i, np.zeros(numtaps // 2) / 2)
rx_filt_q = signal.lfilter(b1, 1, rx_mixed_q)
rx_filt_q = np.append(rx_filt_q, np.zeros(numtaps // 2) / 2)

plt.figure()
plt.plot(rx_filt_i, color="blue", label="In-phase")
plt.plot(rx_filt_q, color="red", label="Quadrature")
plt.show()

rx_bin = np.empty(0)
for i in range(0, Nbits, 2):
    t = (i + 1) * bit_period // 2 + numtaps // 2
    rx_bin = np.append(rx_bin, rx_filt_i[t] > 0)
    rx_bin = np.append(rx_bin, rx_filt_q[t] > 0)

print(tx_bin, "\n", rx_bin)
print(np.sum(rx_bin != tx_bin))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(rx_bin, color="blue", label="rx_bin")
ax2.plot(tx_bin, color="red", label="tx_bin")
fig.tight_layout()
plt.show()