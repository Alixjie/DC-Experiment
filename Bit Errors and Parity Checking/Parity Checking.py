import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy
import komm

# Obtaining Digital Data from file
tx_im = Image.open("/Users/george/Project/DC Project/Bit Errors and Parity Checking/DC4_150x100.pgm")
Npixels = tx_im.size[1] * tx_im.size[0]
plt.figure()
plt.imshow(np.array(tx_im), cmap="gray", vmin=0, vmax=255)
plt.show()
tx_bin = np.unpackbits(np.array(tx_im))
print(tx_bin.shape)

x = np.empty(0)
yber = np.empty(0)
yuber = np.empty(0)
yarq = np.empty(0)

for snrdb in range(27, 91, 1):
    print(snrdb)
    ARQs = 0
    # BPSK
    psk = komm.PSKModulation(2)
    # Quadra - ture Phase Shift Keying (QPSK)
    # psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
    # 4 - QAM
    # qam = komm.QAModulation(4, base_amplitudes=1 / np.sqrt(2))
    # 16 - QAM
    # qam = komm.QAModulation(16, base_amplitudes=1 / np.sqrt(10))
    # 256 - QAM
    # qam = komm.QAModulation(256, base_amplitudes=1 / np.sqrt(170))
    awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10))

    rx_bin = np.empty(0)

    for num in range(0, Npixels * 8, 8):
        # add even parity check bit(8th bit each word)
        tx_bit_stream = np.array(tx_bin[num:num + 8])
        tx_bit_stream[7] = (np.sum(tx_bit_stream) - tx_bit_stream[7]) % 2
        tx_bin[num:num + 8] = tx_bit_stream

        # simulate Noisy Channel
        tx_data_bit_stream = psk.modulate(tx_bit_stream)

        rx_data_bit_stream = awgn(tx_data_bit_stream)
        rx_bin_bit_stream = psk.demodulate(rx_data_bit_stream)

        # parity test
        if (np.sum(rx_bin_bit_stream) % 2):
            # Automatic Repeat-reQuest (ARQ)
            rx_data_bit_stream = awgn(tx_data_bit_stream)
            rx_bin_bit_stream = psk.demodulate(rx_data_bit_stream)
            ARQs += 1

        rx_bin = np.append(rx_bin, rx_bin_bit_stream)
    
    # Calculate ber, uncorrected ber and ratio of ARQs
    ber = np.sum(tx_bin != rx_bin) / (Npixels * 8)
    uber = 0.5 * scipy.special.erfc(np.sqrt(10 ** ((snrdb / 10) / 10) / psk.bits_per_symbol))
    print(uber)
    x = np.append(x, (snrdb / 10))
    yber = np.append(yber, ber)
    yuber = np.append(yuber, uber)
    print(ber, ARQs / (Npixels * 8))
    yarq = np.append(yarq, ARQs / (Npixels * 8))

plt.figure()
# plot ber
plt.scatter(x, yber) 
plt.plot(x, yber, label=f"Ber")  

# plot theoretical curve for uncorrected ber
plt.plot(x, yuber, label=f"Theoretical curve")

# plot ratio of ARQs
plt.plot(x, yarq, label=f"ARQs ratio")

plt.yscale("log")
plt.grid(True)
plt.show()

# x = np.empty(0)
# y1 = np.empty(0)
# y2 = np.empty(0)

# psk = komm.PSKModulation(2)
# tx_data = psk.modulate(tx_bin)

# for snrdb in range(27, 90, 1):
#     awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10))

#     # simulate Noisy Channel
#     rx_data = awgn(tx_data)
#     rx_bin = psk.demodulate(rx_data)

#     ber = np.sum(tx_bin != rx_bin) / (Npixels * 8)
#     print(ber)
#     x = np.append(x, (snrdb / 10))
#     y1 = np.append(y1, ber)
#     ber_nm = 0.5 * scipy.special.erfc(np.sqrt(10 ** ((snrdb / 10) / 10)))
#     y2 = np.append(y2, ber_nm)
#     print(ber_nm)

# plt.figure()
# plt.scatter(x, y1) #plot points
# plt.plot(x, y1)    #plot lines

# plt.plot(x, y2)
# plt.yscale("log")
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.axes().set_aspect("equal")
# plt.scatter(rx_data[:10000].real,rx_data[:10000].imag,s=1,marker=".")
# plt.show()

# rx_im = np.packbits(rx_bin).reshape(tx_im.size[1],tx_im.size[0])
# plt.figure()
# plt.imshow(np.array(rx_im), cmap="gray", vmin=0, vmax=255)
# plt.show()

# print(tx_bin.size, tx_data.size, rx_data.size, rx_bin.size)
# print(np.array(tx_im).dtype, tx_bin.dtype, tx_data.dtype, rx_data.dtype, rx_bin.dtype)

