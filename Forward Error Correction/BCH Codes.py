import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy
import komm

# Obtaining Digital Data from file
tx_im = Image.open("/Users/george/Project/DC-Experiment-/Forward Error Correction/DC4_150x100.pgm")
Npixels = tx_im.size[1] * tx_im.size[0]
plt.figure()
plt.imshow(np.array(tx_im), cmap="gray", vmin=0, vmax=255)
plt.show()
tx_bin = np.unpackbits(np.array(tx_im))
# print(tx_bin, tx_bin.dtype)

code = komm.BCHCode(mu=3, delta=3)
n,k = code.length, code.dimension
encoder = komm.BlockEncoder(code)
decoder = komm.BlockDecoder(code)

print(code, n, k)
print(code.codewords, code.generator_polynomial)
print(code.codewords[1] ^ code.codewords[5])
test = np.array([1, 1, 0, 1])
encoder_test = encoder(test)
print(decoder(encoder_test))

x = np.empty(0)
y_BCH_QPSK_ber = np.empty(0)
y_QPSK_ber = np.empty(0)

for snrdb in range(30, 91, 2):
    print(snrdb)
    # Quadra - ture Phase Shift Keying (QPSK)
    psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
    awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10), signal_power=1.0)

    rx_bin = np.empty(0)
    rx_bin_withoutBCH = np.empty(0)

    for num in range(0, Npixels * 8, 8):
        # Using BCH and QPSK methods to encode the information
        tx_BCH_stream = encoder(tx_bin[num:num + 4])
        tx_BCH_stream = np.append(tx_BCH_stream, encoder(tx_bin[num + 4:num + 8]))
        # add one bit at the end of the tx_BCH_stream to 
        # ensure tx_BCH_stream can divide by psk.bits_per_symbol
        # tx_8bit_BCH_stream = np.append(tx_BCH_stream, 0)
        # tx_QPSK_BCH_stream = psk.modulate(tx_8bit_BCH_stream)
        # print(tx_BCH_stream)
        tx_QPSK_BCH_stream = psk.modulate(tx_BCH_stream)

        # simulate Noisy Channel
        rx_QPSK_BCH_stream = awgn(tx_QPSK_BCH_stream)

        # Demodulate(QPSK)
        rx_BCH_stream = psk.demodulate(rx_QPSK_BCH_stream)
        rx_bin_bit_stream = decoder(rx_BCH_stream[:7])
        rx_bin_bit_stream = np.append(rx_bin_bit_stream, decoder(rx_BCH_stream[7:]))

        rx_bin = np.append(rx_bin, rx_bin_bit_stream)

        # without BCH
        tx_bit_stream = np.array(tx_bin[num:num + 8])
        tx_bit_stream[7] = (np.sum(tx_bit_stream) - tx_bit_stream[7]) % 2
        tx_bin[num:num + 8] = tx_bit_stream

        tx_data_bit_stream = psk.modulate(tx_bit_stream)

        # simulate Noisy Channel
        rx_data_bit_stream = awgn(tx_data_bit_stream)

        rx_bit_stream = psk.demodulate(rx_data_bit_stream)

        # parity test
        if (np.sum(rx_bit_stream) % 2):
            # Automatic Repeat-reQuest (ARQ)
            rx_data_bit_stream = awgn(tx_data_bit_stream)
            rx_bit_stream = psk.demodulate(rx_data_bit_stream)

        rx_bin_withoutBCH = np.append(rx_bin_withoutBCH, rx_bit_stream)

    
    BCH_QPSK_ber = np.sum(tx_bin != rx_bin) / (Npixels * 8)
    QPSK_ber = np.sum(tx_bin != rx_bin_withoutBCH) / (Npixels * 8)
    print(BCH_QPSK_ber, QPSK_ber)

    y_BCH_QPSK_ber = np.append(y_BCH_QPSK_ber, BCH_QPSK_ber)
    y_QPSK_ber = np.append(y_QPSK_ber, QPSK_ber)

    x = np.append(x, (snrdb / 10))

plt.figure()

# plot ber
plt.scatter(x, y_BCH_QPSK_ber, label="BCH_QPSK_BER")  

plt.scatter(x, y_QPSK_ber, marker="v", label="ARQ_QPSK_BER") 

plt.title("BER in different SNR")
plt.xlabel("SNR(db)")   
plt.ylabel("BER") 

plt.yscale("log")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
    
