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

# Concolutional Encode and Decode
code_one_third = komm.ConvolutionalCode(feedforward_polynomials=[[0o155, 0o117, 0o127]])
encoder_one_third = komm.ConvolutionalStreamEncoder(code_one_third)
tblen = 7 * 6
decoder_one_third = komm.ConvolutionalStreamDecoder(code_one_third, traceback_length=tblen)

# BCH encode and decode
code = komm.BCHCode(mu=3, delta=3)
encoder = komm.BlockEncoder(code)
decoder = komm.BlockDecoder(code)

x = np.empty(0)
y_Conv_One_Third = np.empty(0)

for snrdb in range(0, 50, 2):
    print(snrdb)
    # Quadra - ture Phase Shift Keying (QPSK)
    psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
    awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10), signal_power=1.0)

    rx_bin_one_third = np.empty(0)
    
    # append tblen zeros to the input binary stream
    tx_bin = np.append(tx_bin, np.zeros(tblen))

    # The rate 1/3 code
    tx_Conv_coding_one_third = encoder_one_third(tx_bin)
    print(tx_Conv_coding_one_third, tx_Conv_coding_one_third.size)

    tx_Conv_coding_one_third = np.append(tx_Conv_coding_one_third, np.zeros(2))
    print(tx_Conv_coding_one_third.size)

    rx_bin_Conv = np.empty(0)

    for num in range(0, 360128, 8):
        # Using BCH and QPSK methods to encode the information
        tx_BCH_stream = encoder(tx_Conv_coding_one_third[num:num + 4])
        tx_BCH_stream = np.append(tx_BCH_stream, encoder(tx_Conv_coding_one_third[num + 4:num + 8]))
        tx_QPSK_BCH_stream = psk.modulate(tx_BCH_stream)

        # simulate Noisy Channel
        rx_QPSK_BCH_stream = awgn(tx_QPSK_BCH_stream)

        # Demodulate(QPSK)
        rx_BCH_stream = psk.demodulate(rx_QPSK_BCH_stream)
        rx_bin_bit_stream = decoder(rx_BCH_stream[:7])
        rx_bin_bit_stream = np.append(rx_bin_bit_stream, decoder(rx_BCH_stream[7:]))

        rx_bin_Conv = np.append(rx_bin_Conv, rx_bin_bit_stream)

    rx_bin_Conv = rx_bin_Conv[:360126]

    # The rate 1/3 code demodulate and decoding
    rx_bin_one_third = decoder_one_third(rx_bin_Conv)
    print(rx_bin_one_third.size)

    tx_bin = tx_bin[:Npixels * 8]

    Conv_Coding_One_third = np.sum(tx_bin != rx_bin_one_third[tblen:]) / (Npixels * 8)
    print(Conv_Coding_One_third)

    y_Conv_One_Third = np.append(y_Conv_One_Third, Conv_Coding_One_third)

    x = np.append(x, (snrdb / 10))

plt.figure()

# plot ber
plt.scatter(x, y_Conv_One_Third, label="FEC BER(BCH(7, 3) + 7 convolutional codes)")  

plt.title("BER in different SNR")
plt.xlabel("SNR(db)")   
plt.ylabel("BER") 

plt.yscale("log")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()