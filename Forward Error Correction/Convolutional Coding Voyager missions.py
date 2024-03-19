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

code_one_twice = komm.ConvolutionalCode(feedforward_polynomials=[[0o155, 0o117]])
tblen = 7 * 6
encoder_one_twice = komm.ConvolutionalStreamEncoder(code_one_twice)

decoder_one_twice = komm.ConvolutionalStreamDecoder(code_one_twice, 
                                               traceback_length=tblen, input_type="soft")

code_one_third = komm.ConvolutionalCode(feedforward_polynomials=[[0o155, 0o117, 0o127]])
encoder_one_third = komm.ConvolutionalStreamEncoder(code_one_third)

decoder_one_third = komm.ConvolutionalStreamDecoder(code_one_third, 
                                               traceback_length=tblen, input_type="soft")

x = np.empty(0)
y_Conv_One_Twice = np.empty(0)
y_Conv_One_Third = np.empty(0)

for snrdb in range(-30, 55, 2):
    print(snrdb)
    # Quadra - ture Phase Shift Keying (QPSK)
    psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
    awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10), signal_power=1.0)

    rx_bin_one_twice = np.empty(0)
    rx_bin_one_third = np.empty(0)
    
    # append tblen zeros to the input binary stream
    tx_bin = np.append(tx_bin, np.zeros(tblen))

    # The rate 1/2 code
    tx_Conv_coding_one_twice = encoder_one_twice(tx_bin)
    tx_QPSK_Conv_Coding_one_twice = psk.modulate(tx_Conv_coding_one_twice)

    # simulate Noisy Channel
    rx_QPSK_Conv_Coding_one_twice = awgn(tx_QPSK_Conv_Coding_one_twice)

    # The rate 1/2 code "soft" demodulate and decoding
    rx_Conv_Coding_soft_one_twice = psk.demodulate(rx_QPSK_Conv_Coding_one_twice, decision_method="soft")
    rx_bin_one_twice = decoder_one_twice(rx_Conv_Coding_soft_one_twice)

    # The rate 1/3 code
    tx_Conv_coding_one_third = encoder_one_third(tx_bin)
    tx_QPSK_Conv_Coding_one_third = psk.modulate(tx_Conv_coding_one_third)

    # simulate Noisy Channel
    rx_QPSK_Conv_Coding_one_third = awgn(tx_QPSK_Conv_Coding_one_third)

    # The rate 1/3 code "soft" demodulate and decoding
    rx_Conv_Coding_soft_one_third = psk.demodulate(rx_QPSK_Conv_Coding_one_third, decision_method="soft")
    rx_bin_one_third = decoder_one_third(rx_Conv_Coding_soft_one_third)

    tx_bin = tx_bin[:Npixels * 8]

    Conv_Coding_One_Twice = np.sum(tx_bin != rx_bin_one_twice[tblen:]) / (Npixels * 8)
    Conv_Coding_One_third = np.sum(tx_bin != rx_bin_one_third[tblen:]) / (Npixels * 8)

    print(Conv_Coding_One_Twice, Conv_Coding_One_third)

    y_Conv_One_Twice = np.append(y_Conv_One_Twice, Conv_Coding_One_Twice)
    y_Conv_One_Third = np.append(y_Conv_One_Third, Conv_Coding_One_third)

    x = np.append(x, (snrdb / 10))

plt.figure()

# plot ber
plt.scatter(x, y_Conv_One_Twice, label="Convolutional Coding BER (1/2 rate)") 
plt.scatter(x, y_Conv_One_Third, label="Convolutional Coding BER(1/3 rate)")  

plt.title("BER in different SNR")
plt.xlabel("SNR(db)")   
plt.ylabel("BER") 

plt.yscale("log")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()


    
