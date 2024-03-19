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

code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
tblen = 18
encoder = komm.ConvolutionalStreamEncoder(code)
decoder_hard = komm.ConvolutionalStreamDecoder(code,
                                               traceback_length=tblen, input_type="hard")
decoder_soft = komm.ConvolutionalStreamDecoder(code, 
                                               traceback_length=tblen, input_type="soft")

# Test convolutional coding
# print(code)
# test = np.array([1, 1, 0, 1])
# test = np.append(test, np.zeros(18))
# encode_test = encoder(test)
# print(encode_test)
# print(decoder(encode_test)[18:])

x = np.empty(0)
y_Conv_Hard = np.empty(0)
y_Conv_Soft = np.empty(0)
y_Conv_Arq = np.empty(0)

for snrdb in range(30, 91, 2):
    print(snrdb)
    # Quadra - ture Phase Shift Keying (QPSK)
    psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
    awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10), signal_power=1.0)

    rx_bin_hard = np.empty(0)
    rx_bin_soft = np.empty(0)
    rx_bin_ARQ = np.empty(0)

    # append tblen zeros to the input binary stream
    tx_bin = np.append(tx_bin, np.zeros(tblen))
    tx_Conv_coding = encoder(tx_bin)
    tx_QPSK_Conv_Coding = psk.modulate(tx_Conv_coding)

    # simulate Noisy Channel
    rx_QPSK_Conv_Coding = awgn(tx_QPSK_Conv_Coding)

    # "hard" demodulate and decoding
    rx_Conv_Coding_hard = psk.demodulate(rx_QPSK_Conv_Coding, decision_method="hard")
    rx_hard_bin = decoder_hard(rx_Conv_Coding_hard)

    # "soft" demodulate and decoding
    rx_Conv_Coding_soft = psk.demodulate(rx_QPSK_Conv_Coding, decision_method="soft")
    rx_soft_bin = decoder_soft(rx_Conv_Coding_soft)

    # ARQ Coding
    for num in range(0, Npixels * 8, 8):
        # ARQ
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

        rx_bin_ARQ = np.append(rx_bin_ARQ, rx_bit_stream)
    
    tx_bin = tx_bin[:Npixels * 8]
    Conv_Coding_Hard = np.sum(tx_bin != rx_hard_bin[tblen:]) / (Npixels * 8)
    Conv_Coding_Soft = np.sum(tx_bin != rx_soft_bin[tblen:]) / (Npixels * 8)
    Conv_Coding_Arq = np.sum(tx_bin != rx_bin_ARQ) / (Npixels * 8)
    print(Conv_Coding_Hard, Conv_Coding_Soft, Conv_Coding_Arq)

    y_Conv_Hard = np.append(y_Conv_Hard, Conv_Coding_Hard)
    y_Conv_Soft = np.append(y_Conv_Soft, Conv_Coding_Soft)
    y_Conv_Arq = np.append(y_Conv_Arq, Conv_Coding_Arq)

    x = np.append(x, (snrdb / 10))

plt.figure()

# plot ber
plt.scatter(x, y_Conv_Hard, label="Convolutional Coding Hard BER") 
plt.scatter(x, y_Conv_Soft, label="Convolutional Coding Soft BER")  
plt.scatter(x, y_Conv_Arq, label="ARQ_QPSK_BER") 

plt.title("BER in different SNR")
plt.xlabel("SNR(db)")   
plt.ylabel("BER") 

plt.yscale("log")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()


    
