import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy
import komm
import threading


class parityChecker(threading.Thread):
    Npixels = 0
    ARQs = 0
    tx_im = np.empty(0)
    x = np.empty(0)
    yber = np.empty(0)
    yuber = np.empty(0)
    yarq = np.empty(0)

    def __init__(self, imagePath):
        # Obtaining Digital Data from file
        self.tx_im = Image.open(imagePath)
        self.Npixels = self.tx_im.size[1] * self.tx_im.size[0]

    def getTxBin(self):
        return np.unpackbits(np.array(self.tx_im))

    def drawImage(self):
        plt.figure()
        plt.imshow(np.array(self.tx_im), cmap="gray", vmin=0, vmax=255)
        plt.show()

    def parityCheck(self, qam, snrdb):
        ARQs = 0
        rx_bin = np.empty(0)
        tx_bin = self.getTxBin()
        awgn = komm.AWGNChannel(snr=10 ** ((snrdb / 10) / 10))
        for num in range(0, self.Npixels * 8, 8):
            # add even parity check bit(8th bit each word)
            tx_bit_stream = np.array(tx_bin[num:num + 8])
            tx_bit_stream[7] = (np.sum(tx_bit_stream) - tx_bit_stream[7]) % 2
            tx_bin[num:num + 8] = tx_bit_stream

            # simulate Noisy Channel
            tx_data_bit_stream = qam.modulate(tx_bit_stream)

            rx_data_bit_stream = awgn(tx_data_bit_stream)
            rx_bin_bit_stream = qam.demodulate(rx_data_bit_stream)

            # parity test
            if (np.sum(rx_bin_bit_stream) % 2):
                # Automatic Repeat-reQuest (ARQ)
                rx_data_bit_stream = awgn(tx_data_bit_stream)
                rx_bin_bit_stream = qam.demodulate(rx_data_bit_stream)
                ARQs += 1

            rx_bin = np.append(rx_bin, rx_bin_bit_stream)

        # Calculate ber
        ber = np.sum(tx_bin != rx_bin) / (self.Npixels * 8)
        return ARQs, ber

    def qamBerInRangeSNR(self, qamkind, start, end, step):
        # BPSK
        # psk = komm.PSKModulation(2)
        # Quadra - ture Phase Shift Keying (QPSK)
        # psk = komm.PSKModulation(4, phase_offset=np.pi / 4)
        # 4 - QAM
        if qamkind == 4:
            qam = komm.QAModulation(4, base_amplitudes=1 / np.sqrt(2))
        # 16 - QAM
        elif qamkind == 16:
            qam = komm.QAModulation(16, base_amplitudes=1 / np.sqrt(10))
        # 256 - QAM
        elif qamkind == 256:
            qam = komm.QAModulation(256, base_amplitudes=1 / np.sqrt(170))
        for snrdb in range(start, end, step):
            ARQs, ber = self.parityCheck(qam, snrdb)

            # Calculate ratio of ARQs
            self.x = np.append(self.x, (snrdb / 10))
            self.yber = np.append(self.yber, ber)
            print(ber, ARQs / (self.Npixels * 8))

        self.drawImage()

    def drawImage(self):
        plt.figure()
        # plot ber
        plt.scatter(self.x, self.yber)
        plt.plot(self.x, self.yber)

        # plot ratio of ARQs
        plt.plot(self.x, self.yarq)

        plt.yscale("log")
        plt.grid(True)
        plt.show()

    def drawRecImage(self):
        rx_im = np.packbits(self.getTxBin()).reshape(self.tx_im.size[1], self.tx_im.size[0])
        plt.figure()
        plt.imshow(np.array(rx_im), cmap="gray", vmin=0, vmax=255)
        plt.show()

if __name__ == '__main__':


# plt.figure()
# plt.axes().set_aspect("equal")
# plt.scatter(rx_data[:10000].real,rx_data[:10000].imag,s=1,marker=".")
# plt.show()