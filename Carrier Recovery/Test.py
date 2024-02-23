import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from numpy import random


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.bool_)


def costasLoop(ran):
    # import 24 bit digital data
    id_num = 2802461
    Nbits = 24
    tx_bin = bin_array(id_num, Nbits)

    prelude = np.array(np.random.randint(2, size=16), dtype="bool")
    tx_bin = np.append(prelude, tx_bin)

    # VCO Cordic Digital Clock and initialise constants and variables
    myClock = np.array([1.0, 0.0])
    bit_period = 256
    fc = 1 / 32
    fref = fc * (1 + ran * 0.005 * (random.rand() - 0.5))
    pref = 2 * np.pi * random.rand()
    volt = 0
    vout = np.array(volt)
    cout = myClock[0]
    rout = np.cos(pref)
    dout0 = np.empty(0)
    tx_mod = np.empty(0)

    # BPSK modulation
    for i in range(Nbits):
        for j in range(bit_period):
            tx_mod = np.append(tx_mod, (2 * tx_bin[i] - 1) *
                               np.cos(2 * np.pi * fc * (i * bit_period + j)))

    # low-pass filter
    numtaps = 128
    cutoff = 0.005
    b1 = signal.firwin(numtaps, cutoff)

    # Costas Loop
    mixed = np.zeros((2, numtaps))
    lpmixed = np.empty(2)

    for i in range(bit_period * (prelude.size + Nbits) + numtaps // 2):
        mixed[0, :] = np.append(mixed[0, 1:],
                                myClock[0] * (2 * tx_bin[(i // bit_period) % (prelude.size + Nbits)] - 1) * np.cos(
                                    pref + 2 * np.pi * fref * i))
        mixed[1, :] = np.append(mixed[1, 1:],
                                -myClock[1] * (2 * tx_bin[(i // bit_period) % (prelude.size + Nbits)] - 1) * np.cos(
                                    pref + 2 * np.pi * fref * i))

        for j in range(2):
            lpmixed[j] = np.sum(b1 * mixed[j, :])
        volt = lpmixed[0] * lpmixed[1]

        c = np.cos(2 * np.pi * fc * (1 + 0.25 * volt))
        s = np.sin(2 * np.pi * fc * (1 + 0.25 * volt))
        myClock = np.matmul(np.array([[c, -s], [s, c]]), myClock)

        vout = np.append(vout, volt)
        cout = np.append(cout, myClock[0])
        rout = np.append(rout, np.cos(pref + 2 * np.pi * fref * i))
        dout0 = np.append(dout0, lpmixed[0])

    rx_bin = np.empty(0)

    for i in range(prelude.size + Nbits):
        rx_bin = np.append(rx_bin, np.uint8(np.heaviside(dout0[(2 * i + 1) * bit_period // 2 + numtaps // 2], 0)))

    tx_bin = tx_bin[prelude.size:]
    rx_bin = rx_bin[prelude.size:]

    return np.sum(rx_bin != tx_bin)


for iran in range(1, 20):
    num = 0
    for j in range(100):
        test = costasLoop(iran)
        if ((test != 0) & (test != 24)):
            num += 1
    print("carrier frequency range is", iran * 0.005 * 0.5 * 100, "%, Error rate is ", num, "%")
