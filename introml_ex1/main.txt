import numpy as np
import matplotlib.pyplot as plt

from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal


# TODO: Test the functions imported in lines 1 and 2 of this file.
def main():
    #chirp signal
    samplingrate = 200
    duration = 1
    freqfrom = 1
    freqto = 10
    linear_chirp = createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=True)
    exp_chirp = createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=False)
    print(linear_chirp)
    print(exp_chirp)

    #Fourier Decomposition
    samples = 200
    frequency = 2
    k_max = 10000
    amplitude = 1

    triangle_signal = createTriangleSignal(samples, frequency, k_max)
    square_signal = createSquareSignal(samples, frequency, k_max)
    sawtooth_signal = createSawtoothSignal(samples, frequency, k_max, amplitude)

    time_array = np.linspace(0, 1, samples, endpoint=False)

    plt.plot(time_array, triangle_signal, label="Triangle Signal")
    plt.plot(time_array, square_signal, label="Square Signal")
    plt.plot(time_array, sawtooth_signal, label="Sawtooth Signal")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()