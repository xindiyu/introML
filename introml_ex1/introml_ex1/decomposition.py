import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_array = np.linspace(0, 1, samples, endpoint=False)
    signal = np.zeros(samples)
    for k in range(0, k_max + 1):
        signal += (8 / np.pi ** 2) * ((-1) ** k) * (np.sin(2 * np.pi * (2 * k + 1) * frequency * time_array)) / ((2 * k + 1) ** 2)
    return signal

def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_array = np.linspace(0, 1, samples, endpoint=False)
    signal = np.zeros(samples)
    for k in range(1, k_max + 1):
        signal += (4 / np.pi) * np.sin(2 * np.pi * (2 * k - 1) * frequency * time_array) / (2 * k - 1)
    return signal


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_array = np.linspace(0, 1, samples, endpoint=False)
    signal = (amplitude / 2) + np.zeros(samples)
    for k in range(1, k_max + 1):
        signal += -(amplitude / np.pi) * np.sin(2 * np.pi * k * frequency * time_array) / k
    return signal
