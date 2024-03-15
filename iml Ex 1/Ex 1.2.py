import numpy as np

def createTriangleSignal(samples, frequency, kMax):
    t = np.linspace(0, 1, samples, endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, kMax+1, 2):
        signal += (-1)**((k-1)//2) * np.sin(2 * np.pi * k * frequency * t) / k**2
    signal *= 8 / np.pi**2
    return t, signal

def createSquareSignal(samples, frequency, kMax):
    t = np.linspace(0, 1, samples, endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, kMax+1, 2):
        signal += np.sin(2 * np.pi * k * frequency * t) / k
    signal *= 4 / np.pi
    return t, signal

def createSawtoothSignal(samples, frequency, kMax, amplitude):
    t = np.linspace(0, 1, samples, endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, kMax+1):
        signal += np.sin(2 * np.pi * k * frequency * t) / k
    signal *= -2 * amplitude / np.pi
    return t, signal
