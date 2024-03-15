import numpy as np

def createChirpSignal(samplingrate, duration, freqfrom, freqto, linear):
    t = np.linspace(0, duration, int(samplingrate * duration), False)
    if linear:
        # Linear Chirp
        k = (freqto - freqfrom) / duration
        chirp_signal = np.sin(2.0 * np.pi * (freqfrom * t + 0.5 * k * t**2))
    else:
        # Exponential Chirp
        k = np.log(freqto / freqfrom) / duration
        chirp_signal = np.sin(2.0 * np.pi * freqfrom * ((np.exp(k * t) - 1) / k))
    return chirp_signal

# main.py
