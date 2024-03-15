#from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration = 4*44100, offset = 44100//10):
    # Complete this function
    #load sound file
    data = np.load(filename)
    peak = np.argmax(np.abs(data))
    start = peak + offset
    end = start + duration
    return data[start:end]

def compute_frequency(signal, min_freq=20, sampling_frequency=44100):
    magnitude = np.abs(np.fft.fft(signal))
    #d=1/sampling_frequency the space between ferquency values should correspond to the reciprocal of sampling value
    frequencies = np.fft.fftfreq(len(signal), d=1 / sampling_frequency)
    #peak_freqs = frequencies[np.argmax(magnitude):]
    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_magnitudes = magnitude[:len(magnitude) // 2]
    valid_freqs = positive_freqs[positive_freqs > min_freq]
    main_freq = valid_freqs[np.argmax(positive_magnitudes[positive_freqs > min_freq])]
    return main_freq

if __name__ == '__main__':
    # Implement the code to answer the questions here
    sound_folder = 'D:\XindiYu\introML\introml_ex1\introml_ex1\sounds'
    sound_files = os.listdir(sound_folder)

    for file in tqdm(sound_files):
        file_path = os.path.join(sound_folder, file)
        sample = load_sample(file_path)
        frequency = compute_frequency(sample)

        print(f"{file}: {frequency} Hz")

        # Plotting the sound waveform
        plt.figure()
        plt.plot(sample)
        plt.title(file)
        plt.show()

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
