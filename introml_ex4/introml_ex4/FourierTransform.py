'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    y = shape[0] // 2 + r * np.sin(theta)
    x = shape[1] // 2 + r * np.cos(theta)
    return y, x
    pass


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    magnitude_spectrum = np.abs(spectrum)
    magnitude_spectrum = 20 * np.log10(magnitude_spectrum + 1)  # Convert to decibel scale
    return magnitude_spectrum
    pass


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    height, width = magnitude_spectrum.shape
    ring_features = np.zeros(k)

    for i in range(1, k + 1):
        for theta in np.linspace(0, np.pi, sampling_steps):
            for r in range((i - 1) * k, i * k + 1):
                y, x = polarToKart((height, width), r, theta)
                y = int(y)
                x = int(x)
                ring_features[i - 1] += magnitude_spectrum[y, x]
    return ring_features
    pass


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    height, width = magnitude_spectrum.shape
    len = min(height, width)
    fan_features = np.zeros(k)
    for i in range(1, k + 1):
        for theta in np.linspace(i-1, i, sampling_steps):
            for r in range(0, len//2):
                y, x = polarToKart((height, width), r, theta*np.pi/k)
                y = int(np.round(y))
                x = int(np.round(x))
                fan_features[i-1] += magnitude_spectrum[y, x]
    return fan_features
    pass


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    return R, T
    pass
