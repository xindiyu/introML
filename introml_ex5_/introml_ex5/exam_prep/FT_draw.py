import numpy as np
import matplotlib.pyplot as plt

# Define the time values
t = np.linspace(0, 1, 1000)  # Time values from 0 to 1 second

# Define the signal f(t)
f_t = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 800 * t)

# Perform the Fourier transform
N = len(t)
frequencies = np.fft.fftfreq(N)
frequencies_hz = frequencies * N  # Convert frequencies to Hz
frequencies_hz = frequencies_hz[:N // 2]  # Take only positive frequencies

fft_result = np.fft.fft(f_t)
amplitudes = np.abs(fft_result[:N // 2])

# Create the frequency domain signal f(ω) with relative strengths
f1 = 1.0
f2 = 0.6 * f1
f3 = 0.4 * f1

f_omega = np.zeros_like(frequencies_hz)
f_omega[np.abs(frequencies_hz - 200) < 1] = f1
f_omega[np.abs(frequencies_hz - 400) < 1] = f2
f_omega[np.abs(frequencies_hz - 800) < 1] = f3

# Plot the original signal and the frequency domain representation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, f_t)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal in Time Domain')

plt.subplot(1, 2, 2)
plt.plot(frequencies_hz, amplitudes)
plt.stem([200, 400, 800], [f1, f2, f3], markerfmt='ro', basefmt=' ')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Domain Representation')
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()