import numpy as np
import matplotlib.pyplot as plt

def load_sample(filename, offset, duration):
    # Load the numpy array from the .npy file
    data = np.load(filename)

    # Find the position of the highest absolute value of the signal
    high_note = np.argmax(np.abs(data))

    # Add offset to this position as start of the signal
    start_index = high_note + offset

    # Keep only the 'duration' following values
    end_index = start_index + duration
    short_signal = data[start_index:end_index]

    return short_signal

# Test the function with one of the numpy files
short_signal = load_sample('Piano.ff.A2.npy', offset=1000, duration=5000)

# Plot the short part of the signal
plt.figure(figsize=(10, 4))
plt.plot(short_signal)
plt.title("Short part of the signal after its loudest peak")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()
