import numpy as np
import matplotlib.pyplot as plt

# Complete this function such that returns the clipped values of the 1D
# numpy array passed as input to the given minimum and maximum. Example:
#    1 2 3 4 5 6 7 8
# clipped to a minimum of 3 and a maximum of 6 should give
#    3 3 3 4 5 6 6 6
# Note that the input array must not be modified.

def clip(array, minimum, maximum):
    # copy the array in order not to modify the original input
    clipped_array = np.copy(array)

    # compare each value in the array with the given minimum and change every value less than the minimum to the minimum
    clipped_array = np.where(clipped_array < minimum, minimum, clipped_array)

    # compare each value in the array with the given maximum and change every value more than the maximum to the maximum
    clipped_array = np.where(clipped_array > maximum, maximum, clipped_array)

    return clipped_array

if __name__ == '__main__':
    array = np.random.rand(100)
    result = clip(array, 0.2, 0.8)
    plt.plot(array, result, '.')
    plt.show()
