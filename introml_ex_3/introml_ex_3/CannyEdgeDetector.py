import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    kernel = make_kernel(ksize, sigma)
    filtered = convolve(img_in, kernel)
    filtered = filtered.astype(int)
    return kernel, filtered

def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel

def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.flip([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    gx = convolve(img_in, sobel_x)
    gy = convolve(img_in, sobel_y)
    gx = gx.astype(int)
    gy = gy.astype(int)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    g = g.astype(int)
    theta = theta.astype(float)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    angle = np.degrees(angle) % 180
    if angle < 22.5 or angle >= 157.5:
        return 0
    elif 22.5 <= angle < 67.5:
        return 45
    elif 67.5 <= angle < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    max_sup = np.zeros_like(g)
    row, column = g.shape

    for x in range(1, row - 1):
        for y in range(1, column - 1):
            angle = convertAngle(theta[x, y])
            if angle == 0:
                if g[x, y] >= g[x, y - 1] and g[x, y] >= g[x, y + 1]:
                    max_sup[x, y] = g[x, y]
            elif angle == 45:
                if g[x, y] >= g[x - 1, y + 1] and g[x, y] >= g[x + 1, y - 1]:
                    max_sup[x, y] = g[x, y]
            elif angle == 90:
                if g[x, y] >= g[x - 1, y] and g[x, y] >= g[x + 1, y]:
                    max_sup[x, y] = g[x, y]
            elif angle == 135:
                if g[x, y] >= g[x - 1, y - 1] and g[x, y] >= g[x + 1, y + 1]:
                    max_sup[x, y] = g[x, y]

    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    row, column = max_sup.shape
    edge = np.zeros((row, column), dtype=int)

    for i in range(row):
        for j in range(column):
            if max_sup[i, j] >= t_high:
                edge[i, j] = 2  # Set as strong edge
            elif t_low <= max_sup[i, j] < t_high:
                edge[i, j] = 1  # Set as weak edge
            else:
                edge[i, j] = 0
    #neighbors
    for i in range(1, row - 1):
        for j in range(1, column - 1):
            if edge[i, j] == 1:
                if (edge[i - 1:i + 2, j - 1:j + 2] == 2).any():
                    edge[i, j] = 2

    edge[edge == 2] = 255
    return edge


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

if __name__ == '__main__':
    img = plt.imread('contrast.jpg')

    if len(img.shape) > 2:
        img = np.mean(img, axis=2)

    result = canny(img)

    plt.imshow(result, 'gray')
    plt.title('Canny Image')
    plt.colorbar()
    plt.show()
