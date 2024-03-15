import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    hist = np.zeros(256, dtype=int)
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            pixel_value = img[y, x]
            hist[pixel_value] += 1

    return hist


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    binarize_img = np.zeros_like(img)
    binarize_img[img > t] = 255
    return binarize_img


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    p0 = np.sum(hist[:theta+1])
    p1 = np.sum(hist[theta+1:])
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    mu0 = 0
    mu1 = 0
    for n in range(len(hist)):
        if n <= theta:
            mu0 += hist[n] * n
        else:
            mu1 += hist[n] * n

    if p0 == 0:
        mu0 = 0
    else:
        mu0 = mu0 / p0
    if p1 == 0:
        mu1 = 0
    else:
        mu1 = mu1 / p1
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    max_var = 0.0
    threshold = 0

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    sum = np.sum(hist)
    hist = hist / sum

    # TODO loop through all possible thetas

        # TODO compute p0 and p1 using the helper function

        # TODO compute mu and m1 using the helper function

        # TODO compute variance

        # TODO update the threshold
    for theta in range(len(hist)):
        p0, p1 = p_helper(hist, theta)
        mu0, mu1 = mu_helper(hist, theta, p0, p1)
        var = p0 * p1 * (mu1 - mu0) ** 2
        if var > max_var:
            max_var = var
            threshold = theta
    return threshold

def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    b_img = binarize_threshold(img, threshold)
    return b_img
