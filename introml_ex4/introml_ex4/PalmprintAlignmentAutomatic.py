'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    #Binarize Image
    _, binarized = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    #Smoothe Image
    smoothed = cv2.GaussianBlur(binarized, (5, 5), 0)
    return smoothed
    pass


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(img)
    contour_img = cv2.drawContours(contour_img, [largest_contour], 0, 255, 2)

    return contour_img
    pass


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''

    '''
    k = np.zeros(6)
    number = 0
    shape = contour_img.shape
    i = 1
    while(i < shape[0] and number < 6):
        if(contour_img[i][x] == 255):
            k[number] = contour_img[i][x]
            number += 1
        i += 1
    return k
    '''
    y = np.zeros(6)
    counter = -1
    # calculate the width of white => helps to find the middle value
    #print(contour_img.shape)
    #print(len(contour_img))
    white_counter = 1
    for i in range(1, len(contour_img)):
        # if white->white
        if contour_img[i, x] == 255 and contour_img[i - 1, x] == 255:
            white_counter = white_counter + 1
        # if black->white
        elif contour_img[i - 1, x] == 0 and contour_img[i, x] == 255:
            # reset white_counter
            white_counter = 1
        # if white->black
        elif contour_img[i - 1, x] == 255 and contour_img[i, x] == 0:
            if (counter >= 0) and (counter <= 5):
                y[counter] = i - 1 - white_counter // 2
            # calculate the times passing a white line
            counter = counter + 1
    return y

    pass


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    height, width = img.shape
    #same column
    if x1 == x2:
        for y in range(max(y1, y2), height):
            if img[y, x1] == 255 :
                return y, x1
    #same row
    if y1 == y2:
        for x in range(max(x1, x2), width):
            if img[y1, x] == 255:
                return y1, x

    #y=kx+b
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    for x in range(max(x1, x2), width):
        y = int(k * x + b)
        if img[y, x] == 255:
            return y, x
    # If no intersection point found, return None
    return None
    pass


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    y1, x1 = k1
    y2, x2 = k2
    y3, x3 = k3

    #k1, k3 in the same column
    if x1 == x3:
        p2 = 0
        origin_x = x1
        origin_y = y2

    # otherwise, l1-k1k3(x-axis): y = p1 * x + q1, l2-k2-Intersection(y-axis): y = p2 * x + q2
    else:
        p1 = (y1 - y3) / (x1 - x3)
        q1 = y1 - p1 * x1
        p2 = -1 / p1
        q2 = y2 - p2 * x2
        origin_x = (q2 - q1) / (p1 - p2)
        origin_y = p1 * origin_x + q1

    theta = np.arctan(p2)

    matrix = cv2.getRotationMatrix2D((origin_y, origin_x), np.degrees(theta), scale=1.0)
    return matrix

    pass


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    blured = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    largest_contour = drawLargestContour(blured)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 15
    x2 = 25
    intersections1 = getFingerContourIntersections(largest_contour, x1)
    intersections2 = getFingerContourIntersections(largest_contour, x2)
    intersection_img = np.zeros(largest_contour.shape)
    intersection_img = intersection_img + largest_contour
    intersection_img[:, x1] = 255
    intersection_img[:, x2] = 255
    plt.imshow(intersection_img, cmap='gray')
    plt.title('check intersections')
    plt.show()

    # TODO compute middle points from these contour intersections
    mid_points1 = np.zeros(3).astype(int)
    mid_points2 = np.zeros(3).astype(int)
    #print(len(intersections1), len(intersections2))
    for i in range(3):
        mid_points1[i] = (intersections1[i * 2] + intersections1[i * 2 + 1]) // 2
        mid_points2[i] = (intersections2[i * 2] + intersections2[i * 2 + 1]) // 2

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(largest_contour, mid_points1[0], x1, mid_points2[0], x2)
    k2 = findKPoints(largest_contour, mid_points1[1], x1, mid_points2[1], x2)
    k3 = findKPoints(largest_contour, mid_points1[2], x1, mid_points2[2], x2)
    contour_img = drawCircle(largest_contour, k1[1], k1[0])
    contour_img = drawCircle(contour_img, k2[1], k2[0])
    contour_img = drawCircle(contour_img, k3[1], k3[0])
    plt.imshow(contour_img, cmap='gray')
    plt.title('Intersections')
    plt.show()

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    M = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    height, width = np.shape(img)
    rotated_pic = cv2.warpAffine(img, M, (width, height))

    return rotated_pic

    pass
