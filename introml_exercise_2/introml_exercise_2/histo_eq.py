# Implement the histogram equalization in this file
import numpy as np
import cv2

# Load the image using OpenCV
image = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)

# Compute the intensity histogram
hist = np.zeros(256, dtype=int)
height, width = image.shape
for i in range(height):
    for j in range(width):
        pixel_value = image[i, j]
        hist[pixel_value] += 1
# test the sum of the first 90 values in hist
print(np.sum(hist[:90]))

# Compute the cumulative distribution function (CDF)
cdf = np.zeros(256, dtype=float)
cdf[0] = hist[0] / np.sum(hist)
for i in range(1, len(hist)):
    cdf[i] = cdf[i-1] + hist[i] / np.sum(hist)
# test the sum of the first 90 values in cdf
print(np.sum(cdf[:90]))

# Apply histogram equalization to each pixel
equalized_image = np.zeros_like(image)
c_min = np.min(cdf)
for i in range(height):
    for j in range(width):
        pixel_value_old = image[i, j]
        pixel_value_new = ((cdf[pixel_value_old] - c_min) / (1 - c_min)) * 255
        equalized_image[i, j] = int(pixel_value_new)

# (e) Save the result as a new image using OpenCV
cv2.imwrite('kitty.png', equalized_image)
