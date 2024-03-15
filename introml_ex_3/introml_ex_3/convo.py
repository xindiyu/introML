from PIL import Image
import numpy as np


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

def slow_convolve(arr, k):
    # Get the dimensions of the image and kernel
    image_height, image_width = arr.shape
    kernel_height, kernel_width = k.shape

    # Flip the kernel
    flipped_kernel = np.flip(k)

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create a padded image with zeros
    padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an empty output image
    output_image = np.zeros_like(arr)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Apply the convolution operation
            output_image[i, j] = np.sum(roi * flipped_kernel)

    return output_image


if __name__ == '__main__':
    k = make_kernel(3, 1)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    # im = np.array(Image.open('input1.jpg'))
    im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    # Perform convolution
    blurred = np.zeros_like(im)

    for i in range(3):
        blurred[:, :, i] = slow_convolve(im[:, :, i], k)

    # Subtract the blurred image from the input
    difference = im - blurred

    # Add the difference to the input and clip the values
    result = np.clip(im + difference, 0, 255).astype(np.uint8)

    # Save the result
    output_image = Image.fromarray(result)
    output_image.save('output.jpg')
