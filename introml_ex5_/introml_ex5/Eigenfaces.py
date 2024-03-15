import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC

# image size
N = 64

# Define the classifier in clf - Try a Support Vector Machine with C = 0.025 and a linear kernel
# DON'T change this!
clf = SVC(kernel="linear", C=0.025)


def create_database_from_folder(path):
    '''
    DON'T CHANGE THIS METHOD.
    If you run the Online Detection, this function will load and reshape the
    images located in the folder. You pass the path of the images and the function returns the labels,
    training data and number of images in the database
    :param path: path of the training images
    :return: labels, training images, number of images
    '''
    labels = list()
    filenames = np.sort(path)
    num_images = len(filenames)
    train = np.zeros((N * N, num_images))
    for n in range(num_images):
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (N, N))
        assert img.shape == (N, N), 'Image {0} of wrong size'.format(filenames[n])
        train[:, n] = img.reshape((N * N))
        labels.append(filenames[n].split("eigenfaces/")[1].split("_")[0])
    print('Database contains {0} images'.format(num_images))
    labels = np.asarray(labels)
    return labels, train, num_images


def process_and_train(labels, train, num_images, h, w):
    '''
    Calculate the essentials: the average face image and the eigenfaces.
    Train the classifier on the eigenfaces and the given training labels.
    :param labels: 1D-array
    :param train: training face images, 2D-array with images as row vectors (e.g. 64x64 image ->  4096 vector)
    :param num_images: number of images, int
    :param h: height of an image
    :param w: width of an image
    :return: the eigenfaces as row vectors (2D-array), number of eigenfaces, the average face
    '''
    # Compute the average face
    avg = calculate_average_face(train)

    # Calculate the maximum number of eigenfaces
    # ??? why should the number of eigenfaces be set like this
    max_eigenfaces = min(num_images-1, h * w)

    # Calculate the eigenfaces
    eigenfaces = calculate_eigenfaces(train, avg, max_eigenfaces, h, w)

    # Calculate the coefficients/features for all images
    features = get_feature_representation(train, eigenfaces, avg, max_eigenfaces)

    # Train the classifier using the calculated features
    clf.fit(features, labels)

    # Return the requested number of eigenfaces as a matrix
    num_eigenfaces = eigenfaces.shape[0]
    return eigenfaces[:num_images, :], num_eigenfaces, avg

    pass


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''
    avg = np.mean(train, 0)
    return avg
    pass


def calculate_eigenfaces(train, avg, num_eigenfaces, h, w):
    '''
    Calculate the eigenfaces from the given training set using SVD
    :param train: training face images, 2D-array with images as row vectors
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to return from the computed SVD
    :param h: height of an image in the training set
    :param w: width of an image in the training set
    :return: the eigenfaces as row vectors, 2D-array --> shape(num_eigenfaces, #pixel of an image)
    '''

    # subtract the average face from every training sample
    # compute the eigenfaces using svd
    # You might have to swap the axes so that the images are represented as column vectors
    #height, width = train.shape
    #reshaped_avg = avg.reshape(height*width, 1)
    #print("train", train)
    #print("avg:", avg)
    reshaped_avg = np.expand_dims(avg, axis=0)
    #print("reshaped:", reshaped_avg)
    repeated_avg = reshaped_avg.repeat(train.shape[0], axis=0)
    #print("repeated: ", repeated_avg)

    X = train - repeated_avg

    # compute the eigenfaces using SVD
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    # represent your eigenfaces as row vectors in a 2D-matrix & crop it to the requested amount of eigenfaces
    eigenfaces = vh[:num_eigenfaces, :]

    # plot one eigenface to check whether you're using the right axis
    # comment out when submitting your exercise via studOn
    #???
    plt.imshow(eigenfaces, cmap='gray')
    plt.axis('off')
    plt.show()

    return eigenfaces
    pass


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    '''
    For all images, compute their eigenface-coefficients with respect to the given amount of eigenfaces
    :param images: 2D-matrix with a set of images as row vectors, shape (#images, #pixels)
    :param eigenfaces: 2D-array with eigenfaces as row vectors, shape(#pixels, #pixels)
                       -> only use the given number of eigenfaces
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to compute coefficients for
    :return: coefficients/features of all training images, 2D-matrix (#images, #used eigenfaces)
    '''

    # compute the coefficients for all images and save them in a 2D-matrix
    # 1. iterate through all images (one image per row)
    # 1.1 compute the zero mean image by subtracting the average face
    # 1.2 compute the image's coefficients for the expected number of eigenfaces

    # Compute the zero mean image by subtracting the average face
    zero_mean_images = images - np.expand_dims(avg, axis=0)
    # Compute the image's coefficients for the expected number of eigenfaces
    coefficients = np.dot(zero_mean_images, eigenfaces.T[:, :num_eigenfaces])

    return coefficients

    pass


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Reconstruct the given image by weighting the eigenfaces according to their coefficients
    :param img: input image to be reconstructed, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used to reconstruct the input image
    :param h: height of a original image
    :param w: width of a original image
    :return: the reconstructed image, 2D array (shape of a original image)
    '''
    # Reshape the input image to fit in the feature helper method
    img_reshape = img.reshape((h * w,))

    # Compute the coefficients to weight the eigenfaces
    coefficients = get_feature_representation(img_reshape, eigenfaces, avg, num_eigenfaces)

    # Use the average image as the starting point to reconstruct the input image
    reconstructed_img = avg.copy()

    #print("coefficients: ", coefficients)
    #print("reconstructed_img: ", reconstructed_img)
    #print("eigenfaces: ", eigenfaces)

    # Reconstruct the input image using the coefficients
    for i in range(num_eigenfaces):
        reconstructed_img += coefficients[0, i] * eigenfaces[i, :]

    # Reshape the reconstructed image back to its original shape
    reconstructed_img = reconstructed_img.reshape((h, w))

    return reconstructed_img
    pass


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Classify the given input image using the trained classifier
    :param img: input image to be classified, 1D-array
    :param eigenfaces: all given eigenfaces, 2D array with the eigenfaces as row vectors
    :param avg: the average image, 1D array
    :param num_eigenfaces: number of eigenfaces used to extract the features
    :param h: height of a original image
    :param w: width of a original image
    :return: the predicted labels using the classifier, 1D-array (as returned by the classifier)
    '''

    # Reshape the input image as a matrix with the image as a row vector
    img_reshape = img.reshape((h * w,))

    # Extract the features/coefficients for the eigenfaces of this image
    coefficients = get_feature_representation(img_reshape, eigenfaces, avg, num_eigenfaces)

    # Predict the label of the given image by feeding its coefficients to the classifier
    predicted_labels = clf.predict(coefficients)

    return predicted_labels

    pass


