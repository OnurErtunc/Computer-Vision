import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

###################################### Question 1 ######################################

# Load the image in grayscale (as binary)
image = cv2.imread('Q1.png', cv2.IMREAD_GRAYSCALE)
# Threshold the image to make sure it's binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


def dilation(image, struct_elem):
    output = np.zeros_like(image)
    pad = struct_elem.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            if np.sum(struct_elem * padded_image[i-pad:i+pad+1, j-pad:j+pad+1]) >= 1:
                output[i-pad, j-pad] = 255
    return output


def erosion(image, struct_elem):
    output = np.zeros_like(image)
    pad = struct_elem.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            if np.all(struct_elem * padded_image[i-pad:i+pad+1, j-pad:j+pad+1]):
                output[i-pad, j-pad] = 255
    return output


dilated = dilation(binary_image, structuring_element)
eroded = erosion(dilated, structuring_element)

cv2.imwrite('result.png', eroded)


###################################### Question 2 ######################################

image_q2a = plt.imread('Q2_a.jpg')
image_q2b = plt.imread('Q2_b.png')


def generate_histogram(image):
    # If the image is not already in 'uint8' format with range [0, 255], scales it accordingly
    if image.dtype != np.uint8:
        image = np.array(image * 255, dtype=np.uint8)
    else:
        image = np.array(image, dtype=np.uint8)

    histogram = [0] * 256  # Initialize a list of 256 zeros for the histogram
    for pixel in image.flatten():
        histogram[pixel] += 1  # Increment the count for this pixel intensity

    return histogram

def plot_histogram(histogram, title='Histogram'):
    plt.bar(range(len(histogram)), histogram, color='black')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()


# Apply the functions to the images
histogram_q2a = generate_histogram(image_q2a)
histogram_q2b = generate_histogram(image_q2b)

# Plot the histograms
plot_histogram(histogram_q2a, title='Histogram for Q2a')
plot_histogram(histogram_q2b, title='Histogram for Q2b')

# save the histogram plots as images
plt.figure()
plt.bar(range(len(histogram_q2a)), histogram_q2a, color='black')
plt.title('Histogram for Q2a')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('histogram_q2a.png')

plt.figure()
plt.bar(range(len(histogram_q2b)), histogram_q2b, color='black')
plt.title('Histogram for Q2b')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('histogram_q2b.png')


###################################### Question 3 ######################################

def histogram_equalization(image, histogram):
    # Calculate the cumulative distribution function from the histogram
    cdf = np.cumsum(histogram)
    cdf_normalized = (cdf - cdf[0]) * 255 / (cdf[-1] - cdf[0])  # Normalize the CDF

    # I used the normalized CDF to map the old pixel values to new pixel values
    equalized_pixels = np.interp(image.flatten(), range(0, 256), cdf_normalized)

    # Reshape the flat array into the original image shape
    equalized_image = equalized_pixels.reshape(image.shape)

    return equalized_image.astype('uint8')


image_q2a = plt.imread('Q2_a.jpg')
image_q2b = plt.imread('Q2_b.png')

# Ensuring the images are in the range 0-255
image_q2a = np.array(image_q2a * 255, dtype=int)
image_q2b = np.array(image_q2b * 255, dtype=int)

# Generate histograms
histogram_q2a = generate_histogram(image_q2a)
histogram_q2b = generate_histogram(image_q2b)

# Apply histogram equalization
image_q2a_equalized = histogram_equalization(image_q2a, histogram_q2a)
image_q2b_equalized = histogram_equalization(image_q2b, histogram_q2b)

# Plot original q2a histogram
plt.figure()
plt.bar(range(len(histogram_q2a)), histogram_q2a, color='black')
plt.title('Histogram for Q2a')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('histogram_q2a.png')

# Plot original q2a equalized histogram
plt.figure()
plt.bar(range(len(histogram_q2a)), generate_histogram(image_q2a_equalized), color='black')
plt.title('Equalized Histogram for Q2a')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('equalized_histogram_q2a.png')


# Plot original q2b histogram
plt.figure()
plt.bar(range(len(histogram_q2b)), histogram_q2b, color='black')
plt.title('Histogram for Q2b')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('histogram_q2b.png')


# Plot original q2b equalized histogram
plt.figure()
plt.bar(range(len(histogram_q2b)), generate_histogram(image_q2b_equalized), color='black')
plt.title('Equalized Histogram for Q2b')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('equalized_histogram_q2b.png')

###################################### Question 4 ######################################

image_q4a = plt.imread('Q4_a.png')
image_q4b = plt.imread('Q4_b.png')

# Ensure the images are in the range 0-255
image_q4a = np.array(image_q4a * 255, dtype=int)
image_q4b = np.array(image_q4b * 255, dtype=int)

def otsu_threshold(image):
    # Flatten the image to 1D array for histogram computation
    pixels = image.flatten()

    # Compute the histogram
    histogram = np.bincount(pixels, minlength=256)
    # Normalize the histogram
    histogram = histogram / np.sum(histogram)

    # Compute cumulative sum and cumulative mean
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    # Compute the between-class variance for each threshold and find the threshold that maximizes it
    sigma_b_squared = (cumulative_mean[-1] * cumulative_sum - cumulative_mean) ** 2 / (
                cumulative_sum * (1 - cumulative_sum) + 1e-10)
    optimal_threshold = np.argmax(sigma_b_squared)

    # Create the binary image
    binary_image = image.copy()
    binary_image[image <= optimal_threshold] = 0
    binary_image[image > optimal_threshold] = 255

    return binary_image, optimal_threshold


if len(image_q4a.shape) == 3:
    image_q2a = np.dot(image_q4a[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Now apply Otsu's thresholding
binary_image_q4a, optimal_threshold_q4a = otsu_threshold(image_q4a)
binary_image_q4b, optimal_threshold_q4b = otsu_threshold(image_q4b)

# Plot the original and the thresholded image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_q4a, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image_q4a, cmap='gray')
plt.title(f'Binary Image (Threshold: {optimal_threshold_q4a})')
plt.axis('off')

plt.show()
plt.imsave('binary_image_q4a.png', binary_image_q4a, cmap='gray')

# Plot the original and the thresholded image for Q4b
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_q4b, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image_q4b, cmap='gray')
plt.title(f'Binary Image (Threshold: {optimal_threshold_q4b})')
plt.axis('off')

plt.show()
plt.imsave('binary_image_q4b.png', binary_image_q4b, cmap='gray')


###################################### Question 5 ######################################

# This question utilizes previously defined functions for otsu thresholding and dilation/erosion

def label_and_count_objects(binary_image):
    # Label connected components
    labeled_image2, num_features = label(binary_image)
    return labeled_image2, num_features


def process_image_for_objects(image):
    # Apply Otsu's method to get the binary image
    binary_image, _ = otsu_threshold(image)

    binary_image = erosion(binary_image, structuring_element)
    binary_image = dilation(binary_image, structuring_element)

    # Label and count distinct objects
    labeled_image2, num_objects2 = label_and_count_objects(binary_image)

    return labeled_image2, num_objects2


image = cv2.imread('Q5.png', cv2.IMREAD_GRAYSCALE)

if image.ndim == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

labeled_image, num_objects = process_image_for_objects(image)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')


plt.subplot(122)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title(f'Number of Objects: {num_objects}')
plt.axis('off')
plt.savefig('Q5-result.png')


plt.show()

###################################### Question 6 ######################################

def load_image(image_path):
    return plt.imread(image_path)

def combine_edges(edge_x, edge_y):
    return np.sqrt(edge_x ** 2 + edge_y ** 2)

image = load_image('Q6.png')

# Ensure the image is in grayscale
if image.ndim == 3:
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def convolve2d(image, kernel):
    # Flip the kernel (comment out if you want to perform correlation instead)
    kernel = np.flipud(np.fliplr(kernel))

    # Get the dimensions of the image and the kernel
    xImage, yImage = image.shape
    xKernel, yKernel = kernel.shape

    # Calculate the padding size
    pad_x = xKernel // 2
    pad_y = yKernel // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)

    # Initialize the output convolved image
    new_image = np.zeros_like(image)

    # Perform the convolution operation
    for x in range(xImage):
        for y in range(yImage):
            # Extract the current region of interest
            region = padded_image[x:x + xKernel, y:y + yKernel]
            new_image[x, y] = np.sum(region * kernel)

    return new_image

# Define the Sobel operator kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Define the Prewitt operator kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
# Apply Sobel operator (edge detection)
sobel_edges_x = convolve2d(image, sobel_x)
sobel_edges_y = convolve2d(image, sobel_y)
sobel_edges = combine_edges(sobel_edges_x, sobel_edges_y)

# Apply Prewitt operator (edge detection)
prewitt_edges_x = convolve2d(image, prewitt_x)
prewitt_edges_y = convolve2d(image, prewitt_y)
prewitt_edges = combine_edges(prewitt_edges_x, prewitt_edges_y)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(sobel_edges_x, cmap='gray')
plt.title('Sobel Edges X')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sobel_edges_y, cmap='gray')
plt.title('Sobel Edges Y')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(prewitt_edges_x, cmap='gray')
plt.title('Prewitt Edges X')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(prewitt_edges_y, cmap='gray')
plt.title('Prewitt Edges Y')
plt.axis('off')

plt.show()

# Save the images
plt.imsave('sobel_edges_x.png', sobel_edges_x, cmap='gray')
plt.imsave('sobel_edges_y.png', sobel_edges_y, cmap='gray')
plt.imsave('prewitt_edges_x.png', prewitt_edges_x, cmap='gray')
plt.imsave('prewitt_edges_y.png', prewitt_edges_y, cmap='gray')
# Save the combined edge images
plt.imsave('sobel_edges.png', sobel_edges, cmap='gray')
plt.imsave('prewitt_edges.png', prewitt_edges, cmap='gray')


