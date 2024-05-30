import cv2
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt

"""
Author: Onur Ertunç
Bilkent ID: 21802760
CS 484 - Spring 2024 - HW2
Date: 26/04/2024
"""

# Define the directory path where your images are stored
template_images_path = 'template images'
rotated_images_path = 'rotated images'

canny_save_path = 'canny edge images'  # replace with the path for saving Canny edges
hough_save_path = 'hough transform results'  # replace with the path for saving Hough lines


# Ensure the save directories exist
os.makedirs(canny_save_path, exist_ok=True)
os.makedirs(hough_save_path, exist_ok=True)

# Function to apply Canny edge detection to an image
def canny_edge_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply Canny edge detector
    canny_edges = cv2.Canny(blurred_image, 50, 150)
    return canny_edges

# Function to perform line detection using Hough transform
def hough_line_detection(image, edges):
    # Detect lines in the edge image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    # Draw lines on a copy of the original image
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_image


# Function to calculate the orientation of a line given two points (x1, y1) and (x2, y2)
def calculate_orientation(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


# Function to calculate the length of a line given two points (x1, y1) and (x2, y2)
def calculate_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to compute the line orientation histogram
def line_orientation_histogram(lines, num_bins=36):
    # Initialize histogram with the given number of bins
    histogram = np.zeros(num_bins)
    bin_width = 2 * np.pi / num_bins  # Bin width to cover the range from -π to +π

    # Iterate over each detected line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate orientation and length of the line
        orientation = calculate_orientation(x1, y1, x2, y2)
        length = calculate_length(x1, y1, x2, y2)
        # Normalize orientation to be in the range [0, 2π]
        orientation = orientation % (2 * np.pi)
        # Find the corresponding bin
        bin_index = int(orientation // bin_width)
        # Increment the bin by the length of the line
        histogram[bin_index] += length

    return histogram



# Function to compute the line orientation histogram and save it as an image
def process_images_with_histogram(image_files, num_bins=36, save_path='histograms'):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    for image_file in image_files:
        edges = canny_edge_detection(image_file)
        original_image = cv2.imread(image_file)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            histogram = line_orientation_histogram(lines, num_bins)
            np.save(os.path.join(save_path, os.path.basename(image_file).replace('.png', '_histogram')), histogram)
            # Plot the histogram
            plt.figure()
            theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
            plt.bar(theta, histogram, width=2 * np.pi / num_bins, align='edge')
            plt.xlim([0, 2 * np.pi])
            plt.xlabel('Orientation (radians)')
            plt.ylabel('Length')
            plt.title(f'Line Orientation Histogram for {os.path.basename(image_file)}')
            # Save the histogram plot as an image
            plt.savefig(os.path.join(save_path, os.path.basename(image_file).replace('.png', '_histogram.png')))
            plt.close()  # Close the figure to free memory

# Example usage for template images
process_images_with_histogram(glob.glob(os.path.join(template_images_path, '*.png')), save_path='histograms/template')

# Example usage for rotated images
process_images_with_histogram(glob.glob(os.path.join(rotated_images_path, '*.png')), save_path='histograms/rotated')


def circular_shift_histogram(histogram, shift):
    """
    Shifts a histogram by a specified number of bins circularly.
    """
    return np.roll(histogram, shift)

def euclidean_distance(h1, h2):
    """
    Computes the Euclidean distance between two histograms.
    """
    return np.linalg.norm(h1 - h2)

def find_best_match(original_histograms, rotated_histogram, num_bins):
    """
    Finds the best match for a rotated histogram by comparing it with each original histogram
    using circular shifts and calculates the approximate angle of rotation.
    """
    best_distance = float('inf')
    best_match_index = -1
    best_shift = 0

    # Try shifting the rotated histogram by each possible bin amount
    for shift in range(num_bins):
        shifted_histogram = circular_shift_histogram(rotated_histogram, shift)
        # Compare with each original histogram
        for i, original_histogram in enumerate(original_histograms):
            distance = euclidean_distance(shifted_histogram, original_histogram)
            if distance < best_distance:
                best_distance = distance
                best_match_index = i
                best_shift = shift

    # Calculate the angle of rotation based on the bin shift
    # Assuming each bin corresponds to an angle increment of (2 * pi / num_bins)
    angle_of_rotation = best_shift * (360 / num_bins)  # convert to degrees if needed

    return best_match_index, angle_of_rotation, best_distance


def load_histograms_from_folder(folder_path):
    histograms = []
    file_names = []
    # Load all .npy files in the folder
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('_histogram.npy'):
            # Load histogram from the numpy binary file
            histogram = np.load(os.path.join(folder_path, file_name))
            histograms.append(histogram)
            # Store the file name without the '_histogram.npy' part for matching purposes
            file_names.append(file_name.replace('_histogram.npy', ''))
    return histograms, file_names


# Load the template (original) and rotated histograms
template_histograms, template_names = load_histograms_from_folder('histograms/template')
rotated_histograms, rotated_names = load_histograms_from_folder('histograms/rotated')

# Make sure the number of bins in the histogram matches what was used to create them
num_bins = 36

# This dictionary will hold the matches and estimated rotation angles
matches_and_angles = {}

# Match each rotated histogram with the template histograms
for rotated_histogram, rotated_name in zip(rotated_histograms, rotated_names):
    match_index, rotation_angle, min_distance = find_best_match(template_histograms, rotated_histogram, num_bins)
    matches_and_angles[rotated_name] = {
        'matched_template': template_names[match_index],
        'rotation_angle': rotation_angle,
        'euclidean_distance': min_distance
    }

# Output the matches and rotation angles
for rotated_name, info in matches_and_angles.items():
    print(f"Rotated Image: {rotated_name}")
    print(f"Matched Template: {info['matched_template']}")
    print(f"Rotation Angle: {info['rotation_angle']} degrees")
    print(f"Euclidean Distance: {info['euclidean_distance']}")
    print("-" * 30)
