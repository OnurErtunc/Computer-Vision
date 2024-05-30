import os
from skimage import io, segmentation, color, filters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from skimage.measure import regionprops, label

"""
Author: Onur Ertunç
Bilkent ID: 21802760
CS 484 - Spring 2024 - HW3
Date: 14/05/2024
"""

# Define folders
input_folder = 'images'
segmentation_output_folder = 'segmented_images'
gabor_output_folder = 'gabor_features'
gabor_features_txt_folder = 'gabor_features_txt'
clustered_output_folder = 'clustered_images'
contextual_output_folder = 'contextual_features'
reclustered_output_folder = 'reclustered_images'


# Step 1: Superpixel Segmentation
def process_image_segmentation(image_path, output_folder):
    try:
        image = io.imread(image_path)
        segments = segmentation.slic(image, n_segments=450, compactness=10, sigma=1,
                                     start_label=1)  # Adjusted parameters

        # Add borders to the segments
        bordered_image = segmentation.mark_boundaries(image, segments)

        # Convert bordered_image to uint8 format for saving
        bordered_image = (bordered_image * 255).astype(np.uint8)

        segmented_image_path = os.path.join(output_folder, f'segmented_{os.path.basename(image_path)}')
        io.imsave(segmented_image_path, bordered_image)

        segments_array_path = os.path.join(output_folder,
                                           f'segments_{os.path.splitext(os.path.basename(image_path))[0]}.npy')
        np.save(segments_array_path, segments)

        print(f'Successfully processed segmentation for {image_path}')

    except Exception as e:
        print(f'Error in process_image_segmentation for {image_path}: {e}')


def process_all_images_segmentation(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(1, 11):
        image_path = os.path.join(input_folder, f'{i}.jpg')
        process_image_segmentation(image_path, output_folder)


# Step 2: Compute Gabor Features
frequencies = [0.1, 0.2, 0.3, 0.4]
orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def compute_gabor_features(image_path, segments, frequencies, orientations):
    try:
        image = io.imread(image_path)
        gray_image = color.rgb2gray(image)

        gabor_features = np.zeros((gray_image.shape[0], gray_image.shape[1], len(frequencies) * len(orientations)))
        for i, frequency in enumerate(frequencies):
            for j, theta in enumerate(orientations):
                real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
                magnitude = np.sqrt(real ** 2 + imag ** 2)
                gabor_features[:, :, i * len(orientations) + j] = magnitude

        num_superpixels = np.max(segments) + 1
        superpixel_gabor_features = np.zeros((num_superpixels, gabor_features.shape[2]))
        for sp in range(num_superpixels):
            mask = segments == sp
            if np.any(mask):
                for k in range(gabor_features.shape[2]):
                    superpixel_gabor_features[sp, k] = gabor_features[:, :, k][mask].mean()
            else:
                superpixel_gabor_features[sp, :] = np.nan  # Handle empty superpixels

        print(f'Successfully computed Gabor features for {image_path}')

        return superpixel_gabor_features

    except Exception as e:
        print(f'Error in compute_gabor_features for {image_path}: {e}')
        return None


def process_all_images_gabor(input_folder, segments_folder, output_folder, txt_output_folder, frequencies,
                             orientations):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(txt_output_folder):
        os.makedirs(txt_output_folder)
    all_superpixel_features = []
    image_superpixel_features = {}  # Store superpixel features for each image
    for i in range(1, 11):
        try:
            image_path = os.path.join(input_folder, f'{i}.jpg')
            segments_path = os.path.join(segments_folder, f'segments_{i}.npy')
            segments = np.load(segments_path)
            gabor_features = compute_gabor_features(image_path, segments, frequencies, orientations)
            if gabor_features is not None:
                np.save(os.path.join(output_folder, f'gabor_features_{i}.npy'), gabor_features)

                # Save Gabor features to a text file
                txt_output_path = os.path.join(txt_output_folder, f'gabor_features_{i}.txt')
                np.savetxt(txt_output_path, gabor_features, fmt='%f')

                # Collect all superpixel features for clustering
                all_superpixel_features.append(gabor_features)
                image_superpixel_features[i] = gabor_features

        except Exception as e:
            print(f'Error in process_all_images_gabor for {image_path}: {e}')

    all_superpixel_features = np.vstack(all_superpixel_features)
    return all_superpixel_features, image_superpixel_features


# Step 3: Clustering Superpixels
def cluster_superpixels(superpixel_features, n_clusters=10):
    # Handle missing values using imputer
    imputer = SimpleImputer(strategy='mean')
    superpixel_features = imputer.fit_transform(superpixel_features)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(superpixel_features)
    return labels


def map_clusters_to_segments(segments, cluster_labels):
    clustered_image = np.zeros_like(segments)
    for sp in range(np.max(segments) + 1):
        clustered_image[segments == sp] = cluster_labels[sp]
    return clustered_image


def visualize_clustered_image(image_path, clustered_image, output_folder):
    image = io.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(color.label2rgb(clustered_image, image, kind='avg'))
    plt.title(f'Clustered Superpixels for {os.path.basename(image_path)}')
    plt.axis('off')

    # Save the clustered image
    clustered_image_path = os.path.join(output_folder, f'clustered_{os.path.basename(image_path)}')
    plt.savefig(clustered_image_path)
    plt.close()


def process_all_images_clustering(input_folder, gabor_folder, segments_folder, clustered_output_folder, cluster_labels):
    if not os.path.exists(clustered_output_folder):
        os.makedirs(clustered_output_folder)
    start_index = 0
    for i in range(1, 11):
        try:
            image_path = os.path.join(input_folder, f'{i}.jpg')
            segments_path = os.path.join(segments_folder, f'segments_{i}.npy')
            segments = np.load(segments_path)
            num_superpixels = np.max(segments) + 1
            image_cluster_labels = cluster_labels[start_index:start_index + num_superpixels]
            start_index += num_superpixels

            clustered_image = map_clusters_to_segments(segments, image_cluster_labels)
            visualize_clustered_image(image_path, clustered_image, clustered_output_folder)

        except Exception as e:
            print(f'Error in process_all_images_clustering for {image_path}: {e}')


# Step 4: Contextual Representation and Re-clustering
def get_superpixel_centroids(segments):
    labeled_segments = label(segments)
    regions = regionprops(labeled_segments)
    centroids = {region.label: region.centroid for region in regions}
    return centroids


def find_neighbors(segments, centroids, radius_factor, threshold):
    labeled_segments = label(segments)
    regions = regionprops(labeled_segments)
    neighbors = {region.label: [] for region in regions}
    expected_diameter = np.sqrt(segments.size / len(centroids))

    for region in regions:
        y, x = centroids[region.label]
        for neighbor in regions:
            if neighbor.label != region.label:
                ny, nx = centroids[neighbor.label]
                distance = np.sqrt((ny - y) ** 2 + (nx - x) ** 2)
                radius = radius_factor * expected_diameter
                if distance <= radius and np.sum(segments == neighbor.label) >= threshold:
                    neighbors[region.label].append(neighbor.label)

    return neighbors


def compute_contextual_features(superpixel_features, neighbors, level1_radius, level2_radius, threshold):
    contextual_features = []
    for sp in range(len(superpixel_features)):
        features = superpixel_features[sp]
        first_level_feats = [superpixel_features[n] for n in neighbors[sp] if np.linalg.norm(
            np.array(superpixel_features[sp]) - np.array(superpixel_features[n])) <= level1_radius]
        second_level_feats = [superpixel_features[n] for n in neighbors[sp] if level1_radius < np.linalg.norm(
            np.array(superpixel_features[sp]) - np.array(superpixel_features[n])) <= level2_radius]

        first_level_avg = np.mean(first_level_feats, axis=0) if first_level_feats else np.zeros_like(features)
        second_level_avg = np.mean(second_level_feats, axis=0) if second_level_feats else np.zeros_like(features)

        contextual_features.append(np.concatenate([features, first_level_avg, second_level_avg]))

    return np.array(contextual_features)


def process_all_images_contextual(input_folder, gabor_folder, segments_folder, contextual_output_folder,
                                  reclustered_output_folder, level1_radius_factor=1.5, level2_radius_factor=2.5,
                                  threshold=20, n_clusters=10):
    if not os.path.exists(contextual_output_folder):
        os.makedirs(contextual_output_folder)
    if not os.path.exists(reclustered_output_folder):
        os.makedirs(reclustered_output_folder)

    all_contextual_features = []
    for i in range(1, 11):
        try:
            image_path = os.path.join(input_folder, f'{i}.jpg')
            segments_path = os.path.join(segments_folder, f'segments_{i}.npy')
            segments = np.load(segments_path)
            gabor_features = np.load(os.path.join(gabor_folder, f'gabor_features_{i}.npy'))

            centroids = get_superpixel_centroids(segments)
            first_level_neighbors = find_neighbors(segments, centroids, level1_radius_factor, threshold)
            second_level_neighbors = find_neighbors(segments, centroids, level2_radius_factor, threshold)

            # Combine first and second level neighbors
            neighbors = {sp: list(set(first_level_neighbors[sp] + second_level_neighbors[sp])) for sp in
                         first_level_neighbors}

            # Compute contextual features
            contextual_features = compute_contextual_features(gabor_features, neighbors, level1_radius_factor,
                                                              level2_radius_factor, threshold)
            np.save(os.path.join(contextual_output_folder, f'contextual_features_{i}.npy'), contextual_features)

            # Collect all contextual features for clustering
            all_contextual_features.append(contextual_features)

        except Exception as e:
            print(f'Error in process_all_images_contextual for {image_path}: {e}')

    if all_contextual_features:
        all_contextual_features = np.vstack(all_contextual_features)

        # Re-cluster superpixels using contextual features
        cluster_labels = cluster_superpixels(all_contextual_features, n_clusters)
        process_all_images_clustering(input_folder, contextual_output_folder, segments_folder,
                                      reclustered_output_folder, cluster_labels)
    else:
        print("No contextual features were computed.")


# Execute steps sequentially
print("Step 1: Superpixel Segmentation")
process_all_images_segmentation(input_folder, segmentation_output_folder)

print("\nStep 2: Compute Gabor Features")
all_superpixel_features, image_superpixel_features = process_all_images_gabor(input_folder, segmentation_output_folder,
                                                                              gabor_output_folder,
                                                                              gabor_features_txt_folder, frequencies,
                                                                              orientations)

print("\nStep 3: Clustering Superpixels")
cluster_labels = cluster_superpixels(all_superpixel_features, n_clusters=10)
process_all_images_clustering(input_folder, gabor_output_folder, segmentation_output_folder, clustered_output_folder,
                              cluster_labels)

print("\nStep 4: Contextual Representation and Re-clustering")
process_all_images_contextual(input_folder, gabor_output_folder, segmentation_output_folder, contextual_output_folder,
                              reclustered_output_folder, level1_radius_factor=1.5, level2_radius_factor=2.5,
                              threshold=20, n_clusters=10)
