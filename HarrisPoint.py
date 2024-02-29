from __future__ import annotations

import matplotlib.pyplot as plt
# set pyplot to macosx backend
import numpy as np

plt.switch_backend('macosx')

from scipy.ndimage import convolve1d

from Utils import *

BLUR_KERNEL = np.array([1, 5, 10, 10, 5, 1]) / 32

LOWES_RATIO_THRESHOLD = .6


def generate_gaussian_weights(sigma=4):
    """Generate a 16x16 matrix of Gaussian weights."""
    size = 16  # Patch size
    kernel = np.zeros((size, size), dtype=np.float32)

    for x in range(size):
        for y in range(size):
            dx = x - (size - 1) / 2
            dy = y - (size - 1) / 2
            kernel[x, y] = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of all elements is 1
    kernel /= 2 * np.pi * sigma ** 2
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

    return kernel


class HarrisPoint:
    # Generate the Gaussian weights
    gaussian_weights = generate_gaussian_weights()

    def __init__(self, x: int, y: int, response: float):
        self.x, self.y, = x, y
        self.pt = (x, y)
        self.response = response

        self.original_patch = None
        self.gradient_patch_8x8 = None
        self.descriptor = None
        self.theta = None
        self.nearest_neighbor_ratio: float = np.Inf
        self.nn_distance = np.Inf
        self.matched_point = None

    def __repr__(self):
        return f"({self.x},{self.y})"

    def position(self):
        return self.x, self.y

    def response(self):
        return self.response

    def ssd(self, other) -> float:
        # Euclidean distance over the descriptors
        return float(np.sum((self.descriptor - other.descriptor) ** 2))

    def set_descriptor(self, descriptor: np.ndarray):
        self.descriptor = descriptor

    def set_patch_40x40(self, patch: np.ndarray):
        self.original_patch = patch.copy()
        self.descriptor =  blur(patch, 2)

        # magnitude, orientation = calculate_gradients(patch)
        # histogram = create_orientation_histogram(magnitude, orientation)
        # dominant_orientation = find_dominant_orientation(histogram)
        # self.theta = 0
        # print(f"Dominant Orientation: {0} degrees")

        # Adjust the magnitude and orientation arrays according to the canonical orientation
        # rotated_orientation = rotate_gradients(magnitude, orientation, 0)

        # Then, create the SIFT descriptor
        # self.descriptor = (magnitude+orientation).flatten()

    def create_orientation_histogram(self, magnitude, orientation, bin_size=10):
        """
        Create a histogram of orientations.

        Args:
        - magnitude: A 16x16 array of gradient magnitudes.
        - orientation: A 16x16 array of gradient orientations in radians.
        - bin_size: The size of each bin in the histogram in degrees.

        Returns:
        - histogram: A numpy array representing the histogram of orientations.
        """
        gauss_weights = generate_gaussian_weights()
        gauss_weights = (gauss_weights - gauss_weights.min()) * 255 / (gauss_weights.max() - gauss_weights.min())
        output_image = np.concatenate((magnitude, orientation, gauss_weights, self.original_patch), axis=1)
        save_image(output_image, f"patch_{self.x},{self.y}")

        gauss_weights = gauss_weights.flatten()

        num_bins = int(360 / bin_size)
        histogram = np.zeros(num_bins)

        # Convert orientation from radians to degrees and flatten arrays
        orientation_degrees = np.degrees(orientation).flatten()
        magnitude = magnitude.flatten()

        # Normalize orientations to be within [0, 360) degrees
        orientation_degrees = orientation_degrees % 360

        for i in range(len(orientation_degrees)):
            bin_index = int(orientation_degrees[i] // bin_size)
            histogram[bin_index] += magnitude[i] * gauss_weights[i]

        return histogram

    def find_2_nearest_neighbors(self, second_img_points):
        """
        Find the 2 nearest neighbors of the point in the second image and apply Lowe's ratio test to filter matches.
        :param second_img_points: the points in the second image
        """
        if not second_img_points:
            print("No points provided in the second image.")
            return

        # Calculate SSD for each point in the second image.
        neighbors_distances = {point: self.ssd(point) for point in second_img_points}

        # Sort neighbors by distance.
        sorted_neighbors = sorted(neighbors_distances.items(), key=lambda x: x[1])

        # Ensure there are at least two neighbors for the ratio test.
        if len(sorted_neighbors) < 2:
            print("Not enough neighbors for ratio test.")
            return

        # Calculate the nearest neighbor ratio.
        nearest_neighbor_distance, second_nearest_neighbor_distance = sorted_neighbors[0][1], sorted_neighbors[1][1]
        self.nearest_neighbor_ratio = nearest_neighbor_distance / second_nearest_neighbor_distance

        # Lowe's ratio test to filter out weak matches.

        if self.nearest_neighbor_ratio < LOWES_RATIO_THRESHOLD:
            # Considered a good match.
            matched_point = sorted_neighbors[0][0]
            self.matched_point = matched_point
            self.nn_distance = nearest_neighbor_distance
            self.matched_point.set_twin_point(self, self.nn_distance, self.nearest_neighbor_ratio)
            print(
                f"Point {self.x},{self.y} : Matched with ({self.matched_point.x},{self.matched_point.y}) with ratio: {self.nearest_neighbor_ratio}")
        else:
            # The match is not good enough.
            print(
                f"Point {self.x},{self.y} : Below threshold:: {self.nearest_neighbor_ratio} <{LOWES_RATIO_THRESHOLD} ")

    # Note: This refactored function is designed to be a method within a class that `HarrisPoint` instances belong to.
    # The `ssd` method should calculate the squared sum of differences between this point and another point.
    # The `set_twin_point` method is assumed to link two points as matching pairs along with their distance.

    def set_twin_point(self, point: HarrisPoint, distance: float, ratio: float):
        if distance < self.nn_distance and ratio < self.nearest_neighbor_ratio:
            self.matched_point = point
            self.nn_distance = distance
            self.nearest_neighbor_ratio = ratio


def blur(patch: np.ndarray, iterations: int = 1):
    for _ in range(iterations):
        patch = convolve1d(patch, BLUR_KERNEL, axis=1, mode='reflect')
        patch = convolve1d(patch, BLUR_KERNEL, axis=0, mode='reflect')
    return patch


def visualize_descriptor(descriptor: np.ndarray, point: HarrisPoint):
    """
    Visualizes a 4x4x8 SIFT descriptor by plotting arrows for the most dominant direction in each sub-patch.

    Parameters:
    - descriptor: A 4x4x8 numpy array representing the histograms of orientations for 4x4 sub-patches.
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True)
    ax.set_aspect('equal')
    fig.suptitle(f"SIFT Descriptor for Point ({point.x}, {point.y})")

    # Iterate through each sub-patch
    for i in range(4):
        for j in range(4):
            # Find the bin with the highest value (most dominant orientation)
            dominant_orientation_bin = np.argmax(descriptor[i, j, :])
            # Calculate the angle (in radians) of the dominant orientation
            angle = np.deg2rad(dominant_orientation_bin * 45)  # Each bin represents 45 degrees

            # Calculate the direction of the arrow
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Plot the arrow onto the grid
            ax.arrow(j + 0.5, 4 - (i + 0.5), dx * 0.4, dy * 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Save the figure
    plt.savefig('Images/Descriptors/' + f"Pt_{point.x},{point.y}_descriptor_arrows.png")
    plt.close()


def calculate_gradients(patch):
    """
    Calculate the gradients of the patch.

    Args:
    - patch: A 16x16 numpy array representing a grayscale image patch.

    Returns:
    - magnitude: A 16x16 array of gradient magnitudes.
    - orientation: A 16x16 array of gradient orientations in radians.
    """
    # Sobel operators for gradient in X and Y direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Compute the gradients in X and Y direction
    grad_x = np.convolve(patch.flatten(), sobel_x.flatten(), 'same').reshape(patch.shape)
    grad_y = np.convolve(patch.flatten(), sobel_y.flatten(), 'same').reshape(patch.shape)
    return grad_x, grad_y

    # Compute gradient magnitude and orientation
    # magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # orientation = np.arctan2(grad_y, grad_x)
    #
    # return magnitude, orientation





def find_dominant_orientation(histogram, bin_size=10):
    """
    Find the dominant orientation based on the histogram.

    Args:
    - histogram: A numpy array representing the histogram of orientations.
    - bin_size: The size of each bin in the histogram in degrees.

    Returns:
    - dominant_orientation: The dominant orientation in degrees.
    """
    dominant_bin = np.argmax(histogram)
    dominant_orientation = dominant_bin * bin_size + bin_size / 2  # Center of the bin

    return dominant_orientation


from scipy.ndimage import gaussian_filter

"""*********************** SIFT Descriptor ****************************"""


def compute_descriptor(patch):
    """
    Compute a SIFT descriptor for the given patch.

    Args:
    - patch: A 16x16 numpy array representing a grayscale image patch.

    Returns:
    - descriptor: A 128-element numpy array representing the SIFT descriptor.
    """
    # Preprocess the patch: Apply Gaussian smoothing
    patch_smoothed = gaussian_filter(patch, sigma=1)

    # Compute gradients
    magnitude, orientation = calculate_gradients(patch_smoothed)

    # Initialize the descriptor
    descriptor = np.zeros(128)  # 4x4 sub-regions, each with an 8-bin histogram

    # Process each 4x4 sub-region
    for i in range(4):
        for j in range(4):
            sub_magnitude = magnitude[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            sub_orientation = orientation[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]

            # Create orientation histogram for this sub-region with 8 bins
            hist = create_orientation_histogram(sub_magnitude, sub_orientation, bin_size=45)

            # Add to descriptor
            descriptor[(i * 4 + j) * 8:(i * 4 + j + 1) * 8] = hist

    # Normalize the descriptor to have unit length
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)

    # Thresholding values above 0.2 (to reduce the effect of illumination changes)
    descriptor[descriptor > 0.2] = 0.2

    # Normalize again to have unit length
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)

    return descriptor


# Update the create_orientation_histogram function for 8 bins and assuming the input orientations are in radians
def create_orientation_histogram(magnitude, orientation, bin_size: int = 45):
    """
    Create a histogram of orientations for a sub-region.

    Args:
    - magnitude: An array of gradient magnitudes for the sub-region.
    - orientation: An array of gradient orientations (in radians) for the sub-region.
    - bin_size: The size of each bin in the histogram in degrees.

    Returns:
    - histogram: A numpy array representing the histogram of orientations with 8 bins.
    """
    num_bins = 360 // bin_size
    histogram = np.zeros(num_bins, dtype=np.float32)

    orientation_degrees = np.degrees(orientation).flatten() % 360
    magnitude = magnitude.flatten()

    for o, m in zip(orientation_degrees, magnitude):
        bin_index = int(o // bin_size)
        histogram[bin_index % num_bins] += m

    return histogram


def rotate_gradients(magnitude, orientation, canonical_orientation):
    """
    Rotate gradient orientations based on the canonical orientation of the keypoint.

    Args:
    - magnitude: A 16x16 array of gradient magnitudes.
    - orientation: A 16x16 array of gradient orientations in radians.
    - canonical_orientation: The canonical orientation in degrees.

    Returns:
    - rotated_orientation: A 16x16 array of rotated gradient orientations in radians.
    """
    # Convert canonical orientation to radians
    canonical_orientation_rad = np.radians(canonical_orientation)

    # Rotate orientations
    rotated_orientation = orientation - canonical_orientation_rad

    # Normalize the rotated orientations to be within [-pi, pi)
    rotated_orientation = (rotated_orientation + np.pi) % (2 * np.pi) - np.pi

    return rotated_orientation


def create_sift_descriptor(magnitude, rotated_orientation):
    """
    Create a SIFT descriptor from the magnitude and rotated orientations of gradients.

    Args:
    - magnitude: A 16x16 array of gradient magnitudes.
    - rotated_orientation: A 16x16 array of rotated gradient orientations in radians.

    Returns:
    - descriptor: A 128-element numpy array representing the SIFT descriptor.
    """
    # Initialize descriptor
    descriptor = np.zeros((4, 4, 8))  # 4x4 subregions, 8 orientation bins

    # Bin size for orientations (45 degrees in radians)
    bin_size = np.pi / 4

    # Process each 4x4 subregion
    for i in range(4):
        for j in range(4):
            # Extract subregion
            sub_magnitude = magnitude[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            sub_orientation = rotated_orientation[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]

            # Flatten subregion arrays for processing
            sub_magnitude_flat = sub_magnitude.flatten()
            sub_orientation_flat = sub_orientation.flatten()

            # Calculate contributions to bins
            for k in range(len(sub_magnitude_flat)):
                orientation = sub_orientation_flat[k]
                mag = sub_magnitude_flat[k]

                # Normalize orientation to be within [0, 2*pi)
                orientation_normalized = orientation % (2 * np.pi)

                # Determine bin index
                bin_index = int(orientation_normalized // bin_size)
                if bin_index == 8:  # Handle edge case
                    bin_index = 7

                # Accumulate magnitude in corresponding bin
                descriptor[i, j, bin_index] += mag

    # Normalize the descriptor to unit length to ensure invariance to illumination changes
    descriptor = descriptor.flatten()
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm

    # Clip the values in the descriptor to a max of 0.2 to reduce the influence of large gradients
    # descriptor = np.clip(descriptor, a_min=0, a_max=0.2)

    # Normalize again after clipping
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm

    return descriptor
