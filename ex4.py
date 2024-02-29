import cv2

import multi_color_heatmap
from HarrisPointDetection import *

""" ********************************* -  UTILITIES  - *************************************************"""

""" ********************************* -  CODE  - *************************************************"""


# def capture_point_detection_process(original_img,  harris: MultiScaleHarrisPointDetector, name:str, colors_dict: dict):
#     corners = harris.all_corners
#     final_corners = harris.points()
#
#     original_img = CLAMP_TO_UINT8(original_img)
#     capture = np.stack([original_img] * 3, axis=-1)/4
#     print(f'Number of corners detected: {np.count_nonzero(corners)}')
#     print(f'Number of corners after non-maximum suppression: {len(final_corners)}')
#
#
#
#     for point, color in colors_dict.items():
#         capture[point[0], point[1]] = color
#
#     save_image(capture, f"{name}_{thresh_proportion}__dist_{min_distance}", "Images")


def too_close_to_edge_func(point, img_shape, patch_size):
    half_patch = patch_size // 2
    x, y = point
    return x - half_patch < 0 or x + half_patch > img_shape[1] or y - half_patch < 0 or y + half_patch > img_shape[0]


def extract_patch_func(point, image, patch_size):
    half_patch = patch_size // 2
    x, y = point
    return image[y - half_patch:y + half_patch, x - half_patch:x + half_patch]



def match_features(descriptors1: [HarrisPoint], descriptors2: [HarrisPoint], threshold=.99):
    """
    Match feature descriptors between two sets.

    Parameters:
    - descriptors1: Feature descriptors from the first image.
    - descriptors2: Feature descriptors from the second image.
    - threshold: Distance ratio threshold for filtering matches.

    Returns:
    - A list of tuples (i, j) where descriptor i in descriptors1
      is matched with descriptor j in descriptors2.
    """
    matches = []
    for i, desc1 in enumerate(descriptors1):
        # Calculate distances to all descriptors in the second image
        distances = np.array([desc1.ssd(desc2) for desc2 in descriptors2])
        print(f"point ({desc1} dists: {sorted(distances)})")

        # Find the best and second-best match
        if len(distances) < 2: continue  # Ensure there are at least 2 descriptors to compare
        idx_sorted = np.argsort(distances)
        best, second_best = distances[idx_sorted[0]], distances[idx_sorted[1]]

        # Apply the Lowe's ratio test to filter out weak matches
        if best < threshold * second_best:
            matches.append((i, idx_sorted[0]))

    return matches


def draw_interest_points(image: np.ndarray, interest_points: [HarrisPoint]):

    # Convert the image to RGB if it is grayscale to ensure we can draw colored points
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create a copy of the image to draw on
    result_image = image.copy()

    for point in interest_points:
        y, x, theta = point.x, point.y, point.theta
        # Calculate the end point of the line indicating orientation
        length = 10  # Length of the line
        end_x = int(x + length * np.cos(theta))
        end_y = int(y + length * np.sin(theta))

        print(f"point: {point} theta: {theta} end: ({end_x}, {end_y})")

        # Draw the interest point as a circle and the orientation as a line

        cv2.circle(result_image, (x, y), 3, (0, 255, 0), -1)  # Green point
        cv2.line(result_image, (x, y), (end_x, end_y), (255, 0, 0), 2)  # Blue line

    return result_image


def add_point_to_image(image: np.ndarray, pt: HarrisPoint, color: tuple[int, int, int]):
    y, x, theta = pt.x, pt.y, pt.theta
    # Calculate the end point of the line indicating orientation
    length = 10  # Length of the line
    # end_x = int(x + length * np.cos(theta))
    # end_y = int(y + length * np.sin(theta))
    cv2.circle(image, (x, y), 3, color, -1)
    # cv2.line(image, (x, y), (end_x, end_y), color, 2)





def add_matching_points_to_images(pt_a: HarrisPoint, pt_b: HarrisPoint, img_a: np.ndarray, img_b: np.ndarray, rand_color: tuple[int, int, int]):
    add_point_to_image(img_a, pt_a, rand_color)
    add_point_to_image(img_b, pt_b, rand_color)
    return img_a, img_b



match_color_map = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]

matches = 0
if __name__ == "__main__":
    from ImageFFT import *



    desert_low = load_image(DESERT_LOW_RES, "L")
    desert_hi = load_image(DESERT_HIGH_RES, "L")

    edges_a_points: [HarrisPoint] = MultiScaleHarrisPointDetector(desert_low, "desert_low").get_interest_points()
    edges_b_points: [HarrisPoint] = MultiScaleHarrisPointDetector(desert_hi, "desert_hi").get_interest_points()


    #
    for point in edges_a_points:
        point.find_2_nearest_neighbors(edges_b_points)



    img_a = np.stack([desert_low, desert_low, desert_low], axis=-1)
    img_a = clamp255(img_a)
    img_b = np.stack([desert_hi]*3, axis=-1)
    img_b = clamp255(img_b)



    for i, pt_a in enumerate(edges_a_points):
        color = multi_color_heatmap.pixel(i % 255)
        pt_b = pt_a.matched_point

        if pt_a is None or pt_b is None:
            continue
        matches += 1

        print(f'pt_a is {pt_a} and its match is {pt_b}, pt_b is {pt_b} and its match is {pt_a} | color is {color}')
        img_a, img_b = add_matching_points_to_images(pt_a, pt_b, img_a, img_b, color)

    # img_a = draw_interest_points(img_a, edges_a_points)
    # img_b = draw_interest_points(img_b, edges_b_points)


    # save_image(img_a, "img_a", "Images")
    # save_image(img_b, "img_b", "Images")

    # save_image(img_b, "img_b", "Images")
    side_by_side_result = np.concatenate((img_a, img_b), axis=1)
    save_image(side_by_side_result, "all_interest_points", "Images")

    print(f"Found {matches} matches")
