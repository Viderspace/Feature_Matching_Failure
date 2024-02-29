import colorsys
import os
import sys
from collections import Counter
from functools import partial
from scipy.ndimage import convolve1d, gaussian_filter
from multi_color_heatmap import pixel

import numpy as np
import scipy
from PIL import Image

CLAMP_TO_UINT8 = lambda a: (255.1 * (a - np.min(a)) / np.ptp(a)).astype(np.uint8)

# setting the numpy print options so the lines are not truncated
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

""" ********************************* -  CONSTANTS  - *************************************************"""

DESERT_LOW_RES = "Images/Exercise_Inputs/desert_low_res.jpg"
DESERT_HIGH_RES = "Images/Exercise_Inputs/desert_high_res.png"
LAKE_LOW_RES = "Images/Exercise_Inputs/lake_low_res.jpg"
LAKE_HIGH_RES = "Images/Exercise_Inputs/lake_high_res.png"

SAMPLE_HARRIS_NOISED = "Images/Exercise_Inputs/sample_harris_noised005.jpg"
SAMPLE_HARRIS_CLEAN = "Images/Exercise_Inputs/sample_harris_clean.jpg"
SAMPLE_HARRIS_PROJECTED = "Images/Exercise_Inputs/Sample_squared_projected.jpg"

EDGES_EXAMPLE = "Images/Exercise_Inputs/EDGES.jpg"
EDGES_PROJECTED = "Images/Exercise_Inputs/EDGES_PROJECTED.jpg"

BLUE_PIXEL = [0, 0, 255]
RED_PIXEL = [255, 0, 0]
GREEN_PIXEL = [0, 255, 0]
WHITE_PIXEL = [255, 255, 255]
BLACK_PIXEL = [0, 0, 0]

""" ********************************* -  FUNCTIONS  - *************************************************"""


def get_colorful_generator(match_idx):
    hue = match_idx / 12
    (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    R, G, B = int(255 * r), int(255 * g), int(255 * b)
    return R, G, B


def print_link_to_console(absolute_path: str, name: str = "") -> None:
    print(f"New image created as '{name}' at (file://{absolute_path})")


def load_image(path: str, mode="RGBA") -> np.ndarray:
    try:
        image = Image.open(path).convert(mode)
        np_image = np.array(image, dtype=np.float64)
        image.close()
        return np_image
    except FileNotFoundError:
        print(f"File not found for image at path {path}", file=sys.stderr)
        return exit(1)


def save_image(image: np.ndarray, name: str = "output_img", dir_path: str = "Images"):
    relative_path = f"{dir_path}/{name}.png"
    full_path = os.getcwd() + "/" + relative_path
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(np.uint8(image))
    pil_image = pil_image.convert("RGB")
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the image and print the link to the console
        pil_image.save(relative_path)
        print_link_to_console(full_path, name)

    except OSError:
        print(f"Error saving image {name} at path {relative_path}", file=sys.stderr)
