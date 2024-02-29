from Utils import *
from HarrisPoint import HarrisPoint

MIN_NEIGHBOR_DIST = 40
THRESHOLD_PERCENTAGE = 0.03
SIGMA = 5.
K = 0.06


class MultiScaleHarrisPointDetector:
    def __init__(self, image: np.ndarray, name: str = "Harris"):
        self.image = image
        self.name = name
        self.all_corners: [HarrisPoint] = self.find_all_suspected_points(self.image)
        print(f"Harris found {len(self.all_corners)} corners before non-maximum suppression")

        self.all_corners.sort(key=lambda point: point.response, reverse=True)
        self.interest_points = non_maximum_suppression(image.shape, self.all_corners)
        print(f"Harris found {len(self.interest_points)} corners after non-maximum suppression")
        # save_heatmap_of_interest_points(self.interest_points, image, self.name)

        get_image_gradient(image, self.interest_points, self.name)
        self.attach_patches_to_points(self.interest_points)

    def get_interest_points(self) -> [HarrisPoint]:
        return self.interest_points

    def find_all_suspected_points(self, image):
        # Apply Sobel filters to the image
        Gx, Gy = sobel_filters(image)

        # Calculate the products of the gradients at each pixel
        Gx2, Gy2, GxGy = calculate_gradient_products(Gx, Gy)
        # Apply Gaussian filters to the products of the gradients
        gx2_gauss = gaussian_filter(Gx2, SIGMA)
        gy2_gauss = gaussian_filter(Gy2, SIGMA)
        gxgy_gauss = gaussian_filter(GxGy, SIGMA)

        # Calculate the response of the Harris detector at each pixel
        response = make_response_map(gx2_gauss, gy2_gauss, gxgy_gauss)
        self.response = response

        # Apply the threshold to the response map
        variable_threshold = THRESHOLD_PERCENTAGE * response.max()

        # Find the corners in the response map that are above the threshold
        corners = np.argwhere(response > variable_threshold)

        save_heatmap_with_image_overlay(response, image, self.name)
        return [HarrisPoint(p[0], p[1], response[p[0], p[1]]) for p in corners]

    def attach_patches_to_points(self, interest_points: [HarrisPoint], patch_size=16):
        for pt in interest_points:

            # note - new method via gradient patch
            x0, y0, x1, y1 = get_patch_bounds(pt.x, pt.y, patch_size, self.image.shape)
            # gx_patch = gx[x0:x1, y0:y1]
            # gy_patch = gy[x0:x1, y0:y1]
            # assert gx_patch.shape == gy_patch.shape == (patch_size,
            #                                             patch_size), f"gx shape={gx_patch.shape}, gy shape={gy_patch.shape}, point={pt.x, pt.y}, x0={x0}, y0={y0}, x1={x1}, y1={y1}"
            # pt.set_gradient_patch_16x16(gx_patch, gy_patch)
            pt.set_patch_40x40(self.image[x0:x1, y0:y1])


def get_image_gradient(image: np.ndarray, interest_points: [HarrisPoint] = None, name: str = ''):
    Gx, Gy = sobel_filters(image)
    gradient = clamp255(Gx + Gy)

    if interest_points is not None:
        gradient_ref = np.stack([gradient.copy()] * 3, axis=-1)
        max_response, min_response = interest_points[0].response, interest_points[-1].response
        clamp = partial(clamp_func, max_response=max_response, min_response=min_response)

        for point in interest_points:
            dim_neighbors(gradient_ref, point.x, point.y)
            rgb = pixel(clamp(point.response))
            gradient_ref[point.x, point.y] = rgb

        save_image(gradient_ref, f"{name}_clean_gradient", "Images")

    return gradient


def save_heatmap_with_image_overlay(response: np.ndarray, image: np.ndarray, name=""):
    response = clamp255(response)
    heatmap = np.zeros((response.shape[0], response.shape[1], 3), dtype=np.uint8)

    # Apply the lookup table to fill in the heatmap
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            heatmap[i, j] = pixel(response[i, j])

    image_gray = np.stack((image, image, image), axis=-1)
    image_gray[heatmap > 20] = heatmap[heatmap > 20]
    save_image(image_gray, f"heatmap_image_overlay{name}.png")


def clamp_func(r, max_response, min_response):
    if max_response == min_response:
        return 255
    lerped = int(255 * (r - min_response) / (max_response - min_response))
    return min(max(lerped + 100, 127), 255)


def save_heatmap_of_interest_points(points: [HarrisPoint], image: np.ndarray, name=""):
    heatmap = np.stack((image, image, image), axis=-1)
    max_response, min_response = points[0].response, points[-1].response
    clamp = partial(clamp_func, max_response=max_response, min_response=min_response)

    for point in points:
        dim_neighbors(heatmap, point.x, point.y)
        rgb = pixel(clamp(point.response))
        heatmap[point.x, point.y] = rgb

    save_image(heatmap, f"selected_corners_heatmap_image_overlay{name}.png")


def dim_neighbors(map: np.ndarray, x, y, color=(0, 0, 0)):
    map[x - 1:x + 2, y - 1:y + 2] = color
    return map


def clamp255(a):
    return (255.1 * (a - np.min(a)) / np.ptp(a)).astype(np.uint8)


def sobel_filters(image):
    # Define Sobel operator kernels
    blur_k = np.array([1, 2, 1], np.float64) / 4
    deriv_k = np.array([-1, 0, 1], np.float64)

    Gx = convolve1d(image, deriv_k, axis=0, mode='nearest')
    Gx = convolve1d(Gx, blur_k, axis=1, mode='nearest')

    Gy = convolve1d(image, deriv_k, axis=1, mode='nearest')
    Gy = convolve1d(Gy, blur_k, axis=0, mode='nearest')

    return Gx, Gy


def calculate_gradient_products(Gx, Gy):
    # Calculate the products of the gradients
    Gx2 = np.square(Gx)
    Gy2 = np.square(Gy)
    GxGy = np.multiply(Gx, Gy)
    return Gx2, Gy2, GxGy


def make_response_map(gx2_gaussian, gy2_gaussian, gxgy_gaussian):
    # Calculate the structure tensor
    A = gx2_gaussian
    B = gxgy_gaussian
    C = gy2_gaussian
    # Calculate the determinant and trace of the structure tensor
    det = (A * C) - (B ** 2)
    trace = A + C
    r = det - K * (trace ** 2)
    return r


def get_patch_bounds(x, y, patch_size, img_dim):
    x0 = max(x - patch_size // 2, 0)
    y0 = max(y - patch_size // 2, 0)
    x1 = min(x + patch_size // 2, img_dim[0])
    y1 = min(y + patch_size // 2, img_dim[1])
    return x0, y0, x1, y1


def non_maximum_suppression(img_dim, all_corners_sorted: [HarrisPoint]):
    # Keep a list for final corners after NMS
    final_corners = []
    suppressed = np.zeros(img_dim, dtype=bool)

    # all the corners are sorted by response (Highest -> ... -> Lowest)
    for corner in all_corners_sorted:
        x, y = corner.position()
        if not suppressed[x, y] and not is_too_close_to_edge(x, y, img_dim, 16):
            # This pixel has the highest response in its neighborhood
            final_corners.append(corner)
            # Suppress the neighborhood
            x0, y0, x1, y1 = get_patch_bounds(x, y, MIN_NEIGHBOR_DIST, img_dim)
            suppressed[x0:x1, y0:y1] = True

    return final_corners


def is_too_close_to_edge(x, y, img_dim, patch_size=16):
    return x + patch_size > img_dim[0] or \
           y + patch_size > img_dim[1] or \
           x - patch_size < 0 or \
           y - patch_size < 0
