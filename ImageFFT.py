from scipy.fft import ifft2, fft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from Utils import *

def get_radial_gauss_filters(shape: tuple[int, int], sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a low-pass and high-pass filter for the given shape and cut-off frequency (sigma)
    :param shape: The dimensions of the 2D filters
    :param sigma: The cut-off frequency of the filters (size of the Gaussian)
    :return: A tuple containing the low-pass and high-pass filters (in that order)
    """
    rows, cols = shape
    x, y = np.indices((rows, cols))
    center_x, center_y = rows / 2, cols / 2
    # Using a Gaussian function for gradual fade
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    lpf = np.exp(-radius ** 2 / (2 * sigma ** 2))
    # lpf = np.stack([lpf] * 3, axis=-1)
    return lpf, 1 - lpf

def get_hps(shape: tuple[int, int], sigma: float) -> np.ndarray:
    _, hpf = get_radial_gauss_filters(shape, sigma)
    return hpf


def sharpen_image(image: np.ndarray):
    amount = 0.03
    hpf = get_hps(image.shape, 50)

    fft_image = fftshift(fft2(image))

    fft_image += fft_image * hpf * amount

    return np.real(ifft2(ifftshift(fft_image)))


def apply_laplacian_filter(image: np.ndarray):
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return convolve2d(image, laplacian, mode='same', boundary='wrap')

def blurr_image(image: np.ndarray, sigma: float):
    return gaussian_filter(image, sigma)


def get_image_blurness(image: np.ndarray):
    f = fft2(image)
    magnitude = np.abs(f)
    magnitude = apply_laplacian_filter(magnitude)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    return np.count_nonzero(magnitude)

def save_image_as_fft(image: np.ndarray, name):
    f = fft2(image)
    magnitude = np.real(f)
    magnitude = apply_laplacian_filter(magnitude)

    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    non_black_pixels = np.count_nonzero(magnitude)
    print(f"Non-black pixels: {non_black_pixels}")

    # threshold_mag = np.zeros_like(magnitude)
    # threshold_mag[magnitude > 0] = 255
    save_image(magnitude, f"FFT_Thresh_Magnitude_Image_{name}", "Images")
    return magnitude


def find_image_highest_frequency(image: np.ndarray):
    f = fftshift(fft2(image))
    magnitude = np.log(np.abs(f) + 1)
    save_image(magnitude, "FFT_Magnitude_Image", "Images")
    # Search for the highest frequency in the image
    max_magnitude = 0
    max_magnitude_index = (0, 0)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] > max_magnitude:
                max_magnitude = magnitude[i, j]
                max_magnitude_index = (i, j)
