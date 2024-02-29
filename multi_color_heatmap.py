import math
from PIL import Image

# A map of rgb points in your distribution
# [distance, (r, g, b)]
# distance is percentage from left edge
map = [
    [0.0, (0, 0, 0)],
    [0.20, (0, 0, .5)],
    [0.40, (0, .5, 0)],
    [0.60, (.5, 0, 0)],
    [0.80, (.75, .75, 0)],
    [0.90, (1.0, .75, 0)],
    [1.00, (1.0, 1.0, 1.0)],
]

# A map of bright rgb point to mark sets of two matching points:
# green, red, blue, yellow, cyan, magenta,


def match_color_generator():
    num = 0
    while True:
        yield num
        num += 1
        num %= 6




def gaussian(x, a, b, c, d=0):
    return a * math.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def make_lookup_table():
    lookup = {}
    for i in range(256):
        r, g, b = calc_pixel(i, 255)
        lookup[i] = (r, g, b)
    return lookup


def calc_pixel(response, max_response=255, spread=1):
    width = float(max_response)
    r = g = b = 0
    for p in map:
        gauss_value = gaussian(response, 1, p[0] * width, width / (spread * len(map)))
        r += gauss_value * p[1][0]
        g += gauss_value * p[1][1]
        b += gauss_value * p[1][2]
    return int(max(0, min(1, r)) * 255), int(max(0, min(1, g)) * 255), int(max(0, min(1, b)) * 255)


lut = make_lookup_table()


def pixel(val: int):
    return lut[val]
