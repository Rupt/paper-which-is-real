"""
Prepare data for the 2d map example.
"""
import os

import numba
import numpy
from PIL import Image

BASE_MAP = "images/mapdraw_poster.png"

FOLDER = "data_map"


def main():
    img = imload(BASE_MAP)
    unique = numpy.unique(img)

    for x in unique:
        blobs = find_isolated(img, x)
        for i, mask in enumerate(blobs):
            imdump(mask, "mask_%d_%d.png" % (x, i))


def find_isolated(img, level):
    """Return a list of image masks for isolated blobs at level."""
    select = (img == level).astype(numpy.uint8)

    blobs = []
    while select.any():
        mask = find_a_blob(select)
        select[mask] = 0
        blobs.append(mask)

    return blobs


@numba.njit(numba.bool_[:, :](numba.uint8[:, :]))
def find_a_blob(img):
    """Return a mask for a contiguous blob in img.

    Image should contain 1 in allowed regions, 0 in disallowed.

    Horrible algorithm, but simple and fit for purpose.
    """
    img = img.copy()
    height, width = img.shape

    def find_seed():
        for y in range(height):
            for x in range(width):
                if img[y, x]:
                    return y, x
        return -1, -1

    y, x = find_seed()

    if y < 0:
        return numpy.zeros(img.shape, numba.bool_)

    img = img.copy()
    img[y, x] = 2

    def update(y, x):
        hit = img[y, x] == 1
        img[y, x] += hit
        return hit

    while True:
        change = False

        for y in range(height):
            for x in range(width):
                # 0 -> denied
                # 1 -> allowed
                # 2 -> in our blob
                if img[y, x] != 2:
                    continue

                # infect up, down, left, right directions
                change |= update(max(0, y - 1), x)
                change |= update(min(height - 1, y + 1), x)
                change |= update(y, max(0, x - 1))
                change |= update(y, min(width - 1, x + 1))

        if not change:
            break

    return img == 2


# image utilities


def imload(filename):
    """Return an image array loaded from filename."""
    return numpy.array(Image.open(filename))


def imdump(img, filename, *, log=True):
    """Write a numpy array as image filename."""
    fullpath = os.path.join(FOLDER, filename)
    img = Image.fromarray(img)
    img.save(fullpath)


if __name__ == "__main__":
    main()
