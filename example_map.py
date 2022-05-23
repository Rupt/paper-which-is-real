"""
Example using sksym with a two-dimensional landmap.

"""
import functools
import glob
import os

import lightgbm
import matplotlib
import numpy
from matplotlib import pyplot
from PIL import Image

import sksym

RNG = numpy.random.Generator(numpy.random.Philox(0xF00D1))

PREFIX = os.path.basename(__file__)[:-3]

CMAP = matplotlib.cm.inferno


def main():
    ndata = 5_000

    def violate(scale, dist):
        x = numpy.linspace(1, 10, dist.shape[1])
        dist *= 1 - scale * numpy.sin(x) ** 2

    def wave(dist):
        for i in range(dist.shape[0]):
            rot = int(numpy.sin(i / 20) * 100)
            dist[i] = numpy.roll(dist[i], rot)

    def filter_(shape):
        ngap = 30
        scale = 0.2
        filt = numpy.ones(shape)
        filt[:ngap] = scale
        filt[-ngap:] = scale
        filt[:, :ngap] = scale
        filt[:, -ngap:] = scale
        return filt

    for frac in (0.0, 0.5, 0.9):
        suffix = "%dk_%.1f" % (ndata // 1000, frac)
        suffix = suffix.replace(".", "p")

        def seasons(d):
            return violate(frac, d)

        example_map(suffix, ndata, seasons)
        example_map(suffix + "_filtered", ndata, seasons, filter_)

    frac = 0.9
    suffix = "%dk_%.1f" % (ndata // 1000, frac)
    suffix = suffix.replace(".", "p")

    def seasons(d):
        return violate(frac, d)

    example_map(suffix + "_filtered_n10", ndata, seasons, filter_, nfakes=10)


def example_map(suffix, ndata, violate=None, filter_=None, *, nfakes=1):
    """Made data, fit a model, and output diagnostics."""
    print("suffix:", suffix, flush=True)

    images, heights = load_images()

    # build map from pixel to blob index
    shape = images[0].shape  # (y, x)
    pixel_index = numpy.empty(shape, numpy.int_)
    for i, img in enumerate(images):
        pixel_index[img] = i

    # make initial distribution scale with height
    dist = numpy.empty(shape, numpy.float_)
    min_height = min(heights)
    for i, img in enumerate(images):
        dist[img] = (heights[i] - min_height) ** 1.5

    # violate its symmetry
    if violate is not None:
        violate(dist)

    # apply filtering
    if filter_ is None:
        filt = numpy.ones(dist.shape)
    else:
        filt = filter_(dist.shape)
        dist *= filt

    map_cdf = dist.ravel().cumsum()
    map_cdf /= map_cdf[-1]

    # make un-violated distributions for contours
    cdfs = []
    for img in images:
        pdf = img * filt
        cdf = pdf.ravel().cumsum()
        if cdf[-1]:
            cdf /= cdf[-1]
        cdfs.append(cdf)

    # sample data (y, x)
    def sample_blob(cdf):
        # discrete cdf sample
        yxi = numpy.searchsorted(cdf, RNG.uniform())
        y = yxi // shape[1]
        x = yxi % shape[1]
        # assign within pixel
        y += RNG.uniform()
        x += RNG.uniform()
        return y, x

    data = numpy.empty((ndata * 2, 2))
    for i in range(len(data)):
        data[i] = sample_blob(map_cdf)

    x_train = data[:ndata]
    x_test = data[ndata:]

    # fit model
    def transform(data):
        new = numpy.empty_like(data)
        for i, (y0, x0) in enumerate(data):
            index = pixel_index[int(y0), int(x0)]
            new[i] = sample_blob(cdfs[index])
        return new

    blobber = sksym.WhichIsReal(transform, nfakes)

    model = lightgbm.LGBMRegressor(
        objective=blobber.objective(),
        max_depth=2,
        random_state=RNG.integers(2**31),
    )

    sksym.fit(model, blobber.pack(x_train))

    # score
    x_pack = blobber.pack(x_test)

    print("mean llr: %.3f +- %.3f" % sksym.score(model, x_pack, and_std=True))

    # figure: data
    nscat = 10_000

    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(9, 5),
        tight_layout=(0, 0, 0),
    )

    axis.scatter(
        x_train[:nscat, 1],
        x_train[:nscat, 0],
        c="k",
        s=1,
        marker=",",
        lw=0,
    )

    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0], 0)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_aspect("equal")

    save_fig(figure, "data_%s.png" % suffix)

    # figure: classifier by blob
    scores = sksym.predict_log_proba(model, x_pack)[..., 0] - numpy.log(0.5)
    if nfakes > 1:
        scores = scores.mean(axis=0)
    blob_scores = numpy.zeros(len(images))
    for i, score in enumerate(scores):
        y, x = x_train[i]
        index = pixel_index[int(y), int(x)]
        blob_scores[index] += score

    blob_scores /= len(scores)

    influence = numpy.empty(shape)
    for i, bs in enumerate(blob_scores):
        img = images[i]
        influence[img] = bs

    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(9, 5),
        tight_layout=(0, 0, 0),
    )

    size = 0.007
    im = axis.imshow(
        influence,
        vmin=-size,
        vmax=size,
        cmap=CMAP,
        interpolation="None",
    )

    figure.colorbar(im)

    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0] - 1, 0)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_aspect("equal")

    save_fig(figure, "score_%s.png" % suffix)

    # figure: model output
    ygrid = numpy.linspace(0, shape[0], 100)
    xgrid = numpy.linspace(0, shape[1], 200)
    xgrid, ygrid = numpy.meshgrid(xgrid, ygrid)

    grid = numpy.stack([ygrid.ravel(), xgrid.ravel()], axis=-1)
    zgrid = model.predict(grid).reshape(xgrid.shape)

    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(9, 5),
        tight_layout=(0, 0, 0),
    )

    cont = axis.contourf(xgrid, ygrid, zgrid, cmap=CMAP, levels=255)
    figure.colorbar(cont, ax=axis)

    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0], 0)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_aspect("equal")

    save_fig(figure, "zeta_%s.png" % suffix)


@functools.lru_cache
def load_images():
    """Return lists of input images and height values."""
    images = []
    heights = []
    shape = None
    for path in glob.glob("data_map/mask_*_*.png"):
        img = numpy.array(Image.open(path))
        if shape is None:
            shape = img.shape
        assert img.shape == shape
        assert img.dtype == numpy.bool_
        images.append(img)
        heights.append(int(path.split("_")[-2]))
    return images, heights


# utilities


def save_fig(figure, path):
    fullpath = os.path.join(PREFIX, path)
    figure.savefig(fullpath)
    pyplot.close(figure)


if __name__ == "__main__":
    main()
