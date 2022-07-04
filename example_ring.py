"""
Example using sksym on a cylinder.

"""
import math
import os
from math import pi

import lightgbm
import matplotlib
import numpy
from matplotlib import pyplot

import sksym

RNG = numpy.random.Generator(numpy.random.Philox(0xDADA1))

PREFIX = os.path.basename(__file__)[:-3]

CMAP = matplotlib.cm.inferno

# draw maps with dimensions (y, x) = (y, phi) to match image conventions
# y is scaled to [0, 1), phi is scaled to [0, 2pi)
# precision for sampling
NPIXEL_Y = 1024
NPIXEL_PHI = 1024
# precision for drawing
NGRID = 64
# prevision for circular lines
POINTS_PHI = 512
# precision for pyplot plot_surface
RCOUNT = 64
CCOUNT = 64
# for pretty-but-slow plot
SUPER_RCOUNT = 512
SUPER_CCOUNT = 256

# model details
BANDS = [
    (6, 7, 1),
    (8, 9, 1),
    (32, 33, 1),
    (55, 56, 1),
]

CORNER_L = 0.1

FILTERS = [
    # front back
    (0, 64, 10, 22, 1),
    (0, 64, 42, 54, 1),
    # corners
    (0, 64, 5, 9, CORNER_L),
    (0, 64, 23, 27, CORNER_L),
    (0, 64, 37, 41, CORNER_L),
    (0, 64, 55, 59, CORNER_L),
    # holes
    (50, 60, 12, 16, 0),
]

HOLE = (4, 14, 44, 48, 0)


def main():
    os.makedirs(__file__[:-3], exist_ok=True)

    ndata = 5_000

    # data sampling distribution
    dist = numpy.zeros((NPIXEL_Y, NPIXEL_PHI))

    for y0, y1, value in BANDS:
        dist[map_y(y0) : map_y(y1)] = value

    dist_no_wave = dist + 0.02

    map_wave(dist, 1, 8 * pi / 9, 0.02)

    # add background noise
    dist += 0.02

    # data filter: set which areas are non-zero
    filter_ = numpy.zeros_like(dist)

    for y0, y1, phi0, phi1, value in FILTERS:
        filter_[map_y(y0) : map_y(y1), map_phi(phi0) : map_phi(phi1)] = value

    # one filter hypothesised, one for data
    filter_true = filter_.copy()

    y0, y1, phi0, phi1, value = HOLE
    filter_true[map_y(y0) : map_y(y1), map_phi(phi0) : map_phi(phi1)] = value

    # prepare data
    data = map_sample(dist * filter_true, ndata * 2)

    x_train = data[:ndata]
    x_test = data[ndata:]

    example_ring(
        "abstract",
        x_train,
        x_test,
        filter_,
    )

    example_ring(
        "fix_hole",
        x_train,
        x_test,
        filter_true,
        draw_filters=FILTERS + [HOLE],
    )

    # fix the offset
    data = map_sample(dist_no_wave * filter_true, ndata * 2)

    x_train = data[:ndata]
    x_test = data[ndata:]

    example_ring(
        "no_wave",
        x_train,
        x_test,
        filter_true,
        draw_filters=FILTERS + [HOLE],
    )


def example_ring(
    suffix,
    x_train,
    x_test,
    filter_,
    *,
    nfakes=1,
    draw_filters=FILTERS,
):
    """Made data, fit a model, and output diagnostics."""
    print("suffix:", suffix, flush=True)

    # make data
    # prepare filtered transform; rotational symmetry means uniform in phi
    circles = filter_.cumsum(axis=1)
    circles /= circles[:, -1, numpy.newaxis]

    def transform(data):
        new = data.copy()
        # keep Z, resample phi from its circle
        phi_scale = 2 * pi / circles.shape[1]
        for i, (y, _) in enumerate(new):
            circ = circles[int(y * NPIXEL_Y)]
            new_iphi = numpy.searchsorted(circ, RNG.uniform())
            new[i, 1] = (new_iphi + RNG.uniform()) * phi_scale
        return new

    # fit model
    rotor = sksym.WhichIsReal(transform, nfakes)

    model = lightgbm.LGBMRegressor(
        objective=rotor.objective(),
        subsample=0.5,
        subsample_freq=1,
        random_state=RNG.integers(2**31),
    )

    sksym.fit(model, rotor.pack(x_train))

    # score
    x_pack = rotor.pack(x_test)

    quality, quality_std = sksym.score(model, x_pack, and_std=True)
    print("mean llr: %.3f +- %.3f" % (quality, quality_std))

    # plot
    nscat = 10_000

    # figures: data and transformed data
    def plot_data(x_data):
        figure, axis = pyplot.subplots(
            dpi=120,
            figsize=(6, 5),
            tight_layout=(0, 0, 0),
        )

        axis.scatter(
            x_data[:nscat, 1],
            x_data[:nscat, 0],
            c="k",
            s=1,
            marker=",",
            lw=0,
        )

        # filter outlines
        for y0, y1, phi0, phi1, _ in FILTERS:
            axis.plot(
                numpy.array([phi0, phi1, phi1, phi0, phi0]) * (2 * pi / NGRID),
                numpy.array([y0, y0, y1, y1, y0]) / NGRID,
                c="k",
                lw=1,
            )

        axis.set_xlim(0, 2 * pi)
        axis.set_ylim(0, 1)
        axis.set_xlabel(r"$\phi$")
        axis.set_ylabel(r"$y$")
        axis.set_aspect(2 * pi)
        return figure

    figure = plot_data(x_train)
    save_fig(figure, "data_%s.png" % suffix)

    figure = plot_data(transform(x_train))
    save_fig(figure, "transformed_%s.png" % suffix)

    # figure: mean llr contribution by orbit
    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(6, 5),
        tight_layout=(0, 0, 0),
    )

    llr = sksym.predict_log_proba(model, x_pack)[..., 0] - math.log(0.5)

    if len(llr.shape) == 2:  # => nfakes > 1
        llr = llr.mean(axis=0)

    nbins = 64
    hrange = (0, 1)
    counts, bins = numpy.histogram(x_test[:, 0], nbins, hrange)
    totals, _ = numpy.histogram(x_test[:, 0], nbins, hrange, weights=llr)
    var, _ = numpy.histogram(x_test[:, 0], nbins, hrange, weights=llr**2)

    err = var**0.5
    yields_lo = numpy.append(totals - err, 0)
    yields_hi = numpy.append(totals + err, 0)

    # total
    axis.hist(
        bins[:-1],
        nbins,
        hrange,
        weights=totals / len(llr),
        histtype="step",
        color="k",
        lw=2,
    )

    axis.fill_between(
        bins,
        yields_lo / len(llr),
        yields_hi / len(llr),
        step="post",
        alpha=0.1,
        color="k",
        lw=0,
    )

    axis.axhline(y=0, c="k", lw=axis.spines.bottom.get_linewidth())

    _, ymax = axis.get_ylim()
    axis.set_xlim(0, 1)
    axis.set_ylim(-ymax, ymax)
    axis.set_xlabel(r"$y$")
    axis.set_ylabel(r"$Q(y)$")

    save_fig(figure, "orbit_%s.png" % suffix)

    # figure: model output
    ygrid = numpy.linspace(0, 1, NPIXEL_Y)
    phigrid = numpy.linspace(0, 2 * pi, NPIXEL_PHI)
    ygrid, phigrid = numpy.meshgrid(ygrid, phigrid)

    grid = numpy.stack([ygrid.ravel(), phigrid.ravel()], axis=-1)
    zetagrid = model.predict(grid).reshape(ygrid.shape)

    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(6, 5),
        tight_layout=(0, 0, 0),
    )

    cont = axis.contourf(phigrid, ygrid, zetagrid, cmap=CMAP)
    figure.colorbar(cont, ax=axis)

    axis.set_xlim(0, 2 * pi)
    axis.set_ylim(0, 1)
    axis.set_xlabel(r"$\phi$")
    axis.set_ylabel(r"$y$")
    axis.set_aspect(2 * pi)

    save_fig(figure, "zeta_%s.png" % suffix)


# map (2d array) utilities


def map_wave(map_, rate, phase, amplitude, fill=0.0):
    """Mutate map_ by sine shifts around the phi axis.

    Set pixels from incoming edges to fill.
    """
    amp = amplitude * NPIXEL_Y
    for i in range(NPIXEL_PHI):
        angle = (i / NPIXEL_PHI) * (2 * numpy.pi)
        shift = int(math.sin(rate * (angle - phase)) * amp)
        if shift > 0:
            map_[shift:, i] = map_[:-shift, i]
            map_[:shift, i] = fill
        if shift < 0:
            map_[:shift, i] = map_[-shift:, i]
            map_[shift:, i] = fill
        # no action for zero shift


def map_sample(map_, n):
    """Return n samples from map_."""
    pdf = map_.ravel()
    pdf /= pdf.sum()

    yxi = RNG.choice(len(pdf), size=n, p=pdf).astype(float)
    y = yxi // NPIXEL_PHI
    x = yxi % NPIXEL_PHI

    # assign within pixel
    y += RNG.uniform(size=n)
    x += RNG.uniform(size=n)

    # re-scale to unit square
    y /= NPIXEL_Y
    x *= 2 * pi / NPIXEL_PHI
    return numpy.stack([y, x], axis=-1)


def map_y(iy):
    """Return the index for grid point y."""
    return (NPIXEL_Y // NGRID) * iy


def map_phi(iphi, npixel=NGRID):
    """Return the index for grid point iphi."""
    return (NPIXEL_PHI // NGRID) * iphi


# 3d plotting utilities


def ring_square(iy0, iy1, iphi0, iphi1):
    """Return x, y, z arrays for a square wrapped onto a ring."""
    y0 = iy0 / NGRID
    y1 = iy1 / NGRID
    phi0 = iphi0 * (2 * pi / NGRID)
    phi1 = iphi1 * (2 * pi / NGRID)
    delta_phi = min(abs(phi1 - phi0), 2 * pi - abs(phi1 - phi0))
    n = 2 + int(POINTS_PHI * delta_phi / (2 * pi))

    # there and back again
    phi_one_way = numpy.linspace(phi0, phi1, n)
    phi = numpy.concatenate((phi_one_way, phi_one_way[::-1], [phi0]))

    y = numpy.concatenate(([y0] * n, [y1] * n, [y0]))
    x, z = phi_to_ring(phi)
    return x, y, z


def phi_to_ring(phi):
    """Return x and z arrays for phi wrapped onto a 3d ring.

    The angular origin is at the bottom; 0 -> (0, -1)
    """
    return numpy.sin(phi), -numpy.cos(phi)


def set_axis3d_equal(axis):
    """Make a box in axis square based on its limits."""
    xlo, xhi = axis.get_xlim()
    ylo, yhi = axis.get_ylim()
    zlo, zhi = axis.get_zlim()
    return axis.set_box_aspect(
        (abs(xhi - xlo), abs(yhi - ylo), abs(zhi - zlo))
    )


# utilities


def save_fig(figure, path):
    fullpath = os.path.join(PREFIX, path)
    figure.savefig(fullpath)
    pyplot.close(figure)


if __name__ == "__main__":
    main()
