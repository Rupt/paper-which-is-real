"""
Example using sksym to find asymmetry in a step function distribution.

"""
import os

import numpy
import lightgbm
from matplotlib import pyplot

import sksym

RNG = numpy.random.Generator(numpy.random.Philox(0xD1CE))

PREFIX = os.path.basename(__file__)[:-3]


def main():
    ndata = 5_000

    def filter_(x):
        filt = numpy.ones_like(x)
        filt[(0.1 < x) & (x < 0.3)] = 0
        filt[(0.6 < x) & (x < 0.9)] *= 0.2
        return filt

    for frac_lo in (0.0, 0.5, 0.9):
        suffix = "%dk_%.1f" % (ndata // 1000, frac_lo)
        suffix = suffix.replace(".", "p")
        example_step(suffix, ndata, frac_lo)
        example_step(suffix + "_filtered", ndata, frac_lo, filter_)

    frac_lo = 0.9
    suffix = "%dk_%.1f" % (ndata // 1000, frac_lo)
    suffix = suffix.replace(".", "p")
    example_step(suffix + "_filtered_n10", ndata, frac_lo, filter_, nfakes=10)


def example_step(suffix, ndata, frac_lo, filter_=None, nfakes=1):
    """Made data, fit a model, and output diagnostics."""
    print("suffix:", suffix, flush=True)

    data = make_filtered_data(
        ndata * 2,
        frac_lo,
        filter_,
    )

    x_train = data[:ndata]
    x_test = data[ndata:]

    rotor = sksym.WhichIsReal(filtered_uniform(filter_), nfakes)

    model = lightgbm.LGBMRegressor(
        objective=rotor.objective(),
        max_depth=1,
        random_state=RNG.integers(2 ** 31),
    )

    sksym.fit(model, rotor.pack(x_train))

    # score
    x_pack = rotor.pack(x_test)

    print("mean llr: %.3f +- %.3f" % sksym.score(model, x_pack, and_std=True))

    # figure: data
    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(5, 5),
        tight_layout=(0, 0, 0),
    )

    bins = 20
    range_ = (0, 1)
    colors = ("xkcd:orange", "xkcd:blue")
    axis.hist(
        [x_pack[1, :, 0], x_pack[0, :, 0]],
        bins,
        range_,
        histtype="step",
        color=colors,
        linewidth=2,
        label=["fake", "real"],
    )
    axis.hist(
        [x_pack[1, :, 0], x_pack[0, :, 0]],
        bins,
        range_,
        histtype="stepfilled",
        color=colors,
        linewidth=2,
        alpha=0.05,
    )

    axis.legend(loc="upper left", frameon=False)
    axis.set_xlim(0, 1)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel("count")

    save_fig(figure, "data_%s.png" % suffix)

    # figure: model output 1d
    figure, axis = pyplot.subplots(
        dpi=120,
        figsize=(5, 5),
        tight_layout=(0, 0, 0),
    )

    xgrid = numpy.linspace(0, 1, 256)
    ygrid = model.predict(xgrid.reshape(-1, 1))

    axis.plot(xgrid, ygrid, c="k", lw=1)

    axis.set_xlim(0, 1)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$\zeta(x)$")

    save_fig(figure, "zeta_%s.png" % suffix)


# filtering


def make_filtered_data(ndata, frac_lo, filter_, *, nbatch=2 ** 10):
    """Return data samples from a step function, possibly subject to filtering."""

    def make_data(n):
        return numpy.concatenate(
            [
                RNG.uniform(0.0, 1.0, nlo),
                RNG.uniform(0.8, 1.0, n - nlo),
            ]
        )

    if filter_ is None:
        nlo = RNG.binomial(ndata, frac_lo)
        data = make_data(ndata)
    else:
        shards = []
        ngot = 0

        while ngot < ndata:
            nlo = RNG.binomial(nbatch, frac_lo)
            data = make_data(nbatch)

            filt = filter_(data)
            keep = RNG.binomial(1, filt).astype(bool)
            kept = data[keep]

            shards.append(kept)
            ngot += len(kept)

        data = numpy.concatenate(shards)[:ndata]

    RNG.shuffle(data)
    return data.reshape(-1, 1)


def filtered_uniform(filter_, *, nbatch=2 ** 10):
    """Return a transform function using the given filter function."""

    if filter_ is None:
        return lambda data: RNG.uniform(0.0, 1.0, data.shape)

    def transform(data):
        ndata = len(data)
        shards = []
        ngot = 0

        while ngot < ndata:
            x = RNG.uniform(0.0, 1.0, nbatch)

            filt = filter_(x)
            keep = RNG.binomial(1, filt).astype(bool)
            kept = x[keep]

            shards.append(kept)
            ngot += len(kept)

        return numpy.concatenate(shards)[:ndata].reshape(-1, 1)

    return transform


# utilities


def save_fig(figure, path):
    fullpath = os.path.join(PREFIX, path)
    figure.savefig(fullpath)
    pyplot.close(figure)


if __name__ == "__main__":
    main()
