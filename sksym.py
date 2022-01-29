"""Challenge symmetries with the sklearn interface."""
import math
from dataclasses import dataclass
from typing import Optional, Callable
from numbers import Integral

import numpy
import scipy


@dataclass
class WhichIsReal:
    """Package a symmetry testing setup with methods to hook into sklearn."""

    transform: Optional[Callable]
    nfakes: Integral

    def objective(self):
        return logistic_difference(self.nfakes)

    def pack(self, data):
        return pack(data, self.transform, self.nfakes)

    def stack(self, data, fakes):
        fakes = list(fakes)
        assert len(fakes) == self.nfakes, "got len(fakes) == %d, but nfakes == %d" % (
            len(fakes),
            self.nfakes,
        )
        return stack(data, fakes)


def logistic_difference(nfakes=1):
    """Return an objective for the antisymmetric logistic loss with nfakes.

    It returns first and second derivatives of the negative log likelihood.

    Since it is self-supervised, y_true labels are assumed to be
    [1]*ndata + [0]*ndata*nfakes, and ignored.

    y_pred are model predictions with shape ((1 + nfakes)*ndata,).
    The first n entries are real data; the others are fakes.
    """

    def obj(_, y_pred):
        zeta = y_pred.reshape(1 + nfakes, -1)
        phi = zeta[0] - zeta[1:]

        dxdash = scipy.special.expit(-phi)
        if nfakes > 1:
            dxdash *= 1 / nfakes
        d2xdash = dxdash * scipy.special.expit(phi)

        jac = numpy.concatenate([-dxdash.sum(axis=0), dxdash.ravel()])
        hess = numpy.concatenate([d2xdash.sum(axis=0), d2xdash.ravel()])

        return jac, hess

    return obj


def pack(data, transform, nfakes=1):
    """Return data packed with transformed fake versions.

    shape: (1 + nfakes, ndata, ndim)
    """
    fakes = [transform(data) for _ in range(nfakes)]
    return stack(data, fakes)


def stack(data, fakes):
    """Return data stacked with fakes.

    shape: (1 + len(fakes), ndata, ndim)
    """
    return numpy.stack([data, *fakes])


# sklearn interface


def fit(model, packed, *args, **kwargs):
    """Fit model to packed data.

    packed[0] is x, packed[1:] is sx.

    args and kwargs are forwarded to the model.fit(...).
    """
    data = packed.reshape(-1, packed.shape[-1])
    labels = numpy.zeros(data.shape[0], bool)
    return model.fit(data, labels, *args, **kwargs)


def score(model, packed, *, and_std=False):
    """Return the mean log likelihood ratio vs 50:50."""
    return score_log_proba(
        predict_log_proba(model, packed),
        and_std=and_std,
    )


def predict_proba(model, packed):
    """Return probabilities in shape (nfakes, ndata, 2).

    shape: (ndata, 2) if nfakes == 1, else (nfakes, ndata, 2)
    """
    return numpy.exp(predict_log_proba(model, packed))


def predict_log_proba(model, packed):
    """Return log probabilities at packed data.

    shape: (ndata, 2) if nfakes == 1, else (nfakes, ndata, 2)
    """
    zet = predict_zeta(model, packed)
    phi = zet[0] - zet[1:]

    if phi.shape[0] == 1:
        phi = phi.reshape(phi.shape[1:])

    return -numpy.logaddexp(0, numpy.stack([-phi, phi], axis=-1))


# utility


def score_log_proba(log_proba, *, and_std=False):
    """Return the mean log likelihood ratio vs 50:50."""
    size = log_proba.shape[-2]
    r = log_proba[..., 0] - math.log(0.5)
    if and_std:
        return r.mean(), r.std() / size ** 0.5
    return r.mean()


def predict_zeta(model, packed):
    """Return model outputs in shape (ntot, ndata)."""
    ntot, _, ndim = packed.shape
    data = packed.reshape(-1, ndim)
    return model.predict(data).reshape(ntot, -1)
