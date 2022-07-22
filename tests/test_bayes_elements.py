import numpy as np

from stats_learn import random
from stats_learn.bayes.elements import Dirichlet, NormalLinear

RE_set = [
    (
        NormalLinear,
        dict(
            prior_mean=np.ones(2),
            prior_cov=10 * np.eye(2),
            basis=[[1, 0], [0, 1], [1, 1]],
            cov=np.eye(3),
        ),
    ),
    (
        Dirichlet,
        dict(prior_mean=random.elements.FiniteGeneric(["a", "b"], [0.2, 0.8]), alpha_0=10),
    ),
]


def test():
    for cls, kwargs in RE_set:
        b = cls(**kwargs)

        try:
            e = b.random_model()
            d = e.sample(5)
        except NotImplementedError:
            d = b.sample(5)

        b.fit(d)
        b.posterior
        b.posterior_model


if __name__ == "__main__":
    test()
