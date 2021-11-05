import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random import elements as rand_elements
from stats_learn.bayes.elements import NormalLinear, Dirichlet

plt.matplotlib.interactive(False)


RE_set = [
    (NormalLinear, dict(prior_mean=np.ones(2), prior_cov=10*np.eye(2), basis=[[1, 0], [0, 1], [1, 1]], cov=np.eye(3))),
    (Dirichlet, dict(prior_mean=rand_elements.Finite(['a', 'b'], [.2, .8]), alpha_0=10)),
]


def test():
    for cls, kwargs in RE_set:
        b = cls(**kwargs)

        try:
            e = b.random_model()
            d = e.rvs(5)
        except NotImplementedError:
            d = b.rvs(5)

        b.fit(d)
        b.posterior
        b.posterior_model


if __name__ == '__main__':
    test()
