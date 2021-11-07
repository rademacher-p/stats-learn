import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes.models import NormalLinear, Dirichlet

plt.matplotlib.interactive(False)


RM_set = [
    (NormalLinear, dict(prior_mean=np.zeros(1), prior_cov=np.eye(1), basis_y_x=None, cov_y_x=1.,
                        model_x=rand_elements.Normal())),
    (Dirichlet, dict(prior_mean=rand_models.NormalLinear(weights=(2,), basis_y_x=(lambda x: 1,), cov_y_x=.1,
                                                         model_x=rand_elements.Finite(np.linspace(0, 1, 10,
                                                                                                  endpoint=False))),
                     alpha_0=1)),
]


def test():
    for cls, kwargs in RM_set:
        b = cls(**kwargs)

        try:
            e = b.random_model()
            d = e.sample(5)
        except NotImplementedError:
            d = b.sample(5)

        b.fit(d)
        b.posterior
        b.posterior_model


if __name__ == '__main__':
    test()
