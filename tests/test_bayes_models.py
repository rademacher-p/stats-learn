import numpy as np
from matplotlib import pyplot as plt

from stats_learn import random
from stats_learn.bayes.models import Dirichlet, NormalLinear

RM_set = [
    (
        NormalLinear,
        dict(
            prior_mean=np.zeros(1),
            prior_cov=np.eye(1),
            basis_y_x=None,
            cov_y_x=1.0,
            model_x=random.elements.Normal(),
        ),
    ),
    (
        Dirichlet,
        dict(
            prior_mean=random.models.NormalLinear(
                weights=(2,),
                basis_y_x=(lambda x: 1,),
                cov_y_x=0.1,
                model_x=random.elements.FiniteGeneric(np.linspace(0, 1, 10, endpoint=False)),
            ),
            alpha_0=1,
        ),
    ),
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


if __name__ == "__main__":
    test()
