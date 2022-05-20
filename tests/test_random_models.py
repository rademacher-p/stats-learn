import numpy as np
from matplotlib import pyplot as plt

from stats_learn import random
from stats_learn.random.models import (
    ClassConditional,
    DataConditional,
    DataEmpirical,
    MixinRVy,
    Mixture,
    NormalLinear,
)

RM_set = [
    (
        DataConditional,
        {
            "dists": [random.elements.Normal(mean) for mean in [0, 1]],
            "model_x": random.elements.FiniteGeneric([0, 1]),
        },
    ),
    (
        ClassConditional,
        {
            "dists": [random.elements.Normal(mean) for mean in [0, 1]],
            "model_y": random.elements.FiniteGeneric(["a", "b"]),
        },
    ),
    (
        NormalLinear,
        dict(
            basis_y_x=(
                lambda x: np.ones_like(x),
                lambda x: x**2,
            ),
            weights=(1, 2),
            cov_y_x=0.01,
            model_x=random.elements.Normal(4),
        ),
    ),
    (
        NormalLinear,
        dict(weights=(1, 2), cov_y_x=0.01, model_x=random.elements.Normal([0, 0])),
    ),
    (DataEmpirical.from_data, {"d": NormalLinear().sample(10)}),
    (
        Mixture,
        {
            "dists": [
                NormalLinear(basis_y_x=(lambda x: x,), weights=(w,), cov_y_x=10) for w in [0, 4]
            ],
            "weights": [5, 8],
        },
    ),
]


def test():
    for cls, kwargs in RM_set:
        m = cls(**kwargs)

        m.mode_x

        d = m.sample(5)
        x = d["x"]
        m.mode_y_x(x)
        # m.plot_mode_y_x()

        if isinstance(m, MixinRVy):
            m.mean_y_x(x)
            m.cov_y_x(x)


if __name__ == "__main__":
    test()
