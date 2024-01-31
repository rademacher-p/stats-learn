import numpy as np

from stats_learn.random.elements import (
    Beta,
    DirichletEmpirical,
    DirichletEmpiricalScalar,
    Empirical,
    FiniteGeneric,
    Normal,
)

rng = np.random.default_rng()

RV_set = [
    (FiniteGeneric, dict(values=rng.random(5), p=None)),
    (FiniteGeneric, {"values": rng.random((3, 3, 2)), "p": np.full((3, 3), 1 / 9)}),
    (Empirical, {"mean": np.full((3,), 1 / 3), "n": 10}),
    (DirichletEmpirical, {"mean": np.full((3,), 1 / 3), "alpha_0": 10, "n": 10}),
    (DirichletEmpiricalScalar, {"mean": 0.8, "alpha_0": 5, "n": 10}),
    (Normal, dict(mean=1, cov=1)),
    (Normal, dict(mean=-3, cov=5)),
    (Normal, dict(mean=[-1, 2], cov=3)),
    (Beta, dict(a=1, b=1)),
    (Beta, dict(a=10, b=10)),
    #     Mixture,
    #     {
    #         "dists": [Normal(), Normal(5)],  # TODO: need to update lims for approx.
    #         "weights": [2, 10],
    #     },
    # ),
]


def test_expectation():
    for cls, kwargs in RV_set:
        rv = cls(**kwargs)
        mean_approx = rv.expectation(lambda x: x)
        mean_analytic = rv.mean
        assert np.allclose(mean_approx, mean_analytic, rtol=1e-3)


def test_argmax():
    for cls, kwargs in RV_set:
        rv = cls(**kwargs)
        mode_approx = rv.space.argmax(rv.prob)
        mode_analytic = rv.mode
        if mode_analytic is None:
            continue
        assert np.allclose(rv.prob(mode_approx), rv.prob(mode_analytic), rtol=1e-3)


if __name__ == "__main__":
    test_expectation()
    test_argmax()
