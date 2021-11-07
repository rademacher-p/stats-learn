import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random.elements import BaseRV, Deterministic, Finite, Beta, Dirichlet, Empirical, DirichletEmpirical, \
    DirichletEmpiricalScalar, Normal, NormalLinear, DataEmpirical, Mixture
from stats_learn import spaces

plt.matplotlib.interactive(False)

rng = np.random.default_rng()


RE_set = [
    (Deterministic, {'val': np.arange(6).reshape(3, 2)}),
    (Deterministic, {'val': ['a', 'b', 'c']}),
    (Finite, {'supp': rng.random((3, 3, 2)), 'p': np.full((3, 3), 1/9)}),
    (Finite, {'supp': ['a', 'b', 'c']}),
    (Dirichlet, {'mean': np.full((3,), 1/3), 'alpha_0': 10}),
    (Empirical, {'mean': np.full((3,), 1/3), 'n': 10}),
    (DirichletEmpirical, {'mean': np.full((3,), 1/3), 'alpha_0': 10, 'n': 10}),
    (DirichletEmpiricalScalar, {'mean': .8, 'alpha_0': 5, 'n': 10}),
    (Normal, {'mean': 1, 'cov': 1}),
    (Normal, {'mean': [1, 1], 'cov': np.eye(2)}),
    (NormalLinear, {'weights': np.ones(2), 'basis': [[1, 0], [0, 1], [1, 1]], 'cov': np.eye(3)}),
    (DataEmpirical, {'values': ['a', 'b'], 'counts': [5, 10], 'space':spaces.FiniteGeneric(['a', 'b'])}),
    (Mixture, {'dists': [Normal(), DataEmpirical.from_data(Normal().sample(10))], 'weights': [2, 10]})
]


def test():
    for cls, kwargs in RE_set:
        e = cls(**kwargs)

        e.mode
        x = e.sample(5)
        e.pf(x)
        # e.plot_pf(x)

        if isinstance(e, BaseRV):
            e.mean
            e.cov

        # TODO: check PF sum to approx unity. Use `space` integration method!


if __name__ == '__main__':
    test()
