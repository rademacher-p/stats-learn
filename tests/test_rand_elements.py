import numpy as np

from stats_learn.random.elements import BaseRV, Deterministic, Finite, Beta, Dirichlet, Empirical, DirichletEmpirical, \
    DirichletEmpiricalScalar, Normal, NormalLinear, DataEmpirical, Mixture
from stats_learn import spaces

rng = np.random.default_rng()

RE = [
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
    (Mixture, {'dists': [Normal(), DataEmpirical.from_data(Normal().rvs(10))], 'weights': [2, 10]})
]


def test_re():
    for cls, kwargs in RE:
        re = cls(**kwargs)

        re.mode
        rvs = re.rvs(5)
        re.pf(rvs)

        if isinstance(re, BaseRV):
            re.mean
            re.cov

        # TODO: check PF sum to approx unity. Use `space` integration method!
