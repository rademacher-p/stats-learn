import numpy as np


def make_inv_trig(shape=()):
    def sin_orig(x):
        axis = tuple(range(-len(shape), 0))
        return 1 / (2 + np.sin(2 * np.pi * x.mean(axis)))
    return sin_orig


def make_rand_discrete(n, rng):
    rng = np.random.default_rng(rng)
    _rand_vals = dict(zip(np.linspace(0, 1, n), rng.random(n)))

    def rand_discrete(x):
        return _rand_vals[x]
    return rand_discrete
