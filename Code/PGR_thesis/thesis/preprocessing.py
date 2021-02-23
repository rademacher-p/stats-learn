import numpy as np


def discretizer(vals):  # TODO: use sklearn.preprocessing.KBinsDiscretizer?
    def func(x):
        x = np.array(x)
        delta = np.abs(x - vals[:, np.newaxis])
        return vals[delta.argmin(axis=0)]

    return func


if __name__ == '__main__':
    values = np.random.default_rng().random(10)
    print(values)

    func_ = discretizer(np.linspace(0, 1, 11, endpoint=True))
    vals_discrete = func_(values)
    print(vals_discrete)
