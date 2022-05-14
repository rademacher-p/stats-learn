import numpy as np

from stats_learn.preprocessing import make_clipper, make_discretizer
from stats_learn.spaces import Box

np.set_printoptions(precision=3)


def test_main():  # TODO: test with assertions
    # test discretizer
    x = np.random.default_rng().random(10)
    print(x)

    vals = np.linspace(0, 1, 11)
    func_ = make_discretizer(vals)
    x_d = func_(x)
    print(x_d)

    x = np.random.default_rng().random((10, 2))
    print(x)

    vals = Box.make_grid([[0, 1], [0, 1]], 11, True).reshape(-1, 2)
    func_ = make_discretizer(vals)
    x_d = func_(x)
    print(x_d)

    # test clipper
    # lims = np.array((0, 1))
    lims = np.array([(0, 1), (0, 1)])
    clipper = make_clipper(lims)

    x = np.random.default_rng().uniform(-1, 2, (10, *lims.shape[:-1]))
    print(x)

    x_c = clipper(x)
    print(x_c)


if __name__ == "__main__":
    test_main()
