import numpy as np
from scipy.stats._multivariate import multi_rv_generic
from scipy.special import gammaln, xlogy
import matplotlib.pyplot as plt
from util.util import check_data_shape, check_valid_pmf, outer_gen, diag_gen, simplex_grid, simplex_round



class BaseRP(multi_rv_generic):
    """
    Base class for generic random process objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)      # may be None or int for legacy numpy rng

    @property
    def mode(self):
        return self._mode

    def _mode(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def rps(self, size=None, random_state=None):
        random_state = self._get_random_state(random_state)

        if size is None:
            return self._rps(1, random_state)[0]
        else:
            return self._rps(size, random_state)

    def _rps(self, size=None, random_state=None):
        raise NotImplementedError("Method must be overwritten.")
        pass


class BaseRPV(BaseRP):
    """
    Base class for generic random process (numeric) objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def _mean(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def _cov(self, x_i, x_j):
        raise NotImplementedError("Method must be overwritten.")
        pass


class DiscreteRP(BaseRP):
    """
    Base class for discrete random process objects.
    """

    # def pmf(self, y):
    #     y, set_shape = check_data_shape(y, self._data_shape)
    #     return self._pmf(y).reshape(set_shape)

    def pmf(self, y):
        _out = []
        # for y_i in y.reshape((-1,) + self._data_shape):
        for y_i in y.flatten():
            _out.append(self._pmf_single(y_i))
        # return np.asarray(_out)         # returned array may be flattened over 'set_shape'
        return np.asarray(_out).reshape(y.shape)

    def _pmf_single(self, y):
        raise NotImplementedError("Method must be overwritten.")
        pass


class DiscreteRPV(DiscreteRP, BaseRPV):
    """
    Base class for discrete random variable (numeric) objects.
    """


class ContinuousRPV(BaseRPV):
    """
    Base class for continuous random element objects.
    """

    # def pdf(self, y):
    #     y, set_shape = check_data_shape(y, self._data_shape)
    #     return self._pdf(y).reshape(set_shape)

    def pdf(self, y):
        _out = []
        # for y_i in y.reshape((-1,) + self._data_shape):
        for y_i in y.flatten():
            _out.append(self._pdf_single(y_i))
        # return np.asarray(_out)     # returned array may be flattened
        return np.asarray(_out).reshape(y.shape)

    def _pdf_single(self, y):
        raise NotImplementedError("Method must be overwritten.")
        pass





def _dirichlet_check_alpha_0(alpha_0):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar.")
    return alpha_0


def _dirichlet_check_input(x, alpha_0, mean):
    x = check_valid_pmf(x, data_shape=mean.shape)

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its mean is less than 1 / alpha_0.")

    return x


class DirichletRPV(ContinuousRPV):
    """
    Dirichlet random process, finite-domain realizations.
    """

    def __init__(self, alpha_0, mean, rng=None):
        super().__init__(rng)
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Input properties
    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        if np.min(self._mean) > 1 / self._alpha_0:
            self._mode = (self._mean - 1 / self._alpha_0) / (1 - self._data_size / self._alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / (self._alpha_0 + 1)

        self._log_pdf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

    def _rvs(self, size=(), random_state=None):
        return random_state.dirichlet(self._alpha_0 * self._mean.flatten(), size).reshape(size + self._data_shape)

    def _pdf(self, x):
        x = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pdf = self._log_pdf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self._data_size), -1)
        return np.exp(log_pdf)

    def plot_pdf(self, n_plt, ax=None):

        if self._data_size in (2, 3):
            x_plt = simplex_grid(n_plt, self._data_shape, hull_mask=(self.mean < 1 / self.alpha_0))
            pdf_plt = self.pdf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            # pdf_plt.sum() / (n_plt ** (self._data_size - 1))

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pdf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pdf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')





def _empirical_check_n(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input 'n' must be a positive integer.")
    return n


def _empirical_check_input(x, n, mean):
    x = check_valid_pmf(x, data_shape=mean.shape)

    # if ((n * x) % 1 > 0).any():
    if (np.minimum((n * x) % 1, (-n * x) % 1) > 1e-9).any():
        raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

    return x


class EmpiricalRPV(DiscreteRPV):
    """
    Empirical random process, finite-domain realizations.
    """

    def __init__(self, n, mean, rng=None):
        super().__init__(rng)
        self._n = _empirical_check_n(n)
        # self._mean = check_valid_pmf(mean)
        self._mean = mean
        self._update_attr()

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = _empirical_check_n(n)
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        # self._mean = check_valid_pmf(mean)
        self._mean = mean
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        self._mode = ((self._n * self._mean) // 1) + simplex_round((self._n * self._mean) % 1)

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / self._n

        self._log_pmf_coef = gammaln(self._n + 1)

    def _rps(self, size=None, random_state=None):
        _out = []
        for _ in range(size):
            d = self._mean.rvs(self._n)
            def
            _out.append()


        return random_state.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._data_shape) / self._n

    def _pmf(self, x):
        x = _empirical_check_input(x, self._n, self._mean)

        log_pmf = self._log_pmf_coef + (xlogy(self._n * x, self._mean)
                                        - gammaln(self._n * x + 1)).reshape(-1, self._data_size).sum(axis=-1)
        return np.exp(log_pmf)