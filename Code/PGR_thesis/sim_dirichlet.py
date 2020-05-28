import itertools
import functools
import math

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from scipy import stats
# from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
# from mpl_toolkits.mplot3d import Axes3D

from main import learn_sim

from RE_obj import DeterministicRE, FiniteRE, DirichletRV
from SL_obj import YcXModel
from bayes import BaseBayes, FiniteDirichletBayes
from loss_functions import loss_01, loss_se
from learn_functions import DirichletClassifier, DirichletEstimator
from util.generic import empirical_pmf



supp_y = np.array(['a', 'b'])
# supp_y = np.arange(2) / 2
supp_x = np.arange(2) / 2
# supp_x = np.arange(6).reshape(3, 2)
# supp_x = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)

i_split_y, i_split_x = supp_y.ndim, supp_x.ndim - 0

supp_shape_y, data_shape_y = supp_y.shape[:i_split_y], supp_y.shape[i_split_y:]
supp_shape_x, data_shape_x = supp_x.shape[:i_split_x], supp_x.shape[i_split_x:]

supp_x_s = np.array(list(itertools.product(supp_x.reshape((-1,) + data_shape_x))),
                    dtype=[('x', supp_x.dtype, data_shape_x)]).reshape(supp_shape_x)

supp_y_s = np.array(list(itertools.product(supp_y.reshape((-1,) + data_shape_y))),
                    dtype=[('y', supp_y.dtype, data_shape_y)]).reshape(supp_shape_y)



# alpha_0 = supp_x_s.size * supp_y_s.size
alpha_0 = 1




# p_x = np.ones(supp_x_s.shape) / supp_x_s.size
# def p_y_x(x): return np.ones(supp_y_s.shape) / supp_y_s.size
# model_kwargs = {'supp_x': supp_x_s['x'], 'supp_y': supp_y_s['y'], 'p_x': p_x, 'p_y_x': p_y_x, 'rng': None}
# model_gen = YcXModel.finite_model
# bayes_model = BaseBayes(model_gen, model_kwargs)




n_train_plot = list(range(0, 6))

# alpha_0_plot = list(np.arange(0.5, 4, 0.5))
alpha_0_plot = supp_x_s.size * supp_y_s.size * np.array([.1, 1, 10])

mean = np.ones(supp_x_s.shape + supp_y_s.shape) / (supp_x_s.size * supp_y_s.size)


risk_plot = np.empty(tuple(map(len, [n_train_plot, alpha_0_plot])))
n_iter = risk_plot.size
for i, (n_train, alpha_0) in enumerate(itertools.product(n_train_plot, alpha_0_plot)):
    print(f"Simulation {i+1}/{n_iter}")

    # bayes_model.prior.alpha_0 = alpha_0
    bayes_model = FiniteDirichletBayes(supp_x_s, supp_y_s, alpha_0, mean, rng_prior=random.default_rng())

    learner = DirichletClassifier(supp_x_s, supp_y_s, alpha_0, mean)
    # learner = DirichletEstimator(supp_x_s, supp_y_s, alpha_0, mean)

    risk_plot[np.unravel_index([i], risk_plot.shape)] = learn_sim(bayes_model, learner, n_train, n_mc=1000, verbose=True)

fig, ax = plt.subplots(num='risk', clear=True)
ax.plot(n_train_plot, risk_plot)
ax.grid(True)
