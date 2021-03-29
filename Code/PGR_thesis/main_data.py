from pathlib import Path
import pickle
from time import strftime

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, plot_risk_eval_sim_compare, plot_predict_stats_compare,
                               risk_eval_sim_compare, plot_risk_eval_comp_compare, plot_risk_disc)
from thesis.random import elements as rand_elements, models as rand_models
from thesis.preprocessing import discretizer, prob_disc
from thesis.util.base import vectorize_func
from thesis.util.plotting import box_grid
from thesis.util import spaces


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format


#%%

# TODO: data normalization for inverse stability, easy discretization?

model = rand_models.DataSet.from_csv('data/CCPP/data_1.csv', 'PE', iter_mode='repeat', shuffle_mode='repeat')


w_prior = np.zeros(3)
model_x = rand_elements.Normal(model.data['x'].mean(0))
# model_x = spaces.Euclidean(model.shape['x'])
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=100,
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=model_x), name=r'$\mathcal{N}$')


n_train = np.arange(10, 410, 10)


temp = [
    # (opt_predictor, None),
    # (dir_predictor, dir_params),
    # *(zip(dir_predictors, dir_params_full)),
    (norm_predictor, None),
]
predictors, params = list(zip(*temp))

plot_risk_eval_sim_compare(predictors, model, params, n_train, n_test=1000, n_mc=50, verbose=True, ax=None)


#%% Save image and Figure
time_str = strftime('%Y-%m-%d_%H-%M-%S')
image_path = Path('./images/temp/')

fig = plt.gcf()
fig.savefig(image_path.joinpath(f"{time_str}.png"))
with open(image_path.joinpath(f"{time_str}.mpl"), 'wb') as fid:
    pickle.dump(fig, fid)

print('Done')


#%%

# df = pd.read_csv('data/CCPP/data_1.csv')
# df_x = df.copy()
# df_y = df_x.pop('PE')
# d_x, d_y = df_x.to_numpy(), df_y.to_numpy()
#
# df_x.hist()
# __, ax = plt.subplots()
# df_y.hist(ax=ax)


# ds_1 = np.loadtxt('data/CCPP/data_1.csv', delimiter=',')
# ds_1 = np.genfromtxt('data/CCPP/data_1.csv', delimiter=',')
# ds_1 = pd.read_csv('data/CCPP/data_1.csv').to_numpy()

# ds_2 = pd.read_csv('data/CCPP/data_2.csv').to_numpy()

# for samp in ds_1:
#     assert samp in ds_2
# for samp in ds_2:
#     assert samp in ds_1
# assert np.all(np.sort(ds_1[:, 0]) == np.sort(ds_2[:, 0]))

