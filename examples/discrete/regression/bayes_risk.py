import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import BayesRegressor
from stats_learn.random import elements as rand_elements, models as rand_models


def main(log_path, img_dir, seed):
    # %% Model and optimal predictor
    n_x = n_y = 128
    var_y_x_const = 1 / 5
    w_model = [.5]

    model_x = rand_elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

    alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
    prior_mean = rand_models.DataConditional.from_mean_poly_emp(alpha_y_x, n_x, w_model, model_x)
    model = bayes_models.Dirichlet(prior_mean, alpha_0=4e2)

    # %% Dirichlet Learner
    dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)
    dir_predictor = BayesRegressor(dir_model, name=r'$\mathrm{Dir}$')

    # %% Results
    n_test = 100
    n_mc = 1000

    # Bayes Squared-Error vs. N
    n_train = np.arange(0, 4050, 50)
    dir_params = {'alpha_0': [40, 400, 4000]}

    img_path = None if img_dir is None else img_dir + 'risk_bayes_N_leg_a0.png'
    dir_predictor.assess(model, dir_params, n_train, n_test, n_mc, verbose=True, plot_loss=True, print_loss=False,
                         log_path=log_path, img_path=img_path, rng=seed)

    # Bayes Squared-Error vs. prior localization alpha_0
    n_train = [0, 100, 200, 400, 800]
    dir_params = {'alpha_0': np.sort(np.concatenate((np.logspace(-0., 5., 60), [model.alpha_0])))}

    img_path = None if img_dir is None else img_dir + 'risk_bayes_a0_leg_N.png'
    dir_predictor.assess(model, dir_params, n_train, n_test, n_mc, verbose=True, plot_loss=True, print_loss=False,
                         log_path=log_path, img_path=img_path, rng=seed)

    plt.gca().set_xscale('log')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Consistent regressor example on a discrete domain')
    parser.add_argument('-l', '--log_path', type=str, default=None, help='Path to log file')
    parser.add_argument('-i', '--save_img', action="store_true", help='Save images to log')
    parser.add_argument('--style', type=str, default=None, help='Matplotlib style')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed')
    args = parser.parse_args()

    log_path_ = args.log_path
    if log_path_ is not None and args.save_img:
        log_path_ = Path(log_path_)
        from stats_learn.util import get_now
        img_dir_ = str(log_path_.parent / f"images/{get_now()}/")
    else:
        img_dir_ = None

    if args.style is not None:
        plt.style.use(args.style)

    main(log_path_, img_dir_, args.seed)
