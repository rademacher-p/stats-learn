import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from stats_learn import results
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.random import elements as rand_elements, models as rand_models


def main(log_path=None, img_dir=None, seed=None):
    # %% Model and optimal predictor
    n_x = n_y = 128
    var_y_x_const = 1 / 5

    def clairvoyant_func(x):
        return 1 / (2 + np.sin(2 * np.pi * x))

    model_x = rand_elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

    alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
    model = rand_models.DataConditional.from_mean_emp(alpha_y_x, n_y, clairvoyant_func, model_x)

    opt_predictor = ModelRegressor(model, name=r'$f^*(\theta)$')

    # %% Learners
    w_prior = [.5, 0]

    # Dirichlet
    prior_mean = rand_models.DataConditional.from_mean_poly_emp(alpha_y_x, n_y, w_prior, model_x)
    dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

    dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')

    dir_params = {'alpha_0': [10, 1000]}

    # Normal-prior LR
    norm_model = bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=.1, cov_y_x=.1, model_x=model_x,
                                           allow_singular=True)
    norm_predictor = BayesRegressor(norm_model, space=model.space, name=r'$\mathcal{N}$')

    norm_params = {'prior_cov': [.1, .001]}

    #
    temp = [
        (opt_predictor, None),
        (dir_predictor, dir_params),
        (norm_predictor, norm_params),
    ]
    predictors, params = zip(*temp)

    # %% Results
    n_test = 1000
    n_mc = 50

    # Sample regressor realizations
    n_train = 30
    d = model.sample(n_train + n_test, rng=seed)
    d_train, d_test = np.split(d, [n_train])

    img_path = None if img_dir is None else img_dir / 'fit.png'
    results.assess_single_compare(predictors, d_train, d_test, params, log_path=log_path, img_path=img_path)

    # Prediction mean/variance, comparative
    n_train = 400

    img_path = None if img_dir is None else img_dir / 'predict_a0.png'
    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                           plot_stats=True, print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

    # Dirichlet-based prediction mean/variance, varying N
    n_train = [0, 800, 4000]

    img_path = None if img_dir is None else img_dir / 'predict_N.png'
    dir_predictor.assess(model, {'alpha_0': [1000]}, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                         plot_stats=True, print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

    # Squared-Error vs. training data volume N
    n_train = np.arange(0, 4050, 50)

    img_path = None if img_dir is None else img_dir / 'risk_N_leg_a0.png'
    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True,
                           print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

    # Squared-Error vs. prior localization alpha_0
    n_train = [0, 100, 200, 400, 800]

    img_path = None if img_dir is None else img_dir / 'risk_a0_leg_N.png'
    dir_predictor.assess(model, {'alpha_0': np.logspace(0., 5., 100)}, n_train, n_test, n_mc, verbose=True,
                         plot_loss=True, print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

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
        img_dir_ = log_path_.parent / f"images/{get_now()}"
    else:
        img_dir_ = None

    if args.style is not None:
        plt.style.use(args.style)

    main(log_path_, img_dir_, args.seed)
