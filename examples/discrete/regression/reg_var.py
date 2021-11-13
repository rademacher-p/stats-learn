import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn import results
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.predictors.torch import LitMLP, LitPredictor, reset_weights
from stats_learn.preprocessing import make_clipper
from stats_learn.random import elements as rand_elements, models as rand_models


def main(log_path=None, img_dir=None, seed=None):
    # # Model and optimal predictor
    n_x = n_y = 32

    freq = 2

    # def clairvoyant_func(x):
    #     # y = np.sin(2*np.pi*freq*x)
    #     # y = np.where(y > 0, .75, .25)
    #     # return y
    #     return .5 + .35 * np.sin(2 * np.pi * freq * x)

    def clairvoyant_func(x):
        y = np.sin(2 * np.pi * freq * x)
        return .5 + np.where(y > 0, .3, -.3) - .3 * y

    var_y_x_const = 1 / 2

    model_x = rand_elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

    alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
    model = rand_models.DataConditional.from_mean_emp(alpha_y_x, n_y, clairvoyant_func, model_x)

    opt_predictor = ModelRegressor(model, name=r'$f^*(\theta)$')

    # # Learners

    # Dirichlet
    # def prior_func(x):
    #     # return .5 + .35*np.sin(2*np.pi*freq*x)
    #     y = np.sin(2 * np.pi * freq * x)
    #     y = np.where(y > 0, .75, .25)
    #     return y

    def prior_func(x):
        # return .5 + .35*np.sin(2*np.pi*freq*x)
        y = np.sin(2 * np.pi * freq * x)
        a = .25
        return np.where(y > 0, .5 + a, .5 - a)

    prior_mean = rand_models.DataConditional.from_mean_emp(alpha_y_x, n_y, prior_func, model_x)
    dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

    dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')

    # dir_params = {'alpha_0': [8e-5, 800]}
    # dir_params = {'alpha_0': [5e-5, 500]}
    # dir_params = {'alpha_0': [1e-5, 120]}
    # dir_params = {'alpha_0': [1e-5, 2e3]}
    # dir_params = {'alpha_0': [1e-5, 6e2]}  # 32pt, var_c=.8, a=.15 prior
    # dir_params = {'alpha_0': [1e-5, 220]}  # 32pt, var_c=.8, a=.25 prior
    dir_params = {'alpha_0': [1e-5, 125]}  # 32pt, var_c=.5, a=.25 prior

    # PyTorch
    if seed is not None:
        seed_everything(seed)

    weight_decays = [0, 3e-3]  # controls L2 regularization

    proc_funcs = {'pre': [], 'post': [make_clipper([min(model_x.values), max(model_x.values)])]}

    lit_predictors = []
    for weight_decay in weight_decays:
        layer_sizes = [500, 500, 500, 500]
        optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

        if log_path is None:
            logger = False
        else:
            logger_path = str(log_path_.parent / 'logs/')
            logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
            logger = pl_loggers.TensorBoardLogger(logger_path, name=logger_name)
        trainer_params = {
            'max_epochs': 100000,
            'callbacks': EarlyStopping('train_loss', min_delta=1e-4, patience=10000, check_on_train_epoch_end=True),
            'checkpoint_callback': False,
            # 'logger': False,
            'logger': logger,
            'weights_summary': None,
            'gpus': torch.cuda.device_count(),
        }

        lit_model = LitMLP([model.size['x'], *layer_sizes, 1], optim_params=optim_params)

        def reset_func(model_):
            model_.apply(reset_weights)
            # with torch.no_grad():
            #     model_.model[-1].bias.fill_(.5)

        lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"
        lit_predictor = LitPredictor(lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name)
        lit_predictors.append(lit_predictor)

    #
    temp = [
        (opt_predictor, None),
        (dir_predictor, dir_params),
        *((predictor, None) for predictor in lit_predictors),
    ]
    predictors, params = zip(*temp)

    # # Results
    n_test = 1000
    n_mc = 5

    # Sample regressor realizations
    n_train = 128

    d = model.sample(n_train + n_test, rng=seed)
    d_train, d_test = np.split(d, [n_train])

    img_path = None if img_dir is None else img_dir / 'fit.png'
    results.assess_single_compare(predictors, d_train, d_test, params, log_path=log_path, img_path=img_path)

    # Prediction mean/variance, comparative
    n_train = 128

    img_path = None if img_dir is None else img_dir / 'predict_full.png'
    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                           plot_stats=True, print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

    # Squared-Error vs. training data volume N
    n_train = np.insert(2**np.arange(12), 0, 0)

    img_path = None if img_dir is None else img_dir / 'risk_N.png'
    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True,
                           print_loss=True, log_path=log_path, img_path=img_path, rng=seed)

    # Squared-Error vs. prior localization alpha_0
    n_train = [0, 100, 200, 400]

    img_path = None if img_dir is None else img_dir / 'risk_a0_leg_N.png'
    dir_predictor.assess(model, {'alpha_0': np.logspace(0., 5., 60)}, n_train, n_test, n_mc, verbose=True,
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
