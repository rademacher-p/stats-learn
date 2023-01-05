"""Learning predictors using PyTorch networks."""

from copy import deepcopy
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from stats_learn.loss_funcs import loss_se
from stats_learn.predictors.base import Base


def build_mlp(layer_sizes, activation=nn.ReLU, last_act=False):
    """
    PyTorch sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
        The activation function.
    last_act : bool, optional
        Include final activation function.

    Returns
    -------
    nn.Sequential

    """
    layers = []
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if last_act or i < len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class LitMLP(pl.LightningModule):
    """
    PyTorch-Lightning sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
        The activation function.
    loss_func : callable, optional
        The loss function for network training.
    optim_cls : class, optional
        The optimizer class.
    optim_params : dict, optional
        Keyword arguments for optimizer instantiation.

    """

    def __init__(
        self,
        layer_sizes,
        activation=nn.ReLU,
        loss_func=functional.mse_loss,
        optim_cls=torch.optim.Adam,
        optim_params=None,
    ):
        super().__init__()

        self.model = build_mlp(layer_sizes, activation)
        self.loss_func = loss_func
        self.optim_cls = optim_cls
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

    def forward(self, x):
        y = self.model(x)
        # low = torch.tensor(0., dtype=torch.float32).to(y.device)  # TODO: delete clipping?
        # high = torch.tensor(1., dtype=torch.float32).to(y.device)
        # y = torch.where(y < 0., low, y)
        # y = torch.where(y > 1., high, y)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optim_cls(self.parameters(), **self.optim_params)


def reset_weights(model):
    """Reset weights of PyTorch module."""
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()


class LitPredictor(Base):
    r"""
    Regressor using PyTorch module.

    Parameters
    ----------
    model : pl.LightningModule
        The PyTorch-Lightning module used for prediction.
    space : dict, optional
        The domain for :math:`\xrm` and :math:`\yrm`. Defaults to the model's space.
    trainer_params : dict, optional
        Keyword arguments for `pl.Trainer` instantiation.
    dl_kwargs : dict, optional
        Keyword arguments for `DataLoader` instantiation.
    reset_func : callable, optional
        Function that calls `model` and resets to unfit state.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(
        self,
        model,
        space,
        trainer_params=None,
        dl_kwargs=None,
        reset_func=None,
        proc_funcs=(),
        name=None,
    ):
        loss_func = loss_se  # TODO: Generalize!
        super().__init__(loss_func, space, proc_funcs, name)

        self.model = model
        self.trainer_params = trainer_params
        if dl_kwargs is None:
            dl_kwargs = {}
        self.dl_kwargs = dl_kwargs

        if reset_func is None:
            self.reset_func = lambda model_: model_.apply(reset_weights)
        elif callable(reset_func):
            self.reset_func = reset_func
        else:
            raise TypeError(
                "Reset function must be a callable for application to `nn.Module.apply`."
            )

        self.can_warm_start = False
        # TODO: actually can, but `assess` results are better with full datasets!

        self.reset()

    @property
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.model, key, value)

    def _reset_trainer(self):
        self.trainer = pl.Trainer(**deepcopy(self.trainer_params))

    def reset(self):
        self.reset_func(self.model)
        self._reset_trainer()

    @staticmethod
    def _unscalar(x):
        return x[..., np.newaxis] if x.ndim == 1 else x

    def _fit(self, d):
        x, y = map(self._unscalar, (d["x"], d["y"]))
        x, y = map(partial(torch.tensor, dtype=torch.float32), (x, y))
        ds = TensorDataset(x, y)

        batch_size = len(x)  # TODO: no mini-batching! Allow user specification.

        dl = DataLoader(
            ds,
            batch_size,
            shuffle=True,
            **self.dl_kwargs,
        )

        self.trainer.fit(self.model, dl)

    def _predict(self, x):
        x = self._unscalar(x)
        x = torch.tensor(x, requires_grad=False, dtype=torch.float32)
        y_hat = self.model(x).detach().numpy()
        y_hat = y_hat.reshape(-1, *self.shape["y"])
        return y_hat
