"""Learning predictors using PyTorch networks."""

from copy import deepcopy
from functools import partial

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader

from stats_learn.loss_funcs import loss_se
from stats_learn.predictors.base import Base

num_workers = 0
# num_workers = os.cpu_count()

pin_memory = True
# pin_memory = False


def _build_mlp(layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None):
    """
    PyTorch-Lightning sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
    start_layer : nn.Module, optional
    end_layer : nn.Module, optional

    Returns
    -------
    nn.Sequential

    """
    layers = []
    if start_layer is not None:
        layers.append(start_layer)
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if i < len(layer_sizes) - 2:
            layers.append(activation)
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


class LitMLP(pl.LightningModule):
    """
    PyTorch-Lightning sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
    start_layer : nn.Module, optional
    end_layer : nn.Module, optional
    loss_func : callable, optional
        The loss function for network training.
    optim_cls : class, optional
        The optimizer class.
    optim_params : dict, optional
        Keyword arguments for optimizer instantiation.

    """
    def __init__(self, layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None,
                 loss_func=functional.mse_loss, optim_cls=torch.optim.Adam, optim_params=None):
        super().__init__()

        self.model = _build_mlp(layer_sizes, activation, start_layer, end_layer)
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
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optim_cls(self.parameters(), **self.optim_params)


def reset_weights(model):
    """Reset weights of PyTorch module."""
    if hasattr(model, 'reset_parameters'):
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
    reset_func : callable, optional
        Function that calls `model` and resets to unfit state.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """
    def __init__(self, model, space, trainer_params=None, reset_func=None, proc_funcs=(), name=None):
        loss_func = loss_se  # TODO: Generalize!
        super().__init__(loss_func, space, proc_funcs, name)

        self.model = model
        self.trainer_params = trainer_params

        if reset_func is None:
            self.reset_func = lambda model_: model_.apply(reset_weights)
        elif callable(reset_func):
            self.reset_func = reset_func
        else:
            raise TypeError("Reset function must be a callable for application to `nn.Module.apply`.")

        self.can_warm_start = False  # TODO: actually can, but `assess` results are better with full datasets!

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

    def _reshape_batches(self, *arrays):
        shape_x = self.shape['x']
        if shape_x == ():  # cast to non-scalar shape
            shape_x = (1,)
        return tuple(map(lambda x: x.reshape(-1, *shape_x), arrays))

    def _fit(self, d):
        x, y = self._reshape_batches(d['x'], d['y'])
        x, y = map(partial(torch.tensor, dtype=torch.float32), (x, y))
        ds = TensorDataset(x, y)

        batch_size = len(x)  # TODO: no mini-batching! Allow user specification.

        dl = DataLoader(ds, batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        self.trainer.fit(self.model, dl)

    def _predict(self, x):
        x, = self._reshape_batches(x)
        x = torch.tensor(x, requires_grad=False, dtype=torch.float32)
        y_hat = self.model(x).detach().numpy()
        y_hat = y_hat.reshape(-1, *self.shape['y'])
        return y_hat
