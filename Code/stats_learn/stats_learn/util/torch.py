import os
from copy import deepcopy
from functools import partial

import torch
from torch import nn
from torch.nn import functional
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from stats_learn.loss_funcs import loss_se
from stats_learn.predictors import Base

num_workers = 0
# num_workers = os.cpu_count()

pin_memory = True
# pin_memory = False


def _build_torch_net(sizes):
    layers = [nn.Flatten()]
    for in_out in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(*in_out))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-1], 1))
    return nn.Sequential(*layers)


class LitMLP(pl.LightningModule):
    def __init__(self, layer_sizes, optim_class=torch.optim.Adam, optim_params=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.model = _build_torch_net(self.layer_sizes)

        self.optim_class = optim_class
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optim_class(self.parameters(), **self.optim_params)
        return optimizer


def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class LitWrapper(Base):  # TODO: move to submodule to avoid excess imports
    def __init__(self, model, space, trainer_params=None, proc_funcs=(), name=None):
        loss_func = loss_se  # TODO: Generalize!

        super().__init__(loss_func, proc_funcs, name)
        self.model = model
        self._space = space
        self.trainer_params = trainer_params
        self._reset_trainer()

    space = property(lambda self: self._space)

    @property
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self.model, key, val)

    def _reset_trainer(self):
        self.trainer = pl.Trainer(**deepcopy(self.trainer_params))

    def reset(self):  # TODO: add reset method to predictor base class?
        self.model.apply(reset_weights)
        self._reset_trainer()

    def _reshape_batches(self, *arrays):
        shape_x = self.shape['x']
        if shape_x == ():  # cast to non-scalar shape
            shape_x = (1,)
        return tuple(map(lambda x: x.reshape(-1, *shape_x), arrays))

    def _fit(self, d, warm_start):
        if not warm_start:
            self.reset()

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
