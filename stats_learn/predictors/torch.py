from copy import deepcopy
from functools import partial

import torch
from torch import nn
from torch.nn import functional
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from stats_learn.loss_funcs import loss_se
from stats_learn.predictors.base import Base

NUM_WORKERS = 0
# NUM_WORKERS = os.cpu_count()

PIN_MEMORY = True
# PIN_MEMORY = False


def _build_torch_net(layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None):
    layers = []
    if start_layer is not None:
        layers.append(start_layer)
    for in_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(*in_out))
        layers.append(activation)
    layers.pop()
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


class LitMLP(pl.LightningModule):
    def __init__(self, layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None,
                 loss_func=functional.mse_loss, optim_cls=torch.optim.Adam, optim_params=None):
        super().__init__()

        self.model = _build_torch_net(layer_sizes, activation, start_layer, end_layer)
        self.loss_func = loss_func
        self.optim_cls = optim_cls
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

    def forward(self, x):
        y = self.model(x)
        # low = torch.tensor(0., dtype=torch.float32).to(y.device)
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
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class LitPredictor(Base):
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
        for key, val in kwargs.items():
            setattr(self.model, key, val)

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

        dl = DataLoader(ds, batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        self.trainer.fit(self.model, dl)

    def _predict(self, x):
        x, = self._reshape_batches(x)
        x = torch.tensor(x, requires_grad=False, dtype=torch.float32)
        y_hat = self.model(x).detach().numpy()
        y_hat = y_hat.reshape(-1, *self.shape['y'])
        return y_hat
