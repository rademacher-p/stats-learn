from typing import Optional


import torch
from torch import nn
from torch.nn import functional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def _build_torch_net(sizes):
    layers = [nn.Flatten()]
    # for in_out in zip([math.prod(shape_x), *sizes[:-1]], sizes):
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
