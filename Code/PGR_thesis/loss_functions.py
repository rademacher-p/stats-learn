"""
Loss functions.
"""
import numpy as np


def loss_se(y_est, y):
    y_est, y = np.asarray(y_est), np.asarray(y)
    return ((y_est - y)**2).sum()


def loss_01(y_hyp, y):
    return 1 - (y_hyp == y)
