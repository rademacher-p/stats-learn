"""
Loss functions.
"""


def loss_se(y_est, y):
    return sum((y_est - y)**2)


def loss_01(y_hyp, y):
    return 1 - (y_hyp == y)
