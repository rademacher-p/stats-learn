import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random import elements as rand_elements
from stats_learn import spaces

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{PhDmath}")

# n_x = 9
# model = rand_elements.EmpiricalScalar(.75, n_x)
# alpha = rand_elements.EmpiricalScalar(.25, n_x)

# Model and prior mean
n_x = 60
model = rand_elements.EmpiricalScalar(0.7, n_x)
alpha = rand_elements.EmpiricalScalar(0.3, n_x)

n = 1
alpha_0 = 10

#%% Make axes, plot model
space = spaces.check_spaces([model, alpha])
y = space.x_plt
__, ax = plt.subplots()

model_pf = model.pf(y)
space.plot_xy(
    y, model_pf, y_std=np.zeros(y.shape), ax=ax, label=r"$\Prm_{\yrm | \xrm, \uptheta}$"
)

#%% Plot learner bias/var

# for n in [1, 100]:
for alpha_0 in [0.1, 10]:

    gamma = 1 / (1 + n / alpha_0)
    mean = gamma * alpha.pf(y) + (1 - gamma) * model_pf
    cov = n / (alpha_0 + n) ** 2 * model_pf * (1 - model_pf)
    bias = mean - model_pf

    # cov_lo = cov_hi = cov

    cov_lo = np.zeros(y.shape)
    if n > 0:
        for i, p in enumerate(model_pf):
            emp = rand_elements.EmpiricalScalar(p, n)
            supp = emp.space.x_plt
            emp_pf = emp.pf(supp)

            idx = supp <= p
            cov_lo[i] = np.dot(emp_pf[idx], (supp[idx] - p) ** 2)

    cov_lo *= (1 - gamma) ** 2
    cov_hi = cov - cov_lo
    # cov_hi = cov_lo

    err = np.sqrt((bias**2 + cov).sum())

    # label = r'$\mathrm{P}_{\mathrm{y} | \mathrm{x}, \uppsi}$'
    # label = r'$\mathrm{P}_{\mathrm{y} | \mathrm{x}, \uppsi}$, ' + f'$N={n}$'
    label = r"$\Prm_{\yrm | \xrm, \uppsi}$, " + r"$\alpha_0={}$".format(alpha_0)
    space.plot_xy(y, mean, np.sqrt(cov_lo), np.sqrt(cov_hi), ax=ax, label=label)

ax.legend(loc="upper left")
ax.set_ylim(bottom=0.0)

# title = r'$\alpha_0 = {}$; $N={}$; '.format(alpha_0, n) + '$\mathrm{Error} = ' + f'{err:.3f}$'
# title = r'$\alpha_0 = {}$'.format(alpha_0)
title = f"$N = {n}$"
ax.set(xlabel="$y$", title=title)
