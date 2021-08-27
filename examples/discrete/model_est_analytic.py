import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random import elements as rand_elements
from stats_learn.util import spaces

plt.style.use('../../style.mplstyle')


str_p_opt = r'$\mathrm{P}_{\mathrm{y} | \mathrm{x}, \uptheta}$'
str_p_bayes = r'$\mathrm{P}_{\mathrm{y} | \mathrm{x}, \uppsi}$'


#%% Model and prior mean
n_x = 60

n = 500
model = rand_elements.EmpiricalScalar(.7, n_x)
p_x = .5

alpha_0 = 10
alpha = rand_elements.EmpiricalScalar(.3, n_x)
alpha_x = .5


#%% Make axes, plot model
space = spaces.check_spaces([model, alpha])
y = space.x_plt
__, ax = plt.subplots()

model_pf = model.pf(y)
space.plot_xy(y, model_pf, y_std=np.zeros(y.shape), ax=ax, label=str_p_opt)


#%% Plot learner bias/var

# for n in [1, 100]:
for alpha_0 in [.1, 10]:

    n_pf = np.arange(n + 1)
    model_x_pf = rand_elements.Binomial(p_x, n).pf(n_pf)
    e_gamma = np.dot(model_x_pf, 1 / (1 + n_pf / (alpha_0 * alpha_x)))

    mean = e_gamma * alpha.pf(y) + (1 - e_gamma) * model_pf
    bias = mean - model_pf

    cov_lo = np.zeros(n_pf.shape + y.shape)
    cov_hi = np.zeros(n_pf.shape + y.shape)
    for n in n_pf:
        if n > 0:
            for i, p in enumerate(model_pf):
                emp = rand_elements.EmpiricalScalar(p, n)
                supp = emp.space.x_plt
                emp_pf = emp.pf(supp)

                idx = supp <= p
                cov_lo[n, i] = np.dot(emp_pf[idx], (supp[idx] - p)**2)
                idx = supp > p
                cov_hi[n, i] = np.dot(emp_pf[idx], (supp[idx] - p) ** 2)

        gamma = 1 / (1 + n / (alpha_0*alpha_x))
        cov_lo[n] *= (1 - gamma) ** 2
        cov_hi[n] *= (1 - gamma) ** 2

    cov_lo = np.tensordot(model_x_pf, cov_lo, axes=(0, 0))
    cov_hi = np.tensordot(model_x_pf, cov_hi, axes=(0, 0))

    cov = cov_lo + cov_hi
    err = np.sqrt((bias ** 2 + cov).sum())

    label = str_p_bayes + r', $\alpha_0={}$'.format(alpha_0)
    # label = str_p_bayes + f', $N={n}$'
    space.plot_xy(y, mean, np.sqrt(cov_lo), np.sqrt(cov_hi), ax=ax, label=label)

ax.legend()
ax.set_ylim(bottom=0.)

title = f'$N = {n}$'
# title = r'$\alpha_0 = {}$'.format(alpha_0)

ax.set(xlabel='$y$', title=title)
