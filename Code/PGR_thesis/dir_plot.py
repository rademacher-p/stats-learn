import numpy as np
import matplotlib.pyplot as plt

from thesis.random import elements as rand_elements, models as rand_models
from thesis.bayes import models as bayes_models

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath}")


mean = np.array([.4, .3, .3])
alpha_0 = 6
dir = rand_elements.Dirichlet(mean, alpha_0)

psi = np.array([0, 1/3, 2/3])
n = 3

alpha_0_post = alpha_0 + n
mean_post = (alpha_0 * mean + n * psi) / alpha_0_post
dir_post = rand_elements.Dirichlet(mean_post, alpha_0_post)


dir.space.n_plot = 100
dir_post.space.n_plot = 100

mean = np.round(np.array(mean) * 100) / 100
psi = np.round(np.array(psi) * 100) / 100
titles = [r'$\mathrm{Prior}: \alpha_\mathrm{c}(x) = ' + f"{tuple(mean)}$",
          r'$\mathrm{Posterior}: \psi_\mathrm{c}(x) = ' + f"{tuple(psi)}$"]

fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# for r, ax in zip([dir], [axes]):
for r, ax, title in zip([dir, dir_post], axes, titles):
    lims = (0, 1)
    ax.set_xlim3d(*lims)
    ax.set_ylim3d(*lims)
    ax.set_zlim3d(*lims)

    ax.set_box_aspect((1, 1, 1))

    # ticks = []
    ticks = [0, .5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    pane_color = (1.0, 1.0, 1.0, 1.0)
    ax.w_xaxis.set_pane_color(pane_color)
    ax.w_yaxis.set_pane_color(pane_color)
    ax.w_zaxis.set_pane_color(pane_color)

    ax.view_init(45, 45)

    ax.set_xlabel(r'$\theta_\mathrm{c}(\mathcal{Y}_1; x)$')
    ax.set_ylabel(r'$\theta_\mathrm{c}(\mathcal{Y}_2; x)$')
    ax.set_zlabel(r'$\theta_\mathrm{c}(\mathcal{Y}_3; x)$')

    r.plot_pf(ax=ax)

    ax.set_title(title)

# ax[1].set_xlim(0, 1)
# ax[1].set_ylim(0, 1)
# ax[1].set_zlim(0, 1)
# ax[1].view_init(45, 45)
#
# dir_post.plot_pf(ax=ax[1])

# fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# dir.plot_pf(ax=ax[0])
# dir_post.plot_pf(ax=ax[1])

