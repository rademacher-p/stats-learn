# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from thesis.random import elements as rand_elements
from thesis.util import spaces

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath}")

mean = np.array([.4, .3, .3])
alpha_0 = 6
dir_prior = rand_elements.Dirichlet(mean, alpha_0)

psi = np.array([0, 1 / 3, 2 / 3])
n = 3

alpha_0_post = alpha_0 + n
mean_post = (alpha_0 * mean + n * psi) / alpha_0_post
dir_post = rand_elements.Dirichlet(mean_post, alpha_0_post)

dir_prior.space.n_plot = 150
dir_post.space.n_plot = 150

mean = np.round(np.array(mean) * 100) / 100
psi = np.round(np.array(psi) * 100) / 100
titles = [r'$\mathrm{Prior}: \alpha_\mathrm{c}(x) = ' + f"{tuple(mean)}$",
          r'$\mathrm{Posterior}: \psi_\mathrm{c}(x) = ' + f"{tuple(psi)}$"]

fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

space = spaces.check_spaces([dir_prior, dir_post])
x = space.x_plt
y = dir_prior.pf(x)
y_post = dir_post.pf(x)

y_max = np.concatenate((y, y_post)).max()

c = [y / y_max, y_post / y_max]
c = [plt.cm.viridis(i) for i in c]

# for r, ax in zip([dir_prior], [axes]):
for i, (r, ax, title) in enumerate(zip([dir_prior, dir_post], axes, titles)):
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

    # r.plot_pf(ax=ax)
    space.plot(r.pf, x, ax=ax, c=c[i])

    ax.set_title(title)

# ax[1].set_xlim(0, 1)
# ax[1].set_ylim(0, 1)
# ax[1].set_zlim(0, 1)
# ax[1].view_init(45, 45)
#
# dir_post.plot_pf(ax=ax[1])

# fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# dir_prior.plot_pf(ax=ax[0])
# dir_post.plot_pf(ax=ax[1])
