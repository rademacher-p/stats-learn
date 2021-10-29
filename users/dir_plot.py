from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from stats_learn.random import elements as rand_elements
from stats_learn import spaces

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{PhDmath}")


def plot_dirs(dirs, n_plot=None, titles=None, orient='h', same_cm=True, cm_hack=None):

    n_dirs = len(dirs)
    if titles is None:
        titles = [None for __ in range(n_dirs)]

    if orient == 'h':
        subplot_shape = (1, n_dirs)
    else:
        subplot_shape = (n_dirs, 1)
    fig, axes = plt.subplots(*subplot_shape, subplot_kw={'projection': '3d'})

    space = spaces.check_spaces(dirs)

    if n_plot is None:
        n_plot = space.n_plot
    if n_plot is not None:
        space.n_plot = n_plot

    for dir_i in dirs:
        dir_i.space.x_plt = space.simplex_grid(space.n_plot, space.shape, hull_mask=(dir_i.mean < 1 / dir_i.alpha_0))

    x_vec = [dir_i.space.x_plt for dir_i in dirs]
    y_vec = [dir_i.pf(dir_i.space.x_plt) for dir_i in dirs]
    for i in range(n_dirs):
        i_sort = np.argsort(y_vec[i])
        x_vec[i] = x_vec[i][i_sort]
        y_vec[i] = y_vec[i][i_sort]

    if same_cm:
        y_max = np.max(np.concatenate(y_vec))
        c_maps = [plt.cm.viridis(y / y_max) for y in y_vec]
    else:
        c_maps = [plt.cm.viridis(y / np.max(y)) for y in y_vec]

        if cm_hack is not None:
            warn("CM hack!")
            for idx in cm_hack:
                z = y_vec[idx]
                z /= np.max(z)
                z *= 8
                z = np.where(z <= 1, z, 1)
                c_maps[idx] = plt.cm.viridis(z)

    for dir_i, x, ax, title, c in zip(dirs, x_vec, axes, titles, c_maps):
        lims = (0, 1)
        ax.set_xlim3d(*lims)
        ax.set_ylim3d(*lims)
        ax.set_zlim3d(*lims)

        ax.set_box_aspect((1, 1, 1))

        ticks = [0, .5, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)

        # pane_color = (1.0, 1.0, 1.0, 1.0)
        pane_color = (.5, .5, .5, 1.)
        ax.w_xaxis.set_pane_color(pane_color)
        ax.w_yaxis.set_pane_color(pane_color)
        ax.w_zaxis.set_pane_color(pane_color)

        ax.view_init(45, 45)

        ax.set_xlabel(r'$\thetac(\Ycal_1; x)$')
        ax.set_ylabel(r'$\thetac(\Ycal_2; x)$')
        ax.set_zlabel(r'$\thetac(\Ycal_3; x)$')

        space.plot(dir_i.pf, x, ax=ax, c=c, s=3)

        ax.set_title(title)


def prior_post():
    mean = np.array([.4, .3, .3])
    alpha_0 = 6
    dir_prior = rand_elements.Dirichlet(mean, alpha_0)

    psi = np.array([0, 1 / 3, 2 / 3])
    n = 3

    alpha_0_post = alpha_0 + n
    mean_post = (alpha_0 * mean + n * psi) / alpha_0_post
    dir_post = rand_elements.Dirichlet(mean_post, alpha_0_post)

    dirs = [dir_prior, dir_post]

    titles = [r'$\text{Prior}: \alphac(x) = ' + f'{tuple(np.round(np.array(mean) * 100) / 100)}$',
              r'$\text{Posterior}: \psic(x) = ' + f'{tuple(np.round(np.array(psi) * 100) / 100)}$']

    plot_dirs(dirs, n_plot=150, titles=titles, orient='h', same_cm=True)


def localization():
    mean = np.array(np.ones(3) / 3)
    alpha_0_vec = [1e2, 1e-2]
    dirs = [rand_elements.Dirichlet(mean, alpha_0) for alpha_0 in alpha_0_vec]

    # titles = [r'$\alpha_0 \alpha_\mathrm{m}(x) \to 0$',
    #           r'$\alpha_0 \alpha_\mathrm{m}(x) \to \infty$']
    titles = [r'$\alpha_0 \alpham(x) \to \infty$',
              r'$\alpha_0 \alpham(x) \to 0$']
    # titles = [r'$\alpha_0 \alpha_\mathrm{m}(x) = ' + f'{alpha_0}$' for alpha_0 in alpha_0_vec]

    plot_dirs(dirs, n_plot=150, titles=titles, orient='v', same_cm=False, cm_hack=[1])


if __name__ == '__main__':
    # prior_post()
    localization()


# ax[1].set_xlim(0, 1)
# ax[1].set_ylim(0, 1)
# ax[1].set_zlim(0, 1)
# ax[1].view_init(45, 45)
#
# dir_post.plot_pf(ax=ax[1])

# fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# dir_prior.plot_pf(ax=ax[0])
# dir_post.plot_pf(ax=ax[1])
