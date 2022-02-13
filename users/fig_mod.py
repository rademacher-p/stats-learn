import pickle
from pathlib import Path

from matplotlib import pyplot as plt

# plt.style.use('images/style.mplstyle')
plt.style.use(['images/style.mplstyle', 'images/IEEE.mplstyle'])

plt.matplotlib.interactive(False)


def main(dirs):
    for dir_ in dirs:
        dir_ = Path(dir_)
        subdir = dir_ / 'temp'
        subdir.mkdir()
        for filepath in dir_.glob('*.mpl'):
            with filepath.open('rb') as f:
                fig = pickle.load(f)

            # Modifications

            fig.set_size_inches(*plt.rcParams['figure.figsize'])
            for ax in fig.axes:
                ax.grid(plt.rcParams['axes.grid'])

                ax.tick_params(which='both', reset=True)
                if plt.rcParams['xtick.minor.visible'] and plt.rcParams['ytick.minor.visible']:
                    ax.minorticks_on()

                ax.set_xlabel(ax.get_xlabel(), fontsize=plt.rcParams['axes.labelsize'])
                ax.set_ylabel(ax.get_ylabel(), fontsize=plt.rcParams['axes.labelsize'])
                ax.set_title(ax.get_title(), fontsize=plt.rcParams['axes.titlesize'])

                # TODO: redraw artists with new linewidths, markersizes, etc.

                handles, labels = ax.get_legend_handles_labels()
                if labels[-1] == '$D$':
                    handles = handles[-1:] + handles[:-1]
                    labels = labels[-1:] + labels[:-1]
                ax.legend(handles, labels)

            # End mods

            fig.savefig(subdir / filepath.stem)
            mpl_file = subdir / filepath.name
            with open(mpl_file, 'wb') as f:
                pickle.dump(fig, f)

        # plt.show()


def label_update(label):
    label = label.replace(r'\Dir', r'\mathrm{Dir}')
    if label == r'$f_{\Theta}(\theta)$':
        label = r'$f^*(\theta)$'

    if label == r'$f^*(\theta)$':
        label = r'$f^*(\rho)$'

    return label


if __name__ == '__main__':
    # dirs_ = [
    #     '../docs/Figures/Discrete/SE/reg_func',
    #     # '../docs/Figures/Discrete/SE/reg_var',
    #     # '../docs/Figures/Discretization/SE/reg_var',
    # ]
    # main(dirs_)

    import argparse

    parser = argparse.ArgumentParser(description='Update `matplotlib` figures')
    parser.add_argument('dirs', type=str, nargs='+', help='Directories in which to find MPL files')
    args = parser.parse_args()

    main(args.dirs)
