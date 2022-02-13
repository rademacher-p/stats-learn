import pickle
from pathlib import Path

from matplotlib import pyplot as plt

# plt.style.use('images/style.mplstyle')
plt.style.use(['images/style.mplstyle', 'images/double.mplstyle'])

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

                plt.setp(ax.spines.values(), linewidth=plt.rcParams['axes.linewidth'])

                ax.tick_params(which='both', reset=True)
                if plt.rcParams['xtick.minor.visible'] and plt.rcParams['ytick.minor.visible']:
                    ax.minorticks_on()

                ax.set_xlabel(ax.get_xlabel(), fontsize=plt.rcParams['axes.labelsize'])
                ax.set_ylabel(ax.get_ylabel(), fontsize=plt.rcParams['axes.labelsize'])
                ax.set_title(ax.get_title(), fontsize=plt.rcParams['axes.titlesize'])

                for line in ax.get_lines():
                    line.set_linewidth(plt.rcParams['lines.linewidth'])
                    line.set_markersize(plt.rcParams['lines.markersize'])

                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                ax.legend(handles, labels)
            # End mods

            fig.savefig(subdir / filepath.stem)
            fig.savefig(subdir / f"{filepath.stem}.png")
            mpl_file = subdir / filepath.name
            with open(mpl_file, 'wb') as f:
                pickle.dump(fig, f)


if __name__ == '__main__':
    # dirs_ = [
    #     '../docs/Figures/Continuous/SE',
    #     '../docs/Figures/Discrete/model_est',
    #     '../docs/Figures/Discrete/SE/consistency',
    #     '../docs/Figures/Discrete/SE/reg_func',
    #     '../docs/Figures/Discrete/SE/reg_var',
    #     '../docs/Figures/Discretization/SE/consistency',
    #     '../docs/Figures/Discretization/SE/reg_var',
    # ]
    # main(dirs_)

    import argparse

    parser = argparse.ArgumentParser(description='Update `matplotlib` figures')
    parser.add_argument('dirs', type=str, nargs='+', help='Directories in which to find MPL files')
    args = parser.parse_args()

    main(args.dirs)
