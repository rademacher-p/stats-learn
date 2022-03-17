import pickle
from pathlib import Path

from matplotlib import pyplot as plt

plt.style.use(['images/style.mplstyle'])
# plt.style.use(['images/style.mplstyle', 'images/double.mplstyle'])
plt.matplotlib.interactive(False)


def main(dirs):
    for dir_ in dirs:
        dir_ = Path(dir_) / 'temp_dicts'
        subdir = dir_ / 'temp'
        subdir.mkdir()
        for filepath in dir_.glob('*.mpl'):
            with filepath.open('rb') as f:
                a = pickle.load(f)

            fig, ax = plt.subplots()

            for line in a['lines']:
                # ax.add_line(line)
                ax.plot(*line.get_data(), color=line.get_color(), linestyle=line.get_linestyle(),
                        marker=line.get_marker(), label=line.get_label())

            for c in a['collections']:
                ax.add_collection(c)

            ax.set_title(a['title'])
            ax.set_xlabel(a['xlabel'])
            ax.set_ylabel(a['ylabel'])
            ax.set_xlim(a['xlim'])
            ax.set_ylim(a['ylim'])
            ax.set_xscale(a['xscale'])
            ax.set_yscale(a['yscale'])
            ax.legend()


            fig.savefig(subdir / filepath.stem)
            # fig.savefig(subdir / f"{filepath.stem}.png")
            mpl_file = subdir / filepath.name
            with open(mpl_file, 'wb') as f:
                pickle.dump(fig, f)


if __name__ == '__main__':
    dirs_ = [
        # '../docs/Figures/Continuous/SE',
        # '../docs/Figures/Discrete/model_est',
        '../docs/Figures/Discrete/SE/consistency',
        # '../docs/Figures/Discrete/SE/reg_func',
        # '../docs/Figures/Discrete/SE/reg_var',
        # '../docs/Figures/Discretization/SE/consistency',
        # '../docs/Figures/Discretization/SE/reg_var',
    ]
    main(dirs_)

    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Update `matplotlib` figures')
    # parser.add_argument('dirs', type=str, nargs='+', help='Directories in which to find MPL files')
    # args = parser.parse_args()
    #
    # main(args.dirs)
