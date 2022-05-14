import pickle
from pathlib import Path

from matplotlib import pyplot as plt

plt.style.use(["images/style.mplstyle"])
# plt.style.use(['images/style.mplstyle', 'images/double.mplstyle'])


def main(dirs):
    for dir_ in dirs:
        dir_ = Path(dir_)
        subdir = dir_ / "temp_dicts"
        subdir.mkdir()
        for filepath in dir_.glob("*.mpl"):
            with filepath.open("rb") as f:
                fig = pickle.load(f)

            # Modifications
            ax = fig.axes[0]
            a = dict(
                lines=ax.get_lines(),
                collections=ax.collections,
                xlabel=ax.get_xlabel(),
                ylabel=ax.get_ylabel(),
                title=ax.get_title(),
                # legend=ax.get_legend(),
                xlim=ax.get_xlim(),
                xscale=ax.get_xscale(),
                # xticks=ax.get_xticks(),
                # xticklabels=ax.get_xticklabels(),
                # xmajorticklabels=ax.get_xmajorticklabels(),
                # xminorticklabels=ax.get_xminorticklabels(),
                ylim=ax.get_ylim(),
                yscale=ax.get_yscale(),
                # yticks=ax.get_yticks(),
                # yticklabels=ax.get_yticklabels(),
                # ymajorticklabels=ax.get_ymajorticklabels(),
                # yminorticklabels=ax.get_yminorticklabels(),
            )

            for line in a["lines"]:
                line.remove()
            # a['legend'].remove()
            for c in a["collections"]:
                # c.remove()
                c.axes = None

            mpl_file = subdir / filepath.name
            with open(mpl_file, "wb") as f:
                pickle.dump(a, f)


if __name__ == "__main__":
    dirs_ = [
        # '../docs/Figures/Continuous/SE',
        # '../docs/Figures/Discrete/model_est',
        "../docs/Figures/Discrete/SE/consistency",
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
