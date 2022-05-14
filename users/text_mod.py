import pickle
from pathlib import Path

from matplotlib import pyplot as plt

plt.style.use("images/style.mplstyle")


def main(dirs):
    for dir_ in dirs:
        dir_ = Path(dir_)
        subdir = dir_ / "temp"
        subdir.mkdir()
        for filepath in dir_.glob("*.mpl"):
            with filepath.open("rb") as f:
                fig = pickle.load(f)

            # Modifications
            for ax in fig.axes:
                ylabel = ax.get_ylabel()
                ylabel = ylabel.replace(r"\Rcal", r"\mathcal{R}")
                if ylabel == r"$\mathcal{R}_{\Theta}(f;\theta)$":
                    ax.set_ylabel(r"$R(f;\theta)$")
                elif ylabel == r"$\mathcal{R}(f)$":
                    ax.set_ylabel(r"$R_\uptheta(f)$")
                elif ylabel == r"$f(x)$":
                    ax.set_ylabel(r"$y$")

                if ylabel == r"$R(f;\theta)$":
                    ax.set_ylabel(r"$R(f;\rho)$")

                for line in ax.get_lines():
                    label = label_update(line.get_label())
                    line.set_label(label)

                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                ax.legend(handles, labels)

                title = ax.get_title()
                title.replace(r"\Dir", r"\mathrm{Dir}")
                ax.set_title(title)
            # End mods

            fig.savefig(subdir / filepath.stem)
            fig.savefig(subdir / f"{filepath.stem}.png")
            mpl_file = subdir / filepath.name
            with open(mpl_file, "wb") as f:
                pickle.dump(fig, f)


def label_update(label):
    label = label.replace(r"\Dir", r"\mathrm{Dir}")
    if label == r"$f_{\Theta}(\theta)$":
        label = r"$f^*(\theta)$"
    if label == r"$f^*(\theta)$":
        label = r"$f^*(\rho)$"

    return label


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser(description="Update `matplotlib` figures")
    parser.add_argument(
        "dirs", type=str, nargs="+", help="Directories in which to find MPL files"
    )
    args = parser.parse_args()

    main(args.dirs)
