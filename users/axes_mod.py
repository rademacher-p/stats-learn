from pathlib import Path
import pickle

from matplotlib import pyplot as plt

plt.style.use('../images/style.mplstyle')
plt.matplotlib.interactive(False)


dirs = [
    'C:/Users/paulg/Documents/PhD/Research/docs/Figures/Discrete/SE/reg_func',
]


def label_update(label):
    label = label.replace(r'\Dir', r'\mathrm{Dir}')
    if label == r'$f_{\Theta}(\theta)$':
        label = r'$f^*(\theta)$'
    return label


for dir_ in dirs:
    dir_ = Path(dir_)
    subdir = dir_ / 'new'
    subdir.mkdir()
    for filepath in dir_.glob('*.mpl'):
        filename = filepath.name

        with filepath.open('rb') as f:
            fig = pickle.load(f)

        for ax in fig.axes:
            ylabel = ax.get_ylabel()
            ylabel = ylabel.replace(r'\Rcal', r'\mathcal{R}')
            if ylabel == r'$\mathcal{R}_{\Theta}(f;\theta)$':
                ax.set_ylabel(r'$R(f;\theta)$')
            elif ylabel == r'$\mathcal{R}(f)$':
                ax.set_ylabel(r'$R_\uptheta(f)$')
            elif ylabel == r'$f(x)$':
                ax.set_ylabel(r'$y$')

            handles, labels = ax.get_legend_handles_labels()
            labels = list(map(label_update, labels))
            if labels[-1] == '$D$':
                handles = handles[-1:] + handles[:-1]
                labels = labels[-1:] + labels[:-1]
            ax.legend(handles, labels)

            title = ax.get_title()
            title.replace(r'\Dir', r'\mathrm{Dir}')
            ax.set_title(title)

            # TODO: title replace \Dir, ylabel replace bayes risk

        fig.savefig(subdir / f"{filepath.stem}.png")
        mpl_file = subdir / filename
        with open(mpl_file, 'wb') as f:
            pickle.dump(fig, f)

    # plt.show()
