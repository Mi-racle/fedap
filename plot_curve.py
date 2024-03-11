import matplotlib.pyplot as plt
import numpy as np

SETTINGS = {
    'iid': {
        'accuracy': [
            ('results/iid-fedavg/accuracy_fedavg.txt', 'FedAvg', '^'),
            ('results/iid-fedacc/accuracy_fedacc.txt', 'FedAcc', 'o'),
        ],
        'loss': [
            ('results/iid-fedavg/loss_fedavg.txt', 'FedAvg', '^'),
            ('results/iid-fedacc/loss_fedacc.txt', 'FedAcc', 'o'),
        ],
    },
    'dirichlet': {
        'accuracy': [
            ('results/dirichlet-fedavg/accuracy_fedavg.txt', 'FedAvg', '^'),
            ('results/dirichlet-fedprox/accuracy_fedprox.txt', 'FedAcc', 'o'),
            ('results/dirichlet-fedap/accuracy_cosine.txt', 'FedCAP', 'x'),
        ],
        'loss': [
            ('results/dirichlet-fedavg/loss_fedavg.txt', 'FedAvg', '^'),
            ('results/dirichlet-fedprox/loss_fedprox.txt', 'FedAcc', 'o'),
            ('results/dirichlet-fedap/loss_cosine.txt', 'FedCAP', 'x'),
        ],
    },
    'label': {
        'accuracy': [
            ('results/label-fedavg/accuracy_fedavg.txt', 'FedAvg', '^'),
            ('results/label-fedprox/accuracy_fedprox.txt', 'FedAcc', 'o'),
            ('results/label-fedap/accuracy_cosine.txt', 'FedCAP', 'x'),
        ],
        'loss': [
            ('results/label-fedavg/loss_fedavg.txt', 'FedAvg', '^'),
            ('results/label-fedprox/loss_fedprox.txt', 'FedAcc', 'o'),
            ('results/label-fedap/loss_cosine.txt', 'FedCAP', 'x'),
        ]
    }
}

YLIMS = {
    'iid': {
        'accuracy': [.6, 1],
        'loss': [0, 50],
    },
    'dirichlet': {
        'accuracy': [.6, .9],
        'loss': [0, 1],
    },
    'label': {
        'accuracy': [.6, .8],
        'loss': [0, 2]
    }
}

OUTPUT_DIR = 'results/'


def read_data(path):
    with open(path, 'r') as f:
        raw_data = f.read()
        data = raw_data.split()

    return np.asfarray(data, float)


if __name__ == '__main__':

    for key in SETTINGS:

        metric_settings = SETTINGS[key]

        for metric_key in metric_settings:

            settings = metric_settings[metric_key]

            plt.figure()

            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.ylabel(metric_key)
            plt.xlabel('round')

            ylim = YLIMS[key][metric_key]
            plt.ylim(ylim[0], ylim[1])

            for setting in settings:
                ys = read_data(setting[0])
                xs = range(1, len(ys) + 1)

                plt.plot(xs, ys, color='black', marker=setting[2], markersize=4, linewidth=1, label=setting[1])

            plt.legend()
            plt.title(f'{metric_key} curve')
            # plt.show()
            plt.savefig(OUTPUT_DIR + f'{key}_{metric_key}_curve.png')
