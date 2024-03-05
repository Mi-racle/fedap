import matplotlib.pyplot as plt
import numpy as np


SETTINGS = [
    ('/home/miracle/fedyolo/global_map_noniid_avg30.txt', 'FedAvg', '^', '-'),
    ('/home/miracle/fedyolo/global_map_noniid_avg30_es.txt', 'FedAvg with early stopping', '^', ':'),
    ('/home/miracle/fedyolo/global_map_noniid_prox30.txt', 'FedProx', 'o', '-'),
    ('/home/miracle/fedyolo/global_map_noniid_prox30_es.txt', 'FedProx with early stopping', 'o', ':'),
    ('/home/miracle/fedyolo/global_map_noniid_mavg30.txt', 'FedAvg-mAP', 'x', '-'),
    ('/home/miracle/fedyolo/global_map_noniid_mavg30_es.txt', 'FedAvg-mAP with early stopping', 'x', ':'),
]


def read_data(path):

    with open(path, 'r') as f:

        raw_data = f.read()
        data = raw_data.split()

    return np.asfarray(data, float)


if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel('mAP')
    plt.xlabel('round')

    plt.ylim(0.9, 1.0)

    for setting in SETTINGS:

        ys = read_data(setting[0])
        xs = range(1, len(ys) + 1)

        plt.plot(xs, ys, color='black', marker=setting[2], markersize=4, linewidth=1, linestyle=setting[3], label=setting[1])

        print(ys.max())

    plt.legend()
    plt.title('mAP curve')
    # plt.show()
    plt.savefig('mapcurve.png')
